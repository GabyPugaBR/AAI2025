#simple sample with LangChain + LangGraph, OpenAI for LLM + embeddings, “knowledge base” is just a few uploaded PDFs (policies, runbooks, etc.), 4 agents: Intake, Knowledge (RAG), Workflow, Escalation
# use of chroma https://docs.trychroma.com/docs/overview/getting-started and public mcp server https://www.mcpserverfinder.com/servers/sirmews/mcp-pinecone

# step1 setup and ingest PDFs
# pip install langchain langchain-openai langchain-community chromadb pypdf

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter

from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def build_pdf_vectorstore(pdf_paths: List[str], persist_dir: str = "./chroma_pdf_kb") -> Chroma:
    """
    Load a few PDF files, chunk them, embed with OpenAI, and store in Chroma.
    """
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        # Attach some useful metadata
        for p in pages:
            p.metadata.setdefault("source", "pdf")
            p.metadata.setdefault("file_name", os.path.basename(path))
        docs.extend(pages)

    split_docs = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    return vectorstore

# Example usage:
pdf_paths = [
    "policies/vpn_policy.pdf",
    "runbooks/password_resets.pdf",
    "howto/onboarding_it_guide.pdf",
]
chroma = build_pdf_vectorstore(pdf_paths)
kb_retriever = chroma.as_retriever(search_kwargs={"k": 5})


# Pydantic GraphState, GraphState.retrieved_docs will hold chunks coming from PDFs.

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class RetrievedDoc(BaseModel):
    id: str
    source: str                     # "confluence", "kb", "itsm", etc.
    title: str
    snippet: str
    score: float
    url: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class WorkflowAction(BaseModel):
    action_type: Literal[
        "RESET_PASSWORD",
        "UNLOCK_ACCOUNT",
        "GET_LOGS",
        "OPEN_TICKET",
        "CUSTOM"
    ]
    system: Optional[str] = None     # "VPN", "Email", etc.
    parameters: Dict[str, str] = Field(default_factory=dict)
    status: Literal["PENDING", "RUNNING", "SUCCESS", "FAILED"] = "PENDING"
    result_summary: Optional[str] = None
    error_message: Optional[str] = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    payload: Dict = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Intent(BaseModel):
    name: Optional[str] = None       # "reset_password", "vpn_issue", etc.
    confidence: float = 0.0
    request_type: Literal[
        "informational",
        "action",
        "incident",
        "unknown"
    ] = "unknown"
    system: Optional[str] = None
    severity: Optional[Literal["low", "medium", "high"]] = None


class GraphState(BaseModel):
    # conversation messages for LangGraph state
    messages: List[Dict] = Field(default_factory=list)
    # each item: {"role": "user"/"assistant"/"system", "content": "..."}

    # user/session
    user_id: Optional[str] = None
    user_department: Optional[str] = None
    user_roles: List[str] = Field(default_factory=list)

    # intake
    intent: Intent = Field(default_factory=Intent)
    entities: Dict[str, str] = Field(default_factory=dict)

    # RAG
    rag_query: Optional[str] = None
    retrieved_docs: List[RetrievedDoc] = Field(default_factory=list)
    knowledge_answer: Optional[str] = None

    # workflow
    workflow_actions: List[WorkflowAction] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)

    # escalation
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    escalation_ticket_id: Optional[str] = None
    escalation_summary: Optional[str] = None

    # final user-facing answer
    final_answer: Optional[str] = None

# intake agent node

from pydantic import BaseModel
from typing import Any, Dict, Literal

class IntakeSchema(BaseModel):
    intent_name: str
    request_type: Literal["informational", "action", "incident", "unknown"]
    system: str
    severity: Literal["low", "medium", "high", "unknown"]
    entities: Dict[str, Any]
    confidence: float

def intake_node(state: GraphState) -> GraphState:
    last_user_msg = next(
        (m for m in reversed(state.messages) if m["role"] == "user"),
        None,
    )
    if not last_user_msg:
        return state

    prompt = f"""
You are an IT support intake agent.

Task:
1. Determine the intent name (e.g. "reset_password", "vpn_issue", "how_to_request_access").
2. Determine request_type: ["informational", "action", "incident", "unknown"].
3. Determine system (e.g. "VPN", "Email", "SSO", "Laptop", "Unknown").
4. Extract entities like username, device, application, times, etc.
5. Estimate severity: "low", "medium", "high" (or "unknown").
6. Provide a confidence (0-1).

User message:
{last_user_msg["content"]}
"""

    structured_llm = llm.with_structured_output(IntakeSchema)
    result: IntakeSchema = structured_llm.invoke(prompt)

    state.intent = Intent(
        name=result.intent_name,
        confidence=result.confidence,
        request_type=result.request_type,
        system=None if result.system.lower() == "unknown" else result.system,
        severity=None if result.severity == "unknown" else result.severity,
    )
    state.entities = result.entities or {}
    return state

# knowledge agent node, RAG over PDFs

from langchain_core.prompts import ChatPromptTemplate

knowledge_prompt = ChatPromptTemplate.from_template("""
You are an IT knowledge agent answering user questions using internal PDF documentation.

User question:
{question}

Context from documents:
{context}

Instructions:
- Answer based ONLY on the provided context.
- If multiple procedures exist, pick the one most relevant to the user's department and system.
- If you are not sure, say so and suggest next steps.
- Provide a clear, concise answer; include a short numbered procedure if appropriate.
""")

def knowledge_node(state: GraphState) -> GraphState:
    # Only run for informational/incident types
    if state.intent.request_type not in ["informational", "incident"]:
        return state

    last_user_msg = next(
        (m for m in reversed(state.messages) if m["role"] == "user"),
        None,
    )
    if not last_user_msg:
        return state

    system_part = f" (system: {state.intent.system})" if state.intent.system else ""
    rag_query = last_user_msg["content"] + system_part
    state.rag_query = rag_query

    # ---- Retrieve from your uploaded PDFs ----
    docs = kb_retriever.invoke(rag_query)

    state.retrieved_docs = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        state.retrieved_docs.append(
            RetrievedDoc(
                id=str(meta.get("id", f"pdf-{i}")),
                source=meta.get("source", "pdf"),
                title=meta.get("file_name", "PDF Document"),
                snippet=d.page_content[:400],
                score=meta.get("score", 0.0),
                url=None,   # local PDFs usually don’t have URLs
                metadata=meta,
            )
        )

    context_text = "\n\n".join(
        f"[Doc {i+1} - {doc.title}]\n{doc.snippet}"
        for i, doc in enumerate(state.retrieved_docs)
    )

    chain = knowledge_prompt | llm
    answer_msg = chain.invoke(
        {"question": last_user_msg["content"], "context": context_text}
    )

    state.knowledge_answer = answer_msg.content

    # If purely informational, we can answer directly
    if state.intent.request_type == "informational":
        state.final_answer = state.knowledge_answer

    return state

# workflow agent node, LangChain + MCP tools over Pinecone/IDP/ITSM

# helper 
def plan_workflow_actions(state: GraphState) -> List[WorkflowAction]:
    intent = state.intent

    if intent.request_type == "action" and intent.name == "reset_password":
        return [
            WorkflowAction(
                action_type="RESET_PASSWORD",
                system=intent.system or "Unknown",
                parameters={"user_id": state.user_id},
            ),
            WorkflowAction(
                action_type="OPEN_TICKET",
                system=intent.system or "Unknown",
                parameters={"reason": "Password reset performed automatically"},
            ),
        ]

    if intent.request_type == "incident":
        return [
            WorkflowAction(
                action_type="GET_LOGS",
                system=intent.system or "Unknown",
                parameters={
                    "service": intent.system or "VPN",
                    "timeframe": "24h",
                    "user_id": state.user_id,
                },
            )
        ]

    return []

# implementation

def workflow_node(state: GraphState) -> GraphState:
    """
    LangGraph node: Workflow Agent
    Uses mcp_client to call tools (idp.reset_password, logs.get_recent, itsm.create_ticket, etc.).
    """

    # Plan only once
    if not state.workflow_actions:
        state.workflow_actions = plan_workflow_actions(state)

    for action in state.workflow_actions:
        if action.status not in ["PENDING", "RUNNING"]:
            continue

        action.status = "RUNNING"
        tool_name = "unknown"

        try:
            if action.action_type == "RESET_PASSWORD":
                tool_name = "idp.reset_password"
                payload = mcp_client.call(
                    tool_name,
                    user_id=action.parameters["user_id"],
                    system=action.system,
                )

            elif action.action_type == "GET_LOGS":
                tool_name = "logs.get_recent"
                payload = mcp_client.call(
                    tool_name,
                    service=action.parameters["service"],
                    timeframe=action.parameters["timeframe"],
                    user_id=action.parameters["user_id"],
                )

            elif action.action_type == "OPEN_TICKET":
                tool_name = "itsm.create_ticket"
                payload = mcp_client.call(
                    tool_name,
                    summary=f"Auto action for {action.system}",
                    description=action.parameters.get("reason", "Automated action"),
                    user_id=state.user_id,
                    severity=state.intent.severity or "low",
                )

            else:
                payload = {"error": f"Unsupported action_type: {action.action_type}"}
                raise ValueError(payload["error"])

            success = not payload.get("error")
            action.status = "SUCCESS" if success else "FAILED"
            action.result_summary = payload.get("summary") or str(payload)

            state.tool_results.append(
                ToolResult(
                    tool_name=tool_name,
                    success=success,
                    payload=payload,
                    error=payload.get("error"),
                )
            )

            if tool_name.startswith("itsm.") and payload.get("ticket_id"):
                state.escalation_ticket_id = payload["ticket_id"]

        except Exception as e:
            action.status = "FAILED"
            action.error_message = str(e)
            state.tool_results.append(
                ToolResult(
                    tool_name=tool_name,
                    success=False,
                    payload={},
                    error=str(e),
                )
            )

    # If all actions succeeded → build final_answer
    if state.workflow_actions and all(a.status == "SUCCESS" for a in state.workflow_actions):
        summary_lines = [
            f"- {a.action_type} on {a.system}: {a.result_summary}"
            for a in state.workflow_actions
        ]
        ticket_part = (
            f"\nRelated ticket: {state.escalation_ticket_id}"
            if state.escalation_ticket_id
            else ""
        )
        state.final_answer = (
            "I've completed the requested actions:\n" +
            "\n".join(summary_lines) +
            ticket_part
        )
    # If any failed, mark for escalation
    elif any(a.status == "FAILED" for a in state.workflow_actions):
        state.escalation_required = True
        state.escalation_reason = "One or more workflow actions failed."

    return state

# Escalation Agent Node, LangChain + MCP ITSM tool

from langchain_core.prompts import ChatPromptTemplate

escalation_prompt = ChatPromptTemplate.from_template("""
You are an IT escalation agent.

Summarize the situation for a human IT engineer. Include:
- User problem description
- Steps already taken (by user and by automation)
- Any diagnostic data or logs that seem relevant
- Your best hypothesis if applicable
- Suggested next steps for the human engineer.

Conversation (last turns):
{conversation}

Tool results:
{tool_results}
""")

def escalation_node(state: GraphState) -> GraphState:
    """LangGraph node: Escalation Agent."""

    should_escalate = (
        state.escalation_required
        or (state.intent.severity == "high")
        or (state.intent.request_type == "incident" and state.intent.confidence < 0.6)
    )

    if not should_escalate:
        return state

    convo_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in state.messages[-10:]
    )

    tool_summaries = "\n".join(
        f"{i+1}. {tr.tool_name} (success={tr.success}) -> {tr.payload}"
        for i, tr in enumerate(state.tool_results)
    )

    chain = escalation_prompt | escalation_llm
    summary_msg = chain.invoke(
        {"conversation": convo_text, "tool_results": tool_summaries}
    )

    summary = summary_msg.content
    state.escalation_summary = summary

    payload = mcp_client.call(
        "itsm.create_incident",
        summary=f"AI Escalation: {state.intent.name or 'IT Issue'}",
        description=summary,
        user_id=state.user_id,
        severity=state.intent.severity or "medium",
    )

    ticket_id = payload.get("incident_id") or payload.get("ticket_id")
    state.escalation_ticket_id = ticket_id

    state.tool_results.append(
        ToolResult(
            tool_name="itsm.create_incident",
            success=not payload.get("error"),
            payload=payload,
            error=payload.get("error"),
        )
    )

    state.final_answer = (
        f"I’ve escalated this issue to our IT team for further investigation.\n\n"
        f"Reference: {ticket_id or 'pending'}\n\n"
        "They’ll review the diagnostics I’ve attached and follow up with you."
    )
    state.escalation_required = True

    return state

# langgraph orchestration

from langgraph.graph import StateGraph, END

graph = StateGraph(GraphState)

graph.add_node("intake", intake_node)
graph.add_node("knowledge", knowledge_node)
graph.add_node("workflow", workflow_node)
graph.add_node("escalation", escalation_node)

graph.set_entry_point("intake")

def route_from_intake(state: GraphState) -> str:
    if state.intent.request_type == "informational":
        return "knowledge"
    if state.intent.request_type in ["action", "incident"]:
        return "workflow"
    return "escalation"

graph.add_conditional_edges("intake", route_from_intake)

def route_after_knowledge(state: GraphState) -> str:
    if not state.knowledge_answer:
        return "escalation"
    return END

graph.add_conditional_edges("knowledge", route_after_knowledge)

def route_after_workflow(state: GraphState) -> str:
    if state.escalation_required:
        return "escalation"
    return END

graph.add_conditional_edges("workflow", route_after_workflow)

graph.add_edge("escalation", END)

app = graph.compile()
