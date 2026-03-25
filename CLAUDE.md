# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Ad Optimization Agent** workshop — a Jupyter notebook demonstrating how to build AI agents for marketing automation using LangGraph and LangChain. The agent reads real campaign data from a CSV, allocates daily budget across channels, and logs optimization decisions with rationale. It is an educational project, not a production application.

## Setup & Running

**Install dependencies:**
```bash
pip install langgraph langchain langchain_openai pydantic pandas
```

**Required environment variable:**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Run:** Open and execute `Hands_on_Ad_Agent.ipynb` sequentially. The notebook is designed for Google Colab but works in any Jupyter environment.

## Dataset

**File:** `marketing_campaign_dataset.csv`

**Columns:** Campaign_ID, Company, Campaign_Type, Target_Audience, Duration, Channel_Used, Conversion_Rate, Acquisition_Cost, ROI, Location, Language, Clicks, Impressions, Engagement_Score, Customer_Segment, Date

**All channels in CSV:** Google Ads, YouTube, Instagram, Facebook, Website, Email

**Agent scope:** Only the four **paid channels** are used — Google Ads, Facebook, Instagram, YouTube. Website and Email are excluded.

**Date filter:** Agent uses only **Q4 data (Oct, Nov, Dec 2021)** — ~50k rows across the four paid channels.

**Aggregated metrics per channel:** avg Conversion_Rate (CVR), avg ROI, avg Acquisition_Cost (CPA), CTR = total_clicks / total_impressions.

## Architecture

The notebook implements a **stateful agentic loop** using LangGraph:

```
load_csv_data → agent_reasoning ⟶ execute_tools → agent_reasoning → END
                                ↘ END (direct)
```

### Core Components

- **`AgentState`** — TypedDict holding `messages` (conversation history), `channel_data` (aggregated Q4 per-channel metrics), `channel_allocations` (current % per channel), and `next_action` (routing decision).

- **LLM** — `ChatOpenAI("gpt-4o-mini", temperature=0)` bound with tools for deterministic reasoning.

- **Tools** (action nodes the LLM can invoke):
  - `allocate_budget(channel, new_pct, reason)` — shifts budget percentage to a channel (enforces ±20% cap and minimum floor guardrails)
  - `pause_channel(channel, reason)` — flags a channel as paused for underperformance (never >2 consecutive days)
  - `request_new_creatives(channel, reason)` — requests new creative assets for a channel

- **Graph Nodes**:
  - `load_csv_data` — reads `marketing_campaign_dataset.csv`, aggregates metrics by channel (avg Conversion_Rate, avg ROI, CTR = Clicks/Impressions), formats as analysis prompt
  - `agent_reasoning` — LLM analyzes channel performance and selects a tool or decides to end
  - `execute_tools` — runs the tool chosen by the LLM
  - `decide_next_step` — conditional router: routes to `execute_tools` or `END`

### Budget Allocation Rules (Guardrails)
- **Daily budget:** $1,000 total across the four paid channels
- **Starting split:** equal (25% = $250/day per channel)
- **Daily cap:** ±20% change max per channel per day (auto-enforced in `allocate_budget`)
- **Minimum floor:** 5% per channel — never 0% (auto-enforced in `allocate_budget`)
- **Pause limit:** no channel paused for >2 consecutive days (enforced in `pause_channel`)
- **Log:** every action appended to `BUDGET_STATE['log']` with `from_pct`, `to_pct`, `daily_amount`, `reason`, and any guardrail notes

### BUDGET_STATE (global mutable state)
```python
BUDGET_STATE = {
    'allocations': {},   # {channel: float pct} — sums to 100%
    'log': [],           # ordered list of all decisions
    'paused_days': {}    # {channel: consecutive_days_paused}
}
```
Shared across all tool calls in one agent run. `initialize_budget()` resets it before each run.

### Optimization Metric
- Primary: Conversion Rate (CVR = Conversion_Rate column)
- Secondary: ROI
- Evaluation: total conversions, average CPA (Acquisition_Cost), average ROI logged after each run

### Data Flow

`load_csv_data` reads and aggregates the CSV by channel. The LLM receives per-channel performance summaries, reasons about which channel is over/underperforming, selects a budget allocation tool, executes it, then produces a final summary with logged rationale.

## Key Files

| File | Purpose |
|------|---------|
| `Hands_on_Ad_Agent.ipynb` | Main notebook — all code, explanations, and agent implementation |
| `marketing_campaign_dataset.csv` | Real campaign data: channels, conversions, ROI, spend metrics |
| `Agentic AI in Marketing and Advertising.pdf` | Background theory and concepts |
| `Ad Optimization Agent Grading Rubric and Submission Checklist.docx` | Workshop assessment criteria |
| `assignment.md` | Assignment brief: deliverables, grading, submission requirements |
