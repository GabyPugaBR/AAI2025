The Big Picture
Imagine you have $1,000 to spend every day across four advertising channels: Google Ads, Facebook, Instagram, and YouTube. Instead of a human marketing analyst manually reviewing spreadsheets and making decisions, this notebook builds an AI assistant that does it automatically — reading real data, thinking through the numbers, and deciding where to put the money.

Step-by-Step Walkthrough
Step 1 — Install the Tools
The first cell installs the software libraries this project needs. Think of it like downloading apps before you can use them. Nothing business-critical here, just setup.

Step 2 — Setting Up the Agent's "Brain" and Memory
This is where the AI is configured:

The AI model used is GPT-4o-mini — a fast, cost-efficient version of OpenAI's model. Setting temperature=0 means it behaves consistently and predictably, not creatively.
AgentState is essentially the agent's short-term memory — it tracks the conversation history, what the ad data looks like, and how the budget is currently split.
BUDGET_STATE is a live scoreboard — it tracks where money is being allocated right now, a log of every decision made, and how many days each channel has been paused.
The budget always starts at 25% per channel ($250/day each) and resets before every run.
Step 3 — The Three Actions the AI Can Take
The AI isn't just talking — it can actually do three things:

Shift the Budget (allocate_budget) — moves money toward a better-performing channel. Built-in guardrails prevent it from being too aggressive:

Can only change a channel by ±20% per day (so no reckless swings)
Every channel must keep at least 5% — nothing can be cut to zero
Pause a Channel (pause_channel) — temporarily stops spending on a channel that's underperforming. A guardrail prevents pausing any channel for more than 2 days in a row, so the AI can't completely abandon a channel indefinitely.

Request New Ad Creatives (request_new_creatives) — flags that a channel needs fresh ad designs. This doesn't move money, but it logs the recommendation so the creative team knows.

Every action is recorded in a log with the reason why it was taken.

Step 4 — The Three "Departments" That Run the Agent
Think of this as three employees working in a loop:

The Data Analyst (load_csv_data) — reads the CSV file with real Q4 2021 campaign data (~50,000 rows), filters it to just the four paid channels, and calculates four key numbers per channel:

CVR (Conversion Rate) — how often an ad click turns into a customer action
ROI — return on investment
CPA (Cost Per Acquisition) — how much it costs to get one customer
CTR (Click-Through Rate) — how often people click the ad at all
It then hands a clean performance summary to the AI.

The Strategist (agent_reasoning) — this is the AI thinking step. It reads the performance data and decides: "Should I move money somewhere? Pause something? Ask for new creatives? Or am I done?" It either picks a tool to use or writes a final summary.

The Executor (execute_tools) — actually carries out whatever the Strategist decided. After executing, it hands the result back to the Strategist to review.

Step 5 — Connecting It All Into a Loop
This is where the three departments are wired together:


Read Data → AI Thinks → Take Action → AI Thinks Again → Take Action → ... → Final Summary
The AI keeps looping — taking actions and reconsidering — until it decides there's nothing more to do. A safety limit prevents it from looping more than 25 times.

Step 6 — Running the Agent
This kicks everything off. The agent:

Resets the budget to 25/25/25/25
Reads the data
Analyzes and acts
Prints a final written summary of what it decided and why
Step 7 — The Report Card
After the run, this prints three things:

Final budget split — e.g., "Google Ads now gets 40%, Instagram gets 15%"
Full decision log — every action taken, with the reasoning and any guardrail notes (e.g., "capped at +20%")
Performance snapshot — side-by-side view of each channel's metrics and its final budget share, so you can judge whether the AI made sensible calls
Why This Matters
Rather than a marketing manager spending hours in spreadsheets, this agent can read weeks of campaign data in seconds and make budget decisions based on objective performance metrics — with automatic guardrails to prevent extreme or reckless moves. Every decision is logged with a reason, so the team can review and audit what happened.