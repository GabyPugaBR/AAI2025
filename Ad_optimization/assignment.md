Build an Ad Optimization Agent
Goal
Prototype a minimal agent that allocates daily budget across channels (e.g., Search, Social, Display) to maximize conversions (or CTRLinks to an external site.) using simple reasoning + a lightweight learning loop. You’ll implement:

(1) a system prompt,

(2) tooling for reading performance data and deciding budget shifts,

(3) an evaluation routine on a small dataset.

What you’ll deliver
Design doc (≤1 page): agent role, inputs/outputs, rules/guardrails (privacy, brand tone), evaluation metric.

Working script/notebook: reads CSV, proposes budget per channel, logs rationale.

Short readme: how to run, assumptions, results snapshot.

Scaling note (≤½ page): what changes to run on cloud reliably (observability, retries, cost caps).

Data
Create a simple CSV (or get datesets online) with columns like:

date,channel,spend,impressions,clicks,conversions
Use 14–30 days of mock data for 3 channels.

Baseline approach (suggestions)
Heuristic + explore/exploit: Start with an equal split; daily, compute each channel’s conversion rate or CTR; shift +10–20% budget toward top performer while keeping a minimum floor for others (to keep learning).

Guardrails: cap per-day changes (±20%), never allocate 0% to a channel for >2 days; log each decision and reason string (eg, Social up by 10% due to higher CVR, stable CPALinks to an external site.).

Evaluation: track total conversions (or clicks), average CPA, etc.

Group sign-up and meeting summary template are in Google Drive(118s- section01)


Ad Optimization Agent Grading Rubric and Submission Checklist.docx Download Ad Optimization Agent Grading Rubric and Submission Checklist.docx


Submission:

1. The GitHub link and the successful outputs of your code, or a workflow and successful run of your workflow in tools such as n8n.

2. A 2-3 minute presentation with slides.