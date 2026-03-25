---
name: Ad Optimization Agent project decisions
description: Key design decisions and customizations made to Hands_on_Ad_Agent.ipynb for the workshop assignment
type: project
---

The notebook was fully rewritten from the generic template to match these agreed decisions:

**Daily budget:** $1,000 allocated across paid channels only (Google Ads, Facebook, Instagram, YouTube). Website and Email excluded.

**Data:** Real data from marketing_campaign_dataset.csv, filtered to Q4 only (Oct, Nov, Dec 2021).

**Starting state:** Equal split — 25% = $250/day per channel. Stored in global `BUDGET_STATE`.

**Guardrails (all auto-enforced in tools):**
- ±20% daily change cap per channel
- 5% minimum floor — never 0%
- Max 2 consecutive days paused per channel

**Allocation log:** Every tool call appends to `BUDGET_STATE['log']` with from_pct, to_pct, daily_amount, reason, and any guardrail notes triggered. Log printed at the end of each run.

**Why:** Workshop assignment requires logged rationale, guardrails, and real CSV data. Assignment.md specifies CVR as primary metric, ROI secondary.

**How to apply:** When continuing work on this notebook, these decisions are fixed. Don't revert to simulated data or remove the guardrail logic. The global BUDGET_STATE pattern is intentional for notebook simplicity.
