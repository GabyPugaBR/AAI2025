# Ad Optimization Agent — Design Document

**Project:** Agentic AI in Marketing and Advertising
**Scope:** Automated daily budget allocation across three paid ad channels using LangGraph + GPT-4o-mini
**Data:** June 2021 campaign data — Instagram, Google Ads, YouTube (~8,200 rows)

---

## What This Agent Does

Instead of a marketing analyst manually reviewing spreadsheets each day, this agent reads campaign performance data, reasons about which channels are over- or underperforming, and reallocates the daily $1,000 ad budget accordingly — all in a single automated pass. Every decision is logged with a written rationale so the team can audit the outcome.

Each notebook run resets to a clean 34/33/33 baseline (Google Ads / Instagram / YouTube), then runs the agent once through a stateful LangGraph loop:

```
load_csv_data → agent_reasoning ⟶ execute_tools → agent_reasoning → END
                                ↘ END (direct)
```

The loop repeats — reasoning, acting, re-evaluating — until the agent decides nothing more needs to change, then produces a final written summary. A hard limit of 25 iterations prevents runaway loops.

---

## Optimization Goal

**Primary goal: maximize Conversion Rate (CVR)**
The agent ranks channels by average CVR and shifts budget toward whichever channel converts the highest share of impressions into actual outcomes (purchases, sign-ups, etc.).

**Secondary goal: maximize ROI**
When CVR differences are minor, the agent uses ROI as a tiebreaker to ensure each dollar of budget generates the most measurable return.

**Tertiary signal: CTR (Click-Through Rate)**
CTR = total Clicks ÷ total Impressions per channel. Used as a leading indicator of ad creative effectiveness. A low CTR channel may trigger a `request_new_creatives` action even if CVR is acceptable.

---

## Inputs

| Input | Description |
|---|---|
| `ad_data_june.csv` | Pre-filtered June 2021 campaign data; 3 paid channels; ~8,200 rows |
| Columns used | `Campaign_ID`, `Target_Audience`, `Duration`, `Channel_Used`, `Conversion_Rate`, `ROI`, `Clicks`, `Impressions`, `Date` |
| Aggregated per channel | avg `Conversion_Rate` (CVR), avg `ROI`, CTR = sum(Clicks) ÷ sum(Impressions) |
| Starting allocations | 34% Google Ads / 33% Instagram / 33% YouTube (reset every run) |
| LLM | `gpt-4o-mini` at `temperature=0` for deterministic, auditable decisions |

---

## Outputs

| Output | Description |
|---|---|
| Revised budget split | Final % allocation per channel after agent actions (e.g., Instagram 53%, YouTube 13%) |
| Decision log | Ordered list of every action taken — channel, from %, to %, dollar amount, reason, guardrail notes |
| Performance snapshot | Side-by-side table of each channel's CVR, ROI, CTR, and final budget share |
| Written summary | Agent-generated narrative explaining what it did and why |

The agent can take three types of actions per run:

- **`allocate_budget(channel, new_pct, reason)`** — shifts budget percentage to a channel
- **`pause_channel(channel, reason)`** — temporarily halts spend on an underperformer
- **`request_new_creatives(channel, reason)`** — flags that a channel needs fresh ad creative

---

## Guardrails

All guardrails are auto-enforced in code — the agent cannot override them regardless of its reasoning.

| Rule | Value | Purpose |
|---|---|---|
| Daily budget | $1,000 fixed total | Prevents overspend |
| Starting split | 34% / 33% / 33% | Resets bias each run |
| Max change per run | ±20% per channel | Prevents sudden extreme moves |
| Minimum floor | 5% per channel (never 0%) | Ensures all channels retain test coverage |
| Max consecutive pause | 2 days per channel | Prevents a channel from being abandoned |

Any allocation that would breach a guardrail is automatically clamped to the nearest valid value, and the adjustment is noted in the log (e.g., `"capped at +20%"`).

---

## Evaluation Metric

**Primary:** Improvement in CVR-weighted budget allocation vs. the equal-split baseline (33.33% each)

A successful run is one where the agent has meaningfully shifted budget toward the highest-CVR channel — demonstrable by comparing the final split against the 33.33% equal-split baseline. The larger the shift toward the top-performing channel, the more the agent has acted on the data signal rather than defaulting to a passive even split.

**Supporting metrics logged per run:**

- Per-channel CVR, ROI, CTR at time of decision
- `from_pct` → `to_pct` delta per channel
- Dollar amounts reallocated
- Any guardrail interventions triggered

---

## Data Note

`ad_data_june.csv` was synthetically updated with the assistance of Claude (Anthropic) to introduce distinct per-channel KPIs. The original Kaggle source produced near-identical metrics across all channels (< 0.2pp CVR spread), making optimization decisions unobservable. The updated file preserves the original schema and value ranges while reflecting a realistic scenario with clear performance differentiation:

| Channel | CVR | ROI | CTR |
|---|---|---|---|
| Instagram | 10.81% | 6.13 | 12.88% |
| Google Ads | 8.06% | 4.97 | 9.79% |
| YouTube | 4.41% | 3.25 | 5.96% |
