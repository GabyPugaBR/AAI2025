# Ad Optimization Agent

## How to Run

1. Install dependencies:
pip install langgraph langchain langchain_openai pydantic pandas python-dotenv

2. Set your API key:
export OPENAI_API_KEY="your-key-here"

3. Place `marketing_campaign_dataset.csv` in the same directory as the notebook.

4. Open and run `Hands_on_Ad_Agent.ipynb` top to bottom.

## Assumptions
- Agent uses Q4 2021 data only (Oct–Dec), ~33,000 rows across 4 paid channels
- Website and Email channels are excluded — paid channels only
- Budget guardrails (±20% cap, 5% floor, 2-day pause max) are auto-enforced in tools
- LLM used: gpt-4o-mini at temperature=0 for deterministic decisions

## Results Snapshot (Q4 2021 Run)
| Channel | CVR | Final Budget |
|---|---|---|
| Facebook | 8.03% | 26.7% ($267/day) |
| Google Ads | 8.02% | 26.7% ($267/day) |
| YouTube | 8.01% | 26.7% ($267/day) |
| Instagram | 7.99% | 20.0% ($200/day) |

Agent actions taken: Facebook and Google Ads budget increased, 
Instagram reduced + new creatives requested, YouTube paused 
(1 day) for lowest CVR.