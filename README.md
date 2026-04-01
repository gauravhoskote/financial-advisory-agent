# Multi-Agent Financial Advisory System

Three agents collaborate to deliver personalised investment advice using LangChain's `create_agent` and LangGraph.

## Architecture

```
                  ┌──────────────────────────────┐
 Client Agent ◄──►│  Advisor Agent               │
 (simulated)      │  ┌────────────────────────┐  │
                  │  │ research tool           │  │
                  │  │  └─► Analyst Agent      │  │
                  │  │       (subagent)        │  │
                  │  │       ├─ search_web            │  │
                  │  │       ├─ search_knowledge_base │  │
                  │  │       ├─ get_fred_data         │  │
                  │  │       └─ search_fred           │  │
                  │  └────────────────────────┘  │
                  └──────────────────────────────┘
```

**Pattern:** LangChain [Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) — the Analyst is a standalone agent wrapped as a tool the Advisor calls. The Client ↔ Advisor loop is a [Custom Workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow) (2-node LangGraph).

**Channel separation is structural:** The Analyst only receives the research query. It never sees the conversation. The Client only sees the Advisor's final responses. Internal tool calls (research, conclude) are filtered out.

## Setup

```bash
cd financial-advisory-agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# Edit .env and fill in your credentials:
# AWS credentials (via env vars, ~/.aws/credentials, or IAM role)
# TAVILY_API_KEY    — https://app.tavily.com
# FRED_API_KEY      — https://fred.stlouisfed.org/docs/api/api_key.html
# BEDROCK_KB_ID     — optional, from AWS Console → Bedrock → Knowledge Bases
# LANGSMITH (optional tracing) — https://smith.langchain.com

python -m financial_advisory_agent.main
```

## Running with client profiles

Client profiles are stored as JSON files in `data/clients/`. Two are included out of the box:

| Profile | File | Description |
|---|---|---|
| Sarah Chen | `sarah_chen.json` | 42yo, moderate risk, $620K net worth, retirement + college planning |
| James Okafor | `james_okafor.json` | 28yo, aggressive risk, $45K net worth, early retirement + house savings |
| Margaret Osei | `margaret_osei.json` | 67yo, conservative, $890K net worth, RMDs, income generation, inheritance |
| Richard Stavros | `richard_stavros.json` | 51yo, moderate, $3.2M net worth, concentrated stock, tax optimisation, philanthropy |
| Priya Nair | `priya_nair.json` | 31yo, moderate, $28K net worth, $62K student debt, debt vs invest tradeoff |

```bash
# Default profile (Sarah Chen)
python -m financial_advisory_agent.main

# Pick a different profile
python -m financial_advisory_agent.main --profile james_okafor

# See available profiles and options
python -m financial_advisory_agent.main --help
```

To add a new client, drop a `.json` file into `data/clients/` matching the `ClientProfile` schema — no code changes needed.

## Analyst Tools

| Tool | Source | Purpose |
|---|---|---|
| `search_web` | Tavily | Live market news, prices, current events |
| `search_bedrock_kb` | AWS Bedrock Knowledge Base | Curated financial planning docs via managed RAG (preferred) |
| `search_knowledge_base` | FAISS + Bedrock Titan (local) | Fallback RAG if Bedrock KB is not configured |
| `get_fred_data` | FRED API | Macro time series: CPI, Fed Funds Rate, GDP, Treasury yields, unemployment |
| `search_fred` | FRED API | Discover series IDs by keyword when the series is unknown |

`search_bedrock_kb` is preferred over `search_knowledge_base` when `BEDROCK_KB_ID` is set. Both cover the same documents — asset allocation, tax-advantaged accounts, risk diversification, common mistakes.

## REST API

A FastAPI app exposes the analyst as a single endpoint:

```bash
uvicorn financial_advisory_agent.api:app --reload
```

```bash
POST /analyst   body: {"query": "..."}
```

Example:
```bash
curl -X POST http://localhost:8000/analyst \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current Fed Funds Rate and CPI trend?"}'
```

## Using agents independently

```python
from financial_advisory_agent.agents import create_analyst_agent, create_advisor_agent, create_client_agent
from financial_advisory_agent.models import load_client_profile

profile = load_client_profile("sarah_chen")

# Use analyst directly
analyst = create_analyst_agent()
result = analyst.invoke({"messages": [{"role": "user", "content": "S&P 500 avg returns"}]})
print(result["messages"][-1].content)

# Use advisor interactively (you play the client)
advisor = create_advisor_agent(analyst, profile)
result = advisor.invoke({"messages": [{"role": "user", "content": "Hi, I need help with my portfolio"}]})
print(result["messages"][-1].content)
```

## Project Structure

```
src/
└── financial_advisory_agent/
    ├── __init__.py          # Package root
    ├── config.py            # Pydantic settings (models, temperatures, limits)
    ├── models.py            # ClientProfile, WorkflowState, profile loader
    ├── tools.py             # search_web (Tavily), search_knowledge_base (FAISS), get_fred_data, search_fred (FRED API)
    ├── agents/
    │   ├── __init__.py      # Re-exports create_*_agent functions
    │   ├── _common.py       # Shared make_llm() helper (ChatBedrockConverse)
    │   ├── analyst.py       # Standalone research subagent
    │   ├── advisor.py       # Orchestrator with research tool + conclude
    │   └── client.py        # Simulated client persona
    ├── workflow.py           # 2-node LangGraph: advisor ↔ client loop
    └── main.py               # Entry point with CLI args and coloured output
data/
├── clients/                  # Client profile JSONs (add new ones here)
│   ├── sarah_chen.json
│   └── james_okafor.json
└── knowledge/                # Financial planning docs for RAG
    ├── asset_allocation.txt
    ├── common_mistakes.txt
    ├── risk_diversification.txt
    └── tax_advantaged_accounts.txt
```

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Framework | LangChain `create_agent` + LangGraph | Production-ready agents with explicit workflow control |
| LLM provider | AWS Bedrock (`ChatBedrockConverse`) | Enterprise-ready, no API keys in code — uses standard AWS credential chain |
| Multi-agent pattern | Subagents | Analyst is stateless research — perfect fit for tool-as-agent |
| Channel separation | Structural (tool boundary) | Analyst never sees messages; client never sees tool calls |
| Routing | Tool calling | `research` and `conclude` are structured actions, not parsed text |
| Knowledge base | FAISS + Bedrock Titan embeddings | Simple, local, no infra for a demo |
| Macro data | FRED API (`fredapi`) | Authoritative, free economic time series — no scraping needed |
| Client profiles | JSON files in `data/clients/` | Add new personas without code changes; validated by Pydantic |
| Termination | `conclude` tool + max turns | Natural conclusion with safety valve |

## LangSmith Tracing

Two separate LangSmith projects are used:

| Project | What it traces |
|---|---|
| `financial-advisory-agent` | Full advisor ↔ client workflow, including `research` tool calls |
| `financial-advisory-analyst` | Every analyst invocation (whether from the advisor or the API directly) |

### Correlating an analyst trace to an advisor tool call

Each time the advisor calls `research`, the analyst trace is tagged with the ID of that tool call. To find the matching pair:

1. **In `financial-advisory-agent`** — open the advisor run → find the `research` tool call step → note its `id` (e.g. `call_abc123`)
2. **In `financial-advisory-analyst`** — click **Filter** → **Metadata** → set `advisor_tool_call_id = call_abc123`

The matching analyst run will appear. This links the two independent traces without nesting them.