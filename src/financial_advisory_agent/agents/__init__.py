"""Agent creation for the financial advisory system (subagents pattern).

Architecture:
  - Analyst: standalone agent with web search + knowledge base tools.
  - Advisor: agent with analyst-as-tool + conclude tool. Runs full ReAct loop.
  - Client:  stateless LLM with fixed persona (no tools).

Each agent is independently usable. The workflow composes advisor + client.
"""

from financial_advisory_agent.agents.advisor import create_advisor_agent
from financial_advisory_agent.agents.analyst import create_analyst_agent
from financial_advisory_agent.agents.client import create_client_agent

__all__ = [
    "create_advisor_agent",
    "create_analyst_agent",
    "create_client_agent",
]
