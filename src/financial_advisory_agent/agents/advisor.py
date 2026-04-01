"""Advisor agent — senior financial advisor with analyst subagent.

The Advisor's ReAct loop: think → research (via analyst) → think → respond.
The analyst is wrapped as a tool so it runs inside the Advisor's loop,
providing structural channel separation.
"""

from __future__ import annotations

from typing import Annotated

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool

from financial_advisory_agent.agents._common import make_llm
from financial_advisory_agent.config import settings
from financial_advisory_agent.models import ClientProfile


def _make_research_tool(analyst_agent):
    """Wrap the analyst agent as a tool the advisor can call."""

    @tool
    def research(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig,
    ) -> str:
        """Research a financial topic. Include anonymised client context in your query
        (e.g. 'asset allocation for 42yo moderate-risk investor with 20yr horizon').
        Never include the client's name."""
        result = analyst_agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"metadata": {"advisor_tool_call_id": tool_call_id}},
        )
        return result["messages"][-1].content

    return research


@tool
def conclude(summary: str) -> str:
    """End the advisory session. Call this ONLY after delivering your
    recommendations and confirming the client is satisfied."""
    return f"[SESSION CONCLUDED]\n{summary}"


def create_advisor_agent(analyst_agent, profile: ClientProfile):
    """Advisor agent with analyst as a subagent tool.

    Multiple research calls can happen in a single turn.
    """
    research_tool = _make_research_tool(analyst_agent)

    return create_agent(
        model=make_llm(settings.advisor_model, settings.advisor_temperature),
        tools=[research_tool, conclude],
        system_prompt=(
            "You are a senior financial advisor in a private meeting with a client.\n\n"
            f"## Client Profile\n{profile.to_summary()}\n\n"
            "## Conversation Flow\n"
            "1. DISCOVERY — Greet the client, ask about goals, situation, concerns.\n"
            "2. ANALYSIS — Use the `research` tool to get data. Call it multiple times if needed.\n"
            "3. RECOMMENDATION — Synthesize research into clear, tailored advice.\n"
            "4. CLOSING — Summarize, confirm satisfaction, call `conclude`.\n\n"
            "## Rules\n"
            "- NEVER guarantee specific returns.\n"
            "- ALWAYS note this is educational — client should consult a licensed advisor.\n"
            "- When using `research`, include anonymised context (age, risk, holdings). "
            "NEVER include the client's name.\n"
            "- Keep responses concise and actionable.\n"
        ),
        name="advisor",
    )
