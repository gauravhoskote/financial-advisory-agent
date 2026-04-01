"""Analyst agent — standalone research subagent.

Usable directly or wrapped as a tool by the Advisor.
Always traces to the 'financial-advisory-analyst' LangSmith project.
"""

from __future__ import annotations

import contextvars

from langchain.agents import create_agent
from langsmith import trace

from financial_advisory_agent.agents._common import make_llm
from financial_advisory_agent.config import settings
from financial_advisory_agent.tools import analyst_tools


def create_analyst_agent():
    """Standalone research agent with web search + knowledge base tools."""
    agent = create_agent(
        model=make_llm(settings.analyst_model, settings.analyst_temperature),
        tools=analyst_tools,
        system_prompt=(
            "You are a financial research analyst. You provide factual findings.\n"
            "- Use search_web for current market news and real-time financial data.\n"
            "- Use search_bedrock_kb for financial planning principles and strategies (preferred over search_knowledge_base if available).\n"
            "- Use search_knowledge_base as a fallback for financial planning principles if search_bedrock_kb returns no results.\n"
            "- Use get_fred_data to fetch specific macroeconomic time series (interest rates, "
            "inflation, GDP, unemployment, Treasury yields). If you know the FRED series ID, "
            "call it directly. Common IDs: FEDFUNDS (Fed Funds Rate), CPIAUCSL (CPI), "
            "DGS10 (10-Year Treasury), UNRATE (Unemployment), GDP.\n"
            "- Use search_fred to discover series IDs when you are unsure which series to use.\n"
            "- Stick to FACTS. Do NOT give opinions or recommendations.\n"
            "- If you can't find information, say so. Do NOT fabricate data.\n"
            "- Structure findings clearly with specific data points."
        ),
        name="analyst",
    )

    original_invoke = agent.invoke

    def invoke(input, **kwargs):
        result = {}

        extra_metadata = {}
        config = kwargs.get("config") or {}
        if isinstance(config, dict):
            extra_metadata = config.get("metadata") or {}

        def _run():
            with trace("analyst", run_type="chain", project_name="financial-advisory-analyst", metadata=extra_metadata):
                result["value"] = original_invoke(input, **kwargs)

        contextvars.Context().run(_run)
        return result["value"]

    agent.invoke = invoke
    return agent
