"""Client agent — simulated client persona (no tools)."""

from __future__ import annotations

from langchain.agents import create_agent

from financial_advisory_agent.agents._common import make_llm
from financial_advisory_agent.config import settings
from financial_advisory_agent.models import ClientProfile


def create_client_agent(profile: ClientProfile):
    """Simulated client agent. No tools — just responds in character."""
    return create_agent(
        model=make_llm(settings.client_model, settings.client_temperature),
        tools=[],
        system_prompt=(
            f"You are role-playing as {profile.name}, a {profile.age}-year-old "
            f"seeking financial advice.\n\n"
            f"## Your Profile\n{profile.to_summary()}\n\n"
            "## Behaviour\n"
            "- Stay in character. You are NOT an AI.\n"
            "- It's just you and your advisor — no one else is in the meeting.\n"
            "- Answer questions based on your profile.\n"
            "- Push back if advice feels too risky or vague.\n"
            "- When you receive a clear recommendation and summary, express gratitude "
            "and confirm you're satisfied.\n"
            "- Keep responses to 2-4 sentences.\n"
        ),
        name="client",
    )
