"""Data models for the advisory system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

CLIENTS_DIR = Path("data/clients")


class ClientProfile(BaseModel):
    """Simulated client attributes."""

    name: str
    age: int = Field(ge=18, le=100)
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    annual_income: float = Field(ge=0)
    net_worth: float = Field(ge=0)
    current_holdings: dict[str, float]
    investment_horizon_years: int = Field(ge=1)
    goals: list[str]
    concerns: list[str] = Field(default_factory=list)

    def to_summary(self) -> str:
        """One-paragraph summary for prompt injection."""
        holdings = ", ".join(f"{k}: {v}%" for k, v in self.current_holdings.items())
        goals = "; ".join(self.goals)
        concerns = "; ".join(self.concerns) if self.concerns else "None stated"
        return (
            f"{self.name}, age {self.age}. Risk tolerance: {self.risk_tolerance}. "
            f"Income: ${self.annual_income:,.0f}, Net worth: ${self.net_worth:,.0f}. "
            f"Holdings: {holdings}. Horizon: {self.investment_horizon_years} years. "
            f"Goals: {goals}. Concerns: {concerns}."
        )


class WorkflowState(TypedDict):
    """Shared state for the client ↔ advisor conversation loop."""

    messages: Annotated[list[BaseMessage], add_messages]
    is_concluded: bool
    turn_count: int


# ---------------------------------------------------------------------------
# Client profile loading from data/clients/*.json
# ---------------------------------------------------------------------------

def list_client_profiles() -> list[str]:
    """Return available profile names (filenames without .json)."""
    if not CLIENTS_DIR.exists():
        return []
    return sorted(p.stem for p in CLIENTS_DIR.glob("*.json"))


def load_client_profile(name: str) -> ClientProfile:
    """Load a ClientProfile from ``data/clients/<name>.json``.

    Raises:
        FileNotFoundError: if the profile JSON does not exist.
        pydantic.ValidationError: if the JSON doesn't match the schema.
    """
    path = CLIENTS_DIR / f"{name}.json"
    if not path.exists():
        available = list_client_profiles()
        raise FileNotFoundError(
            f"Profile '{name}' not found at {path}. "
            f"Available profiles: {available}"
        )
    data = json.loads(path.read_text())
    return ClientProfile(**data)


# ---------------------------------------------------------------------------
# Default demo profile (convenience shortcut)
# ---------------------------------------------------------------------------

DEFAULT_CLIENT = load_client_profile("sarah_chen")
