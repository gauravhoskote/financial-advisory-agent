"""Shared utilities for agent creation."""

from __future__ import annotations

from langchain_aws import ChatBedrockConverse

from financial_advisory_agent.config import settings


def make_llm(model_id: str, temperature: float) -> ChatBedrockConverse:
    """Create a ChatBedrockConverse instance for the given model.

    When *model_id* is an inference-profile ARN, ``provider`` must be
    supplied explicitly (required by langchain-aws >=1.4).
    """
    kwargs: dict = {
        "model": model_id,
        "region_name": settings.aws_region,
        "temperature": temperature,
    }

    if model_id.startswith("arn:"):
        kwargs["provider"] = settings.model_provider

    return ChatBedrockConverse(**kwargs)
