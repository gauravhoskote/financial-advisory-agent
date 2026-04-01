"""Client ↔ Advisor conversation loop (LangGraph custom workflow).

The graph is intentionally thin — two nodes ping-ponging messages.
All agent logic lives in the agents themselves (via create_agent).
The analyst is hidden inside the advisor's tool, invisible to the client.

Message-role remapping:
    Each agent is an LLM that produces AIMessage. Before invoking an agent we
    remap the shared history so the *other* agent's messages appear as
    HumanMessage (user input) and the agent's *own* prior messages appear as
    AIMessage (assistant output). Without this, both agents see only AIMessages
    and the LLM has nothing to respond to.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from financial_advisory_agent.config import settings
from financial_advisory_agent.models import ClientProfile, WorkflowState

logger = logging.getLogger(__name__)


def _remap_messages_for(
    messages: list, agent_name: str
) -> list:
    """Return a copy of *messages* with roles set from *agent_name*'s perspective.

    - Messages whose ``name`` matches *agent_name* → ``AIMessage`` (the agent's
      own prior output).
    - All other messages → ``HumanMessage`` (input from the other party).
    """
    remapped: list = []
    for msg in messages:
        name = getattr(msg, "name", "")
        content = getattr(msg, "content", "")
        if not content:
            continue
        if name == agent_name:
            remapped.append(AIMessage(content=content, name=name))
        else:
            remapped.append(HumanMessage(content=content, name=name))
    return remapped


def build_workflow(advisor_agent, client_agent) -> StateGraph:
    """Build and compile the conversation graph.

    Nodes:
        advisor — invokes advisor agent (which may call analyst internally)
        client  — invokes simulated client agent

    Edges:
        advisor → client | END  (based on conclude tool or max turns)
        client  → advisor | END
    """

    def advisor_node(state: WorkflowState) -> dict:
        advisor_msgs = _remap_messages_for(state["messages"], "advisor")
        result = advisor_agent.invoke({"messages": advisor_msgs})
        new_msgs = result["messages"][len(advisor_msgs):]

        # Check if the advisor called the `conclude` tool in this turn.
        concluded = any(
            tc["name"] == "conclude"
            for msg in new_msgs
            if hasattr(msg, "tool_calls") and msg.tool_calls
            for tc in msg.tool_calls
        )

        # Only surface the advisor's final text response to the client.
        # Internal tool calls (research, conclude) stay hidden.
        final_msg = new_msgs[-1] if new_msgs else None
        visible = []
        if final_msg and getattr(final_msg, "content", ""):
            visible = [AIMessage(content=final_msg.content, name="advisor")]

        logger.info("[Advisor] concluded=%s  msgs=%d", concluded, len(new_msgs))
        return {
            "messages": visible,
            "is_concluded": concluded,
            "turn_count": state["turn_count"] + 1,
        }

    def client_node(state: WorkflowState) -> dict:
        client_msgs = _remap_messages_for(state["messages"], "client")
        result = client_agent.invoke({"messages": client_msgs})
        new_msgs = result["messages"][len(client_msgs):]

        final_msg = new_msgs[-1] if new_msgs else None
        visible = []
        if final_msg and getattr(final_msg, "content", ""):
            visible = [HumanMessage(content=final_msg.content, name="client")]

        logger.info("[Client] response_len=%d", len(final_msg.content) if final_msg else 0)
        return {
            "messages": visible,
            "turn_count": state["turn_count"] + 1,
        }

    def route_after_advisor(state: WorkflowState) -> str:
        if state["is_concluded"] or state["turn_count"] >= settings.max_turns:
            return END
        return "client"

    def route_after_client(state: WorkflowState) -> str:
        if state["turn_count"] >= settings.max_turns:
            return END
        return "advisor"

    # --- Assemble graph ---
    graph = StateGraph(WorkflowState)
    graph.add_node("advisor", advisor_node)
    graph.add_node("client", client_node)
    graph.set_entry_point("advisor")
    graph.add_conditional_edges("advisor", route_after_advisor, {"client": "client", END: END})
    graph.add_conditional_edges("client", route_after_client, {"advisor": "advisor", END: END})

    return graph.compile()
