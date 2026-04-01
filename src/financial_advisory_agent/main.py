"""Entry point: python -m src.main [--profile <name>]"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from financial_advisory_agent.agents import create_advisor_agent, create_analyst_agent, create_client_agent
from financial_advisory_agent.models import WorkflowState, list_client_profiles, load_client_profile
from financial_advisory_agent.workflow import build_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# Colours
BLUE, GREEN, DIM, RESET = "\033[94m", "\033[92m", "\033[2m", "\033[0m"


def _header(label: str, color: str) -> None:
    print(f"\n{color}{'─'*60}\n  {label}\n{'─'*60}{RESET}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Multi-Agent Financial Advisory System")
    parser.add_argument(
        "--profile",
        default="sarah_chen",
        help=(
            "Client profile name (filename without .json from data/clients/). "
            f"Available: {list_client_profiles()}"
        ),
    )
    args = parser.parse_args()
    profile = load_client_profile(args.profile)

    print(f"\n{'═'*60}")
    print("  🏦  Multi-Agent Financial Advisory System")
    print(f"{'═'*60}")
    print(f"  Client: {profile.name}, age {profile.age}")
    print(f"  Risk:   {profile.risk_tolerance}  |  Net worth: ${profile.net_worth:,.0f}")
    print(f"{'═'*60}\n")

    # --- Create standalone agents ---
    analyst = create_analyst_agent()
    advisor = create_advisor_agent(analyst, profile)
    client = create_client_agent(profile)

    # --- Build conversation workflow ---
    workflow = build_workflow(advisor, client)

    initial: WorkflowState = {
        "messages": [],
        "is_concluded": False,
        "turn_count": 0,
    }

    print("📋 Starting session...\n")
    try:
        for event in workflow.stream(initial, {"recursion_limit": 50}):
            for node_name, update in event.items():
                if node_name == "__end__":
                    continue
                for msg in update.get("messages", []):
                    name = getattr(msg, "name", "")
                    content = getattr(msg, "content", "")
                    if not content:
                        continue
                    if name == "advisor":
                        _header("ADVISOR", BLUE)
                        print(f"{BLUE}{content}{RESET}")
                    elif name == "client":
                        _header("CLIENT", GREEN)
                        print(f"{GREEN}{content}{RESET}")

        print(f"\n{'═'*60}")
        print("  ✅  Session complete.")
        print(f"{'═'*60}\n")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(0)
    except Exception as e:
        logging.exception("Session failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
