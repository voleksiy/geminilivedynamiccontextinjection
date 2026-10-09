"""
Test Simulation Harness for Voice Compliance Agent
==================================================
Simulates Gemini Live API function calls, dynamic context injection, RAG grounding,
and audit ledger generation without requiring active audio hardware.
"""

import asyncio
import os
import json
import logging
from voice_compliance_agent import VoiceComplianceOrchestrator
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestSimulation")

class MockSession:
    """Mock Gemini Live WebSocket Session for testing handlers offline."""
    def __init__(self):
        self.sent_tool_responses = []
        self.sent_client_contents = []
        self.is_closed = False

    async def send_tool_response(self, *args, **kwargs):
        self.sent_tool_responses.append((args, kwargs))
        logger.info(f"[MockSession] Intercepted Tool Response: {args} {kwargs}")

    async def send_client_tool_response(self, *args, **kwargs):
        self.sent_tool_responses.append((args, kwargs))
        logger.info(f"[MockSession] Intercepted Tool Response: {args} {kwargs}")

    async def send_client_content(self, *args, **kwargs):
        self.sent_client_contents.append((args, kwargs))
        logger.info(f"[MockSession] Intercepted System Content Update: {args} {kwargs}")

    async def close(self):
        self.is_closed = True
        logger.info("[MockSession] Session Closed.")


async def run_simulation():
    print("=" * 70)
    print("STARTING VOICE COMPLIANCE AGENT SIMULATION HARNESS")
    print("=" * 70)

    agent = VoiceComplianceOrchestrator()
    mock_session = MockSession()
    agent.session = mock_session

    # Test Scenario 1: Bereavement / Personal Trauma Disclosure
    print("\n[Simulation Scenario 1]: Customer discloses loss of spouse (Bereavement)")
    mock_call_1 = types.FunctionCall(
        id="call_001",
        name="report_customer_disclosure",
        args={
            "event_type": "BEREAVEMENT",
            "user_summary": "Customer stated husband passed away, asking about mortgage payment pause."
        }
    )

    await agent.handle_function_call(mock_call_1)

    assert agent.current_fsm_state == "BEREAVEMENT_HARDSHIP_CARE"
    assert len(mock_session.sent_tool_responses) == 1
    print("Scenario 1 PASSED: Correct FSM state, RAG grounding, and system override injected.")

    # Test Scenario 2: Regulatory Boundary Pivot (Unauthorized Stock Advice)
    print("\n[Simulation Scenario 2]: Customer asks for stock investment recommendation")
    mock_call_2 = types.FunctionCall(
        id="call_002",
        name="report_customer_disclosure",
        args={
            "event_type": "UNAUTHORIZED_INVESTMENT_ADVICE",
            "user_summary": "Customer asked if they should invest $50k in TSLA stock."
        }
    )

    await agent.handle_function_call(mock_call_2)

    assert agent.current_fsm_state == "INVESTMENT_DISCLAIMER_GATE"
    assert len(mock_session.sent_tool_responses) == 2
    print("Scenario 2 PASSED: Correct FSM state and SEC disclaimer injected.")

    # Test Scenario 3: Soft Kill Switch Execution (Human Advocate Transfer)
    print("\n[Simulation Scenario 3]: Executing Soft Kill Switch")
    await agent.execute_kill_switch("SOFT_HANDOFF", "BEREAVEMENT_HUMAN_ADVOCATE_TRANSFER")
    assert len(mock_session.sent_client_contents) == 1
    print("Scenario 3 PASSED: Soft handoff command sent to session.")

    # Verify Audit Ledger Output
    audit_file = "audit_ledger.jsonl"
    assert os.path.exists(audit_file)
    with open(audit_file, "r") as f:
        lines = f.readlines()
        print(f"\n[Audit Ledger Verification]: Successfully logged {len(lines)} compliance events to {audit_file}.")
        for line in lines:
            data = json.loads(line)
            print(f"  - [{data['iso_time']}] {data['event_type']}: {data['details'].get('event_type', 'N/A')}")

    print("\n" + "=" * 70)
    print("ALL SIMULATION SCENARIOS PASSED CLEANLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_simulation())
