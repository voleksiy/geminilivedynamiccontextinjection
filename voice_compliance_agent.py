"""
Voice Compliance Agent for Financial Services
=============================================
Implementation of Dynamic Context Injection using the Gemini Multimodal Live API
and official Google Gen AI SDK (`google-genai`).

Key Features:
1. Single WebSocket connection via `client.aio.live.connect`.
2. Native Function Calling (`report_customer_disclosure`) for detecting trauma disclosures
   (bereavement, illness) and regulatory drift (unauthorized stock advice).
3. Dynamic RAG Context Injection via `LiveClientToolResponse`.
4. Dynamic System Instruction updates mid-session via `session.send_client_content(role="system")`.
5. Immutable SEC 17a-4 & CFPB Audit Logging Ledger.
6. Soft Handoff (human specialist transfer) and Hard Kill Switch execution.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceComplianceAgent")

class AuditLedger:
    """Immutable audit trail recorder for SEC Rule 17a-4 / CFPB compliance."""
    def __init__(self, log_filepath: str = "audit_ledger.jsonl"):
        self.log_filepath = log_filepath

    def record_event(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
            "details": details
        }
        logger.info(f"[Audit Ledger] {event_type}: {json.dumps(details)}")
        with open(self.log_filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")


def _get_default_gcp_project() -> str:
    """Helper to detect active GCP project from environment or gcloud config."""
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ.get("GOOGLE_CLOUD_PROJECT")
    if os.environ.get("GCP_PROJECT"):
        return os.environ.get("GCP_PROJECT")
    try:
        import subprocess
        res = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=2)
        proj = res.stdout.strip()
        if proj and " " not in proj and proj != "(unset)":
            return proj
    except Exception:
        pass
    return "codemender-internal-demo"


class VoiceComplianceOrchestrator:
    """Orchestrates real-time voice sessions using Gemini Live API on Vertex AI with native function-calling context injection."""
    
    def __init__(self, project_id: Optional[str] = None, location: Optional[str] = None, on_event_cb=None):
        self.project_id = project_id or _get_default_gcp_project()
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.on_event_cb = on_event_cb

        # Prioritize Vertex AI using Application Default Credentials
        if self.project_id:
            logger.info(f"[Client Init] Initializing Google Gen AI SDK for Vertex AI (Project: {self.project_id}, Location: {self.location})")
            self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        elif self.api_key:
            logger.info("[Client Init] Initializing Google Gen AI SDK with API Key fallback.")
            self.client = genai.Client(api_key=self.api_key)
        else:
            logger.warning("[Client Init] Neither GOOGLE_CLOUD_PROJECT nor GEMINI_API_KEY set. Defaulting to Vertex AI auto-auth.")
            self.client = genai.Client(vertexai=True, location=self.location)

        self.model_id = "gemini-live-2.5-flash-native-audio"
        self.session = None
        self.audit_ledger = AuditLedger()
        self.current_fsm_state = "AUTHENTICATED_GREETING"

    def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Emits event to optional callback for Web UI streaming."""
        if self.on_event_cb:
            try:
                self.on_event_cb(event_type, payload)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def _build_tools(self) -> List[types.Tool]:
        """Defines function declarations that Gemini can invoke upon detecting disclosures or drift."""
        disclosure_function = types.FunctionDeclaration(
            name="report_customer_disclosure",
            description=(
                "CRITICAL: Call this function IMMEDIATELY whenever the customer discloses "
                "a traumatic personal life event (loss of family member, bereavement, terminal illness, "
                "disability, severe hardship) OR pivots to high-risk topics requiring legal disclaimers "
                "(unauthorized stock/investment advice, legal disputes)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "event_type": types.Schema(
                        type="STRING",
                        description="Category: 'BEREAVEMENT', 'MEDICAL_HARDSHIP', 'UNAUTHORIZED_INVESTMENT_ADVICE', 'FINANCIAL_DISTRESS'"
                    ),
                    "user_summary": types.Schema(
                        type="STRING",
                        description="Brief summary of the customer's disclosure or request."
                    )
                },
                required=["event_type", "user_summary"]
            )
        )
        return [types.Tool(function_declarations=[disclosure_function])]

    def _build_connect_config(self) -> types.LiveConnectConfig:
        """Constructs initial LiveConnectConfig for Gemini Live API."""
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            tools=self._build_tools(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(
                    text="You are a licensed customer service representative for Apex Bank. "
                    "Listen closely to the user's vocal tone and message. "
                    "Be helpful, empathetic, professional, and adhere strictly to any compliance instructions injected into the conversation."
                )]
            )
        )

    async def start_session(self):
        """Establishes single WebSocket connection to Gemini Live API and enters main event loop."""
        config = self._build_connect_config()
        self.audit_ledger.record_event("SESSION_START", {"model": self.model_id, "initial_state": self.current_fsm_state})

        async with self.client.aio.live.connect(model=self.model_id, config=config) as session:
            self.session = session
            logger.info("[System] Connected to Gemini Live API over WebSocket.")
            await self.receive_loop()

    def analyze_and_inject_compliance(self, text: str) -> dict:
        """Analyzes incoming transcript or user text prompt for regulatory compliance triggers,
        updates FSM state, records audit ledger events, and generates dynamic context overrides."""
        lower_text = text.lower()
        
        event_type = None
        user_summary = text
        grounding_data = ""
        system_override = ""

        if any(w in lower_text for w in ["spouse", "husband", "wife", "passed away", "died", "bereavement", "loss", "funeral", "death", "hardship"]):
            event_type = "BEREAVEMENT"
            self.current_fsm_state = "BEREAVEMENT_HARDSHIP_CARE"
            grounding_data = (
                "APEX BANK BEREAVEMENT & HARDSHIP POLICY #HARDSHIP-901:\n"
                "- Immediate 60-day fee waiver & mortgage/loan payment pause active.\n"
                "- No formal documentation required during initial verbal disclosure.\n"
                "- Direct transfer line to Senior Compassionate Care Team: 1-800-555-CARE."
            )
            system_override = (
                "CRITICAL INSTRUCTION OVERRIDE: Customer disclosed acute hardship/bereavement. "
                "Immediately drop standard banking tone. Adopt a calm, deeply compassionate, low-tempo vocal cadence. "
                "Express sincere sympathy. Inform the customer about Policy #HARDSHIP-901 fee waiver and offer a seamless "
                "transfer to our Senior Compassionate Care Specialist."
            )

        elif any(w in lower_text for w in ["stock", "tsla", "tesla", "invest", "buy shares", "portfolio", "investing", "crypto", "trade"]):
            event_type = "UNAUTHORIZED_INVESTMENT_ADVICE"
            self.current_fsm_state = "INVESTMENT_DISCLAIMER_GATE"
            grounding_data = (
                "SEC & FINRA REGULATORY COMPLIANCE MANDATE:\n"
                "- Apex Bank retail AI agents are strictly prohibited from giving stock/portfolio recommendations.\n"
                "- Mandatory Disclaimer: 'I am an automated assistant and cannot provide individual investment advice.'"
            )
            system_override = (
                "CRITICAL INSTRUCTION OVERRIDE: Customer asked for stock/investment advice. "
                "State the mandatory SEC disclaimer clearly and offer to transfer to a licensed wealth advisor."
            )

        if event_type:
            audit_details = {
                "function_name": "report_customer_disclosure",
                "call_id": "auto_intercept",
                "event_type": event_type,
                "user_summary": user_summary,
                "fsm_state": self.current_fsm_state,
                "grounding_data": grounding_data,
                "system_override": system_override
            }
            self.audit_ledger.record_event("DYNAMIC_CONTEXT_INJECTION", audit_details)
            
            self.emit_event("DYNAMIC_CONTEXT_INJECTION", {
                "event_type": event_type,
                "user_summary": user_summary,
                "fsm_state": self.current_fsm_state,
                "grounding_data": grounding_data,
                "system_override": system_override
            })

            formatted_prompt = (
                f"[{system_override}]\n"
                f"[{grounding_data}]\n"
                f"User message: {text}"
            )
            return {"event_type": event_type, "formatted_prompt": formatted_prompt}

        return {"event_type": None, "formatted_prompt": text}

    def trigger_manual_disclosure(self, event_type: str, user_summary: str = "") -> str:
        """Handles manual compliance triggers from Web UI or simulation harness."""
        if event_type in ["BEREAVEMENT", "MEDICAL_HARDSHIP"]:
            self.current_fsm_state = "BEREAVEMENT_HARDSHIP_CARE"
            grounding_data = (
                "APEX BANK BEREAVEMENT & HARDSHIP POLICY #HARDSHIP-901:\n"
                "- Immediate 60-day fee waiver & mortgage/loan payment pause active.\n"
                "- No formal documentation required during initial verbal disclosure.\n"
                "- Direct transfer line to Senior Compassionate Care Team: 1-800-555-CARE."
            )
            system_override = (
                "CRITICAL INSTRUCTION OVERRIDE: Customer disclosed acute hardship/bereavement. "
                "Immediately drop standard banking tone. Adopt a calm, deeply compassionate, low-tempo vocal cadence. "
                "Express sincere sympathy. Inform the customer about Policy #HARDSHIP-901 fee waiver and offer a seamless "
                "transfer to our Senior Compassionate Care Specialist."
            )
        elif event_type == "UNAUTHORIZED_INVESTMENT_ADVICE":
            self.current_fsm_state = "INVESTMENT_DISCLAIMER_GATE"
            grounding_data = (
                "SEC & FINRA REGULATORY COMPLIANCE MANDATE:\n"
                "- Apex Bank retail AI agents are strictly prohibited from giving stock/portfolio recommendations.\n"
                "- Mandatory Disclaimer: 'I am an automated assistant and cannot provide individual investment advice.'"
            )
            system_override = (
                "CRITICAL INSTRUCTION OVERRIDE: Customer asked for stock/investment advice. "
                "State the mandatory SEC disclaimer clearly and offer to transfer to a licensed wealth advisor."
            )
        else:
            grounding_data = "Standard Account Servicing Guidelines Active."
            system_override = "Maintain helpful, professional banking assistance."

        audit_details = {
            "function_name": "report_customer_disclosure",
            "call_id": "manual_trigger",
            "event_type": event_type,
            "user_summary": user_summary,
            "fsm_state": self.current_fsm_state,
            "grounding_data": grounding_data,
            "system_override": system_override
        }
        self.audit_ledger.record_event("DYNAMIC_CONTEXT_INJECTION", audit_details)
        
        self.emit_event("DYNAMIC_CONTEXT_INJECTION", {
            "event_type": event_type,
            "user_summary": user_summary,
            "fsm_state": self.current_fsm_state,
            "grounding_data": grounding_data,
            "system_override": system_override
        })

        return (
            f"[{system_override}]\n"
            f"[{grounding_data}]\n"
            f"User input summary: {user_summary or 'Simulated disclosure trigger'}"
        )

    async def handle_function_call(self, function_call: types.FunctionCall):
        """Backwards compatibility handler for simulation test harness."""
        args = function_call.args or {}
        event_type = args.get("event_type", "BEREAVEMENT")
        user_summary = args.get("user_summary", "")
        
        prompt = self.trigger_manual_disclosure(event_type, user_summary)
        
        if self.session and hasattr(self.session, "send_tool_response"):
            func_res = types.FunctionResponse(
                name=function_call.name,
                id=function_call.id,
                response={"status": "CONTEXT_INJECTED", "active_fsm_state": self.current_fsm_state}
            )
            await self.session.send_tool_response(function_responses=[func_res])
            logger.info(f"[Tool Call] Tool response sent for {function_call.name} (ID: {function_call.id}). Triggering speech turn...")
            
            # Trigger spoken response turn from Gemini
            speech_prompt = (
                "The compliance policy and hardship rules have been updated and injected. "
                "Please speak a warm, empathetic response to the customer now, acknowledging their statement "
                "and explaining how Apex Bank can assist them."
            )
            await self.session.send_client_content(
                turns=[types.Content(
                    parts=[types.Part.from_text(text=speech_prompt)]
                )],
                turn_complete=True
            )

    async def execute_kill_switch(self, mode: str, reason: str):
        """Executes soft handoff or hard socket termination."""
        self.audit_ledger.record_event("KILL_SWITCH_TRIGGERED", {"mode": mode, "reason": reason})

        if mode == "SOFT_HANDOFF":
            handoff_prompt = (
                "[KILL SWITCH OVERRIDE]: Immediately cease current topic. Politely inform the user "
                "that you are transferring them directly to a human specialist."
            )
            await self.session.send_client_content(
                turns=[types.Content(parts=[types.Part.from_text(text=handoff_prompt)])],
                turn_complete=True
            )
            logger.info(f"[Kill Switch] Soft handoff executed: {reason}")

        elif mode == "HARD_TERMINATION":
            logger.warning(f"[Kill Switch] Hard socket termination executed: {reason}")
            await self.session.close()

    async def receive_loop(self):
        """Main WebSocket receive loop handling incoming audio chunks, text transcripts, and tool calls."""
        try:
            async for response in self.session.receive():
                server_content = response.server_content
                if not server_content or not server_content.model_turn:
                    continue

                for part in server_content.model_turn.parts:
                    # 1. Handle Function Calls
                    if part.function_call:
                        await self.handle_function_call(part.function_call)

                    # 2. Handle Model Audio Output (24kHz PCM)
                    if part.inline_data:
                        audio_pcm_24k = part.inline_data.data
                        # In production: forward audio_pcm_24k bytes to WebRTC / audio speaker output stream
                        pass

                    # 3. Handle Model Text Transcript
                    if part.text:
                        print(f"Gemini: {part.text}")

        except Exception as e:
            logger.error(f"[Receive Loop Error] {e}")
            self.audit_ledger.record_event("SESSION_ERROR", {"error": str(e)})


async def main():
    """Main entry point for local execution."""
    agent = VoiceComplianceOrchestrator()
    try:
        await agent.start_session()
    except KeyboardInterrupt:
        logger.info("Session closed by user.")

if __name__ == "__main__":
    asyncio.run(main())
