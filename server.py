"""
Voice Compliance Agent Web Server
=================================
FastAPI backend bridging Browser Microphone / WebAudio with Gemini Multimodal Live API on Vertex AI.
"""

import asyncio
import json
import logging
import os
import struct
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from google import genai
from google.genai import types

from voice_compliance_agent import VoiceComplianceOrchestrator, AuditLedger

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceComplianceServer")

def calc_pcm_rms(pcm_bytes: bytes) -> float:
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"<{count}h", pcm_bytes)
    return (sum(s * s for s in shorts) / count) ** 0.5

app = FastAPI(title="Voice Compliance Agent - Dynamic Context Injection")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.get("/api/status")
async def get_status():
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "cloud-genai-yck"
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
    return {
        "status": "ONLINE",
        "provider": "Vertex AI",
        "project_id": project_id,
        "location": location,
        "model_id": "gemini-live-2.5-flash-native-audio",
        "audit_ledger": "audit_ledger.jsonl"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WebSocket] Browser client connected.")

    loop = asyncio.get_running_loop()

    # Event callback to push UI updates over WebSocket to browser
    def send_ui_event(event_type: str, payload: dict):
        msg = json.dumps({"type": event_type, "payload": payload})
        asyncio.run_coroutine_threadsafe(websocket.send_text(msg), loop)

    # Initialize Compliance Orchestrator targeting Vertex AI
    orchestrator = VoiceComplianceOrchestrator(on_event_cb=send_ui_event)
    config = orchestrator._build_connect_config()

    async def run_live_session():
        async with orchestrator.client.aio.live.connect(model=orchestrator.model_id, config=config) as session:
            orchestrator.session = session
            logger.info("[WebSocket] Gemini Live session connected via Vertex AI.")

            # Notify UI connection ready
            await websocket.send_text(json.dumps({
                "type": "SESSION_READY",
                "payload": {
                    "fsm_state": orchestrator.current_fsm_state,
                    "model": orchestrator.model_id,
                    "project_id": orchestrator.project_id
                }
            }))

            # Shared state to prevent microphone audio from interrupting Gemini while Gemini is speaking
            is_gemini_responding = False

            # Task 1: Receive from Browser microphone -> Send to Gemini Live API
            async def receive_from_browser():
                nonlocal is_gemini_responding
                mic_chunks = 0
                mic_bytes = 0
                try:
                    while True:
                        data = await websocket.receive()
                        if "bytes" in data and data["bytes"]:
                            # If Gemini is currently generating a spoken response, suppress mic forwarding to prevent echo/interruption
                            if is_gemini_responding:
                                continue

                            pcm_data = data["bytes"]
                            rms = calc_pcm_rms(pcm_data)
                            # Stream PCM audio chunk (16kHz mono 16-bit) to Gemini Live API
                            mic_chunks += 1
                            mic_bytes += len(pcm_data)
                            if mic_chunks == 1 or mic_chunks % 100 == 0:
                                logger.info(f"[Mic Stream] Forwarded mic chunk #{mic_chunks} (RMS: {int(rms)}).")

                            try:
                                await session.send_realtime_input(
                                    audio=types.Blob(mime_type="audio/pcm", data=pcm_data)
                                )
                            except Exception as mic_err:
                                logger.error(f"[Mic Forward Error] {mic_err}")
                                if "1011" in str(mic_err) or "closed" in str(mic_err).lower():
                                    raise mic_err

                        elif "text" in data and data["text"]:
                            cmd = json.loads(data["text"])
                            action = cmd.get("action")
                            
                            if action == "ping":
                                # Keepalive heartbeat from client
                                continue

                            elif action == "simulate_disclosure":
                                is_gemini_responding = True
                                event_type = cmd.get("event_type", "BEREAVEMENT")
                                summary = cmd.get("summary", "Simulated disclosure trigger from UI")
                                prompt = orchestrator.trigger_manual_disclosure(event_type, summary)
                                await session.send_client_content(
                                    turns=[types.Content(
                                        parts=[types.Part.from_text(text=prompt)]
                                    )],
                                    turn_complete=True
                                )

                            elif action == "send_text_prompt":
                                prompt_text = cmd.get("text", "")
                                if prompt_text:
                                    is_gemini_responding = True
                                    res = orchestrator.analyze_and_inject_compliance(prompt_text)
                                    final_prompt = res["formatted_prompt"]
                                    logger.info(f"[Text Prompt] Sending user text prompt to Gemini: {final_prompt}")
                                    await session.send_client_content(
                                        turns=[types.Content(
                                            parts=[types.Part.from_text(text=final_prompt)]
                                        )],
                                        turn_complete=True
                                    )

                            elif action == "trigger_kill_switch":
                                mode = cmd.get("mode", "SOFT_HANDOFF")
                                reason = cmd.get("reason", "Manual trigger from UI control panel")
                                await orchestrator.execute_kill_switch(mode, reason)
                                send_ui_event("KILL_SWITCH_TRIGGERED", {"mode": mode, "reason": reason})

                except WebSocketDisconnect:
                    logger.info("[WebSocket] Browser disconnected.")
                except Exception as e:
                    logger.error(f"[Browser Receive Error] {e}")
                    raise e

            # Task 2: Receive from Gemini Live API -> Send audio/text/events to Browser UI
            async def receive_from_gemini():
                nonlocal is_gemini_responding
                is_first_audio_chunk = True
                has_received_transcription = False
                try:
                    while True:
                        async for response in session.receive():
                            logger.info(response)
                            # 1. Handle Direct Tool Calls
                            if hasattr(response, "tool_call") and response.tool_call:
                                calls = getattr(response.tool_call, "function_calls", []) or []
                                for call in calls:
                                    await orchestrator.handle_function_call(call)

                            server_content = getattr(response, "server_content", None)
                            if not server_content:
                                continue

                            if getattr(server_content, "interrupted", False):
                                logger.info("[Gemini] Response generation interrupted.")
                                is_gemini_responding = False

                            # Real-time Transcriptions from Gemini (User input speech & Gemini output speech)
                            if getattr(server_content, "input_transcription", None) and getattr(server_content.input_transcription, "text", None):
                                usr_text = server_content.input_transcription.text
                                # Analyze live spoken input for regulatory disclosures
                                orchestrator.analyze_and_inject_compliance(usr_text)
                                await websocket.send_text(json.dumps({
                                    "type": "TRANSCRIPT_CHUNK",
                                    "payload": {"text": usr_text, "speaker": "You"}
                                }))

                            if getattr(server_content, "output_transcription", None) and getattr(server_content.output_transcription, "text", None):
                                is_gemini_responding = True
                                has_received_transcription = True
                                gem_text = server_content.output_transcription.text
                                await websocket.send_text(json.dumps({
                                    "type": "TRANSCRIPT_CHUNK",
                                    "payload": {"text": gem_text, "speaker": "Gemini"}
                                }))

                            # 2. Handle Turn Parts (Audio PCM & direct text)
                            if server_content.model_turn and server_content.model_turn.parts:
                                is_gemini_responding = True
                                for part in server_content.model_turn.parts:
                                    if hasattr(part, "function_call") and part.function_call:
                                        await orchestrator.handle_function_call(part.function_call)

                                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                                        audio_bytes = part.inline_data.data
                                        await websocket.send_bytes(audio_bytes)
                                        if is_first_audio_chunk and not has_received_transcription:
                                            await websocket.send_text(json.dumps({
                                                "type": "TRANSCRIPT_CHUNK",
                                                "payload": {"text": "🔊 [Gemini Spoken Response]", "speaker": "Gemini"}
                                            }))
                                            is_first_audio_chunk = False

                                    if hasattr(part, "text") and part.text:
                                        has_received_transcription = True
                                        await websocket.send_text(json.dumps({
                                            "type": "TRANSCRIPT_CHUNK",
                                            "payload": {"text": part.text, "speaker": "Gemini"}
                                        }))

                            if server_content.turn_complete:
                                is_gemini_responding = False
                                is_first_audio_chunk = True
                                has_received_transcription = False
                                await websocket.send_text(json.dumps({
                                    "type": "TURN_COMPLETE"
                                }))
                                logger.info("[Gemini] Turn complete. Re-entering session.receive() for next turn.")
                                break

                except Exception as e:
                    logger.error(f"[Gemini Receive Error] {e}")
                    raise e

            # Run both streaming tasks concurrently, canceling the other if one raises
            done, pending = await asyncio.wait(
                [asyncio.create_task(receive_from_browser()), asyncio.create_task(receive_from_gemini())],
                return_when=asyncio.FIRST_EXCEPTION
            )
            for p in pending:
                p.cancel()
            for d in done:
                if d.exception():
                    raise d.exception()

    try:
        await run_live_session()
    except Exception as e:
        logger.error(f"[Server Session Error] {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "ERROR",
                "payload": {"message": str(e)}
            }))
        except Exception:
            pass
    finally:
        logger.info("[WebSocket] Session closed.")


if __name__ == "__main__":
    import uvicorn
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_cert = "cert.pem" if os.path.exists("cert.pem") else None
    
    if ssl_key and ssl_cert:
        logger.info("[SSL] Found cert.pem and key.pem. Launching server with HTTPS / Secure WebSockets (WSS).")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, ssl_keyfile=ssl_key, ssl_certfile=ssl_cert, reload=False)
    else:
        logger.info("[HTTP] Launching server with HTTP.")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
