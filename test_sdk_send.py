import asyncio
import os
import logging
from google import genai
from google.genai import types
from voice_compliance_agent import VoiceComplianceOrchestrator

logging.basicConfig(level=logging.INFO)

async def test_while_receive_loop():
    orchestrator = VoiceComplianceOrchestrator()
    config = orchestrator._build_connect_config()

    async with orchestrator.client.aio.live.connect(model=orchestrator.model_id, config=config) as session:
        print("Connected to Gemini Live via Orchestrator Config!")

        turn_complete_event = asyncio.Event()

        async def receive_loop():
            audio_count = 0
            while True:
                try:
                    async for response in session.receive():
                        if response.server_content:
                            if response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.text:
                                        print(f"💬 [Gemini]: {part.text}")
                                    if part.inline_data:
                                        audio_count += 1
                                        if audio_count == 1 or audio_count % 10 == 0:
                                            print(f"🔊 Audio chunk #{audio_count}")
                            if response.server_content.turn_complete:
                                print(f"✅ Received TURN_COMPLETE from Gemini! Total audio: {audio_count}")
                                audio_count = 0
                                turn_complete_event.set()
                                # BREAK TO RE-ENTER session.receive() FOR NEXT TURN!
                                break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Receive error: {e}")
                    break

        recv_task = asyncio.create_task(receive_loop())

        # TURN 1
        print("\n--- Sending Turn 1 ---")
        turn_complete_event.clear()
        p1 = orchestrator.analyze_and_inject_compliance("Hello! What services do you offer?")["formatted_prompt"]
        await session.send_client_content(turns=[types.Content(parts=[types.Part.from_text(text=p1)])], turn_complete=True)
        await turn_complete_event.wait()
        await asyncio.sleep(1.0)

        # TURN 2
        print("\n--- Sending Turn 2 ---")
        turn_complete_event.clear()
        p2 = orchestrator.analyze_and_inject_compliance("How can I open a bank account?")["formatted_prompt"]
        await session.send_client_content(turns=[types.Content(parts=[types.Part.from_text(text=p2)])], turn_complete=True)
        await turn_complete_event.wait()
        await asyncio.sleep(1.0)

        # TURN 3
        print("\n--- Sending Turn 3 ---")
        turn_complete_event.clear()
        p3 = orchestrator.trigger_manual_disclosure("BEREAVEMENT", "Customer husband passed away.")
        await session.send_client_content(turns=[types.Content(parts=[types.Part.from_text(text=p3)])], turn_complete=True)
        await turn_complete_event.wait()
        await asyncio.sleep(1.0)

        recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_while_receive_loop())
