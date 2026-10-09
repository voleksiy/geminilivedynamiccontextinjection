import asyncio
import json
import websockets

async def test_server_ws():
    import ssl
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    uri = "wss://localhost:8000/ws"
    print(f"Connecting to server WebSocket: {uri}")

    async with websockets.connect(uri, ssl=ssl_context) as ws:
        print("Connected!")

        # 1. Receive SESSION_READY
        init_msg = await ws.recv()
        print(f"Server init message: {init_msg}")

        async def run_turn(turn_num: int, prompt_dict: dict):
            print(f"\n--- Sending Turn {turn_num}: {prompt_dict} ---")
            await ws.send(json.dumps(prompt_dict))

            audio_chunks = 0
            transcript_chunks = 0

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=4.0)
                    if isinstance(msg, bytes):
                        audio_chunks += 1
                        if audio_chunks == 1 or audio_chunks % 10 == 0:
                            print(f"🔊 [Turn {turn_num}] Audio Chunk #{audio_chunks} ({len(msg)} bytes)")
                    else:
                        event = json.loads(msg)
                        msg_type = event.get("type")
                        if msg_type == "TRANSCRIPT_CHUNK":
                            transcript_chunks += 1
                            speaker = event["payload"]["speaker"]
                            text = event["payload"]["text"]
                            print(f"💬 [{speaker}]: {text}")
                        elif msg_type == "TURN_COMPLETE":
                            print(f"✅ [Turn {turn_num}] Received TURN_COMPLETE signal from Gemini! ({audio_chunks} audio chunks, {transcript_chunks} transcript parts)")
                        else:
                            print(f"📩 [Turn {turn_num}] Server Event: {event}")
                except asyncio.TimeoutError:
                    print(f"⏹️ [Turn {turn_num}] No more messages for 4s. Turn {turn_num} complete! Total audio chunks: {audio_chunks}")
                    break

            # Pause 2 seconds between turns to allow Gemini live audio pipeline to clear
            await asyncio.sleep(2.0)

        # TURN 1: Text Prompt
        await run_turn(1, {"action": "send_text_prompt", "text": "Hello! What services do you offer at Apex Bank?"})

        # TURN 2: Text Prompt
        await run_turn(2, {"action": "send_text_prompt", "text": "How can I open a new bank account with Apex Bank?"})

        # TURN 3: Simulate Disclosure
        await run_turn(3, {"action": "simulate_disclosure", "event_type": "BEREAVEMENT", "summary": "Customer husband passed away."})

        print("\n🎉 ALL 3 TURNS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_server_ws())
