import asyncio
import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

async def test_keepalive():
    client = genai.Client(vertexai=True, project="codemender-internal-demo", location="us-central1")
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text="You are Apex Bank AI assistant.")]
        )
    )

    print("Connecting...")
    async with client.aio.live.connect(model="gemini-live-2.5-flash-native-audio", config=config) as session:
        print("Connected!")
        
        async def send_dummy_audio():
            # Send 500 chunks of silent 16kHz PCM audio
            silent_chunk = b'\x00' * 1024 # 512 samples = 32ms
            for i in range(1, 500):
                await asyncio.sleep(0.032) # ~30 chunks per sec
                try:
                    await session.send_realtime_input(
                        audio=types.Blob(mime_type="audio/pcm;rate=16000", data=silent_chunk)
                    )
                    if i % 50 == 0:
                        print(f"Sent {i} silent chunks...")
                except Exception as e:
                    print(f"--> ERROR sending chunk {i}: {e}")
                    break

        async def receive_loop():
            try:
                async for response in session.receive():
                    print(f"Received response from Gemini: {response}")
            except Exception as e:
                print(f"--> ERROR in receive_loop: {e}")

        await asyncio.gather(send_dummy_audio(), receive_loop())

if __name__ == "__main__":
    asyncio.run(test_keepalive())
