import asyncio
import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

async def test_audio_out():
    client = genai.Client(vertexai=True, project="codemender-internal-demo", location="us-central1")
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"]
    )
    model_id = "gemini-live-2.5-flash-native-audio"

    print("Connecting to Gemini Live API...")
    async with client.aio.live.connect(model=model_id, config=config) as session:
        print("Connected! Sending prompt: 'Hello Gemini, please say hello back in spoken audio.'")
        
        # Send text input prompt using session.send
        await session.send(input="Hello Gemini, please introduce yourself in spoken audio.", end_of_turn=True)

        print("Listening for audio responses...")
        audio_chunk_count = 0
        total_bytes = 0

        async for response in session.receive():
            sc = response.server_content
            if sc and sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_chunk_count += 1
                        total_bytes += len(part.inline_data.data)
                        print(f"Received audio chunk #{audio_chunk_count}: {len(part.inline_data.data)} bytes (mime: {part.inline_data.mime_type})")
                    if part.text:
                        print(f"Received text chunk: {part.text}")
            if sc and sc.turn_complete:
                print("Turn complete!")
                break

        print(f"\nTOTAL AUDIO RECEIVED: {audio_chunk_count} chunks, {total_bytes} bytes!")

if __name__ == "__main__":
    asyncio.run(test_audio_out())
