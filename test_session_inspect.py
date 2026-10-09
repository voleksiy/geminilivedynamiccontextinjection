import asyncio
from google import genai
from google.genai import types

async def test_session():
    client = genai.Client(vertexai=True, project="codemender-internal-demo", location="us-central1")
    config = types.LiveConnectConfig(response_modalities=["AUDIO"])
    async with client.aio.live.connect(model="gemini-live-2.5-flash-native-audio", config=config) as session:
        print("Sending turns= Content to Gemini Live API...")
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello Gemini, please say hello back.")]
            ),
            turn_complete=True
        )

        audio_count = 0
        async for response in session.receive():
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_count += 1
                        print(f"Received audio chunk #{audio_count}: {len(part.inline_data.data)} bytes")
            if response.server_content and response.server_content.turn_complete:
                print("Turn complete!")
                break

if __name__ == "__main__":
    asyncio.run(test_session())
