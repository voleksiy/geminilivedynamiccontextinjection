import asyncio
from google import genai
from google.genai import types

async def test_two_turns():
    client = genai.Client(vertexai=True, project="codemender-internal-demo", location="us-central1")
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text="You are Apex Bank AI assistant. Speak warmly to the user in audio.")]
        )
    )

    async with client.aio.live.connect(model="gemini-live-2.5-flash-native-audio", config=config) as session:
        print("Connected!")

        # TURN 1
        print("\n--- TURN 1 ---")
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello! What services do you offer?")]
            ),
            turn_complete=True
        )

        audio_chunks_turn1 = 0
        async for response in session.receive():
            sc = getattr(response, "server_content", None)
            if sc and sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        audio_chunks_turn1 += 1
                        print(f"🔊 Turn 1 Audio Chunk #{audio_chunks_turn1}: {len(part.inline_data.data)} bytes")

            if sc and sc.turn_complete:
                print(f"Turn 1 Complete! Received {audio_chunks_turn1} audio chunks.")
                break

        # TURN 2
        print("\n--- TURN 2 ---")
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Can you tell me about mortgage options?")]
            ),
            turn_complete=True
        )

        audio_chunks_turn2 = 0
        async for response in session.receive():
            sc = getattr(response, "server_content", None)
            if sc and sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        audio_chunks_turn2 += 1
                        print(f"🔊 Turn 2 Audio Chunk #{audio_chunks_turn2}: {len(part.inline_data.data)} bytes")

            if sc and sc.turn_complete:
                print(f"Turn 2 Complete! Received {audio_chunks_turn2} audio chunks.")
                break

if __name__ == "__main__":
    asyncio.run(test_two_turns())
