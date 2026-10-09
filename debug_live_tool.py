import asyncio
from google import genai
from google.genai import types

async def debug_live():
    client = genai.Client(vertexai=True, project="codemender-internal-demo", location="us-central1")
    
    disclosure_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="report_customer_disclosure",
                description="Call this function when the customer discloses personal hardship.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "event_type": types.Schema(type="STRING"),
                        "user_summary": types.Schema(type="STRING")
                    },
                    required=["event_type"]
                )
            )
        ]
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[disclosure_tool],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text="You are Apex Bank AI assistant.")]
        )
    )

    async with client.aio.live.connect(model="gemini-live-2.5-flash-native-audio", config=config) as session:
        print("Connected!")
        await session.send_client_content(
            turns=[types.Content(
                role="user",
                parts=[types.Part.from_text(text="I need help, my husband passed away yesterday.")]
            )],
            turn_complete=True
        )

        audio_count = 0
        tool_call_resolved = False

        async for response in session.receive():
            print("\n[LOOP TICK] Received message from Gemini:")
            if hasattr(response, "tool_call") and response.tool_call:
                for call in response.tool_call.function_calls:
                    print(f"  --> [1] Tool Call Received: {call.name} (id: {call.id})")
                    func_res = types.FunctionResponse(
                        name=call.name,
                        id=call.id,
                        response={"status": "HARDSHIP_POLICY_901_ACTIVE"}
                    )
                    await session.send_tool_response(function_responses=[func_res])
                    tool_call_resolved = True
                    print("  --> [2] send_tool_response sent!")

            sc = getattr(response, "server_content", None)
            if sc and sc.turn_complete and tool_call_resolved:
                print("  --> [3] Tool call turn completed! Testing send_realtime_input with audio/text...")
                tool_call_resolved = False
                await session.send_realtime_input(text="Please speak a compassionate message regarding mortgage pause.")

            if sc and sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        audio_count += 1
                        print(f"  🔊 [AUDIO CHUNK #{audio_count}]: {len(part.inline_data.data)} bytes")

            if audio_count >= 5:
                print(f"\n🎉 SUCCESS! RECEIVED {audio_count} AUDIO CHUNKS AFTER TOOL CALL!")
                break

if __name__ == "__main__":
    asyncio.run(debug_live())
