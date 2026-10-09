import asyncio
from google import genai
from google.genai import types

async def test_tool_response_flow():
    client = genai.Client(vertexai=True, project="codemender-internal-demo", location="us-central1")
    
    disclosure_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="report_customer_disclosure",
                description="Report when customer discloses personal hardship or asks about investment advice.",
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
            parts=[types.Part.from_text(text="You are Apex Bank AI. Speak warmly to the customer.")]
        )
    )

    async with client.aio.live.connect(model="gemini-live-2.5-flash-native-audio", config=config) as session:
        print("Connected! Prompting user disclosure...")
        
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text="I need help with my mortgage, my husband passed away yesterday.")]
            ),
            turn_complete=True
        )

        audio_count = 0
        total_bytes = 0

        async for response in session.receive():
            if hasattr(response, "tool_call") and response.tool_call:
                for call in response.tool_call.function_calls:
                    print(f"\n[EVENT] Tool Call Received: {call.name} (id: {call.id})")
                    
                    # 1. Send tool_response
                    func_res = types.FunctionResponse(
                        name=call.name,
                        id=call.id,
                        response={
                            "result": "Policy #HARDSHIP-901 active: 60-day fee waiver & mortgage pause granted."
                        }
                    )
                    print("[1/2] Sending send_tool_response...")
                    await session.send_tool_response(function_responses=[func_res])
                    
                    # 2. Prompt Gemini to speak turn
                    print("[2/2] Sending turn_complete client content to trigger spoken output...")
                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part.from_text(text="Policy #HARDSHIP-901 is active. Please speak a compassionate message to the customer.")]
                        ),
                        turn_complete=True
                    )

            sc = getattr(response, "server_content", None)
            if sc and sc.model_turn:
                for part in sc.model_turn.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        audio_count += 1
                        total_bytes += len(part.inline_data.data)
                        print(f"🔊 Received Audio Chunk #{audio_count}: {len(part.inline_data.data)} bytes")
                    if hasattr(part, "text") and part.text:
                        print(f"💬 Text: {part.text}")

            if audio_count >= 10:
                print(f"\nSUCCESS! STREAMING AUDIO RESUMED AND PRODUCED {audio_count} CHUNKS ({total_bytes} bytes)!")
                break

if __name__ == "__main__":
    asyncio.run(test_tool_response_flow())
