import asyncio
import os
import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

async def test_conn():
    project_id = "codemender-internal-demo"
    location = "us-central1"
    
    # Let's test candidate models for Live Multimodal connection on Vertex AI
    candidate_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-001",
        "gemini-2.5-flash",
        "gemini-live-2.5-flash-native-audio"
    ]

    for model_id in candidate_models:
        print(f"\nTesting Vertex AI model: {model_id}...")
        client = genai.Client(vertexai=True, project=project_id, location=location)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"]
        )

        try:
            async with client.aio.live.connect(model=model_id, config=config) as session:
                print(f"SUCCESS! Connected to model: {model_id}")
                return model_id
        except Exception as e:
            print(f"Failed for model {model_id}:", type(e), e)

if __name__ == "__main__":
    asyncio.run(test_conn())
