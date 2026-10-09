import asyncio
import websockets
import ssl
import json

async def test_ws_audio():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    uri = "wss://localhost:8000/ws"
    try:
        async with websockets.connect(uri, ssl=ssl_context) as ws:
            print("Connected to server Secure WebSocket (WSS)!")
            
            ready_msg = await ws.recv()
            print("Server Ready Event:", ready_msg)
            
            # Send text speech prompt
            prompt_cmd = {
                "action": "send_text_prompt",
                "text": "Hello Gemini! Please introduce yourself briefly."
            }
            await ws.send(json.dumps(prompt_cmd))
            print("Sent text prompt to Gemini...")
            
            audio_count = 0
            total_bytes = 0
            while True:
                try:
                    reply = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(reply, bytes):
                        audio_count += 1
                        total_bytes += len(reply)
                        print(f"RECEIVED AUDIO CHUNK #{audio_count}: {len(reply)} bytes")
                    else:
                        print(f"RECEIVED EVENT: {reply}")
                except asyncio.TimeoutError:
                    print("Stream completed or timed out.")
                    break
                    
            print(f"\nSUCCESS! TOTAL AUDIO RECEIVED FROM GEMINI: {audio_count} chunks ({total_bytes} bytes)")
                
    except Exception as e:
        print("WS Test Error:", e)

if __name__ == "__main__":
    asyncio.run(test_ws_audio())
