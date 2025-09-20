import asyncio
import websockets
import urllib.parse
print("🚨 DEBUG: websockets version =", websockets.__version__)
# --- CONFIGURE THESE ---
API_KEY = "teseti"  # 👈 Replace with real key from your YAML
PHONE = "+12345952496"
# SESSION_ID = "e00f8d55-b5c7-4b87-9441-c7bf501e5366"

# URL-encode the phone number — + becomes %2B
ENCODED_PHONE = urllib.parse.quote(PHONE)

RAG_URI = f"ws://0.0.0.0:5003/ws/search/{ENCODED_PHONE}"



async def test_rag_connection():
    headers = {"api-key": API_KEY}
    try:
        async with websockets.connect(RAG_URI, extra_headers=headers, ping_interval=None) as ws:
            print("✅ RAG WebSocket: Connected successfully")
            # Optionally send/receive a test message
            # await ws.send('{"type": "ping"}')
            # response = await ws.recv()
            # print(f"RAG Response: {response}")
    except Exception as e:
        print(f"❌ RAG WebSocket Failed: {type(e).__name__}: {e}")



async def main():
    print("🔍 Starting WebSocket endpoint tests...\n")
    await test_rag_connection()
    print("\n🏁 Test completed.")


if __name__ == "__main__":
    asyncio.run(main())