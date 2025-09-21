import asyncio
import json
import urllib.parse

from websockets.legacy.client import connect as websockets_connect

# --- CONFIGURE THESE ---
API_KEY = "test"
# PHONE = "+12345952496"
PHONE = "123"
QUERY = "السعر بلوتوث"
KB_ID = ["test_kb"]

ENCODED_PHONE = urllib.parse.quote(PHONE)
RAG_URI = f"ws://localhost:5003/ws/search/{PHONE}"


async def test_rag_connection():
    headers = {"api-key": API_KEY}
    print(f"📡 Connecting to: {RAG_URI}")
    try:
        async with websockets_connect(
            RAG_URI, extra_headers=headers, ping_interval=None
        ) as ws:
            print("✅ RAG WebSocket: Connected successfully")

            # --- 1. Wrapped Query Test ---
            request = {
                "query_text": QUERY,
                "kb_id": KB_ID,
                "limit": 3,
            }
            await ws.send(json.dumps(request))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"📡 Wrapped Query Response: {response}")

            # --- 2. Flat Query Test ---
            flat_query = {
                "query_text": QUERY,
                "kb_id": KB_ID,
                "limit": 3,
            }
            await ws.send(json.dumps(flat_query, ensure_ascii=False))
            response = await ws.recv()
            print(f"📡 Flat Query Response: {response}")

    except Exception as e:
        print(f"❌ RAG WebSocket Failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


async def main():
    print("🔍 Starting WebSocket endpoint tests...\n")
    await test_rag_connection()
    print("\n🏁 Test completed.")


if __name__ == "__main__":
    asyncio.run(main())
