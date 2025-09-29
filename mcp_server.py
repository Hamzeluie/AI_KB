# mcp_server.py
import asyncio
import json
import urllib.parse
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

# ✅ MODERN WebSocket import
from websockets.asyncio.client import connect as websockets_connect


# --- CONFIGURE THESE ---
load_dotenv()

API_KEY = "test"
PHONE = "+12345952496"
KB_ID = ["kb+12345952496_en"]
ENCODED_PHONE = urllib.parse.quote(PHONE)
RAG_URI = f"ws://localhost:5003/ws/search/{ENCODED_PHONE}"

# Global WebSocket connection
rag_ws = None


async def connect_to_rag():
    """Establish and maintain WebSocket connection to RAG service."""
    global rag_ws
    headers = {"api-key": API_KEY}
    print(f"📡 Connecting to RAG service: {RAG_URI}")
    try:
        rag_ws = await websockets_connect(
            RAG_URI,
            additional_headers=headers,
            ping_interval=20,      # ✅ Send ping every 20s to keep alive
            ping_timeout=10,       # ✅ Wait 10s for pong before failing
            close_timeout=5,       # ✅ Close cleanly within 5s
        )
        print("✅ Connected to RAG WebSocket successfully")
    except Exception as e:
        print(f"❌ Failed to connect to RAG: {type(e).__name__}: {e}")
        rag_ws = None


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print("🚀 Starting MCP server...")
    await connect_to_rag()
    print("✅ RAG connection initialized")

    yield  # Server runs here

    # Cleanup (optional)
    global rag_ws
    if rag_ws and rag_ws.state == "OPEN":
        await rag_ws.close()
        print("🔌 RAG WebSocket closed on shutdown")


mcp = FastMCP("RAG MCP Server", lifespan=lifespan)

    
@mcp.tool
async def search(query: str) -> str:
    """
    Search the knowledge base, this tool can be called to answer user queries about item's price.
    Sends the query and returns the response as a JSON string.
    """
    global rag_ws

    # Reconnect if needed
    if rag_ws is None or rag_ws.state != "OPEN":
        print("⚠️  WebSocket not connected. Attempting to reconnect...")
        await connect_to_rag()
        # We'll rely on try/except below — no need to double-check state

    try:
        request = {
            "query_text": query,
            "kb_id": KB_ID,
            "limit": 3,
        }

        await rag_ws.send(json.dumps(request, ensure_ascii=False))
        print(f"📤 Sent query: {query}")

        response = await asyncio.wait_for(rag_ws.recv(), timeout=10.0)
        print(f"📥 Received response: {response}")
        return response

    except asyncio.TimeoutError:
        return json.dumps({"error": "Request timeout"})
    except Exception as e:
        print(f"❌ Search error: {type(e).__name__}: {e}")
        return json.dumps({"error": str(e)})
    


if __name__ == "__main__":
    print("⏳ Initializing MCP server with RAG WebSocket...")
    mcp.run(transport="streamable-http", port=6000)  # ✅ Use streamable-http