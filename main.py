import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from langchain_core.documents import Document
from pydantic import BaseModel
from vector_db import MultiTenantVectorDB

# Load YAML configuration first so it can be used in route definitions
load_dotenv()
BACKEND_API_KEY = os.getenv("API_KEY")


if not BACKEND_API_KEY:
    raise RuntimeError("BACKEND_API_KEY environment variable is required")


# async def verify_backend_api_key(request: Request, api_key: str = Header(None)):


async def verify_backend_api_key(api_key: str = Header(..., alias="api-key")):
    if api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key


# app = FastAPI(dependencies=[Depends(verify_backend_api_key)])
app = FastAPI()


db = MultiTenantVectorDB(
    config_path=os.path.join(Path(__file__).parents[0], "emb.yaml")
)


class DocumentInput(BaseModel):
    kb_id: str
    owner_id: str
    document: Union[Dict[str, Any], str]


class SearchRequest(BaseModel):
    query_text: str
    kb_id: List[str] = None
    limit: int = 10


class SearchResult(BaseModel):
    score: float
    content: str
    metadata: dict


class SearchResponse(BaseModel):
    status: str
    owner_id: str
    kb_id: List[str]
    query: str
    results: List[SearchResult]
    count: int


@app.post(f"/db/{{owner_id}}")
async def create_document(doc_input: DocumentInput):
    if db.kb_exists(owner_id=doc_input.owner_id, kb_id=doc_input.kb_id):
        print(f"User with KB {doc_input.kb_id} Exists")
    print(doc_input)

    # Handle document regardless of whether it's string or dict
    if isinstance(doc_input.document, str):
        try:
            document_data = json.loads(doc_input.document)
            if "body" in document_data:
                document_type = "text"
                document = document_data["body"]
            else:
                document_type = "json"
                document = doc_input.document  # Keep as string for JSON case
        except json.JSONDecodeError:
            document_type = "text"
            document = doc_input.document
    else:
        # It's already a dict, convert to string for page_content
        document_type = "json"
        # document = json.dumps(doc_input.document)  # Convert dict to JSON string
        document = json.dumps(
            doc_input.document, ensure_ascii=False
        )  # Convert dict to JSON string

    # Ensure page_content is a string
    document = Document(page_content=str(document), metadata={"source": "history.json"})
    db.add_documents(
        owner_id=doc_input.owner_id,
        kb_id=doc_input.kb_id,
        documents=document,
        doc_type=document_type,
    )
    return {
        "status": "success",
        "owner_id": doc_input.owner_id,
        "kb_id": doc_input.kb_id,
        "document": doc_input.document,
    }


@app.get(f"/db/{{owner_id}}")
async def get_documents(owner_id: str):

    return {
        "status": "success",
        "owner_id": owner_id,
        "documents": owner_id,
        "count": 0,
    }


@app.delete(f"/db/{{owner_id}}/{{kb_id}}")
async def delete_documents(owner_id: str, kb_id: str):
    if not db.kb_exists(owner_id=owner_id, kb_id=kb_id):
        raise HTTPException(status_code=404, detail="User or KB not found")

    db.delete_documents(owner_id=owner_id, kb_id=kb_id)
    return {"status": "success", "owner_id": owner_id, "message": "Document deleted"}


@app.post(
    f"/db/search/{{owner_id}}",
    response_model=SearchResponse,
)
async def search_documents(owner_id: str, request: SearchRequest):
    """
    Process a search request and return results from Qdrant.

    Example request body:
    {
        "query_text": "What is the price?",
        "kb_id": "kb123",
        "limit": 3
    }
    """
    try:
        # Step 1: Process the request (Query Qdrant)
        raw_results = db.search(
            owner_id=owner_id,
            query_text=request.query_text,
            kb_id=request.kb_id,
            limit=request.limit,
        )

        # Step 2: Format results
        formatted_results = [
            SearchResult(
                score=res["score"],
                content=res["content"],
                metadata=res["payload"].get("metadata", {}),
            )
            for res in raw_results
        ]

        # Step 3: Return structured response
        return SearchResponse(
            status="success",
            owner_id=owner_id,
            kb_id=request.kb_id,
            query=request.query_text,
            results=formatted_results,
            count=len(formatted_results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.websocket(f"/ws/search/{{owner_id}}")
async def websocket_search(websocket: WebSocket, owner_id: str):
    logger.info(f"WebSocket connection attempt for owner_id={owner_id}")
    logger.info(f"Headers: {dict(websocket.headers)}")
    # 🛡️ Manually check api-key from headers
    if "api-key" not in websocket.headers:
        await websocket.send_json({"status": "error", "message": "Missing API Key"})
        await websocket.close(code=1008, reason="Unauthorized")
        return

    api_key = websocket.headers["api-key"]

    logger.info(f"Received api-key: {api_key}")
    logger.info(f"Expected BACKEND_API_KEY: {BACKEND_API_KEY}")

    if api_key != BACKEND_API_KEY:
        await websocket.send_json(
            {"status": "error", "message": "Invalid or missing API Key"}
        )
        await websocket.close(code=1008, reason="Unauthorized")
        return

    print(f"✅ WebSocket authenticated for owner_id={owner_id}")
    await websocket.accept()  # ← Accept after auth

    try:
        while True:
            data = await websocket.receive_json()

            try:
                if not all(k in data for k in ["query_text", "kb_id"]):
                    await websocket.send_json(
                        {
                            "status": "error",
                            "message": "Missing required fields: query_text, kb_id",
                        }
                    )
                    continue

                limit = data.get("limit", 3)
                raw_results = db.search(
                    owner_id=owner_id,
                    query_text=data["query_text"],
                    kb_id=data["kb_id"],
                    limit=limit,
                )

                formatted_results = [
                    {
                        "score": res["score"],
                        "content": res["content"],
                        "metadata": res["payload"].get("metadata", {}),
                    }
                    for res in raw_results
                ]

                await websocket.send_json(
                    {
                        "status": "success",
                        "owner_id": owner_id,
                        "kb_id": data["kb_id"],
                        "query": data["query_text"],
                        "results": formatted_results,
                        "count": len(formatted_results),
                    }
                )

            except Exception as e:
                await websocket.send_json({"status": "error", "message": str(e)})

    except WebSocketDisconnect:
        print(f"Client {owner_id} disconnected")
    except Exception as e:
        await websocket.send_json(
            {"status": "error", "message": f"Unexpected error: {str(e)}"}
        )
        await websocket.close()


if __name__ == "__main__":
    # main()
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
