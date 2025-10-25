#!/usr/bin/env python3
import os
import asyncio
import logging
import time
import redis.asyncio as redis
import sys
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI_VAD.models.abstract_models import Features, SessionStatus, transform_priority_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Define SttFeatures inline (as used by your RAG service) ---
class SttFeatures(Features):
    def __init__(self, sid: str, payload: Dict[str, Any], priority: str, created_at: float):
        self.sid = sid
        self.payload = payload
        self.priority = priority
        self.created_at = created_at

    def to_json(self) -> str:
        import json
        return json.dumps({
            "sid": self.sid,
            "payload": self.payload,
            "priority": self.priority,
            "created_at": self.created_at
        })

    @classmethod
    def from_json(cls, data: str):
        import json
        obj = json.loads(data)
        return cls(
            sid=obj["sid"],
            payload=obj["payload"],
            priority=obj["priority"],
            created_at=obj["created_at"]
        )


# Redis configuration (must match your RAG service)
INPUT_CHANNELS = ["STT:high", "STT:low"]
OUTPUT_CHANNELS = ["RAG:high", "RAG:low"]
ACTIVE_SESSIONS_KEY = "call_agent:active_sessions"


async def publish_stt_requests(redis_client, num_requests: int = 5):
    sids = [f"test_sid_{i}" for i in range(num_requests)]

    # Mark sessions as active
    for sid in sids:
        status = SessionStatus(
            sid=sid,
            status="active",
            created_at=time.time(),
            timeout=3000.0
        )
        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, status.to_json())

    tasks = []
    for i in range(num_requests):
        sid = sids[i]
        priority = "high" if i % 2 == 0 else "low"
        channel = f"STT:{priority}"

        stt_payload = {
            "transcript": f"This is a simulated STT result number {i} from session {sid}.",
            "language": "en-US",
            "confidence": 0.92,
            "owner_id": "+12345952496",
            "kb_id": "kb+12345952496_en",
            "limit": 5
        }

        stt_feat = SttFeatures(
            sid=sid,
            payload=stt_payload,
            priority=f"STT:{priority}",
            created_at=time.time()
        )

        logger.info(f"Publishing STT request (sid={sid}, priority={priority}) to {channel}")
        tasks.append(redis_client.lpush(channel, stt_feat.to_json()))

    await asyncio.gather(*tasks)
    logger.info(f"Published {num_requests} STT requests.")


async def listen_for_rag_results(redis_client, expected_count: int, timeout: int = 60):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(*OUTPUT_CHANNELS)

    results = []
    start_time = asyncio.get_event_loop().time()

    logger.info("Listening for RAG results...")
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel = message["channel"].decode()
        logger.info(f"✅ Received RAG result on {channel}")
        results.append(message["data"])

        if len(results) >= expected_count:
            break

        if asyncio.get_event_loop().time() - start_time > timeout:
            logger.warning("Timeout reached while waiting for RAG results.")
            break

    await pubsub.unsubscribe(*OUTPUT_CHANNELS)
    return results


async def main():
    NUM_REQUESTS = 5
    TIMEOUT = 60

    redis_client = await redis.from_url("redis://localhost:6379", decode_responses=False)

    # Start listener before publishing
    listener_task = asyncio.create_task(listen_for_rag_results(redis_client, NUM_REQUESTS, TIMEOUT))

    # Publish simulated STT outputs
    await publish_stt_requests(redis_client, num_requests=NUM_REQUESTS)

    # Wait for results
    try:
        results = await asyncio.wait_for(listener_task, timeout=TIMEOUT)
        logger.info(f"✅ Received {len(results)} / {NUM_REQUESTS} RAG results.")
    except asyncio.TimeoutError:
        logger.error("❌ Timeout: Not all RAG results received.")
        results = listener_task.result() if listener_task.done() else []

    # Cleanup: mark sessions as stopped
    sids = [f"test_sid_{i}" for i in range(NUM_REQUESTS)]
    for sid in sids:
        stop_status = SessionStatus(sid=sid, status="stop", created_at=None, timeout=0.0)
        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, stop_status.to_json())

    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())