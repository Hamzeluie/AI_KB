import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import redis.asyncio as redis
from abstract_models import *
from fastapi import FastAPI
from model_with_inference_engine import ChatSession, LLMConfig, LLMManager
from model_with_inference_engine import config as llm_config
from model_with_inference_engine import llm_manager, vector_search
from vllm import SamplingParams  # <-- ADD THIS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncRagLlmInference(AbstractAsyncModelInference):
    def __init__(self, llm_manager: LLMManager, config: LLMConfig, max_worker: int = 4):
        super().__init__(max_worker=max_worker)
        self.llm_manager = llm_manager
        self.config = config
        self.chat_sessions: Dict[str, ChatSession] = {}
        self.redis_client: Optional[redis.Redis] = None

    async def warmup(self):
        if self.llm_manager.is_initialized():
            await self.llm_manager._warmup_llm()

    async def _process_single(
        self, req: RAGFeatures
    ) -> AsyncGenerator[TextFeatures, None]:
        """Process a single InferenceRequest."""
        try:
            if req.sid not in self.chat_sessions:
                # ================================
                # CALL fro getting system prompt and temprature and ...
                # ================================
                self.chat_sessions[req.sid] = ChatSession(self.llm_manager, self.config)

            session = self.chat_sessions[req.sid]

            # RAG retrieval
            retrieved_docs = await vector_search(
                owner_id=req.owner_id,
                query_text=req.text,
                kb_id=req.kb_id,
                limit=req.kb_limit,
            )

            context_str = (
                "\n\n".join(
                    [
                        f"Context: {doc.content}\nSource: {doc.metadata.get('source', 'N/A')}"
                        for doc in retrieved_docs
                    ]
                )
                or "No relevant context found."
            )

            augmented_prompt = f"**Retrieved Context:**\n{context_str}\n\n**User Question:**\n{req.text}"
            session.add_message("user", augmented_prompt)

            final_prompt = session.get_formatted_prompt()

            sampling_params = SamplingParams(
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
                skip_special_tokens=True,
            )
            request_id = f"inf-{req.sid}-{uuid.uuid4().hex[:8]}"

            current_text = ""
            prev_text = ""
            async for output in self.llm_manager.llm_engine.generate(
                prompt=final_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                current_text = output.outputs[0].text
                delta = current_text[len(prev_text) :]
                if delta:
                    text_feat = TextFeatures(
                        sid=req.sid,
                        agent_name=req.agent_name,
                        text=delta,
                        priority=req.priority,
                        created_at=time.time(),
                    )
                    print(f"Streaming delta for {req.sid}: {delta}")
                    yield text_feat
                prev_text = current_text

            session.add_message("assistant", current_text)
            print(f"🤖=>Completed response for {req.sid}: {current_text}")

        except Exception as e:
            logger.error(f"Error processing request {req.sid}: {e}", exc_info=True)


class RedisQueueManager(AbstractQueueManagerServer):
    """
    Manages Redis-based async queue for inference requests.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        service_name: str = "RAG",
        queue_name: str = "call_agent",
        priorities: List[str] = ["high", "low"],
    ):
        self.redis_url = redis_url
        self.priorities = priorities
        self.service_name = service_name
        self.redis_client = None
        self.pubsub = None
        self.queue_name = queue_name
        self.active_sessions_key = f"{queue_name}:active_sessions"
        self.input_channels: List = [
            f"{self.service_name}:high",
            f"{self.service_name}:low",
        ]

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)

    async def get_status_object(self, sid: str) -> AgentSessions:
        raw = await self.redis_client.hget(self.active_sessions_key, sid)
        if raw is None:
            return None
        return AgentSessions.from_json(raw)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(req.sid)
        if status_obj is None:
            return False
        # change status of the session to 'stop' if the session expired
        if status_obj.is_expired():
            status_obj.status = SessionStatus.STOP
            await self.redis_client.hset(
                self.active_sessions_key, req.sid, status_obj.to_json()
            )
            return False
        elif status_obj.status == SessionStatus.INTERRUPT:
            return False
        return True

    async def get_data_batch(
        self, max_batch_size: int = 8, max_wait_time: float = 0.1
    ) -> List[TextFeatures]:
        batch = []
        start_time = time.time()
        while len(batch) < max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time and batch:
                break

            for input_channel in self.input_channels:
                result = await self.redis_client.brpop(input_channel, timeout=0.01)
                if result:
                    break

            if result:
                _, request_json = result
                try:
                    req = TextFeatures.from_json(request_json)
                    if not await self.is_session_active(req):
                        logger.info(f"Skipped request for stopped session: {req.sid}")
                        continue
                    status_obj = await self.get_status_object(req.sid)
                    batch.append(
                        RAGFeatures(
                            sid=req.sid,
                            agent_name=req.agent_name,
                            text=req.text,
                            owner_id=status_obj.owner_id,
                            kb_id=status_obj.kb_id,
                            kb_limit=status_obj.kb_limit,
                            priority=req.priority,
                            created_at=req.created_at,
                        )
                    )

                except Exception as e:
                    logger.error(f"Error in get_data_batch: {e}", exc_info=True)
            else:
                await asyncio.sleep(0.01)

        return batch

    async def push_result(self, result: TextFeatures):
        """Push inference result back to Redis pub/sub"""
        status_obj = await self.get_status_object(result.sid)
        status_obj.refresh_time()
        await self.redis_client.hset(
            self.active_sessions_key, result.sid, status_obj.to_json()
        )
        # calculate next service and queue name
        next_service = go_next_service(
            current_stage_name=self.service_name,
            service_names=status_obj.service_names,
            channels_steps=status_obj.channels_steps,
            last_channel=status_obj.last_channel,
            prioriry=result.priority,
        )
        await self.redis_client.lpush(next_service, result.to_json())
        logger.info(f"Result pushed for request {result.sid}, to {next_service}")


class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        queue_name: str = "call_agent",
    ):
        super().__init__()
        self.queue_manager = RedisQueueManager(redis_url, queue_name=queue_name)
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = AsyncRagLlmInference(
            llm_manager=llm_manager, config=llm_config, max_worker=max_worker
        )

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(req)

    async def _initialize_components(self):
        await self.queue_manager.initialize()

    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_batches_loop())

    async def process_and_send(self, req: RAGFeatures):
        async for text_feat in self.inference_engine._process_single(req):
            text_feat = TextFeatures(
                sid=text_feat.sid,
                agent_name=text_feat.agent_name,
                priority=text_feat.priority,
                text=text_feat.text,
                created_at=text_feat.created_at,
            )
            await asyncio.sleep(0.01)
            await self.queue_manager.push_result(text_feat)

    async def _process_batches_loop(self):
        logger.info("Starting batch processing loop")
        # Main processing loop @Borhan
        while self.is_running:
            try:
                batch = await self.queue_manager.get_data_batch(
                    max_batch_size=self.batch_manager.max_batch_size,
                    max_wait_time=self.batch_manager.max_wait_time,
                )
                print(f"Got batch of size {len(batch)} and batch = {batch}")
                if batch:
                    start_time = time.time()
                    for req in batch:
                        asyncio.create_task(self.process_and_send(req))
                    processing_time = time.time() - start_time
                    self.batch_manager.update_metrics(len(batch), processing_time)
                    logger.info(
                        f"Processed batch of {len(batch)} requests in {processing_time:.3f}s"
                    )
                else:
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)


chat_sessions: Dict[str, ChatSession] = {}  # Global session storage
inference_engine: Optional[AsyncRagLlmInference] = None  # NEW

from contextlib import asynccontextmanager

service = None  # Global reference to the service for shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, service
    logging.info("Application startup: Initializing LLM Manager...")
    await llm_manager.initialize()
    if llm_manager.is_initialized():
        inference_engine = AsyncRagLlmInference(llm_manager, llm_config, max_worker=4)
        await inference_engine.warmup()
        logging.info("LLM Manager initialized and warmed up.")

        # Instantiate and start the InferenceService here
        service = InferenceService()
        await service.start()
        service.inference_engine.redis_client = service.queue_manager.redis_client
        logging.info("InferenceService started.")
    else:
        logging.error("LLM Manager FAILED to initialize.")
    yield
    # Shutdown logic
    if service:
        service.is_running = False
        if service.processing_task:
            service.processing_task.cancel()
            try:
                await service.processing_task
            except asyncio.CancelledError:
                logging.info("Processing loop cancelled.")
        logging.info("InferenceService stopped.")
    logging.info("Application shutdown...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "LLM RAG server is running."}


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8101)
