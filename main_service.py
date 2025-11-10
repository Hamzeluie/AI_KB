import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
import redis.asyncio as redis
from agent_architect.datatype_abstraction import Features, RAGFeatures, TextFeatures
from agent_architect.models_abstraction import (
    AbstractAsyncModelInference,
    AbstractInferenceServer,
    AbstractQueueManagerServer,
    DynamicBatchManager,
)
from agent_architect.session_abstraction import AgentSessions, SessionStatus
from agent_architect.utils import go_next_service
from fastapi import FastAPI
from model_with_inference_engine import ChatSession, LLMConfig, LLMManager
from model_with_inference_engine import config as llm_config
from model_with_inference_engine import llm_manager, vector_search
from utils import (  # Ensure truncated_json_dumps is available in utils
    get_env_variable,
    truncated_json_dumps,
)
from vllm import SamplingParams  # <-- ADD THIS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DASHBOARD_URL = "https://api-dev.vexu.ai/api/v1/server/user-profile"
HEADER = {
    "Authorization": "Bearer vexu_6W3Qr84dHNJRDHQIYdC3VLLb4eWsdko4MGGNTD99ttV6jvNN1K0PwZcXNGSc8dsO"
}
import re

PUNCTUATION_MARKS = {".", "!", "?", ";", ":", "\n"}
MAX_BUFFER_WORDS = 5  # or use char limit like MAX_BUFFER_CHARS = 100
MAX_BUFFER_CHARS = 5000

SYSTEM_MESSAGE_TEMPLATE_DEV = get_env_variable("SYSTEM_MESSAGE_TEMPLATE_DEV")
SYSTEM_PROMPT = SYSTEM_MESSAGE_TEMPLATE_DEV.format(
    owner_name="Fadi",
    owner_subject_pronouns="he",
    owner_object_pronouns="him",
    owner_possessive_adjectives="his",
)


def clean_text_simple(text):
    # Only allow letters, digits, space, and ?, $, %, ., ,, !, -, (, )
    return re.sub(r"[^a-zA-Z0-9\s?,$%.!()':-]", "", text)


# print(type(SYSTEM_PROMPT))
# print(f"Using SYSTEM PROMPT:\n{SYSTEM_PROMPT}\n")
# Add this near the top


class AsyncRagLlmInference(AbstractAsyncModelInference):
    def __init__(
        self,
        llm_manager: LLMManager,
        config: LLMConfig,
        queue_manager: AbstractQueueManagerServer = None,
        max_worker: int = 4,
    ):
        super().__init__(max_worker=max_worker)
        self.llm_manager = llm_manager
        self.config = config
        self.chat_sessions: Dict[str, ChatSession] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.queue_manager: Optional[AbstractQueueManagerServer] = queue_manager

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
                """
                async with aiohttp.ClientSession() as session:
                    params = {"user_id": str(req.owner_id)}
                    response = await session.get(DASHBOARD_URL, params=params, headers=HEADER)

                response = response.json()
                kb_ids = []
                for kb_id in response.json()['data']['user']['agents'][0]['knowledge_bases']:
                    kb_ids.append(kb_id['vexu_ai_kb_id'])
                system_prompt = response.json()['data']['user']['agents'][req.owner_id]['system_prompt']
                kb_limit = response.get("kb_limit", 5)
                """
                self.chat_sessions[req.sid] = ChatSession(
                    llm_manager=self.llm_manager,
                    system_instructions=SYSTEM_PROMPT,
                    kb_ids=["kb+12345952496_en"],
                    kb_limit=5,
                    config=self.config,
                )
            session = self.chat_sessions[req.sid]
            # RAG retrieval
            retrieved_docs = await vector_search(
                owner_id=req.owner_id,
                query_text=req.text,
                kb_id=session.kb_ids,
                limit=session.kb_limit,
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
                repetition_penalty=0.9,
                skip_special_tokens=True,
                stop=["<|im_end|>", "\n<|im_start|>"],
            )
            # request_id = f"inf-{req.sid}-{uuid.uuid4().hex[:8]}"
            request_id = req.sid
            current_text = ""
            prev_text = ""
            buffer = ""  # accumulates tokens until we decide to yield
            output_word = ""

            async for output in self.llm_manager.llm_engine.generate(
                prompt=final_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):

                if await self.queue_manager.is_session_active(req) == False:
                    await self.llm_manager.llm_engine.abort(request_id)
                    logger.info(
                        f"🟡 Aborted request {request_id} for inactive session {req.sid}"
                    )
                    return

                current_text = output.outputs[0].text
                delta = current_text[len(prev_text) :]
                prev_text = current_text
                delta = clean_text_simple(delta)
                # print("DELTA:", repr(delta))

                if not delta:
                    continue
                buffer += delta

                # Accumulate until we have a complete word (i.e., a space appears)
                try:
                    space_index = buffer.index(" ")
                    output_word += " " + buffer[:space_index]
                    buffer = buffer[space_index + 1 :]
                    if output.finished:
                        output_word += " " + buffer
                        text_feat = TextFeatures(
                            sid=req.sid,
                            agent_type=req.agent_type,
                            is_final=True,
                            text=output_word,
                            priority=req.priority,
                            created_at=time.time(),
                        )
                        buffer = ""
                        output_word = ""
                        yield text_feat
                        break
                except:
                    if output.finished:
                        output_word += " " + buffer

                        text_feat = TextFeatures(
                            sid=req.sid,
                            agent_type=req.agent_type,
                            is_final=True,
                            text=output_word,
                            priority=req.priority,
                            created_at=time.time(),
                        )
                        yield text_feat
                        buffer = ""
                        break
                    else:
                        continue

                # Heuristic 1: ends with sentence-like punctuation?
                stripped = output_word.rstrip()
                should_flush = stripped and stripped[-1] in PUNCTUATION_MARKS

                # Heuristic 2: buffer too long without punctuation?
                if not should_flush:
                    word_count = len(output_word.split())
                    char_count = len(output_word)
                    if word_count >= MAX_BUFFER_WORDS or char_count >= MAX_BUFFER_CHARS:
                        should_flush = True

                if should_flush and buffer:
                    text_feat = TextFeatures(
                        sid=req.sid,
                        agent_type=req.agent_type,
                        is_final=False,
                        text=output_word,
                        priority=req.priority,
                        created_at=time.time(),
                    )
                    output_word = ""
                    yield text_feat
            session.add_message("assistant", current_text)
            output_word = ""
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
        priorities: List[str] = ["high", "low"],
    ):
        self.redis_url = redis_url
        self.priorities = priorities
        self.service_name = service_name
        self.redis_client = None
        self.pubsub = None
        self.active_sessions_key = f"active_sessions"
        self.input_channels: List = [
            f"{self.service_name}:high",
            f"{self.service_name}:low",
        ]

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)

    async def get_status_object(self, req: Features) -> AgentSessions:
        raw = await self.redis_client.hget(
            f"{req.agent_type}:{self.active_sessions_key}", req.sid
        )
        if raw is None:
            return None
        return AgentSessions.from_json(raw)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(req)
        if status_obj is None:
            return False
        # change status of the session to 'stop' if the session expired
        if status_obj.is_expired():
            status_obj.status = SessionStatus.STOP
            await self.redis_client.hset(
                f"{req.agent_type}:{self.active_sessions_key}",
                req.sid,
                status_obj.to_json(),
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
                    status_obj = await self.get_status_object(req)
                    batch.append(
                        RAGFeatures(
                            sid=req.sid,
                            agent_id=status_obj.agent_id,
                            agent_type=status_obj.agent_type,
                            is_final=req.is_final,
                            text=req.text,
                            owner_id=status_obj.owner_id,
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
        # Check if session was stopped during processing
        if not await self.is_session_active(result):
            logger.info(f"❌ Not pushing result for inactive session: {result.sid}")
            return

        status_obj = await self.get_status_object(result)
        if status_obj is None:
            logger.error(f"Session status not found for {result.sid}")
            return

        status_obj.refresh_time()
        await self.redis_client.hset(
            f"{result.agent_type}:{self.active_sessions_key}",
            result.sid,
            status_obj.to_json(),
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
        # logger.info(f"Result pushed for request {result.sid}, to {next_service}")


class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        service_name: str = "RAG",
    ):
        super().__init__()
        self.service_name = service_name
        self.queue_manager = RedisQueueManager(
            redis_url, service_name=self.service_name
        )
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = AsyncRagLlmInference(
            llm_manager=llm_manager,
            config=llm_config,
            queue_manager=self.queue_manager,
            max_worker=max_worker,
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
            await asyncio.sleep(0.01)
            await self.queue_manager.push_result(text_feat)

    async def _process_batches_loop(self):
        logger.info("Starting batch processing loop")
        while self.is_running:
            try:
                batch = await self.queue_manager.get_data_batch(
                    max_batch_size=self.batch_manager.max_batch_size,
                    max_wait_time=self.batch_manager.max_wait_time,
                )
                if batch:
                    start_time = time.time()
                    for req in batch:
                        asyncio.create_task(self.process_and_send(req))
                    processing_time = time.time() - start_time
                    self.batch_manager.update_metrics(len(batch), processing_time)
                    # logger.info(f"Processed batch of {len(batch)} requests in {processing_time:.3f}s")
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
