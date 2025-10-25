from logging import config
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_VAD.models.abstract_models import *

import base64
import numpy as np
import base64
import logging
import numpy as np
import time
import redis.asyncio as redis
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass
from abc import ABC
from typing import Any, Dict, List
from dataclasses import dataclass
import uuid
import asyncio
import logging
from vllm import SamplingParams  # <-- ADD THIS
from model_with_inference_engine import LLMManager, LLMConfig, ChatSession, vector_search
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from model_with_inference_engine import llm_manager
from model_with_inference_engine import config as llm_config

@dataclass
class SttFeatures(Features):
    sid: str
    payload: Dict[str, Any]
    priority: str
    created_at: float

@dataclass
class TextFeatures(Features):
    sid: str
    response_text: str
    retrieved_context: List[Any]  # or a proper Doc type
    priority: str
    created_at: float
    delta: Optional[str] = None
    is_final: bool = False

    def to_json(self) -> str:
        import json
        # Convert retrieved_context to serializable form if needed
        context_serializable = [
            {"content": doc.content, "metadata": doc.metadata} 
            for doc in self.retrieved_context
        ] if self.retrieved_context else []
        return json.dumps({
            "sid": self.sid,
            "response_text": self.response_text,
            "retrieved_context": context_serializable,
            "priority": self.priority,
            "created_at": self.created_at,
            "delta": self.delta,
            "is_final": self.is_final
        })

import base64
import numpy as np
import time
from typing import Any, Dict, List



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  
call_channels = ChannelNames(input_channel=["STT:high", "STT:low"], output_channel=["RAG:high", "RAG:low"])
    
@dataclass
class InferenceRequest:
    """Internal representation of a batch request item."""
    sid: str  # session_id
    owner_id: str
    user_prompt: str
    kb_id: str
    limit: int = 5


class AbstractAsyncModelInference(ABC):
    """
    Abstract base class for asynchronous model inference with dynamic batching.
    Subclasses must implement model-specific logic.
    """

    def __init__(self, max_worker: int = 4):
        # ThreadPool is kept for compatibility but NOT used for vLLM
        from concurrent.futures import ThreadPoolExecutor
        self.thread_pool = ThreadPoolExecutor(max_workers=max_worker)
        self.stats = {
            'total_batches': 0,
            'total_requests': 0,
            'avg_batch_size': 0,
            'avg_inference_time': 0
        }

    async def process_batch(self, batch: List[InferenceRequest]) -> Dict[str, Any]:
        """Process a batch of requests asynchronously."""
        import time
        start_time = time.time()
        try:
            batch_inputs = await self._prepare_batch_inputs(batch)
            # Use async inference — vLLM cannot run in thread pool
            batch_outputs = await self._run_async_model_inference(batch_inputs)
            results = await self._process_batch_outputs(batch_outputs, batch)
            self._update_stats(len(batch), time.time() - start_time)
            return results
        except Exception as e:
            return await self._handle_batch_error(batch, e)

    async def warmup(self):
        """Perform model warm-up (to be implemented by subclass)."""
        raise NotImplementedError

    async def _prepare_batch_inputs(self, batch: List[InferenceRequest]) -> Any:
        raise NotImplementedError

    async def _run_async_model_inference(self, prepared_inputs: Any) -> Any:
        """Run async inference (e.g., with vLLM). Must be implemented."""
        raise NotImplementedError

    async def _process_batch_outputs(self, outputs: Any, batch: List[InferenceRequest]) -> Dict[str, Any]:
        raise NotImplementedError

    async def _handle_batch_error(self, batch: List[InferenceRequest], error: Exception) -> Dict[str, Any]:
        error_results = {}
        for request in batch:
            error_results[request.sid] = {
                'result': None,
                'error': str(error)
            }
        return error_results

    def _update_stats(self, batch_size: int, processing_time: float):
        self.stats['total_batches'] += 1
        self.stats['total_requests'] += batch_size
        self.stats['avg_batch_size'] = (
            self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + batch_size
        ) / self.stats['total_batches']
        self.stats['avg_inference_time'] = (
            self.stats['avg_inference_time'] * (self.stats['total_batches'] - 1) + processing_time
        ) / self.stats['total_batches']


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

    # Override process_batch to accept SttFeatures and return TextFeatures
    async def process_batch(self, batch: List[SttFeatures]) -> List[TextFeatures]:
        """Process a batch of SttFeatures and return TextFeatures results."""
        import time
        start_time = time.time()
        try:
            # Convert SttFeatures → InferenceRequest
            inference_requests = await self._prepare_batch_inputs(batch)
            # Run LLM inference
            batch_outputs = await self._run_async_model_inference(inference_requests, batch)
            # Convert outputs → TextFeatures (using original SttFeatures for priority/sid)
            results = await self._process_batch_outputs(batch_outputs, batch)
            self._update_stats(len(batch), time.time() - start_time)
            return results
        except Exception as e:
            return await self._handle_batch_error(batch, e)

    async def _prepare_batch_inputs(self, batch: List[SttFeatures]) -> List[InferenceRequest]:
        """Convert SttFeatures to InferenceRequest."""
        requests = []
        for feat in batch:
            p = feat.payload
            req = InferenceRequest(
                sid=feat.sid,
                owner_id=p["owner_id"],
                user_prompt=p["transcript"],
                kb_id=p["kb_id"],
                limit=p.get("limit", 5)
            )
            requests.append(req)
        return requests

    async def _run_async_model_inference(self, inference_requests: List[InferenceRequest], original_batch: List[SttFeatures]) -> List[tuple]:
        """Run async inference on a batch of InferenceRequest."""
        tasks = [self._process_single(item, feat) for item, feat in zip(inference_requests, original_batch)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(inference_requests, results))

    async def _process_single(self, item: InferenceRequest, feat: SttFeatures) -> Dict[str, Any]:
        """Process a single InferenceRequest."""
        try:
            if item.sid not in self.chat_sessions:
                self.chat_sessions[item.sid] = ChatSession(self.llm_manager, self.config)

            session = self.chat_sessions[item.sid]

            # RAG retrieval
            retrieved_docs = await vector_search(
                owner_id=item.owner_id,
                query_text=item.user_prompt,
                kb_id=item.kb_id,
                limit=item.limit
            )

            context_str = "\n\n".join([
                f"Context: {doc.content}\nSource: {doc.metadata.get('source', 'N/A')}"
                for doc in retrieved_docs
            ]) or "No relevant context found."

            augmented_prompt = f"**Retrieved Context:**\n{context_str}\n\n**User Question:**\n{item.user_prompt}"
            session.add_message("user", augmented_prompt)

            final_prompt = session.get_formatted_prompt()

            sampling_params = SamplingParams(
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
                skip_special_tokens=True,
            )
            request_id = f"inf-{item.sid}-{uuid.uuid4().hex[:8]}"

            current_text = ""
            prev_text = ""
            async for output in self.llm_manager.llm_engine.generate(
                prompt=final_prompt,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                current_text = output.outputs[0].text
                delta = current_text[len(prev_text):]
                if delta:
                    text_feat = TextFeatures(
                        sid=item.sid,
                        response_text=current_text,
                        retrieved_context=[],
                        priority=feat.priority,
                        created_at=time.time(),
                        delta=delta,
                        is_final=False
                    )
                    print(f"Streaming delta for {item.sid}: {delta}")
                    await self.redis_client.lpush(feat.priority, text_feat.to_json())
                prev_text = current_text

            session.add_message("assistant", current_text)

            text_feat = TextFeatures(
                sid=item.sid,
                response_text=current_text,
                retrieved_context=retrieved_docs,
                priority=feat.priority,
                created_at=time.time(),
                delta="",
                is_final=True
            )
            await self.redis_client.lpush(feat.priority, text_feat.to_json())

            return {
                "error": None
            }

        except Exception as e:
            if item.sid in self.chat_sessions:
                self.chat_sessions[item.sid].reset_user_turn()
            logging.error(f"Error in _process_single for {item.sid}: {e}", exc_info=True)
            text_feat = TextFeatures(
                sid=item.sid,
                response_text="",
                retrieved_context=retrieved_docs if 'retrieved_docs' in locals() else [],
                priority=feat.priority,
                created_at=time.time(),
                delta="",
                is_final=True
            )
            await self.redis_client.lpush(feat.priority, text_feat.to_json())
            return {
                "error": str(e)
            }

    async def _process_batch_outputs(
        self, 
        outputs: List[tuple], 
        original_batch: List[SttFeatures]
    ) -> List[TextFeatures]:
        """Convert raw outputs to TextFeatures using original SttFeatures for metadata."""
        return []

    async def _handle_batch_error(self, batch: List[SttFeatures], error: Exception) -> List[TextFeatures]:
        """Handle batch-level errors by returning empty TextFeatures."""
        return []
                
class RedisQueueManager(AbstractQueueManagerServer):
    """
    Manages Redis-based async queue for inference requests.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", queue_name: str = "call_agent"):
        self.redis_url = redis_url
        self.channels_name = call_channels.get_all_channels()
        self.redis_client = None
        self.pubsub = None
        self.queue_name = queue_name
        self.active_sessions_key = f"{queue_name}:active_sessions"
        self.priorities = get_high_low(call_channels.output_channel)
        self.in_priorities = get_high_low(call_channels.input_channel)

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)
        self.pubsub = self.redis_client.pubsub()
        for channel_name in self.channels_name:
            await self.pubsub.subscribe(channel_name)
        logger.info(f"Redis queue manager initialized for queue: {self.queue_name}")
        
    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        raw = await self.redis_client.hget(self.active_sessions_key, req.sid)
        if raw is None:
            return False
        status_obj = SessionStatus.from_json(raw)
        # change status of the session to 'stop' if the session expired
        if status_obj.is_expired():
            status_obj.status = "stop"
            await self.redis_client.hset(self.active_sessions_key, req.sid, status_obj.to_json())
            return False
        elif status_obj.status == b"interrupt":
            return False
        # update create_at time of the session
        status_obj.refresh_time()
        await self.redis_client.hset(self.active_sessions_key, req.sid, status_obj.to_json())
        return True

    async def get_data_batch(self, max_batch_size: int = 8, max_wait_time: float = 0.1) -> List[SttFeatures]:
        batch = []
        start_time = time.time()
        while len(batch) < max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time and batch:
                break
            
            result = await self.redis_client.brpop(self.in_priorities["high"], timeout=0.01)
            if not result:
                result = await self.redis_client.brpop(self.in_priorities["low"], timeout=0.01)
            
            if result:
                _, request_json = result
                try:
                    req = SttFeatures.from_json(request_json)
                    if not await self.is_session_active(req):
                        logger.info(f"Skipped request for stopped session: {req.sid}")
                        continue
                    req.priority = transform_priority_name(self.priorities, req.priority)
                    batch.append(req)
                except Exception as e:
                    logger.error(f"Error in get_data_batch: {e}", exc_info=True)
            else:
                await asyncio.sleep(0.01)

        return batch
    
    async def push_result(self, result: TextFeatures, channel_name:str, error: str = None):
        """Push inference result back to Redis pub/sub"""
        await self.redis_client.lpush(channel_name, result.to_json())
        logger.info(f"Result pushed for request {result.sid}")
   



chat_sessions: Dict[str, ChatSession] = {} # Global session storage
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

class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        queue_name: str = "call_agent"
    ):
        super().__init__()
        self.queue_manager = RedisQueueManager(redis_url, queue_name=queue_name)
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = AsyncRagLlmInference(llm_manager=llm_manager, config=llm_config,max_worker=max_worker)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(req)
    
    async def _initialize_components(self):
        await self.queue_manager.initialize()
        # await self.inference_engine.model_manager.initialize()
    
    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_batches_loop())
    
    async def _process_batches_loop(self):
        logger.info("Starting batch processing loop")
        # Main processing loop @Borhan
        while self.is_running:
        # try:
            batch = await self.queue_manager.get_data_batch(
                max_batch_size=self.batch_manager.max_batch_size,
                max_wait_time=self.batch_manager.max_wait_time
            )
            print(f"Got batch of size {len(batch)} and batch = {batch}")
            if batch:
                start_time = time.time()
                batch_results = await self.inference_engine.process_batch(batch)
                print(f"Batch results: {batch_results}")
                processing_time = time.time() - start_time
                # No push here since streaming pushes inside _process_single
                self.batch_manager.update_metrics(len(batch), processing_time)
                logger.info(f"Processed batch of {len(batch)} requests in {processing_time:.3f}s")
            else:
                await asyncio.sleep(0.01)
            
        # except Exception as e:
        #     logger.error(f"Error in batch processing loop: {e}")
        #     await asyncio.sleep(0.1)
        


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)