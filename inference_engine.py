from abc import ABC
from typing import Any, Dict, List
from dataclasses import dataclass
import uuid
import asyncio
import logging
from vllm import SamplingParams  # <-- ADD THIS
from model import LLMManager, LLMConfig, ChatSession, vector_search


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

    async def warmup(self):
        if self.llm_manager.is_initialized():
            await self.llm_manager._warmup_llm()

    async def _prepare_batch_inputs(self, batch: List[InferenceRequest]) -> List[InferenceRequest]:
        return batch

    async def _run_async_model_inference(self, batch: List[InferenceRequest]) -> List[tuple]:
        tasks = [self._process_single(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(batch, results))

    async def _process_single(self, item: InferenceRequest) -> Dict:
        try:
            if item.sid not in self.chat_sessions:
                self.chat_sessions[item.sid] = ChatSession(self.llm_manager, self.config)

            session = self.chat_sessions[item.sid]

            # RAG
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

            full_response = ""
            async for output in self.llm_manager.llm_engine.generate(
                prompt=final_prompt,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                full_response = output.outputs[0].text

            session.add_message("assistant", full_response)

            return {
                "response_text": full_response,
                "retrieved_context": retrieved_docs,
                "error": None
            }

        except Exception as e:
            if item.sid in self.chat_sessions:
                self.chat_sessions[item.sid].reset_user_turn()
            logging.error(f"Error in _process_single for {item.sid}: {e}", exc_info=True)
            return {
                "response_text": "",
                "retrieved_context": retrieved_docs if 'retrieved_docs' in locals() else [],
                "error": str(e)
            }

    async def _process_batch_outputs(self, outputs: List[tuple], batch: List[InferenceRequest]) -> Dict[str, Any]:
        results = {}
        for req, res in outputs:
            if isinstance(res, Exception):
                results[req.sid] = {
                    'result': None,
                    'error': f"Processing failed: {str(res)}"
                }
            else:
                results[req.sid] = {
                    'result': {
                        'response_text': res['response_text'],
                        'retrieved_context': res['retrieved_context']
                    },
                    'error': res['error']
                }
        return results