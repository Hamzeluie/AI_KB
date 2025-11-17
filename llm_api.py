from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from pydantic import BaseModel
from typing import List, Dict, Optional, AsyncGenerator
import asyncio
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMConfig:
    """Encapsulates all server configuration constants."""
    # vLLM Model Configuration
    HF_MODEL_ID: str = "Qwen/Qwen3-32B-AWQ"
    MAX_TOKENS: int = 22768  # Max *generation* tokens
    TEMPERATURE: float = 0.7
    DTYPE: str = "float16"
    QUANTIZATION: str = "awq_marlin"
    LOCAL_MODEL_DIR: str = f"./local_models/{HF_MODEL_ID}"
    # Chat History Configuration
    MAX_CONTEXT_TOKENS: int = 32768 # Max tokens for *total* prompt (history + context + query)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = True

class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[Dict]
    model: str

class ChatCompletionStreamResponse(BaseModel):
    id: str
    choices: List[Dict]
    model: str

app = FastAPI(title="vLLM Streaming Service", version="1.0.0")

# Global LLM engine instance
engine = None

def initialize_engine():
    global engine
    try:
        logger.info(f"Initializing AsyncLLMEngine with model: {LLMConfig.HF_MODEL_ID}")
        
        # Check if local model directory exists, otherwise use HF model ID directly
        model_path = LLMConfig.LOCAL_MODEL_DIR if os.path.exists(LLMConfig.LOCAL_MODEL_DIR) else LLMConfig.HF_MODEL_ID
        
        # Create engine arguments
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype=LLMConfig.DTYPE,
            quantization=LLMConfig.QUANTIZATION,
            tensor_parallel_size=1,  # Adjust based on your GPU setup
            gpu_memory_utilization=0.9,
            max_model_len=LLMConfig.MAX_CONTEXT_TOKENS
        )
        
        # Initialize the engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("AsyncLLMEngine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AsyncLLMEngine: {e}")
        raise

async def generate_streaming_response(
    request_id: str,
    formatted_prompt: str,
    sampling_params: SamplingParams
) -> AsyncGenerator[str, None]:
    """Generate streaming response from the model."""
    try:
        # Generate tokens asynchronously
        results_generator = engine.generate(
            formatted_prompt, sampling_params, request_id
        )
        
        async for request_output in results_generator:
            # Extract the latest token from the output
            if request_output.outputs:
                output = request_output.outputs[0]
                token_text = output.text
                
                # Create streaming response chunk
                chunk = {
                    "id": request_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": token_text
                            },
                            "finish_reason": output.finish_reason
                        }
                    ],
                    "model": LLMConfig.HF_MODEL_ID
                }
                
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send end of stream marker
        final_chunk = {
            "id": request_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "model": LLMConfig.HF_MODEL_ID
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        error_chunk = {
            "id": request_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                    "error": str(e)
                }
            ],
            "model": LLMConfig.HF_MODEL_ID
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.on_event("startup")
def startup_event():
    initialize_engine()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Format messages into a single prompt for the model
        formatted_prompt = ""
        for message in request.messages:
            formatted_prompt += f"{message.role}: {message.content}\n"
        formatted_prompt += "assistant: "
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            n=1,
            best_of=1,
            temperature=request.temperature or LLMConfig.TEMPERATURE,
            max_tokens=request.max_tokens or LLMConfig.MAX_TOKENS,
            stop_token_ids=[151643, 151644, 151645]  # Common stop tokens for Qwen models
        )
        
        request_id = f"chatcmpl-{random_uuid()}"
        
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                generate_streaming_response(request_id, formatted_prompt, sampling_params),
                media_type="text/event-stream"
            )
        else:
            # Generate complete response (non-streaming)
            outputs = await engine.generate(formatted_prompt, sampling_params, request_id).__anext__()
            
            # Process the output to get the final result
            # We need to iterate through all the outputs to get the final result
            final_output = None
            async for output in engine.generate(formatted_prompt, sampling_params, request_id):
                final_output = output
            if final_output is None:
                raise ValueError("No output generated")
            
            generated_text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            
            response = ChatCompletionResponse(
                id=request_id,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": finish_reason
                    }
                ],
                model=LLMConfig.HF_MODEL_ID
            )
            
            return response
            
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": LLMConfig.HF_MODEL_ID}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8210)