import os
import json
import torch
import time, uuid
from typing import Optional
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from .sse import sse_stream
from .config import DEVICE, AMP
from .schemas import ChatRequest
from .chat_format import render_chat
from .deps import INFERENCE_ENGINE, INFERENCE_LOCK, TOKENIZER, PREPROCESSOR

router = APIRouter()

@router.get("/v1/models")
def list_models():
    with open(os.path.join("models", "list.json"), 'r') as f:
        available_models = json.load(f)
    return {"object": "list", "data": available_models}

                    
@torch.no_grad()            
def _generate(prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: Optional[int]):
    INFERENCE_ENGINE.update_config(temperature=temperature, top_p=top_p, top_k=top_k)
    with INFERENCE_LOCK:
        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            count = 0
            prompt = PREPROCESSOR.execute(prompt)
            token_ids = TOKENIZER.Encode(prompt, out_type=int)
            for token_id in INFERENCE_ENGINE.complete_with_memory(token_ids):
                token: str = TOKENIZER.IdToPiece(token_id)
                if token:
                    yield token.replace("▁", " ")
                count += 1
                if count >= max_tokens:
                    break

@router.post("/v1/chat/completions")
def chat_completions(req: ChatRequest = Body(...)):
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    prompt = render_chat([m.model_dump() for m in req.messages])

    if req.stream:
        def iterator():
            # role chunk
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            for piece in _generate(prompt, req.max_tokens, req.temperature, req.top_p, req.top_k, req.stop):
                yield {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        
        return StreamingResponse(sse_stream(iterator()), media_type="text/event-stream")

    # non-streaming
    text = "".join(_generate(prompt, req.max_tokens, req.temperature, req.top_p, req.top_k, req.stop))
    prompt_tokens = len(TOKENIZER.Encode(prompt, out_type=int))
    completion_tokens = len(TOKENIZER.Encode(text, out_type=int))
    return JSONResponse({
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })