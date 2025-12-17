import os
import json
import torch
import time, uuid
from typing import Optional
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from .sse import sse_stream
from .config import DEVICE, AMP
from .schemas import (
    ChatRequest,
    ModelListResponse,
    ChatCompletionChunk,
    ChatCompletionResponse,
)
from .chat_format import render_chat
from .deps import INFERENCE_ENGINE, INFERENCE_LOCK, TOKENIZER, PREPROCESSOR

router = APIRouter()

@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Returns all models currently served by this instance.",
    responses={
        200: {
            "description": "List of models currently loaded and available."
        }
    },
)
def list_models():
    with open(os.path.join(os.getcwd(), "app", "models", "list.json"), 'r') as f:
        available_models = json.load(f)
    return {"object": "list", "data": available_models}


@torch.no_grad()
def _generate(prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: Optional[int]):
    INFERENCE_ENGINE.update_config(temperature=temperature, top_p=top_p, top_k=top_k)
    with INFERENCE_LOCK:
        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            count = 0
            prompt = PREPROCESSOR.execute(prompt)
            token_ids = [
                t_id
                for t_id in TOKENIZER.Encode(prompt, out_type=int)
                if t_id != TOKENIZER.PieceToId("▁")
            ]
            for token_id in INFERENCE_ENGINE.complete_with_memory(token_ids):
                token: str = TOKENIZER.IdToPiece(token_id)
                if token:
                    # SentencePiece underline marker -> space
                    yield token.replace("▁", " ").replace("Г", "\n")
                count += 1
                if count >= max_tokens:
                    break

@router.post(
    "/v1/chat/completions",
    summary="Create chat completion",
    description="""
Chat-style completion endpoint with optional streaming.

- If `stream=false` (default): returns one `chat.completion` JSON object.
- If `stream=true`: returns `text/event-stream` (SSE). Each event is a
  `chat.completion.chunk` with incremental deltas, similar to OpenAI.

**Streaming event shape (`stream=true`)**:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1730025600,
  "model": "my-model-name",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "partial text here"
      },
      "finish_reason": null
    }
  ]
}
````

The last event has `finish_reason: "stop"` and `delta: {}`.
""",
    responses={
        200: {
            "description": "Non-streaming chat completion response",
            "model": ChatCompletionResponse,
        },
        206: {
            "description": "Streaming SSE chunks (only if `stream=true`). "
            "Each chunk is `chat.completion.chunk`.",
            "model": ChatCompletionChunk,
            "content": {
                "text/event-stream": {
                    "schema": ChatCompletionChunk.model_json_schema()
                }
            }
        }
    },
    openapi_extra={
        "x-streaming-response": {
            "content-type": "text/event-stream",
            "chunk-schema-ref": "#/components/schemas/ChatCompletionChunk",
        }
    },
)
def chat_completions(req: ChatRequest = Body(...)):
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    prompt = render_chat([m.model_dump() for m in req.messages])

    if req.stream:
        def iterator():
            # First chunk: just role
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }

            # Token chunks
            for piece in _generate(prompt, req.max_tokens, req.temperature, req.top_p, req.top_k):
                yield {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece, "role": "assistant"},
                        "finish_reason": None,
                    }],
                }

            # Final chunk
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }

        return StreamingResponse(
            sse_stream(iterator()),
            media_type="text/event-stream",
            status_code=206,  # helps Swagger docs distinguish streaming
        )

    # -------- non-streaming branch --------
    text = "".join(
        _generate(prompt, req.max_tokens, req.temperature, req.top_p, req.top_k)
    )

    prompt_tokens = len(TOKENIZER.Encode(prompt, out_type=int))
    completion_tokens = len(TOKENIZER.Encode(text, out_type=int))

    body = ChatCompletionResponse(
        id=chat_id,
        created=created,
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )

    return JSONResponse(body.model_dump())
