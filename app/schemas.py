from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

#
# /v1/models
#

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: Optional[str] = None
    # add fields if they already exist in app/models/list.json (e.g. "context_length")


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]


#
# /v1/chat/completions  (non-streaming)
#

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-1234")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., example=1730025600)
    model: str = Field(..., example="my-model-name")
    choices: List[ChatChoice]
    usage: UsageInfo


#
# /v1/chat/completions  (streaming chunks)
#   We won't actually return these as pydantic in code,
#   but we expose them to Swagger via responses / openapi_extra
#

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(..., example="chatcmpl-1234")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., example=1730025600)
    model: str = Field(..., example="my-model-name")
    choices: List[StreamChoice]
