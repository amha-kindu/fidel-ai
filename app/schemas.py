from pydantic import BaseModel
from typing import List, Optional


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