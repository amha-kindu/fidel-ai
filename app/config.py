import os
import torch
import numpy
import random
from pydantic import BaseModel

random.seed(4321)
torch.manual_seed(4321)
numpy.random.seed(4321)

class Settings(BaseModel):
    model_id: str = os.getenv("MODEL_ID", "")
    lora: bool = bool(os.getenv("LORA", ""))
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "")
    max_tokens: int = int(os.getenv("MAX_TOKENS", 256))
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))
    top_p: float = float(os.getenv("TOP_P", 0.9))
    top_k: int | None = int(os.getenv("TOP_K")) if os.getenv("TOP_K") else None
    allow_origins: str = os.getenv("ALLOW_ORIGINS", "*")
    amp: bool = os.getenv("AMP", "true").lower() in {"1", "true", "yes"}

class Config:
    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def update(self, skip: list[str] = [], **kwargs):
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key) \
                and key not in skip:
                setattr(self, key, value)

                
class ModelConfig(Config):
    def __init__(self, **kwargs):
        self.embed_dim: int = kwargs.get("embed_dim", 512)
        self.n_blocks: int = kwargs.get("n_blocks", 6)
        self.vocab_size: int = kwargs.get("vocab_size", 25000)
        self.ff_dim: int = kwargs.get("ff_dim", 2048)
        self.heads: int = kwargs.get("heads", 8)
        self.dropout: float = kwargs.get("dropout", 0.1)
        self.seq_len: int = kwargs.get("seq_len", 50)
        self.metric_dim: int = kwargs.get("metric_dim", 128)
        self.epsilon: float = kwargs.get("metric_epsilon", 1e-02)


class ModelWithLoRAConfig(ModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lora_rank: int = kwargs.get("lora_rank", 16)
        self.lora_alpha: int = kwargs.get("lora_alpha", 32)
        self.lora_targets: dict = kwargs.get("lora_targets", {})
        self.lora_dropout: float = kwargs.get("lora_dropout", 0.05)
        
class InferenceConfig(Config):
    def __init__(self, **kwargs):
        self.top_k: int = kwargs.get("top_k", 0)
        self.top_p: float = kwargs.get("top_p", 1.0)
        self.temperature: float = kwargs.get("temperature", 1.0)
        self.max_temp: float = kwargs.get("max_temp", 2.0)
        self.repetition_penalty: float = kwargs.get("repetition_penalty", 1.15)
        self.presence_penalty: float = kwargs.get("presence_penalty", 0.0)
        self.freq_penalty: float = kwargs.get("freq_penalty", 0.3)
        self.no_repeat_ngram_size: int = kwargs.get("no_repeat_ngram_size", 3)
        self.rep_window: int = kwargs.get("rep_window", 200)
        self.kv_cache_size: int = kwargs.get("kv_cache_size", 0)
        
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
SETTINGS = Settings()

DEVICE = torch.device('cpu')
AMP = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3000)
    DEVICE = torch.device(f'cuda')
    torch.cuda.set_device(DEVICE)
    AMP = torch.amp.autocast_mode.is_autocast_available(DEVICE.type)