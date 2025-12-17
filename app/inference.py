import torch
import torch.nn as nn
from typing import Iterator
import sentencepiece as spm
from collections import Counter
import torch.nn.functional as F

from .config import *
from .deps import AMP, DEVICE
from .kv_cache import SlidingKVCache
from .preprocessor import AmharicPreprocessor
from .utils import Conversation, get_casual_mask


class InferenceEngine:
    def __init__(self, model: nn.Module, tokenizer: spm.SentencePieceProcessor, config: InferenceConfig=DEFAULT_INFERENCE_CONFIG, system_prompt: str = ""):
        
        self.model = model
        self.config = config        
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor()
        self.max_len = self.model.config.seq_len
        self.use_kv_cache = config.kv_cache_size > 0
        self.kv_caches = [
            SlidingKVCache(config.kv_cache_size) for _ in range(self.model.config.n_blocks) \
            if self.use_kv_cache
        ]
        
        self.pad_token = self.tokenizer.PieceToId("[PAD]")
        self.unk_token = self.tokenizer.PieceToId("[UNK]")
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.stop_token = self.tokenizer.PieceToId("[STOP]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")
        
        self.conv = Conversation(os.environ.get("SYSTEM_PROMPT", ""))
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")
        
        self.system_tokens = []
        if self.conv.system_text:
            self.conv.system_text = self.PREPROCESSOR.execute(self.conv.system_text)
            self.system_tokens.extend([
                self.system_token,
                *self.tokenizer.Encode(self.conv.system_text, out_type=int)
            ])
        
        self.model.eval()
        
    def update_config(self, **kwargs):
        self.config.update(**kwargs)
    
    def _ban_tokens(self, logits: torch.Tensor, banned_token_ids: list[int]) -> None:
        for token_id in banned_token_ids:
            if token_id >= 0:
                logits[token_id] = -float("inf")
                
    def _no_repeat_ngrams_ids(self, history: list[int], n: int) -> list[int]:
        if n <= 1 or len(history) < n-1:
            return []
        prefix = tuple(history[-(n-1):])      # last n-1 tokens
        bans = []
        for i in range(len(history) - n + 1):
            if tuple(history[i:i+n-1]) == prefix:
                bans.append(history[i+n-1])   # the token that completed that n-gram before
        return list(set(bans))

    def _apply_penalties(self, logits: torch.Tensor, history: list[int]) -> None:
        if not history: 
            return
        counts = Counter(history[-self.config.rep_window:])
        fp = self.config.freq_penalty
        pp = self.config.presence_penalty
        rp = self.config.repetition_penalty
        if rp <= 1.0 and fp <= 0.0 and pp <= 0.0:
            return

        for token_id, count in counts.items():
            if token_id < 0: 
                continue
            
            val = logits[token_id]
            # HF-style repetition penalty
            logits[token_id] = torch.where(val > 0, val / rp, val * rp)
            
            # frequency + presence penalties (OpenAI-style)
            logits[token_id] = logits[token_id] - (fp * float(count) + pp)

    @torch.no_grad()
    def complete_with_memory(self, token_ids: list[int]) -> Iterator[int]:
        bot_output = []
        for prediction_token in self.complete(token_ids):
            bot_output.append(prediction_token)
            yield prediction_token
        self.conv.add_exchange(token_ids, bot_output)

    @torch.no_grad()
    def complete(self, token_ids: list[int]) -> Iterator[int]:
        token_ids = self.system_tokens + token_ids
        temperature = min(self.config.temperature, self.config.max_temp) + 1e-5
        while token_ids and len(token_ids) < self.max_len:
            decoder_input = torch.tensor(
                [token_ids],
                dtype=torch.int64
            ).to(DEVICE)
                        
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                # (1, SEQ_LEN, VOCAB_SIZE)
                logits: torch.Tensor = self.model(decoder_input, None, self.use_kv_cache, self.kv_caches)

            # (VOCAB_SIZE,)
            # Take logits for the last position and apply temperature scaling
            logits = logits[0, -1, :].float() / temperature

            # Ban control tokens and no-repeat n-gram causing tokens
            self._ban_tokens(
                logits=logits, 
                banned_token_ids=[
                    self.unk_token, self.user_token, self.bot_token, self.system_token, self.pad_token,
                    *self._no_repeat_ngrams_ids(token_ids, self.config.no_repeat_ngram_size)
                ]
            )

            # Apply presence, frequency and repetition penalties based on history
            self._apply_penalties(logits, token_ids)

            # Apply Top-k filtering
            if self.config.top_k and self.config.top_k > 0:
                top_k_logits, top_k_idx = torch.topk(logits, k=min(self.config.top_k, logits.size(0)), dim=0)                
                logits = torch.full_like(logits, float('-inf'))\
                    .scatter(dim=0, index=top_k_idx, src=top_k_logits)
            
            # Apply Top-p (nucleus) filtering
            if self.config.top_p and self.config.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=0)
                sorted_probs = F.softmax(sorted_logits, dim=0)
                cprobs = torch.cumsum(sorted_probs, dim=0)
                mask = cprobs <= self.config.top_p
                mask[0] = True  # ensure at least one token
                logits = torch.full_like(logits, float('-inf'))\
                    .scatter(dim=0, index=sorted_idx[mask], src=sorted_logits)
                
            # Apply softmax & sample next token
            probs = F.softmax(logits, dim=0)
            mass = probs.sum(dim=0, keepdim=True)
            
            if (mass <= 1e-12).any():
                predicted_token = int(logits.argmax(dim=0).item())
            else:
                predicted_token = int(torch.multinomial(probs, num_samples=1).item())
            
            if predicted_token == self.stop_token:
                break
            
            token_ids.append(predicted_token)
            yield predicted_token
