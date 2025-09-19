import os
import json
import torch

import importlib
from threading import Lock
import sentencepiece as spm

from config import SETTINGS, DEVICE, AMP
from .logging import LOGGER
from .models.lora import LoRAdapter
from .inference import InferenceEngine
from .preprocessor import AmharicPreprocessor


def get_model():
    base_model_dir = os.path.join("models", SETTINGS.model_id, "checkpoint.pt")
    base_metadata = os.path.join("models", SETTINGS.model_id, "metadata.json")
    LOGGER.info(f"Loading checkpoint from {base_model_dir}...")
    base_weights: dict = torch.load(base_model_dir, map_location=DEVICE, weights_only=False)
    with open(base_metadata, 'r') as f:
        base_metadata = json.load(f)
    
    model_module = importlib.import_module(f'app.models.{SETTINGS.model_id}.model')
    GPTmodel = getattr(model_module, 'GPTmodel')
    
    if SETTINGS.lora:
        lora_model_dir = os.path.join("models", SETTINGS.model_id, "checkpoint-lora.pt")
        lora_metadata_dir = os.path.join("models", SETTINGS.model_id, "metadata-lora.json")
        LOGGER.info(f"Loading checkpoint from {lora_model_dir}...")
        lora_weights: dict = torch.load(lora_model_dir, map_location=DEVICE, weights_only=False)
        with open(lora_metadata_dir, 'r') as f:
            model_config = json.load(f)
        base_weights.update(lora_weights)

    model: torch.nn.Module = GPTmodel.build(
        model_config,
        weights=base_weights,
    )
    
    if SETTINGS.lora:
        merged = False
        for module in model.modules():
            if isinstance(module, LoRAdapter):
                module.merge()
                merged = True
        if merged:
            LOGGER.info(f"Merged LoRA adapters in model...")

    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Device: {DEVICE}")
    LOGGER.info(f"Total Parameters: {total_params}")
    LOGGER.info(f"Model Size(MB): {total_params * 4 / (1024 ** 2):.2f}MB")
    LOGGER.info(f"Initiating inference with {'mixed-precision' if AMP else 'single-precision'}...")
    
    return model

INFERENCE_LOCK = Lock()
model = get_model().to(DEVICE).eval()
INFERENCE_ENGINE = InferenceEngine(model, SETTINGS.max_tokens, SETTINGS.temperature, SETTINGS.top_p, SETTINGS.top_k)

TOKENIZER = spm.SentencePieceProcessor()
TOKENIZER.LoadFromFile(SETTINGS.tokenizer_path)
PREPROCESSOR = AmharicPreprocessor()