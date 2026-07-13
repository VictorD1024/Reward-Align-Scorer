import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .cache import LRUCache


@dataclass(frozen=True)
class EmbeddingBackend:
    tokenizer: object
    model: object
    model_path: str
    device: torch.device
    dtype: torch.dtype


_MODEL_CACHE = {}


def get_reward_device() -> torch.device:
    """Choose CUDA, then Ascend NPU, then CPU."""
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if torch.cuda.is_available():
        device_id = local_rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")

    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            if npu.is_available():
                device_id = local_rank % max(npu.device_count(), 1)
                npu.set_device(device_id)
                return torch.device(f"npu:{device_id}")
        except Exception:
            pass

    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "npu"} and (os.environ.get("RAISE_FP32") or os.environ.get("REWARD_ALIGN_FP32", "0")) != "1":
        return torch.float16
    return torch.float32


def load_embedding_backend(
    model_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[EmbeddingBackend]:
    device = device or get_reward_device()
    dtype = dtype or default_dtype(device)
    cache_key = (model_path, str(device), str(dtype))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not os.path.exists(model_path):
        _MODEL_CACHE[cache_key] = None
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    backend = EmbeddingBackend(tokenizer=tokenizer, model=model, model_path=model_path, device=device, dtype=dtype)
    _MODEL_CACHE[cache_key] = backend
    return backend


class TextEmbedder:
    def __init__(
        self,
        backend: EmbeddingBackend,
        batch_size: int = 128,
        max_length: int = 128,
        cache_size: int = 8192,
    ):
        self.backend = backend
        self.batch_size = batch_size
        self.max_length = max_length
        # The cache is instance-scoped, so model/device/dtype/max_length are already isolated.
        # Keep cached tensors on CPU to avoid long-running reward workers pinning GPU/NPU memory.
        self.cache: LRUCache[str, torch.Tensor] = LRUCache(capacity=cache_size)

    def _cache_key(self, text: str) -> str:
        return text

    def encode(self, texts: Iterable[str], use_cache: bool = True) -> Optional[torch.Tensor]:
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if not texts:
            return None

        cached: list[Optional[torch.Tensor]] = [None] * len(texts)
        missing_texts = []
        missing_indices = []

        for idx, text in enumerate(texts):
            emb = self.cache.get(self._cache_key(text)) if use_cache else None
            if emb is None:
                missing_indices.append(idx)
                missing_texts.append(text)
            else:
                cached[idx] = emb.to(self.backend.device)

        encoded_missing_batches = []
        for start in range(0, len(missing_texts), self.batch_size):
            batch_texts = missing_texts[start:start + self.batch_size]
            enc = self.backend.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(self.backend.device) for key, value in enc.items()}
            with torch.inference_mode():
                output = self.backend.model(**enc)
                emb = output.last_hidden_state[:, 0].to(torch.float32)
                emb = F.normalize(emb, p=2, dim=1)
            encoded_missing_batches.append(emb)

        encoded_missing = torch.cat(encoded_missing_batches, dim=0) if encoded_missing_batches else None
        if encoded_missing is not None:
            for offset, idx in enumerate(missing_indices):
                emb = encoded_missing[offset]
                cached[idx] = emb
                if use_cache:
                    self.cache.put(self._cache_key(texts[idx]), emb.detach().cpu())

        for idx, emb in enumerate(cached):
            if emb is None:
                raise RuntimeError("internal embedding cache assembly failed")
            cached[idx] = emb

        return torch.stack(cached, dim=0)
