"""Embedding backend loading and batched text encoding."""

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
    pooling: str = "cls"
    query_instruction: str = ""
    passage_instruction: str = ""
    revision: Optional[str] = None


_MODEL_CACHE = {}

_MODEL_PRESETS = {
    "intfloat/multilingual-e5-small": {
        "pooling": "mean",
        "query_instruction": "query: ",
        "passage_instruction": "passage: ",
    },
    "intfloat/multilingual-e5-base": {
        "pooling": "mean",
        "query_instruction": "query: ",
        "passage_instruction": "passage: ",
    },
    "intfloat/multilingual-e5-large": {
        "pooling": "mean",
        "query_instruction": "query: ",
        "passage_instruction": "passage: ",
    },
}


def _model_preset(model_name_or_path: str) -> dict:
    normalized = str(model_name_or_path).rstrip("/").replace("\\", "/")
    for model_id, preset in _MODEL_PRESETS.items():
        if normalized == model_id or normalized.endswith(f"/{model_id.rsplit('/', 1)[-1]}"):
            return preset
    return {}


def _resolve_backend_options(
    model_name_or_path: str,
    pooling: Optional[str],
    query_instruction: Optional[str],
    passage_instruction: Optional[str],
) -> tuple[str, str, str]:
    preset = _model_preset(model_name_or_path)
    resolved_pooling = pooling or preset.get("pooling", "cls")
    if resolved_pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be one of: cls, mean")
    return (
        resolved_pooling,
        preset.get("query_instruction", "") if query_instruction is None else query_instruction,
        preset.get("passage_instruction", "") if passage_instruction is None else passage_instruction,
    )


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
    pooling: Optional[str] = None,
    query_instruction: Optional[str] = None,
    passage_instruction: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> Optional[EmbeddingBackend]:
    """Load a local directory or Hugging Face model ID as an embedding backend."""
    device = device or get_reward_device()
    dtype = dtype or default_dtype(device)
    resolved_pooling, resolved_query, resolved_passage = _resolve_backend_options(
        model_path,
        pooling,
        query_instruction,
        passage_instruction,
    )
    cache_key = (
        model_path,
        str(device),
        str(dtype),
        resolved_pooling,
        resolved_query,
        resolved_passage,
        revision,
        local_files_only,
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    load_kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        # Transformers 5 renamed torch_dtype to dtype. Fall back for the
        # supported Transformers 4.x range.
        try:
            model = AutoModel.from_pretrained(model_path, dtype=dtype, **load_kwargs)
        except TypeError:
            model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, **load_kwargs)
    except OSError:
        return None

    model = model.to(device).eval()
    backend = EmbeddingBackend(
        tokenizer=tokenizer,
        model=model,
        model_path=model_path,
        device=device,
        dtype=dtype,
        pooling=resolved_pooling,
        query_instruction=resolved_query,
        passage_instruction=resolved_passage,
        revision=revision,
    )
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

    def _cache_key(self, text: str, text_type: str) -> str:
        return f"{text_type}\0{text}"

    def _instruction(self, text_type: str) -> str:
        if text_type == "query":
            return self.backend.query_instruction
        if text_type == "passage":
            return self.backend.passage_instruction
        raise ValueError("text_type must be either 'query' or 'passage'")

    def encode(
        self,
        texts: Iterable[str],
        use_cache: bool = True,
        text_types: Optional[Iterable[str]] = None,
    ) -> Optional[torch.Tensor]:
        raw_texts = list(texts)
        if text_types is None:
            raw_types = ["passage"] * len(raw_texts)
        else:
            raw_types = list(text_types)
            if len(raw_types) != len(raw_texts):
                raise ValueError("text_types must have the same length as texts")

        pairs = [
            (str(text).strip(), str(text_type).strip().lower())
            for text, text_type in zip(raw_texts, raw_types)
            if str(text).strip()
        ]
        texts = [text for text, _ in pairs]
        types = [text_type for _, text_type in pairs]
        if not texts:
            return None
        for text_type in types:
            self._instruction(text_type)

        # Deduplicate within the current call before consulting the persistent
        # cache. Without this, repeated reference steps across a rollout batch
        # all look like misses until the first encoder forward has completed.
        unique_pairs = list(dict.fromkeys(zip(texts, types)))
        pair_to_unique_index = {pair: index for index, pair in enumerate(unique_pairs)}
        cached: list[Optional[torch.Tensor]] = [None] * len(unique_pairs)
        missing_inputs = []
        missing_indices = []

        for idx, (text, text_type) in enumerate(unique_pairs):
            emb = self.cache.get(self._cache_key(text, text_type)) if use_cache else None
            if emb is None:
                missing_indices.append(idx)
                missing_inputs.append(f"{self._instruction(text_type)}{text}")
            else:
                cached[idx] = emb.to(self.backend.device)

        encoded_missing_batches = []
        for start in range(0, len(missing_inputs), self.batch_size):
            batch_texts = missing_inputs[start:start + self.batch_size]
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
                hidden = output.last_hidden_state.to(torch.float32)
                if self.backend.pooling == "cls":
                    emb = hidden[:, 0]
                else:
                    attention_mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    emb = (hidden * attention_mask).sum(dim=1)
                    emb = emb / attention_mask.sum(dim=1).clamp_min(1.0)
                emb = F.normalize(emb, p=2, dim=1)
            encoded_missing_batches.append(emb)

        encoded_missing = torch.cat(encoded_missing_batches, dim=0) if encoded_missing_batches else None
        if encoded_missing is not None:
            for offset, idx in enumerate(missing_indices):
                emb = encoded_missing[offset]
                cached[idx] = emb
                if use_cache:
                    text, text_type = unique_pairs[idx]
                    self.cache.put(self._cache_key(text, text_type), emb.detach().cpu())

        for emb in cached:
            if emb is None:
                raise RuntimeError("internal embedding cache assembly failed")

        return torch.stack(
            [cached[pair_to_unique_index[pair]] for pair in zip(texts, types)],
            dim=0,
        )
