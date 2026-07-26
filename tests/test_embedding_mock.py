from types import SimpleNamespace

import pytest
import torch

import raise_scorer.backends.embedding as embedding_module
from raise_scorer.backends import EmbeddingBackend, TextEmbedder, load_embedding_backend


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        values = torch.arange(1, len(texts) + 1, dtype=torch.long).unsqueeze(1)
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
        }


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        del attention_mask
        values = input_ids.to(torch.float32)
        hidden = torch.cat([values, torch.ones_like(values)], dim=1).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=hidden)


def _embedder(pooling="cls", query_instruction="", passage_instruction=""):
    tokenizer = FakeTokenizer()
    backend = EmbeddingBackend(
        tokenizer=tokenizer,
        model=FakeModel(),
        model_path="fake",
        device=torch.device("cpu"),
        dtype=torch.float32,
        pooling=pooling,
        query_instruction=query_instruction,
        passage_instruction=passage_instruction,
    )
    return TextEmbedder(backend, batch_size=16, max_length=8, cache_size=16), tokenizer


def test_encode_deduplicates_repeated_texts_within_batch():
    embedder, tokenizer = _embedder()

    embeddings = embedder.encode(["shared step", "shared step", "unique step"], use_cache=False)

    assert tokenizer.calls == [["shared step", "unique step"]]
    assert embeddings.shape == (3, 2)
    assert torch.equal(embeddings[0], embeddings[1])


def test_encode_reuses_persistent_cache_after_batch_deduplication():
    embedder, tokenizer = _embedder()

    first = embedder.encode(["shared step", "shared step", "unique step"], use_cache=True)
    second = embedder.encode(["unique step", "shared step"], use_cache=True)

    assert len(tokenizer.calls) == 1
    assert torch.equal(first[0], second[1])
    assert torch.equal(first[2], second[0])
    assert embedder.cache.stats()["hits"] == 2


def test_encode_applies_query_and_passage_instructions_and_keeps_roles_distinct():
    embedder, tokenizer = _embedder(
        query_instruction="query: ",
        passage_instruction="passage: ",
    )

    embeddings = embedder.encode(
        ["same text", "same text"],
        use_cache=False,
        text_types=["query", "passage"],
    )

    assert tokenizer.calls == [["query: same text", "passage: same text"]]
    assert embeddings.shape == (2, 2)


def test_encode_validates_text_types():
    embedder, _ = _embedder()

    with pytest.raises(ValueError, match="same length"):
        embedder.encode(["one", "two"], text_types=["query"])
    with pytest.raises(ValueError, match="query.*passage"):
        embedder.encode(["one"], text_types=["unknown"])


class MultiTokenTokenizer:
    def __call__(self, texts, **kwargs):
        assert texts == ["example"]
        return {
            "input_ids": torch.tensor([[1, 3, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        }


class MultiTokenModel:
    def __call__(self, input_ids, attention_mask):
        del attention_mask
        values = input_ids.to(torch.float32)
        hidden = torch.stack([values, torch.ones_like(values)], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_mean_pooling_uses_attention_mask():
    backend = EmbeddingBackend(
        tokenizer=MultiTokenTokenizer(),
        model=MultiTokenModel(),
        model_path="fake",
        device=torch.device("cpu"),
        dtype=torch.float32,
        pooling="mean",
    )
    embedder = TextEmbedder(backend)

    embedding = embedder.encode(["example"], use_cache=False)
    expected = torch.nn.functional.normalize(torch.tensor([[2.0, 1.0]]), dim=1)

    assert torch.allclose(embedding, expected)


class LoadableFakeModel(FakeModel):
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def test_load_backend_accepts_model_id_and_applies_multilingual_e5_preset(monkeypatch):
    calls = []

    def load_tokenizer(model_id, **kwargs):
        calls.append(("tokenizer", model_id, kwargs))
        return FakeTokenizer()

    def load_model(model_id, **kwargs):
        calls.append(("model", model_id, kwargs))
        return LoadableFakeModel()

    monkeypatch.setattr(
        embedding_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=load_tokenizer),
    )
    monkeypatch.setattr(
        embedding_module,
        "AutoModel",
        SimpleNamespace(from_pretrained=load_model),
    )
    embedding_module._MODEL_CACHE.clear()

    backend = load_embedding_backend(
        "intfloat/multilingual-e5-small",
        device=torch.device("cpu"),
        revision="test-revision",
    )

    assert backend is not None
    assert backend.pooling == "mean"
    assert backend.query_instruction == "query: "
    assert backend.passage_instruction == "passage: "
    assert calls[0][1] == "intfloat/multilingual-e5-small"
    assert calls[0][2]["revision"] == "test-revision"


def test_load_backend_allows_preset_overrides(monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
    )
    monkeypatch.setattr(
        embedding_module,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: LoadableFakeModel()),
    )
    embedding_module._MODEL_CACHE.clear()

    backend = load_embedding_backend(
        "intfloat/multilingual-e5-small",
        device=torch.device("cpu"),
        pooling="cls",
        query_instruction="",
        passage_instruction="",
    )

    assert backend.pooling == "cls"
    assert backend.query_instruction == ""
    assert backend.passage_instruction == ""
