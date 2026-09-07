# tests/test_backends.py — model/tokenizer load chain + Backend abstraction (Task 10)
# Hermetic: toy local tokenizer dir (no hub); nonexistent id exercises the failure path
# (AutoTokenizer 404s fast; mistral-common repo probe short-circuits — no weight downloads).
import pytest
from pathlib import Path
from src.backends import load_backend
from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES


def test_tokenizer_only_backend_loads(tmp_path, monkeypatch):
    tok_dir = tmp_path / "tok"
    make_toy_tokenizer(tok_dir, TOY_STYLES["chatml"])
    cfg = {"key": "toy", "id": str(tok_dir), "role": "lite", "family": "qwen35",
           "is_multimodal": False}
    monkeypatch.setattr("src.backends.AutoTokenizer",
                        __import__("transformers").AutoTokenizer)
    b = load_backend(cfg, tokenizer_only=True)
    assert b.loaded and b.model is None
    assert b.chat_template() and "<|im_start|>" in b.chat_template()
    ids = b.ids_from_messages([{"role": "user", "content": "hi"}])
    assert ids["input_ids"]


def test_loader_records_failures(tmp_path, monkeypatch):
    cfg = {"key": "toy", "id": "nonexistent/model-xyz", "role": "lite", "family": "qwen35",
           "is_multimodal": False}
    b = load_backend(cfg, tokenizer_only=True)
    assert not b.loaded and b.failure_reason
