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


# ---- manual no-stop decoder（stub backend）----
import torch
from src.manual_decode import manual_generate


class StubBackend:
    """只有 forward_next_logits 的极简 backend，供 manual_generate 单测。"""

    def __init__(self, logits_fn):
        self._f = logits_fn
        self.model = object()

    def forward_next_logits(self, input_ids):
        return self._f(input_ids)


def _argmax_stub(vocab=32, hot=7):
    def f(input_ids):
        t = torch.zeros(vocab)
        t[hot] = 1.0
        return t
    return f


def test_manual_generate_runs_to_max_tokens():
    b = StubBackend(_argmax_stub())
    out = manual_generate(b, [1, 2], max_new_tokens=5, temperature=0.0, top_p=1.0,
                          seed=0, do_sample=False)
    # 无 EOS 停止，跑满 5 步；hot token 持续 argmax
    assert out["output_ids"] == [1, 2, 7, 7, 7, 7, 7]


def test_manual_generate_greedy_deterministic():
    b1 = StubBackend(_argmax_stub())
    b2 = StubBackend(_argmax_stub())
    a = manual_generate(b1, [1], 5, 0.0, 1.0, 0, False)["output_ids"]
    c = manual_generate(b2, [1], 5, 0.0, 1.0, 0, False)["output_ids"]
    assert a == c
