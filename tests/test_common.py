# tests/test_common.py
from pathlib import Path
import yaml
from src.common import load_yaml

ROOT = Path(__file__).resolve().parents[1]

def test_models_yaml_required_keys():
    cfg = load_yaml(ROOT / "configs/models.yaml")
    models = cfg["models"]
    keys = {m["key"] for m in models}
    assert {"qwen3.5-0.8b", "qwen3.5-0.8b-base", "qwen3.5-2b", "qwen3.5-2b-base",
            "qwen3.5-4b", "qwen3.5-4b-base", "ministral-3-3b-instruct",
            "ministral-3-3b-base"} <= keys
    for m in models:
        assert {"key", "id", "role", "family"} <= set(m)
        assert m["role"] in {"main", "lite", "base", "legacy"}
        assert m["family"] in {"qwen35", "ministral3", "qwen3"}

def test_generation_yaml_phases():
    cfg = load_yaml(ROOT / "configs/generation.yaml")
    assert cfg["smoke"]["n_canaries"] == 20
    assert cfg["formal"]["n_canaries"] >= 100
    assert cfg["formal"]["sampling_seeds"] == [1, 2, 3]
    assert cfg["max_new_tokens"] >= 256

def test_attacks_cover_a0_to_a10():
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    ids = [a["id"] for a in cfg["attacks"]]
    assert ids == [f"A{i}" for i in range(11)]  # A0..A10 有序

def test_canary_strings_match_design_doc():
    # 设计书 §6 E5：BLOCKED 判定；§7.1 用 exact marker
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    assert "BLOCKED" in cons["e5_system"]
    assert "{marker}" in cons["e5_system"]  # SYS_CANARY_<hex> 运行时由 gen_marker 注入 {marker}
    assert "SECRET_SYSTEM_NONCE" in cons["e7_system"]

# 追加到 tests/test_common.py
import json, hashlib, time, random
from src.common import now_tag, make_run_dir, append_jsonl, read_jsonl, sha256_bytes

def test_now_tag_format():
    import re
    assert re.fullmatch(r"\d{8}_\d{6}", now_tag())

def test_make_run_dir_creates_layout(tmp_path):
    d = make_run_dir(tmp_path, "tag1")
    assert (d / "template").is_dir() and (d / "cases").is_dir()

def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "o.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    rows = read_jsonl(p)
    assert [r["a"] for r in rows] == [1, 2]

def test_sha256_bytes():
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
