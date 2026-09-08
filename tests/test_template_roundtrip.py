from src.render_context import render_and_tokenize, annotate_sources
from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES

def test_render_context_rows_and_sources(tmp_path):
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    msgs = [{"role": "system", "content": "SSS sys content"},
            {"role": "user", "content": "UUU usr content"}]
    out = render_and_tokenize(tok, msgs, add_generation_prompt=True)
    assert out["rendered_prompt"] == tok.apply_chat_template(msgs, tokenize=False,
                                                             add_generation_prompt=True)
    assert out["input_ids"] == [r["token_id"] for r in out["token_rows"]] and len(out["input_ids"]) == len(out["token_rows"])
    rows = out["token_rows"]
    sys_rows = [r for r in rows if r["source_guess"] == "system"]
    usr_rows = [r for r in rows if r["source_guess"] == "user"]
    ctl_rows = [r for r in rows if r["source_guess"] == "control"]
    # 内容行 decode 后包含原文（不跳过 special token）
    assert "".join(r["decoded_piece"] for r in sys_rows).find("SSS sys content") >= 0
    assert "".join(r["decoded_piece"] for r in usr_rows).find("UUU usr content") >= 0
    assert any("<|im_start|>" in r["decoded_piece"] for r in ctl_rows)
    # generation prompt 尾部标记
    tail = "".join(r["decoded_piece"] for r in rows if r["source_guess"] == "generation_prompt")
    assert "<|im_start|>assistant" in tail

def test_source_guess_stable_when_marker_repeated(tmp_path):
    """marker 在 system/user 同时出现时位置不串（用唯一 canary 保证）。"""
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    msgs = [{"role": "system", "content": "SYS_CANARY_ABCDEF"},
            {"role": "user", "content": "please echo SYS_CANARY_ABCDEF"}]
    out = render_and_tokenize(tok, msgs, add_generation_prompt=True)
    rows = out["token_rows"]
    sys_dec = "".join(r["decoded_piece"] for r in rows if r["source_guess"] == "system")
    usr_dec = "".join(r["decoded_piece"] for r in rows if r["source_guess"] == "user")
    assert sys_dec.count("SYS_CANARY_ABCDEF") == 1
    assert usr_dec.count("SYS_CANARY_ABCDEF") == 1


# ---- run_attacks 编排（stub backend，纯结构验证）----
import re
from pathlib import Path
from src.run_attacks import run_experiments
from src.common import make_run_dir, read_jsonl, load_yaml


class StubGenBackend:
    """无权重 backend：只有 standard 轨；surfaces 提取走 apply_template 字符串。"""

    backend_name = "stub"
    quant = None
    device = "cpu"
    loaded = True
    model = object()
    model_id = "stub"
    revision = "rev-local"

    def __init__(self):
        self.mode = "nostop_greedy"

    def apply_template(self, msgs, add_generation_prompt=False, tokenize=False):
        return "".join(f"<{m['role']}>{m.get('content', '')}</>" for m in msgs)

    def ids_from_messages(self, msgs):
        return {"input_ids": list(range(1, 9)), "attention_mask": [1] * 8}

    def decode(self, ids, skip_special_tokens=False):
        return "BLOCKED"

    def generate_standard(self, *a, **k):
        return {"output_ids": [1] * 8 + [2] * 5, "meta": {}}


def test_run_experiments_e5_writes_jsonl_and_metrics(tmp_path):
    ROOT = Path(__file__).resolve().parents[1]
    gen = load_yaml(ROOT / "configs/generation.yaml")
    d = make_run_dir(tmp_path, "t1")
    res = run_experiments(StubGenBackend(), ["E5"], "stub", d, gen, n_canaries=3)
    files = list((d / "cases").glob("*.jsonl"))
    assert len(files) == 1
    rows = read_jsonl(files[0])
    assert len(rows) == 3  # stub 无 forward_next_logits → 只有 standard_greedy 轨
    ids = [r["case_id"] for r in rows]
    assert len(set(ids)) == 3 and all(re.match(r"stub_e5_plain_\d{4}", i) for i in ids)
    # metrics 汇总行（Task 17 接线）
    mp = d / "metrics.json"
    assert mp.exists()
    mrows = read_jsonl(mp)
    assert any(r["experiment"] == "E5" and r["attack"] == "plain" and r["n"] == 3
               and r["mode"] == "standard_greedy" for r in mrows)
    assert res["cases_run"] == 3 and res["metrics_rows"] == 1
