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
