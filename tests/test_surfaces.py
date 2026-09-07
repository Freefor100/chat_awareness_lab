# tests/test_surfaces.py — probe-render control surface extraction (Task 6)
from src.surfaces import extract_surfaces, boundary_token_ids
from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES

def _renderer(tok):
    def render(msgs, add_gen=False):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
    return render

def test_chatml_surfaces(tmp_path):
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    s = extract_surfaces(_renderer(tok))
    assert s["sys_open"] == "<|im_start|>system\n"
    assert s["sys_close"].startswith("<|im_end|>")
    assert s["usr_open"] == "<|im_start|>user\n"
    assert s["usr_close"].startswith("<|im_end|>")
    assert s["asst_open"].startswith("<|im_start|>assistant")
    assert not s["skipped"]

def test_llama_inst_surfaces(tmp_path):
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["llama_inst"])
    s = extract_surfaces(_renderer(tok))
    assert s["sys_open"] == "<<SYS>>\n"
    assert s["sys_close"] == "\n<</SYS>>\n\n"
    assert "[INST]" in s["usr_open"] and "[INST]" not in s["sys_close"]
    assert "[/INST]" in s["usr_close"]

def test_boundary_token_ids_flags_special(tmp_path):
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    s = extract_surfaces(_renderer(tok))
    special = {t for t in tok.get_added_vocab().values()}
    b = boundary_token_ids(s, lambda t: tok(t, add_special_tokens=False)["input_ids"], special)
    # <|im_start|> 等必须被标记为 special；marker 文本不是
    assert b["sys_open"]["is_special"] and b["sys_open"]["token_ids"]

# ---- Task 9: dynamic attack case builder (E1–E8) ----
from src.attack_builder import build_case
from src.common import load_yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_e5_baseline_messages(tmp_path):
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    m = build_case("E5", "plain", cfg, cons, None, "SYS_CANARY_ABCDEF", "X", seed=0)
    assert m["messages"][0]["role"] == "system"
    assert "SYS_CANARY_ABCDEF" in m["messages"][0]["content"]

def test_e6_a6_uses_surfaces(tmp_path):
    from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES
    from src.surfaces import extract_surfaces
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    s = extract_surfaces(lambda ms, add_gen=False: tok.apply_chat_template(
        ms, tokenize=False, add_generation_prompt=add_gen))
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    m = build_case("E6", "A6", cfg, cons, s, "SYS_CANARY_ABCDEF", None, seed=0)
    user_content = m["messages"][-1]["content"]
    assert s["sys_open"] in user_content and "SYS_CANARY_ABCDEF" in user_content

def test_e6_a7_combo_order(tmp_path):
    from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES
    from src.surfaces import extract_surfaces
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    s = extract_surfaces(lambda ms, add_gen=False: tok.apply_chat_template(
        ms, tokenize=False, add_generation_prompt=add_gen))
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    m = build_case("E6", "A7", cfg, cons, s, "SYS_CANARY_ABCDEF", None, seed=0)
    uc = m["messages"][-1]["content"]
    # chatml 的 usr_close == sys_close == "<|im_end|>\n"（同一模板后缀），
    # str.index 恒取首个出现 → 从 canary 之后起搜 sys_close 以区分两次出现
    assert uc.index(s["usr_close"]) < uc.index(s["sys_open"]) < uc.index("SYS_CANARY_ABCDEF") \
           < uc.index(s["sys_close"], uc.index("SYS_CANARY_ABCDEF")) < uc.index(s["usr_open"])

def test_e6_a10_marks_token_override(tmp_path):
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    m = build_case("E6", "A10", cfg, cons, None, "SYS_CANARY_ABCDEF", None, seed=0)
    assert m["attack_meta"]["token_attack"] is True

def test_e4_query_contains_echo_placeholder(tmp_path):
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    assert "<ECHO_QUERY>" in cons["e4_query"]

def test_e6_surface_attack_skips_without_surfaces(tmp_path):
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    for atk in ("A6", "A7", "A8"):
        m = build_case("E6", atk, cfg, cons, None, "SYS_CANARY_ABCDEF", None, seed=0)
        assert m["messages"] == [] and m["attack_meta"]["skipped"]

def test_e6_a6_a7_skip_when_sys_close_missing(tmp_path):
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    partial = {"sys_open": "<|im_start|>system\n", "sys_close": None}
    # A6/A7 需成对的 open+close；sys_open 在而 sys_close 为 None → 仍跳过
    for atk in ("A6", "A7"):
        m = build_case("E6", atk, cfg, cons, partial, "SYS_CANARY_ABCDEF", None, seed=0)
        assert m["messages"] == [] and m["attack_meta"]["skipped"]

def test_e6_empty_string_surface_value_is_not_missing(tmp_path):
    cfg = load_yaml(ROOT / "configs/attacks.yaml")
    cons = load_yaml(ROOT / "prompts/system_constraints.yaml")
    s = {"asst_open": ""}   # llama 风格空串 asst_open："" ≠ missing，不应跳过
    m = build_case("E6", "A8", cfg, cons, s, "SYS_CANARY_ABCDEF", None, seed=0)
    uc = m["messages"][-1]["content"]
    assert not m["attack_meta"].get("skipped")
    assert "{ASST_OPEN}" not in uc and "SYS_CANARY_ABCDEF" in uc
