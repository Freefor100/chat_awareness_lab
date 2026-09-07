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
