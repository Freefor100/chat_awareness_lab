from pathlib import Path
from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES

def test_toy_roundtrip_decode(tmp_path):
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    msgs = [{"role": "system", "content": "You are a test."},
            {"role": "user", "content": "Hi"}]
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=True)
    ids = out["input_ids"]
    assert ids[0] == tok.bos_token_id  # 模板以 BOS 开头
    text = tok.decode(ids, skip_special_tokens=False)
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text == rendered  # decode(不跳过) == 渲染串（双 BOS 会出现 != 而失败）

def test_toy_apply_no_double_bos_after_manual_tokenize(tmp_path):
    tok = make_toy_tokenizer(tmp_path, TOY_STYLES["chatml"])
    msgs = [{"role": "user", "content": "hello"}]
    # transformers 5.x apply_chat_template 默认 return_dict=True（4.x 默认返回列表）；
    # 显式 return_dict=False 取回与 4.x 等价的纯 id 列表，语义不变
    ids1 = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   return_dict=False)
    # 禁止"先渲染成 str 再 tokenize 又自动加 BOS"
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids2 = tok(s, add_special_tokens=False)["input_ids"]
    assert ids1 == ids2
