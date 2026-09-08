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


# ---- 采样确定性（同 seed 同 token 序列）----
def test_sampling_same_seed_same_tokens():
    import torch
    from src.manual_decode import _sample_token
    torch.manual_seed(42)
    ids = [_sample_token(torch.tensor([0.1, 0.2, 0.7]), temperature=0.7, top_p=0.9)
           for _ in range(200)]
    torch.manual_seed(42)
    ids2 = [_sample_token(torch.tensor([0.1, 0.2, 0.7]), temperature=0.7, top_p=0.9)
            for _ in range(200)]
    assert ids == ids2


# ---- run_stage CLI 参数解析（dry-run 语义）----
def test_run_stage_dry_run_parse():
    from src.run_stage import parse_args
    args = parse_args(["e0", "--dry-run", "--models", "qwen3.5-0.8b"])
    assert args.stage == "e0" and args.dry_run and args.models == ["qwen3.5-0.8b"]


def test_run_stage_dry_run_does_not_touch_network(tmp_path):
    import subprocess
    import sys
    env = {"PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        [sys.executable, "-m", "src.run_stage", "e0", "--dry-run", "--models",
         "qwen3.5-0.8b", "--runs-root", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "[dry-run] qwen3.5-0.8b: e0 n=20" in r.stdout
