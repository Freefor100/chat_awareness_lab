"""机制补分析:注入块 token 与真实分隔符 token 的结构对比(纯 tokenizer,离线)。

回答:用户输入里的模板字符串,经官方模板二次渲染后,
其 token id 与真实分隔符的 token id 是否相同?最终序列长什么样?
注入块被标成什么来源、出现在什么位置?

用已存实验 case 的 messages(0.8B smoke 的 A6/A7),不重跑 GPU。
"""
import bisect
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MODEL)
special = set(tok.all_special_ids)
added_vals = set(tok.get_added_vocab().values())
CONTROL_NAMES = {v: k for k, v in tok.get_added_vocab().items()}


def describe_id(tid):
    if tid in added_vals or tid in special:
        tag = "special" if tid in special else f"added:{CONTROL_NAMES.get(tid, '?')}"
        return f"id={tid} [{tag}]"
    return f"id={tid} [普通]"


def token_index_at(rendered, ids, char_pos):
    """字符偏移 → token 下标(与 render_context 同法)。"""
    if char_pos <= 0:
        return 0
    enc = tok(rendered, add_special_tokens=False)
    ends = [e for _, e in enc.encodings[0].offsets]
    return bisect.bisect_right(ends, char_pos)


def analyze(tag, attack_file):
    rows = [json.loads(l) for l in open(attack_file)]
    msg = rows[0]["messages"]
    canary = rows[0]["canary"]
    out = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                  return_dict=True)
    ids = list(out["input_ids"])
    text = tok.decode(ids, skip_special_tokens=False)
    print("=" * 78)
    print(f"[{tag}] canary={canary}")
    print("整段渲染文本(→ = 模板自动插入的换行):")
    shown = text
    print(shown[:400])
    print("-" * 78)
    print("control token 在最终序列中的分布(带下标与来源区间):")
    for i, tid in enumerate(ids):
        if tid in added_vals or tid in special:
            piece = tok.decode([tid], skip_special_tokens=False)
            print(f"  idx={i:3d}  {describe_id(tid):28s} {piece!r}")
    # 定位真实与注入边界
    sys_open = "<|im_start|>system\n"
    p_real = text.find(sys_open)
    p_inj = text.find(sys_open, p_real + 1)
    usr_open_marker = "<|im_start|>user\n"
    p_usr = text.find(usr_open_marker)
    print("-" * 78)
    if p_real >= 0:
        print(f"真实 system 轮起始字符偏移 = {p_real}  → token idx {token_index_at(text, ids, p_real)}")
    if p_usr >= 0:
        print(f"真实 user  轮起始字符偏移 = {p_usr}  → token idx {token_index_at(text, ids, p_usr)}")
    if p_inj >= 0:
        i_inj = token_index_at(text, ids, p_inj)
        print(f"注入块(位于 user 正文内)起始 = {p_inj} → token idx {i_inj}  "
              f"(落在 user 轮内部)")
        print(f"\n==> 关键对比:同一个 '{sys_open.strip()}' 字符串")
        print(f"    真实边界处 token id = {ids[token_index_at(text, ids, p_real)]}")
        print(f"    注入处    token id = {ids[i_inj]}")
        print(f"    两者是否相同:"
              f"{ids[token_index_at(text, ids, p_real)] == ids[i_inj]}")
    # 看注入块前一个 token 是 user 轮的什么
    if p_inj >= 0:
        i = token_index_at(text, ids, p_inj)
        prev = ids[i - 1] if i > 0 else None
        print(f"\n注入块紧邻前一个 token = {describe_id(prev)}"
              f"(即 user 内容正文结尾/紧邻 token)")
    print()


analyze("E6 A6 —— 把整段伪 system 写进用户消息", "runs/20260907_205249_qwen3.5-0.8b_attack/cases/E6_A6.jsonl")
analyze("E6 A7 —— combo:先闭 user 再开 system 再开 user", "runs/20260907_205249_qwen3.5-0.8b_attack/cases/E6_A7.jsonl")
