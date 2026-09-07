"""render_context.py——渲染 + tokens.tsv + source_guess 标注。

Design doc §5.2 / task 7:
- render_and_tokenize: apply_chat_template 双跑（tokenize False/True），产出行 token_rows。
- annotate_sources: 纯程序化的 source_guess 标注——内容跨度用字符偏移定位，
  字符边界 → token 边界用「整串 fast-tokenizer 编码的 char offsets」映射（不靠模型自报）。
- run_render: 把渲染结果落盘为 rendered_prompt.txt / input_ids.json / tokens.tsv / meta.json。

边界映射机制（有实证依据，见 task-7-report）：
本仓库 toy tokenizer（ByteLevel BPE + 长 special token 文本）上，增量前缀重编码
`tokenizer(rendered[:p])` 与整串编码不一致（在 <|im_start|> 内部或跨 merge 处切断会
产生不同切分，110 个字符位中 60 处漂移），因此采用 brief 的 fallback：
整串编码 `tokenizer(rendered, add_special_tokens=False).encodings[0].offsets` 的
逐 token 字符区间（transformers 5.x 中 BatchEncoding.encoding 已移除，改为 encodings）。
`token_offset(p)` = 完全落在 rendered[:p] 内的 token 数 = end <= p 的 token 数，
与前缀编码在无漂移处的语义一致。
"""
import bisect
import json
from pathlib import Path

CONTROL = "control"


def render_and_tokenize(tokenizer, messages: list[dict], add_generation_prompt: bool) -> dict:
    # 规范文本 = ids 的 decode(不跳 special)：与 token 序列一一对应，
    # 避免 tokenize=False 再手工重编码（v5 对 mistral-common 后端明确告警 unsafe）。
    out = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=add_generation_prompt,
                                        return_dict=True)
    input_ids = list(out["input_ids"])
    rendered = tokenizer.decode(input_ids, skip_special_tokens=False)
    rows = annotate_sources(rendered, input_ids, tokenizer, messages, add_generation_prompt)
    return {"rendered_prompt": rendered, "input_ids": input_ids, "token_rows": rows}


def annotate_sources(rendered: str, ids: list[int], tokenizer, messages: list[dict],
                     add_generation_prompt: bool) -> list[dict]:
    """给每个 token 行标注 source_guess ∈ {system,user,assistant,control,generation_prompt}。

    内容跨度定位：直接 find 每个 message content（唯一 canary 保证唯一命中；找不到则跳过）。
    字符边界 → token index：用整串 fast 编码的逐 token char offsets（见模块 docstring）。
    """
    # 1) 找每个 role content 在 rendered 中的字符区间
    spans = []  # (start, end, label)
    for m in messages:
        c = m.get("content") or ""
        if not c:
            continue
        start = rendered.find(c)
        if start >= 0:
            spans.append((start, start + len(c), m["role"]))
    if add_generation_prompt:
        # generation prompt = 渲染串末尾的 assistant 引导部分（无 content 与之对应）
        last_end = max((e for _, e, _ in spans), default=0)
        spans.append((last_end, len(rendered), "generation_prompt"))
    spans.sort()
    # 2) 字符边界 → token 边界：整串编码的 offsets（token i 覆盖 [offsets[i][0], offsets[i][1])）
    #    slow tokenizer（MistralCommonBackend 等）无 .encodings → decode 长度二分 fallback
    enc = tokenizer(rendered, add_special_tokens=False)
    offsets = getattr(enc, "encodings", None)
    ends = None
    if offsets is not None and offsets[0] is not None:
        ends = [e for _, e in offsets[0].offsets]

    def token_offset(char_pos: int) -> int:
        if char_pos <= 0:
            return 0
        if ends is not None:
            # 完全落在 rendered[:char_pos] 内的 token 数（end <= char_pos）
            return bisect.bisect_right(ends, char_pos)
        # slow 路径：最小 k 使 decode(ids[:k]) 长度 >= char_pos（解码长度单调不减）
        n = len(ids)
        cache = {}

        def dec_len(k: int) -> int:
            if k not in cache:
                cache[k] = len(tokenizer.decode(ids[:k], skip_special_tokens=False))
            return cache[k]

        if dec_len(n) < char_pos:
            return n
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if dec_len(mid) >= char_pos:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # 3) 逐区间标 source；区间外 = control
    seg_label = {}
    for s, e, label in spans:
        ts, te = token_offset(s), token_offset(e)
        seg_label[(ts, te)] = label
    special_ids = set(tokenizer.all_special_ids)
    rows = []
    for i, tid in enumerate(ids):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        is_special = tid in special_ids
        label = CONTROL
        for (ts, te), lab in sorted(seg_label.items()):
            if ts <= i < te:
                label = lab
                break
        rows.append({"index": i, "token_id": tid, "token_repr": repr(piece),
                     "decoded_piece": piece, "is_special": is_special,
                     "source_guess": label})
    return rows


def run_render(tokenizer, messages, out_dir: Path, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = render_and_tokenize(tokenizer, messages, add_generation_prompt=True)
    (out_dir / "rendered_prompt.txt").write_text(res["rendered_prompt"], encoding="utf-8")
    (out_dir / "input_ids.json").write_text(json.dumps(res["input_ids"]), encoding="utf-8")
    with open(out_dir / "tokens.tsv", "w", encoding="utf-8") as f:
        f.write("index\ttoken_id\ttoken_repr\tdecoded_piece\tis_special\tsource_guess\n")
        for r in res["token_rows"]:
            # decoded_piece 是唯一未转义列：\t、换行/回车都必须是字面反斜杠序列，
            # 否则含 \n 的 piece（如模板换行 token）会把一行拆成多行，破坏 TSV 结构
            piece = r["decoded_piece"].replace(chr(9), "\\t").replace("\n", "\\n") \
                                      .replace("\r", "\\r")
            f.write(f"{r['index']}\t{r['token_id']}\t{r['token_repr']}\t"
                    f"{piece}\t{int(r['is_special'])}\t{r['source_guess']}\n")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return res
