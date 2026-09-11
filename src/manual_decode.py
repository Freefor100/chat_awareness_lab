"""逐 token 手动自回归解码器（no-stop）。

设计书 §5.4：当模型尝试输出 <|im_end|>、</s>、[INST] 等特殊 token 时，
默认 generate() 的 EOS/stop 策略会提前终止，把 serving stop behavior
误判成"模型不能复述 template"。本模块提供不以 EOS 停止的解码循环，
与标准 generate 双轨对照。
"""

import torch


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    sorted_p, idx = probs.sort(descending=True)
    cum = torch.cumsum(sorted_p, dim=0)
    keep = cum <= top_p
    keep[0] = True  # 至少保留一个 token
    out = torch.zeros_like(probs)
    out[idx[keep]] = sorted_p[keep]
    return out / out.sum()


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    probs = _apply_top_p(probs, top_p)
    return int(torch.multinomial(probs, 1).item())


def manual_generate(backend, input_ids: list[int], max_new_tokens: int,
                    temperature: float, top_p: float, seed: int,
                    do_sample: bool) -> dict:
    """逐 token 自回归；不以 EOS 停止，只跑满 max_new_tokens。

    backend 提供 forward_step(token_ids, past) 时走增量 KV 路径（近线性，
    与 generate 同速）；否则回退全量 forward_next_logits。
    返回 {"output_ids": 完整序列（含 prompt）, "meta": {...}}。
    """
    if do_sample:
        torch.manual_seed(seed)
    ids = list(input_ids)
    meta = {"mode": "nostop", "seed": seed, "kv_path": None,
            # 与标准轨对齐：no-stop 行也要能看出跑在什么档位、什么设备上
            "quant": getattr(backend, "quant", None),
            "device": getattr(backend, "device", None),
            "backend": getattr(backend, "backend_name", None)}
    use_kv = hasattr(backend, "forward_step")
    past = None
    for step in range(max_new_tokens):
        if use_kv:
            try:
                if step == 0 or past is not None:
                    inp = ids if step == 0 else [ids[-1]]
                    logits, past = backend.forward_step(inp, past)
                    if step == 0:
                        meta["kv_path"] = "kv"
                else:  # KV 断链后回到全量
                    logits = backend.forward_next_logits(ids)
            except Exception:
                use_kv = False
                meta["kv_path"] = "full-fallback"
                logits = backend.forward_next_logits(ids)
        else:
            logits = backend.forward_next_logits(ids)
        if do_sample:
            tok = _sample_token(logits, temperature, top_p)
        else:
            tok = int(torch.argmax(logits))
        ids.append(tok)
    return {"output_ids": ids, "meta": meta}
