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

    返回 {"output_ids": 完整序列（含 prompt）, "meta": {...}}，
    与 Backend.generate_standard 的返回形状一致。
    """
    if do_sample:
        torch.manual_seed(seed)
    ids = list(input_ids)
    meta = {"mode": "nostop", "seed": seed}
    for _ in range(max_new_tokens):
        logits = backend.forward_next_logits(ids)
        if do_sample:
            tok = _sample_token(logits, temperature, top_p)
        else:
            tok = int(torch.argmax(logits))
        ids.append(tok)
    return {"output_ids": ids, "meta": meta}
