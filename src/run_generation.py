"""单 case 双轨生成：standard generate + manual no-stop decode。

thinking 模式固定（设计书 §9.5）在 Backend.ids_from_messages 内完成
（enable_thinking=False，模板不接受该 kwarg 时自动回退）。
"""

from src.manual_decode import manual_generate


def run_case_generation(backend, messages, gen_cfg: dict, mode: str,
                        input_ids: list[int] | None = None,
                        max_new_tokens: int | None = None) -> dict:
    """按 mode 跑一次生成，返回记录片段。

    mode base ∈ {"standard_greedy", "nostop_greedy", "standard_sampling",
                 "nostop_sampling"}；sampling 轨可用 "base:<seed>" 后缀指定 seed。
    input_ids: 预注入 ids（A10 token attack 由 run_attacks 提供）；
    为空时用 messages 经 backend.ids_from_messages 编码。
    """
    if input_ids is None:
        ids = list(backend.ids_from_messages(messages)["input_ids"])
    else:
        ids = list(input_ids)
    mn = max_new_tokens or gen_cfg["max_new_tokens"]

    seed = 0
    if ":" in mode:
        mode, seed_str = mode.split(":", 1)
        seed = int(seed_str)

    if mode == "standard_greedy":
        r = backend.generate_standard(ids, max_new_tokens=mn, temperature=0.0,
                                      top_p=1.0, seed=0, do_sample=False)
        m = "standard_greedy"
    elif mode == "nostop_greedy":
        r = manual_generate(backend, ids, max_new_tokens=mn, temperature=0.0,
                            top_p=1.0, seed=0, do_sample=False)
        m = "nostop_greedy"
    elif mode == "standard_sampling":
        r = backend.generate_standard(ids, max_new_tokens=mn, temperature=0.7,
                                      top_p=0.9, seed=seed, do_sample=True)
        m = f"standard_sampling:{seed}"
    elif mode == "nostop_sampling":
        r = manual_generate(backend, ids, max_new_tokens=mn, temperature=0.7,
                            top_p=0.9, seed=seed, do_sample=True)
        m = f"nostop_sampling:{seed}"
    else:
        raise ValueError(mode)

    out_ids = list(r["output_ids"])
    gen_ids = out_ids[len(ids):] if len(out_ids) > len(ids) else out_ids
    return {
        "mode": m,
        "generated_ids": gen_ids,
        "full_ids": out_ids,
        "output_raw": backend.decode(gen_ids, skip_special_tokens=False),
        "output_full_raw": backend.decode(out_ids, skip_special_tokens=False),
        "generation_meta": {**r.get("meta", {}), "mode": m},
        "n_prompt_tokens": len(ids),
    }
