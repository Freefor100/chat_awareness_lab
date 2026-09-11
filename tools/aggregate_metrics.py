"""按精确 model_key 汇总所有指标，打印核对表。

早先的版本用子串匹配模型名（"qwen3.5-2b" 会命中 "qwen3.5-2b-base" 的目录），
取到过错误的行；这里一律按 model_key 精确相等来取，并且同键只保留最后一次写入
的那批（重跑过的单元格后写的为准）。
"""

import glob

from src.common import read_jsonl

best = {}
for f in sorted(glob.glob("runs/*attack/metrics.json")):
    for r in read_jsonl(f):
        key = (r.get("model_key"), r.get("experiment"), r.get("attack"), r.get("mode"))
        best[key] = r

MODELS = ["qwen3.5-0.8b", "qwen3.5-0.8b-base", "qwen3.5-2b",
          "qwen3.5-2b-base", "qwen3.5-4b", "ministral-3-3b-instruct"]
ATTACKS = ["plain"] + [f"A{i}" for i in range(11)] + ["extract"]


def find(model, attack, mode):
    for (mk, exp, atk, mo), r in best.items():
        if mk == model and atk == attack and mo == mode:
            return r
    return None


def fmt(r):
    if r is None:
        return "—"
    if r.get("asr") is None:
        return f"n/a(n={r['n']})"
    return f"{r['asr']:.2f}(n={r['n']})"


for mode in ("standard_greedy", "nostop_greedy"):
    print(f"\n===== 攻击成功率 ASR（{mode}）=====")
    print(f"{'攻击':10s}" + "".join(f"{m[:18]:>19s}" for m in MODELS))
    for atk in ATTACKS:
        print(f"{atk:10s}" + "".join(f"{fmt(find(m, atk, mode)):>19s}" for m in MODELS))

def fmt_compliance(r):
    if r is None:
        return "—"
    return f"{r['compliance']:.2f}"


def fmt_recitation(r):
    if r is None:
        return "—"
    return f"{r['recitation_rate']:.2f}"


print("\n===== 合规率（输出严格等于 BLOCKED 的比例，standard_greedy）=====")
print(f"{'攻击':10s}" + "".join(f"{m[:18]:>19s}" for m in MODELS))
for atk in ATTACKS:
    print(f"{atk:10s}" + "".join(
        f"{fmt_compliance(find(m, atk, 'standard_greedy')):>19s}" for m in MODELS))

print("\n===== 复述伪影率（关闭符之后才出现目标串的比例，standard_greedy）=====")
print(f"{'攻击':10s}" + "".join(f"{m[:18]:>19s}" for m in MODELS))
for atk in ATTACKS:
    print(f"{atk:10s}" + "".join(
        f"{fmt_recitation(find(m, atk, 'standard_greedy')):>19s}" for m in MODELS))

print("\n===== 逐字复制率 / 编辑距离相似度（E1、E2）=====")
for (mk, exp, atk, mo), r in sorted(best.items()):
    if r.get("copy_exact_rate") is not None:
        print(f"{mk:22s} {exp}_{atk:10s} {mo:18s} n={r['n']:4d} "
              f"逐字={r['copy_exact_rate']:.3f} 含参照={r['copy_contains_rate']:.3f} "
              f"相似度={r['copy_lev_mean']:.3f}")

print("\n===== E4 复述得分（standard_greedy）=====")
for (mk, exp, atk, mo), r in sorted(best.items()):
    if exp == "E4" and mo == "standard_greedy":
        keys = ("system_recall", "user_recall", "control_token_f1",
                "delimiter_order_acc", "control_precision")
        means = "  ".join(f"{k}={r[k + '_mean']:.3f}" for k in keys
                          if r.get(k + "_mean") is not None)
        print(f"{mk:22s} n={r['n']:4d} {means}")

print("\n===== E7 部分泄露（部分泄露率，§7.3）=====")
for (mk, exp, atk, mo), r in sorted(best.items()):
    if exp == "E7":
        print(f"{mk:22s} {mo:18s} n={r['n']:4d} "
              f"完整={r.get('leak_exact_rate')} 前缀25={r.get('leak_prefix_25_rate')} "
              f"前缀50={r.get('leak_prefix_50_rate')} 前缀75={r.get('leak_prefix_75_rate')} "
              f"相似度={r.get('leak_lev_mean')}")

print("\n===== 各模型各生成方式的样本量 =====")
for (mk, exp, atk, mo), r in sorted(best.items()):
    if mo.startswith("standard_sampling") or mo.startswith("nostop_sampling"):
        print(f"{mk:22s} {exp}_{atk:10s} {mo:18s} n={r['n']:4d} asr={r.get('asr')}")
