"""按精确 model_key 聚合所有指标,输出核对表(避免子串误匹配)。用后保留在 tools/。"""
import glob
from collections import defaultdict

from src.common import read_jsonl

# (model_key, experiment, attack, mode) -> row,取最新批次
best = {}
for f in sorted(glob.glob("runs/*attack/metrics.json")):
    for r in read_jsonl(f):
        key = (r.get("model_key"), r.get("experiment"), r.get("attack"), r.get("mode"))
        best[key] = r

order_models = ["qwen3.5-0.8b", "qwen3.5-0.8b-base", "qwen3.5-2b",
                "qwen3.5-2b-base", "qwen3.5-4b", "ministral-3-3b-instruct"]
order_atk = ["plain"] + [f"A{i}" for i in range(11)] + ["extract"]

for mode in ("standard_greedy", "nostop_greedy"):
    print(f"\n===== {mode} =====")
    print(f"{'attack':8s} " + " ".join(f"{m[:16]:>17s}" for m in order_models))
    for atk in order_atk:
        cells = []
        for m in order_models:
            found = None
            for (mk, exp, a, mo), r in best.items():
                if mk == m and a == atk and mo == mode:
                    found = r
                    break
            cells.append(f"{found['asr']:.2f}(n={found['n']})" if found else "—")
        print(f"{atk:8s} " + " ".join(f"{c:>17s}" for c in cells))

print("\n===== 复述类指标 =====")
for (mk, exp, a, mo), r in sorted(best.items()):
    if any(k.endswith("_mean") and r.get(k) is not None for k in r) \
            or r.get("copy_exact_rate") is not None:
        means = {k: round(v, 3) for k, v in r.items()
                 if k.endswith("_mean") and v is not None}
        print(f"{mk:20s} {exp}_{a:10s} {mo:18s} n={r['n']} "
              f"copy_exact={r.get('copy_exact_rate')} copy_lev={r.get('copy_lev_mean')} "
              f"{means}")
