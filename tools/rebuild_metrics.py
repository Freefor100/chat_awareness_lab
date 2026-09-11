"""从已存的逐条用例记录重建每个 run 的 metrics.json。

metrics.json 是 append-only 的 JSONL：某个单元格重跑之后直接追加，会出现新旧
两行并存；评分口径修正之后，旧行也不会自己更新。本工具按当前口径把整个 run 的
指标行重算一遍，非用例类的行（模型加载记录、E0 解剖记录）原样保留。

与 tools/normalize_records.py 的分工：那个工具修的是"每条记录里的派生字段"，
本工具修的是"由这些字段汇总出来的指标行"。两者都只读已存的 token 序列与文本，
不重新调用模型。
"""

from pathlib import Path

from src.common import append_jsonl, read_jsonl
from src.metrics import build_metric_rows

ROOT = Path(__file__).resolve().parents[1]
# 这些实验的行不是用例汇总出来的，重建时原样保留
KEEP_EXPERIMENTS = ("load", "E0", "e0")


def rebuild_run(run: Path) -> int:
    files = sorted((run / "cases").glob("*.jsonl"))
    if not files:
        return 0
    old = read_jsonl(run / "metrics.json") if (run / "metrics.json").exists() else []
    keep = [r for r in old if r.get("experiment") in KEEP_EXPERIMENTS]
    model_key = (old[0].get("model_key") if old else None) \
        or run.name.split("_", 2)[-1].replace("_attack", "")
    rows = []
    for f in files:
        recs = read_jsonl(f)
        if recs:
            rows += build_metric_rows(recs, run.name, model_key)
    # 排序只为让 diff 稳定：按实验、攻击、生成方式
    rows.sort(key=lambda r: (r["experiment"], r["attack"], r["mode"]))
    mp = run / "metrics.json"
    mp.unlink(missing_ok=True)
    for r in keep + rows:
        append_jsonl(mp, r)
    print(f"{run.name}: {len(keep)} 行保留 + {len(rows)} 行重建")
    return len(rows)


def main() -> None:
    total = 0
    for run in sorted((ROOT / "runs").iterdir()):
        if run.is_dir() and (run / "cases").is_dir():
            total += rebuild_run(run)
    print(f"共重建 {total} 行")


if __name__ == "__main__":
    main()
