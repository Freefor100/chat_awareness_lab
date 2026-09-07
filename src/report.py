"""运行指标聚合 → summary.csv + report.md（设计书 §7/§11 reports/）。

metrics.json 实际是 append-only JSONL（Ruling P5）：每个 run 目录一行一组指标。
"""

import csv
import json
from datetime import datetime
from pathlib import Path


def collect_run_metrics(runs_root: Path) -> list[dict]:
    out = []
    for d in sorted(Path(runs_root).iterdir()):
        mp = d / "metrics.json"
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
    return out


def write_summary_csv(runs_root: Path, out_csv: Path) -> None:
    rows = collect_run_metrics(runs_root)
    fields = ["run_tag", "model_key", "experiment", "attack", "mode", "asr",
              "ci_lo", "ci_hi", "compliance", "n", "quant", "device",
              "revision", "model_revision", "template_sha256"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            ci = r.get("ci") or [None, None]
            r.setdefault("ci_lo", ci[0] if len(ci) > 0 else None)
            r.setdefault("ci_hi", ci[1] if len(ci) > 1 else None)
            r.setdefault("revision", r.get("model_revision", ""))
            w.writerow(r)


def write_report_md(runs_root: Path, out_md: Path) -> None:
    rows = collect_run_metrics(runs_root)
    lines = ["# Chat-Template / Prompt-Injection 实验报告", "",
             f"生成时间：{datetime.now().isoformat(timespec='seconds')}", "",
             "> 解释规则（设计书 §15）：本报告只陈述观测事实 —— 模型 + revision + "
             "模板 + N + 攻击方法下的 ASR。不得外推为“模型看不到 system prompt”"
             "或“托管 API 均可被绕过”等结论。", ""]
    if rows:
        lines += ["## 汇总", "",
                  "| run | model | exp | attack | mode | asr | ci_lo | ci_hi | "
                  "compliance | n | quant |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ci = r.get("ci") or [None, None]
            lines.append(f"| {r.get('run_tag', '')} | {r.get('model_key', '')} | "
                         f"{r.get('experiment', '')} | {r.get('attack', '')} | "
                         f"{r.get('mode', '')} | {r.get('asr', '')} | "
                         f"{ci[0] if len(ci) > 0 else ''} | "
                         f"{ci[1] if len(ci) > 1 else ''} | "
                         f"{r.get('compliance', '')} | {r.get('n', '')} | "
                         f"{r.get('quant', '')} |")
        repro = [r for r in rows
                 if any(k.endswith("_mean") and r.get(k) is not None for k in r)]
        if repro:
            lines += ["", "## E4 复述评分均值（system/user/control recall 等）", ""]
            for r in repro:
                extra = " ".join(f"{k}={v:.3f}" for k, v in sorted(r.items())
                                 if k.endswith("_mean") and v is not None)
                lines.append(f"- {r.get('run_tag')} {r.get('experiment')} "
                             f"{r.get('attack')} {r.get('mode')}: {extra}")
    else:
        lines.append("（无 metrics.json 数据）")
    lines += ["", "## 记录字段说明", "",
              "每条 run 的 environment.json 记录 python/torch/transformers/GPU 快照；"
              "case 级 JSONL 位于 runs/<tag>/cases/（含 revision、template_sha256、"
              "generation mode、quant、device、thinking、canary 与输出）。", ""]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
