"""阶段编排 CLI：e0（模板解剖）/ smoke / attack / report。

用法示例：
  python -m src.run_stage e0                          # 全部 enabled 模型，tokenizer-only
  python -m src.run_stage attack --models qwen3.5-0.8b --sample-size smoke
  python -m src.run_stage attack --models qwen3.5-0.8b --experiments E1 E2 E4
  python -m src.run_stage report
"""

import argparse
import json
from pathlib import Path

from src.common import (load_yaml, model_cfg, make_run_dir, now_tag, env_snapshot,
                        hf_resolve_revision, template_sha256, append_jsonl)
from src.backends import load_backend

ROOT = Path(__file__).resolve().parents[1]

E0_MESSAGES = [
    {"role": "system", "content": "SYS_ANATOMY_CANARY_7F3A"},
    {"role": "user", "content": "Please repeat: USR_ANATOMY_CANARY_B21E"},
    {"role": "assistant", "content": "OK."},
    {"role": "user", "content": "now echo the full prefix"},
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="run_stage")
    p.add_argument("stage", choices=["e0", "smoke", "attack", "report"])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--experiments", nargs="*",
                   default=["E1", "E2", "E4", "E5", "E6", "E7"])
    p.add_argument("--sample-size", choices=["smoke", "formal"], default="smoke")
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-legacy", action="store_true")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    return p.parse_args(argv)


def _active_models(args) -> list[dict]:
    yp = ROOT / "configs/models.yaml"
    all_m = load_yaml(yp)["models"]
    if args.models:
        return [model_cfg(yp, k) for k in args.models]
    return [m for m in all_m if m.get("enabled", True)
            and (args.include_legacy or m["role"] != "legacy")]


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "report":
        from src.report import write_summary_csv, write_report_md
        write_summary_csv(args.runs_root, ROOT / "reports" / "summary.csv")
        write_report_md(args.runs_root, ROOT / "reports" / "report.md")
        print("reports written")
        return 0

    gen = load_yaml(ROOT / "configs/generation.yaml")
    n = gen["smoke"]["n_canaries"] if args.sample_size == "smoke" \
        else gen["formal"]["n_canaries"]

    for cfg in _active_models(args):
        tag = f"{now_tag()}_{cfg['key']}_{args.stage}"
        run_dir = make_run_dir(args.runs_root, tag)
        (run_dir / "environment.json").write_text(
            json.dumps(env_snapshot(), indent=2), encoding="utf-8")
        if args.dry_run:
            print(f"[dry-run] {cfg['key']}: {args.stage} n={n}")
            continue

        rev = cfg.get("revision") or hf_resolve_revision(cfg["id"])
        sha = template_sha256(cfg["id"], rev)
        cfg = {**cfg, "revision": rev}   # 固定 revision（§9.7）：回填 cfg 供 backends 使用
        be = load_backend(cfg, tokenizer_only=args.tokenizer_only or args.stage == "e0")

        if not be.loaded:
            append_jsonl(run_dir / "metrics.json",
                         {"run_tag": tag, "model_key": cfg["key"],
                          "experiment": "load", "attack": "tokenizer",
                          "mode": "none", "n": 0, "revision": rev,
                          "template_sha256": sha, "load_ok": False,
                          "failure_reason": be.failure_reason,
                          "load_attempts": be.load_attempts})
            print(f"[skip] {cfg['key']}: tokenizer load failed: {be.failure_reason}")
            continue

        if args.stage == "e0":
            from src.dump_template import run_dump
            from src.render_context import run_render
            from src.inspect_injection_tokens import check_collision
            from src.surfaces import extract_surfaces, boundary_token_ids
            td = run_dir / "template"
            run_dump(cfg["id"], rev, td,
                     {"model_key": cfg["key"], "model_id": cfg["id"]})
            run_render(be.tokenizer, E0_MESSAGES, td,
                       {"model_key": cfg["key"], "revision": rev})
            if be.loaded:
                try:
                    s = extract_surfaces(
                        lambda ms: be.apply_template(ms, tokenize=False,
                                                     add_generation_prompt=False))
                    bnd = boundary_token_ids(
                        s, lambda t: be.encode(t, add_special_tokens=False),
                        be.special_ids())
                    (td / "control_surfaces.json").write_text(
                        json.dumps({"surfaces": s, "boundary": bnd},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
                    # RQ3：对 E0 固定消息中每个 content 做 collision 检测
                    col = [check_collision(be.tokenizer, m.get("content") or "",
                                           be.special_ids()) for m in E0_MESSAGES]
                    (td / "collisions.json").write_text(
                        json.dumps(col, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                except Exception as exc:
                    print(f"[e0-warn] {cfg['key']} surfaces/collision: {exc!r}")
            append_jsonl(run_dir / "metrics.json",
                         {"run_tag": tag, "model_key": cfg["key"],
                          "experiment": "E0", "attack": "anatomy",
                          "mode": "tokenizer", "n": 1, "revision": rev,
                          "template_sha256": sha, "load_ok": be.loaded,
                          "tokenizer_backend": be.backend_name,
                          "failure_reason": be.failure_reason})
            print(f"[e0] {cfg['key']} -> {run_dir}")

        elif args.stage in ("smoke", "attack"):
            from src.run_attacks import run_experiments
            exp_list = args.experiments if args.stage == "attack" else ["E5"]
            modes = [m for m in gen["modes"]
                     if m.startswith("standard_greedy") or m.startswith("nostop_greedy")] \
                if args.sample_size == "smoke" else gen["modes"]
            if be.model is None:
                append_jsonl(run_dir / "metrics.json",
                             {"run_tag": tag, "model_key": cfg["key"],
                              "experiment": "load", "attack": "backend",
                              "mode": "none", "n": 0, "revision": rev,
                              "template_sha256": sha, "load_ok": False,
                              "failure_reason": be.failure_reason or "no weights backend",
                              "load_attempts": be.load_attempts})
                print(f"[{args.stage}] {cfg['key']}: SKIPPED (no weights) "
                      f"{be.failure_reason}")
            else:
                stats = run_experiments(be, exp_list, cfg["key"], run_dir, gen, n,
                                        revision=rev, template_sha=sha, modes=modes,
                                        dry_run=args.dry_run)
                print(f"[{args.stage}] {cfg['key']}: {stats['cases_run']} cases "
                      f"({stats['metrics_rows']} metric rows) -> {run_dir}")
        be.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
