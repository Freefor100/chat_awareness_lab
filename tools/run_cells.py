"""分批跑单元格，可中断续跑，写进同一个 run 目录。

为什么需要它：`run_stage attack` 每次调用都会新建一个带时间戳的 run 目录，
而且断点续跑的粒度是"整个单元格文件存在就跳过"——中途被打断会留下一个残缺
的 jsonl，下次却会被当成已完成。本工具按 (模型, 攻击) 逐个跑，写进指定的
run 目录，并且在每个单元格跑完后核对行数：行数不对就删掉重跑。

用法：
  python tools/run_cells.py <模型key> <run目录> <攻击列表> <样本数> <生成方式>
例：
  python tools/run_cells.py qwen3.5-0.8b runs/20260911_a11_qwen3.5-0.8b_attack \
      A11a,A11b,A11c,A11d 20 standard_greedy,nostop_greedy
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backends import load_backend            # noqa: E402
from src.common import (load_yaml, model_cfg, template_sha256,  # noqa: E402
                        hf_resolve_revision)
from src.run_attacks import run_experiments      # noqa: E402


def cell_ok(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) == expected


def main() -> None:
    key, run_dir, attacks, n, modes = sys.argv[1:6]
    attacks = attacks.split(",")
    modes = modes.split(",")
    n = int(n)
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)
    if not (run_dir / "environment.json").exists():
        import json
        from src.common import env_snapshot
        (run_dir / "environment.json").write_text(
            json.dumps(env_snapshot(), indent=2), encoding="utf-8")

    todo = []
    for atk in attacks:
        f = run_dir / "cases" / f"E6_{atk}.jsonl"
        if cell_ok(f, n * len(modes)):
            print(f"[跳过] {key} E6:{atk} 已完成（{n * len(modes)} 行）", flush=True)
            continue
        if f.exists():
            print(f"[重跑] {key} E6:{atk} 上次残留 {sum(1 for _ in open(f))} 行，"
                  f"应为 {n * len(modes)} 行", flush=True)
            f.unlink()
        todo.append(atk)
    if not todo:
        print(f"[{key}] 无需运行", flush=True)
        return

    cfg = model_cfg(ROOT / "configs/models.yaml", key)
    # 与 run_stage 一致：固定 revision 并记录模板哈希，保证结论可回溯（§9.7）
    rev = cfg.get("revision") or hf_resolve_revision(cfg["id"])
    sha = template_sha256(cfg["id"], rev)
    cfg = {**cfg, "revision": rev}
    be = load_backend(cfg, tokenizer_only=False)
    if be.model is None:
        print(f"[{key}] 权重未加载：{be.failure_reason}", flush=True)
        return
    # 档位必须落在日志里：同批的 A0–A10 用的是哪一档，这里就得用哪一档，
    # 否则"跨轮 vs 单轮"的对比会混进精度差异（曾经踩过：A11 跑成 bf16，
    # 而 9 月的 A0–A10 是 int4，两边输出形态完全不同）。
    print(f"[{key}] 实际加载档位 quant={be.quant} device={be.device} "
          f"backend={be.backend_name}", flush=True)
    expect = os.environ.get("CHATLAB_EXPECT_QUANT")
    if expect and str(be.quant) != expect:
        print(f"[{key}] 档位不符：期望 {expect}，实际 {be.quant}；中止以免污染数据",
              flush=True)
        return
    gen = load_yaml(ROOT / "configs/generation.yaml")
    print(f"[{key}] run_dir={run_dir.name} 本次跑 {todo}", flush=True)
    stats = run_experiments(be, ["E6"], key, run_dir, gen, n_canaries=n,
                            revision=rev, template_sha=sha,
                            modes=modes, attacks=todo)
    print(f"[{key}] 完成 {stats['per_exp']}", flush=True)
    be.release()
    for atk in todo:
        f = run_dir / "cases" / f"E6_{atk}.jsonl"
        ok = cell_ok(f, n * len(modes))
        print(f"  {'✓' if ok else '✗'} E6:{atk} "
              f"{sum(1 for _ in open(f)) if f.exists() else 0}/{n * len(modes)} 行",
              flush=True)


if __name__ == "__main__":
    main()
