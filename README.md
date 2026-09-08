# chat-template-lab

messages → 官方 chat template → token IDs → 本地模型输出 的全链路可观测实验库。
见 `chat_template_prompt_injection_experiment_design.md`（spec v1.0）。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,mistral]"

# 单元测试（无需模型权重 / 网络）
.venv/bin/python -m pytest tests/ -m "not integration" -q

# E0：tokenizer 级解剖（不下载权重）
.venv/bin/python -m src.run_stage e0 --models qwen3.5-0.8b

# 行为实验（smoke n=20 / formal n=100）
.venv/bin/python -m src.run_stage attack --models qwen3.5-0.8b --sample-size smoke
.venv/bin/python -m src.run_stage attack --models qwen3.5-0.8b --experiments E1 E2 E4

# 汇总报告
.venv/bin/python -m src.run_stage report
```

每个 run 落在 `runs/<timestamp>_<model>_<stage>/`：`template/`（chat_template.txt、
tokens.tsv、control_surfaces.json、collisions.json）、`cases/*.jsonl`（§12 schema +
sidecar）、`environment.json`、`metrics.json`。

## 验收标准对照（spec v1.0 §13）

| # | 验收项 | 实现位置 |
|---|--------|----------|
| 1 | 打印官方 chat template 原文 | src/dump_template.py → runs/*/template/chat_template.txt |
| 2 | messages → rendered → token IDs | src/render_context.py |
| 3 | system/user boundary 对应哪些 token | tokens.tsv 的 source_guess + control_surfaces.json |
| 4 | 检测 user content 内 control-token IDs | src/inspect_injection_tokens.py（RQ3 collision） |
| 5 | JSON copy 与 template reproduction 对照 | E1/E2/E4 → src/score_copy.py |
| 6 | ≥8 类 prompt injection | E6 A0–A10 → configs/attacks.yaml |
| 7 | manual no-stop decoder | src/manual_decode.py（双轨报告） |
| 8 | Base vs Instruct 对照 | models.yaml e8_pair → E8 |
| 9 | Qwen3.5 与 Ministral 3 两种模板家族 | configs/models.yaml |
| 10 | 自动保存 revision/template hash/环境/seed | src/common.py + runs/*/environment.json + case JSONL |
| 11 | CSV/Markdown 汇总 | src/report.py → reports/ |
| 12 | 报告 ASR 与样本数（禁单例结论） | metrics.json + summary.csv（95% bootstrap CI） |

## 解释边界（spec §15）

报告只陈述观测事实（模型 + revision + 模板 + N + 方法下的 ASR）；不外推为
“模型看不到 system prompt”“存在权限系统”或“托管 API 均可被绕过”。
