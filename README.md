# chat-template-lab —— 对话模板可观测性实验库

这个实验库研究一个具体问题:把结构化的消息列表交给本地开源大模型时,官方对话模板到底做了什么;模型能不能复述自己收到的模板内容;以及用户输入里的"模板字符串"有没有可能骗过模型。整条链路(messages → 官方模板渲染 → token 编号 → 模型输出)的每一步都留下可检查的中间文件,结论只用程序观测说话。

完整的研究问题、实验设计和解释规则见根目录的 `chat_template_prompt_injection_experiment_design.md`(设计书,建议先读"最重要的解释规则"一节,它限定了哪些结论可以说、哪些不能说)。

## 快速开始

以下命令在仓库根目录执行:

```bash
# 1. 创建虚拟环境并安装依赖(需要 Python 3.11+)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,mistral]"

# 2. 跑单元测试(不需要模型权重,也不需要联网)
.venv/bin/python -m pytest tests/ -m "not integration" -q

# 3. E0 阶段:只解剖分词器和模板,不下载模型权重
.venv/bin/python -m src.run_stage e0 --models qwen3.5-0.8b

# 4. 行为实验:先用小样本冒烟(smoke,n=20),再上正式样本(formal,n=100)
.venv/bin/python -m src.run_stage attack --models qwen3.5-0.8b --sample-size smoke
.venv/bin/python -m src.run_stage attack --models qwen3.5-0.8b --experiments E1 E2 E4

# 5. 汇总报告:CSV 表格 + Markdown 报告
.venv/bin/python -m src.run_stage report
```

每个实验批次会生成一个独立的运行目录 `runs/<时间戳>_<模型>_<阶段>/`,里面包含:

- `template/`:模板原文、逐 token 表格 `tokens.tsv`、模板控制面提取结果、撞号检测结果;
- `cases/`:逐用例的 JSONL 记录(含完整消息、渲染文本、双轨输出、判定与指标);
- `environment.json`:运行环境快照(软件版本、显卡状态);
- `metrics.json`:该批次按实验/攻击/生成方式聚合的指标行。

## 几个常用的运行参数

| 参数 | 作用 |
|---|---|
| `--models` | 指定要跑的模型(配置名见 `configs/models.yaml`) |
| `--experiments` | 指定实验组,如 `E5 E6`、`E1 E2 E4` |
| `--sample-size` | `smoke`(20 个金丝雀)或 `formal`(100 个金丝雀) |
| `--dry-run` | 只打印将要做什么,不下载不加载 |
| 环境变量 `CHATLAB_QUANT` | 强制量化档位(`int4_bnb`、`bf16`、`cpu` 等),显存紧张的机器推荐 `int4_bnb` |
| 环境变量 `CHATLAB_MAX_NEW_TOKENS` | 临时收紧单次生成的最大 token 数,调参用 |

## 术语对照(中文释义)

- **messages**:传给模型的结构化消息列表,每条带角色(role)与正文(content)。
- **chat template(对话模板)**:模型仓库提供的一段模板文本(通常是 Jinja 语法),负责把消息列表拼成模型真正看到的那一串输入。
- **token / token ID**:分词器把文本切成的片段及其编号。同一个字符串在不同模型里可能切法不同。
- **特殊 token(special token)**:有专门含义的 token,例如 `<|im_start|>`、`[INST]`、`</s>`,常用来标记角色边界或句子结束。
- **金丝雀(canary)**:每条用例随机生成的标识串(如 `SYS_CANARY_A19F`),用于精确判定模型有没有输出受保护内容,不依赖人工阅读。
- **ASR**:攻击成功率,即成功输出受保护标记的样本占比。
- **no-stop 解码**:自写的不停手逐 token 解码,用来区分"模型不会输出结束符"和"服务端看到结束符就截断"。
- **撞号(collision)**:用户内容里的字符串被分词器编成了与真实模板控制 token 相同的编号。

## 实验分层概览

| 阶段 | 内容 | 是否加载权重 |
|---|---|---|
| E0 | 模板与分词器解剖,撞号检测(RQ1/RQ3) | 否 |
| E1/E2 | 普通内容与伪模板的复制能力基线(RQ2) | 是 |
| E4 | 完整输入前缀复述(RQ2 核心) | 是 |
| E5–E7 | 人造约束下的注入攻击矩阵与提取(RQ4/RQ5) | 是 |
| E8 | Base 与 Instruct 同家族对照(RQ6) | 是 |

## 各实验组的实现位置

| 功能 | 代码文件 |
|---|---|
| 模板解剖快照 | `src/dump_template.py` |
| 渲染、逐 token 表格与来源标注 | `src/render_context.py` |
| 撞号检查(RQ3) | `src/inspect_injection_tokens.py` |
| 模板控制面提取(供动态攻击串使用) | `src/surfaces.py` |
| 攻击消息组装(E1–E8 各组的消息构造) | `src/attack_builder.py` |
| 模型加载与量化回退链 | `src/backends.py` |
| 双轨生成(标准 + 不停手解码) | `src/manual_decode.py`、`src/run_generation.py` |
| 实验编排与指标写入 | `src/run_attacks.py` |
| 复制/复述评分、攻击指标 | `src/score_copy.py`、`src/score_attack.py` |
| 汇总报告生成 | `src/report.py` |
| 阶段编排命令行入口 | `src/run_stage.py` |

## 验收标准对照(设计书第十三节)

| # | 验收项 | 实现位置 |
|---|--------|----------|
| 1 | 打印官方对话模板原文 | `src/dump_template.py` → `runs/*/template/chat_template.txt` |
| 2 | messages → 渲染文本 → token 编号 | `src/render_context.py` |
| 3 | system/user 边界对应哪些 token | `tokens.tsv` 的来源列 + `control_surfaces.json` |
| 4 | 检测用户内容内的控制 token 撞号 | `src/inspect_injection_tokens.py` |
| 5 | JSON 复制与模板复述对照 | E1/E2/E4 → `src/score_copy.py` |
| 6 | 至少 8 类提示注入 | E6 的 A0–A10 → `configs/attacks.yaml` |
| 7 | 不停手的手动解码器 | `src/manual_decode.py`(双轨报告) |
| 8 | Base 与 Instruct 对照 | `models.yaml` 的 `e8_pair` 配对 → E8 |
| 9 | Qwen3.5 与 Ministral 3 两种模板家族 | `configs/models.yaml` |
| 10 | 自动保存 revision、模板哈希、环境、种子 | `src/common.py` + `runs/*/environment.json` + 用例记录 |
| 11 | CSV/Markdown 汇总 | `src/report.py` → `reports/` |
| 12 | 报告 ASR 与样本数(禁止单例结论) | `metrics.json` + `summary.csv`(95% 自助法置信区间) |

## 实验结果的读取建议

按下面的顺序读,能最快理解"结论是什么":

1. **`reports/analysis_conclusions.md`** —— 主分析文档。它不罗列数据,而是按"问题→证据→解读→边界"把整套实验的结论讲清楚,直接回答"往用户输入里塞模板文本,模型会不会被骗""注入的分隔符和真实分隔符 token id 是否相同""最终序列长什么样"这类问题,并明确区分哪些是直接观测、哪些是合理推断、哪些超出实验能说的范围。
2. `reports/phase1_manifest.md` —— 8 个模型的分词器与模板解剖结果(模板长什么样、撞号是否成立、家族差异)。
3. `reports/formal_summary.md` —— 行为实验的逐模型结果摘要与方法学说明(如何避免把"续写复述"误判成攻击成功)。
4. `reports/summary.csv` 与 `reports/report.md` —— 机器生成的逐项汇总表,作为核对原始数字的证据索引,不需要通读。

所有结论请对照设计书第十五节的解释规则理解其边界。

## 本机运行环境与覆盖范围(2026-09 实测)

- 实验机:NVIDIA RTX 3050 Laptop 4GB,桌面占用后实际可用约 2.5–3.4GB,故主实验一律使用 4-bit 量化(int4)。
- 行为实验实际完成:Qwen3.5-0.8B(smoke 全套)、Qwen3.5-2B(正式 n=100 全套)、Qwen3.5-4B(仅部分)、0.8B/2B 的 Base 对照(n=100,std 轨)。
- Ministral-3-3B 在本机只完成了分词器/模板层(E0):其官方仓库权重为 FP8 格式(约 3.85GB,超过本卡可用显存),且当前 transformers 的因果语言模型入口不支持 mistral3 架构,故行为实验记录为"设备不可行"。该部分需要 8GB 以上显存或改用 vLLM 等引擎(设计书第十四节 Phase 5 路径)。
- 设计书第十四节的 Phase 5(vLLM/SGLang + OpenAI SDK + LangChain 三层对照)在本机未执行,原因同样是显存不足。
- 以上限制都如实记录在各批次 `runs/` 的 `environment.json` 与 `metrics.json` 中,报告的样本数与设备信息与实际运行一致。
