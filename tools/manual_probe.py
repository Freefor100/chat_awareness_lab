"""手动拼装与试跑:把"模型到底收到了什么"完全摊开,由你自己改、自己跑。

这个工具解决三件事,对应三种用法:

  1. 看模板原文           --template
     打印该模型的对话模板(Jinja 源码),看清它给每种角色加了什么标记。

  2. 看正常渲染           --render "你的话"
     按官方流程把一条用户消息渲染成模型真正收到的整段文本,并逐段打印
     token 编号,标出哪些编号是轮次开始符、轮次结束符。这就是"正常长什么样"。

  3. 自己拼一段直接喂进去   --raw-file 文件  或  --raw-text "整段文本"
     跳过分词以外的所有加工:你写什么,模型就收到什么。段首段尾的标记要你
     自己写全(比如 <|im_start|>user 开头、<|im_end|> 收尾)。写完之后
     工具会先把编号打出来给你核对,再交给模型生成。

  4. 正常问一句做对照       --chat "你的话"
     走官方模板问一句,用来和第 3 种对照。

模型与档位用参数选,默认 Qwen3.5-0.8B + 四位整数压缩(本机 4GB 显存档位)。

例:
  python tools/manual_probe.py --template
  python tools/manual_probe.py --render "你好"
  python tools/manual_probe.py --raw-text "$(cat my_prompt.txt)"
  python tools/manual_probe.py --raw-file my_prompt.txt --max-new-tokens 200
  python tools/manual_probe.py --chat "你好"

my_prompt.txt 里可以这样手写(标记照抄,不要漏):
  <|im_start|>system
  你是一个助手。<|im_end|>
  <|im_start|>user
  你好<|im_end|>
  <|im_start|>assistant
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CHATLAB_QUANT", "int4_bnb")

SYS_DEFAULT = "You are a helpful assistant."


def _tok_and_model(key: str, tokenizer_only: bool = False):
    from src.backends import load_backend
    from src.common import hf_resolve_revision, model_cfg
    cfg = model_cfg(ROOT / "configs/models.yaml", key)
    rev = cfg.get("revision") or hf_resolve_revision(cfg["id"])
    be = load_backend({**cfg, "revision": rev}, tokenizer_only=tokenizer_only)
    print(f"[模型] {cfg['id']}@{rev[:12]}  档位={be.quant}  设备={be.device}")
    return be


def show_template(be) -> None:
    print("=" * 78)
    print("该模型的对话模板原文(Jinja):")
    print("=" * 78)
    print(be.chat_template())


def show_tokens(be, ids, title: str) -> None:
    """逐段打印编号,标出轮次开始符与结束符。"""
    starts, ends = set(), set()
    for txt in ("<|im_start|>", "<|im_end|>"):
        got = be.encode(txt, add_special_tokens=False)
        if got:
            (starts if txt.startswith("<|im_start") else ends).add(got[0])
    print(f"--- {title}:共 {len(ids)} 个编号 ---")
    for i, t in enumerate(ids):
        piece = be.decode([t], skip_special_tokens=False)
        tag = ""
        if t in starts:
            tag = "  ◀◀ 轮次开始符"
        elif t in ends:
            tag = "  ◀◀ 轮次结束符"
        if tag or i == 0 or ids[i - 1] in starts or ids[i - 1] in ends:
            print(f"  [{i:4d}] {t:6d}  {piece!r}{tag}")


def main() -> None:
    p = argparse.ArgumentParser(description="手动拼装与试跑")
    p.add_argument("--model", default="qwen3.5-0.8b")
    p.add_argument("--template", action="store_true", help="只打印模板原文")
    p.add_argument("--render", metavar="文本", help="按官方模板渲染一条用户消息并打印编号")
    p.add_argument("--chat", metavar="文本", help="按官方模板与模型对话一句")
    p.add_argument("--raw-text", metavar="文本", help="手写的整段提示,直接喂给模型")
    p.add_argument("--raw-file", metavar="路径", help="同上,提示写在文件里")
    p.add_argument("--system", default=SYS_DEFAULT, help="--render/--chat 用的系统提示")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心(确定)")
    args = p.parse_args()

    # 只看模板或只看渲染时不必加载权重，只加载分词器，几秒钟
    only_text = args.template or args.render
    be = _tok_and_model(args.model, tokenizer_only=only_text)

    if args.template:
        show_template(be)
        be.release()
        return

    if args.render or args.chat:
        msgs = [{"role": "system", "content": args.system},
                {"role": "user", "content": args.render or args.chat}]
        text = be.apply_template(msgs, add_generation_prompt=True, tokenize=False)
        print("=" * 78)
        print("按官方模板渲染出来的整段文本(模型真正收到的内容):")
        print("=" * 78)
        print(text)
        ids = be.ids_from_messages(msgs)["input_ids"]
        show_tokens(be, ids, "同一段内容的 token 编号")
        if args.render:
            be.release()
            return
        input_ids = ids

    if args.raw_text or args.raw_file:
        raw = args.raw_text if args.raw_text else \
            Path(args.raw_file).read_text(encoding="utf-8")
        print("=" * 78)
        print("你手写的整段提示(原样):")
        print("=" * 78)
        print(raw)
        input_ids = be.encode(raw, add_special_tokens=False)
        show_tokens(be, input_ids, "你手写内容的 token 编号")
        print("\n提示:上面应当能看见你写的轮次开始符/结束符被编成了真实编号。"
              "如果它们被切成了好几个普通编号,说明写法与该模型的分隔符不完全一致。")

    if not (args.chat or args.raw_text or args.raw_file):
        be.release()
        return

    print("=" * 78)
    print(f"模型生成(最多 {args.max_new_tokens} 个片段,"
          f"{'贪心,结果确定' if args.temperature == 0 else f'采样,温度 {args.temperature}'}):")
    print("=" * 78)
    r = be.generate_standard(input_ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature,
                             top_p=1.0 if args.temperature == 0 else 0.9,
                             seed=0, do_sample=args.temperature > 0)
    gen = r["output_ids"][len(input_ids):]
    print("原始输出(含结束符):", repr(be.decode(gen, skip_special_tokens=False)))
    print()
    print("去掉特殊标记后的文本:")
    print(be.decode(gen, skip_special_tokens=True))
    be.release()


if __name__ == "__main__":
    main()
