"""实测:同一个分隔符编号,站在真位置和假位置,模型内部算出来的东西差多少。

背景:机制实验已经证明,用户消息里写的 <|fim_prefix|> 和模板放的真实边界,
分词后是同一个编号(248045)。但"编号相同"不等于"模型内部的表示相同"——
注意力计算中,查询(q)和键(k)会先按位置做旋转(位置编码),而且每个位置
只能注意到它前面的内容。因此:
  - 站在序列开头的真分隔符:位置旋转量小,且只能注意到自己;
  - 站在用户正文中段的假分隔符:位置旋转量不同,能注意到前面全部片段
    (包括整段真系统提示)。
从第一层起,两者的内部表示就应当分道扬镳。本脚本用真实权重把这件事测出来。

测三件事(对实验里那条 A6 攻击序列,Qwen3.5-0.8B,CPU 单次前向):
  1. 嵌入层:真假两个分隔符的向量是否完全相同(应当相同,余弦相似度 1.0);
  2. 第一层注意力:两个位置各自"看"了谁;
  3. 逐层隐藏状态:两个位置的余弦相似度随深度如何变化。
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-0.8B"
CASE = "runs/20260907_205249_qwen3.5-0.8b_attack/cases/E6_A6.jsonl"
# 词表里同时存在 <|object_ref_start|>/<|fim_middle|>(248060/248062,多模态谱系遗留);
# 当前纯文本模板实际使用的是 <|fim_prefix|>/<|fim_middle|> 这一对。
IM_START_NAME = None  # 不再用词表键名(显示层易混淆);直接取序列首位


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()


tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open(CASE)]
ids_list = tok.apply_chat_template(rows[0]["messages"], tokenize=True,
                                   add_generation_prompt=True,
                                   return_dict=True)["input_ids"]
ids = torch.tensor([ids_list])

im_start = ids_list[0]  # 序列第 0 位就是真的 im_start(模板首字符)
occ = [i for i, t in enumerate(ids_list) if t == im_start]
# 两次后接 "system" 的出现:第一次是真 system 轮开头,第二次是用户正文里的假货
sys_occ = [i for i in occ
           if tok.decode([ids_list[i + 1]], skip_special_tokens=False) == "system"]
real_pos, fake_pos = sys_occ[0], sys_occ[1]
print(f"真分隔符位置: {real_pos}(序列开头)")
print(f"假分隔符位置: {fake_pos}(用户消息正文内)")
print(f"序列总长: {len(ids_list)} 个片段")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager")
except TypeError:  # 旧参数名
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()
with torch.no_grad():
    res = model(ids, output_hidden_states=True, output_attentions=True)

# ---- 1. 嵌入层 ----
emb = model.get_input_embeddings()(ids[0])
print(f"\n[1] 嵌入向量余弦相似度(同一编号查同一行,应为 1.000):"
      f" {cos(emb[real_pos], emb[fake_pos]):.6f}")

# ---- 2. 第一层注意力:各自看了谁 ----
att0 = res.attentions[0][0]  # [heads, seq, seq]
row_real = att0[:, real_pos, :].mean(0)
row_fake = att0[:, fake_pos, :].mean(0)
print("\n[2] 第 1 层注意力(全部头平均):")
print(f"    真位置 {real_pos}:落在自己身上的权重 {row_real[real_pos]:.3f}"
      f"(它前面没有任何内容,只能看自己)")
top = torch.topk(row_fake, 3)
print(f"    假位置 {fake_pos}:注意力分散,前三大关注对象:")
for w, idx in zip(top.values.tolist(), top.indices.tolist()):
    piece = tok.decode([ids_list[idx]], skip_special_tokens=False)
    print(f"      权重 {w:.3f}  位置 {idx}  {piece!r}")
print(f"    假位置注意到(权重>0.01)的片段数:{int((row_fake > 0.01).sum())}")

# ---- 3. 逐层隐藏状态相似度 ----
hs = res.hidden_states  # (层数+1, 1, seq, hidden);hs[0] 即嵌入输出
n_layers = len(hs) - 1
pick = sorted({0, 1, 2, 4, 8, 12, 16, n_layers // 2, n_layers - 1, n_layers})
print(f"\n[3] 两个位置隐藏状态的余弦相似度(模型共 {n_layers} 层):")
for li in pick:
    stage = "嵌入(第 0 层前)" if li == 0 else f"第 {li} 层后"
    print(f"    {stage:>16s}: "
          f"{cos(hs[li][0, real_pos], hs[li][0, fake_pos]):.4f}")
