"""复制/复述指标单测（score_copy）。"""

from src.score_copy import (exact_match, char_edit_distance, token_f1,
                            normalized_lev, score_reproduction)


def test_exact_and_edit_distance():
    assert exact_match("abc", "abc") and not exact_match("abc", "abd")
    assert char_edit_distance("kitten", "sitting") == 3
    assert char_edit_distance("abc", "abc") == 0


def test_token_f1():
    # ref=[1,2,3], out=[1,2,3,4]: precision 3/4=0.75, recall 3/3=1 → f1=6/7
    f = token_f1([1, 2, 3], [1, 2, 3, 4])
    assert abs(f - (2 * 0.75 * 1.0 / (0.75 + 1.0))) < 1e-6


def test_normalized_lev():
    assert normalized_lev("abc", "abc") == 1.0
    assert normalized_lev("abc", "xyz") == 0.0


def test_score_reproduction_full():
    # 输出实际时序：control 打头 → system → user（order 列表按真实前缀时序）
    ref = {"system": "sys line", "user": "usr line",
           "control_text": "<|im_start|>", "control_ids": [99],
           "all_control_ids": [98, 99],
           "order": ["control", "system", "user"]}
    out = "<|im_start|>sys line<|im_end|> usr line"
    s = score_reproduction(ref, out, [99], instr=None)   # out_ids = 生成段（不含 prompt）
    assert s["system_recall"] > 0.9 and s["user_recall"] > 0.9
    assert s["control_token_f1"] == 1.0
    assert s["control_precision"] == 1.0
    assert s["delimiter_order_acc"] == 1.0


def test_score_reproduction_echo_normalization():
    ref = {"system": "s", "user": "<ECHO_QUERY>__TARGET", "control_text": "",
           "control_ids": [], "all_control_ids": [], "order": ["system", "user"]}
    instr = "Reproduce everything now"
    out = "Reproduce everything now__TARGET"  # 模型回显了真实指令而非占位符
    s = score_reproduction(ref, out, [], instr)
    assert s["user_recall"] >= 0.9  # 归一化后算对
