# src/surfaces.py — probe-render control-surface extraction (Task 6)
#
# 通过向真实 chat_template 渲染带随机 marker 的探针消息，从渲染结果里剥出
# 模板控制的文本片段（surface），供动态攻击构造器使用——不硬编码任何模板串。
import secrets
from typing import Callable

_SURFACE_KEYS = ("sys_open", "sys_close", "usr_open", "usr_close", "asst_open")


def _marker(tag: str) -> str:
    return f"PROBE_{tag}_{secrets.token_hex(6).upper()}"


def _cut(text: str, marker: str, side: str) -> str | None:
    if marker not in text:
        return None
    if side == "before":
        return text.split(marker, 1)[0]
    return text.split(marker, 1)[1]


def _strip_known(gap: str, known_open: str | None, side: str) -> tuple[str, str | None, bool]:
    """把 gap 里已知的 open 剥出来。返回 (剩余, 剥出的部分, 是否成功剥出)。"""
    if not known_open or known_open not in gap:
        return gap, None, False
    if side == "suffix":  # 已知串在 gap 尾部 → 剩下的是 close
        i = gap.rfind(known_open)
        return gap[:i], gap[i:], True
    # prefix: 已知串在 gap 头部 → 剥掉它，剩下的是 close
    i = gap.find(known_open)
    return gap[i + len(known_open):], known_open, True


def extract_surfaces(render: Callable[[list[dict]], str]) -> dict:
    """探针渲染提取控制 surface。

    探针全部以 user 消息收尾——部分官方模板（如 Qwen3.5）要求最后一条是
    user，assistant-only / system-only 探针会抛 TemplateError。
    asst_open 由 generation-prompt 前后渲染的差值得出：
    tail_gen = 最后 user content 之后(含 generation prompt)的文本，
    tail_nogen = 同样位置(不含 generation prompt)的文本，
    asst_open = tail_gen 去掉 tail_nogen 前缀后的余量。
    """
    ms, mu = _marker("S"), _marker("U")
    r_usr = render([{"role": "user", "content": mu}])
    r_usr_gen = render([{"role": "user", "content": mu}], add_gen=True)
    r_su = render([{"role": "system", "content": ms}, {"role": "user", "content": mu}])

    usr_open = _cut(r_usr, mu, "before")
    sys_open = _cut(r_su, ms, "before")
    gap1 = _cut(r_su, ms, "after")
    if gap1 is not None:
        gap1 = _cut(gap1, mu, "before")
    tail_nogen = _cut(r_usr, mu, "after")
    tail_gen = _cut(r_usr_gen, mu, "after")

    # usr_close = 无 generation prompt 时最后一个 user content 之后的文本
    usr_close = tail_nogen
    # asst_open = 加 generation prompt 后多出来的部分
    asst_open = None
    if tail_nogen is not None and tail_gen is not None:
        if tail_gen.startswith(tail_nogen):
            asst_open = tail_gen[len(tail_nogen):]
        elif tail_gen == "":
            asst_open = ""
        else:
            asst_open = None
    # sys_close = gap1 去掉尾部 usr_open（sys_close 与 usr_open 相邻）
    sys_close, _, _ = _strip_known(gap1 or "", usr_open, "suffix")

    skipped = []
    if sys_open is None: skipped.append("sys_open")
    if usr_open is None: skipped.append("usr_open")
    if asst_open is None: skipped.append("asst_open")
    if gap1 is None: skipped.append("gap1")
    if usr_close is None: skipped.append("usr_close")
    if tail_gen is None: skipped.append("tail")

    return {
        "sys_open": sys_open, "sys_close": sys_close or None,
        "usr_open": usr_open, "usr_close": usr_close or None,
        "asst_open": asst_open, "head": sys_open,
        "tail": tail_gen,
        "render_log": {"user": r_usr, "user_gen": r_usr_gen,
                       "sys_user": r_su},
        "skipped": skipped,
    }


def boundary_token_ids(surfaces: dict, enc: Callable[[str], list[int]],
                       special_ids: set[int]) -> dict:
    out = {}
    for k in _SURFACE_KEYS:
        v = surfaces.get(k)
        if v is None:
            out[k] = {"token_ids": [], "is_special": False}
            continue
        ids = enc(v)
        out[k] = {"token_ids": ids,
                  "is_special": any(i in special_ids for i in ids)}
    return out
