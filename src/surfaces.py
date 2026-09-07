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
    ms, mu, ma = _marker("S"), _marker("U"), _marker("A")
    r_sys = render([{"role": "system", "content": ms}])
    r_usr = render([{"role": "user", "content": mu}])
    r_asst = render([{"role": "assistant", "content": ma}])
    r_su = render([{"role": "system", "content": ms}, {"role": "user", "content": mu}])
    r_ua = render([{"role": "user", "content": mu}, {"role": "assistant", "content": ma}])
    r_su_gen = render([{"role": "system", "content": ms}, {"role": "user", "content": mu}],
                      add_gen=True)

    sys_open = _cut(r_sys, ms, "before")
    usr_open = _cut(r_usr, mu, "before")
    asst_open = _cut(r_asst, ma, "before")

    gap1 = _cut(r_su, ms, "after")
    if gap1 is not None:
        gap1 = _cut(gap1, mu, "before")
    gap2 = _cut(r_ua, mu, "after")
    if gap2 is not None:
        gap2 = _cut(gap2, ma, "before")
    tail = _cut(r_su_gen, mu, "after")

    sys_close, _, _ = _strip_known(gap1 or "", usr_open, "suffix")
    usr_close, _, _ = _strip_known(gap2 or "", asst_open, "suffix")

    skipped = []
    if sys_open is None: skipped.append("sys_open")
    if usr_open is None: skipped.append("usr_open")
    if asst_open is None: skipped.append("asst_open")
    if gap1 is None: skipped.append("gap1")
    if gap2 is None: skipped.append("gap2")
    if tail is None: skipped.append("tail")

    return {
        "sys_open": sys_open, "sys_close": sys_close or None,
        "usr_open": usr_open, "usr_close": usr_close or None,
        "asst_open": asst_open, "head": _cut(r_sys, ms, "before"),
        "tail": tail,
        "render_log": {"sys": r_sys, "user": r_usr, "assistant": r_asst,
                       "sys_user": r_su, "user_assistant": r_ua, "sys_user_gen": r_su_gen},
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
