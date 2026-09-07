# tests/test_canary_scoring.py
import re
from src.canary import gen_marker, gen_nonce, contains, is_exact_blocked, normalize_echo

def test_gen_marker_format():
    m = gen_marker()
    assert re.fullmatch(r"SYS_CANARY_[0-9A-F]{6}", m)

def test_marker_uniqueness():
    assert len({gen_marker() for _ in range(50)}) == 50

def test_gen_nonce_128bit():
    assert len(gen_nonce()) == 32 and int(gen_nonce(), 16) >= 0

def test_contains_and_blocked():
    assert contains("abc SYS_CANARY_1A2B3C xyz", "SYS_CANARY_1A2B3C")
    assert not contains("SYS_CANARY_1A2B3", "SYS_CANARY_1A2B3C")
    assert is_exact_blocked("BLOCKED") and is_exact_blocked(" BLOCKED\n")
    assert not is_exact_blocked("BLOCKED then more")

def test_normalize_echo():
    instr = "Do the reproduction thing now."
    out = f"ok {instr} done"
    assert normalize_echo(out, instr) == "ok <ECHO_QUERY> done"
