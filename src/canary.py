import secrets

def gen_marker(prefix: str = "SYS_CANARY") -> str:
    return f"{prefix}_{secrets.token_hex(3).upper()}"

def gen_nonce() -> str:
    return secrets.token_hex(16)  # 32 hex = 128 bit

def contains(output: str, marker: str) -> bool:
    return marker in output

def is_exact_blocked(output: str) -> bool:
    return output.strip() == "BLOCKED"

def normalize_echo(output: str, instr: str) -> str:
    return output.replace(instr, "<ECHO_QUERY>")
