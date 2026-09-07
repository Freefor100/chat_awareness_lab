# src/common.py
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_dir(root: Path, tag: str) -> Path:
    d = root / tag
    (d / "template").mkdir(parents=True, exist_ok=True)
    (d / "cases" / "sidecars").mkdir(parents=True, exist_ok=True)
    return d

def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

def read_jsonl(path: Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def env_snapshot() -> dict:
    def ver(m):
        try:
            import importlib
            return getattr(importlib.import_module(m), "__version__", "?")
        except Exception as e:
            return f"unavailable: {e}"
    snap = {"python": __import__("sys").version,
            "torch": ver("torch"), "transformers": ver("transformers"),
            "accelerate": ver("accelerate"), "huggingface_hub": ver("huggingface_hub"),
            "torch_cuda_available": False, "gpu": None}
    try:
        import torch
        snap["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        snap["nvidia_smi"] = r.stdout.strip().splitlines() if r.returncode == 0 else []
    except Exception as e:
        snap["nvidia_smi"] = f"error: {e}"
    return snap

def _snapshot_ref(model_id: str) -> str | None:
    """离线 fallback：从 HF 快照缓存读主分支 commit（无网络时仍可固定 revision）。"""
    repo = model_id.replace("/", "--")
    ref = (Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}"
           / "refs" / "main")
    try:
        v = ref.read_text().strip()
        return v or None
    except Exception:
        return None

def hf_resolve_revision(model_id: str) -> str:
    from huggingface_hub import model_info
    try:
        return model_info(model_id).sha  # "0123abcd..." 主分支 commit
    except Exception:
        ref = _snapshot_ref(model_id)
        if ref:
            return ref
        raise

def hf_download(model_id: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo_id=model_id, filename=filename,
                                    revision=revision))
    except Exception:
        # 网络不可用 → 只查本地快照
        return Path(hf_hub_download(repo_id=model_id, filename=filename,
                                    revision=revision, local_files_only=True))

def template_sha256(model_id: str, revision: str) -> str:
    try:
        p = hf_download(model_id, "chat_template.jinja", revision)
    except Exception:
        p = hf_download(model_id, "tokenizer_config.json", revision)
    return sha256_file(p)

def model_cfg(models_yaml: Path, key: str) -> dict:
    cfg = load_yaml(models_yaml)
    for m in cfg["models"]:
        if m["key"] == key:
            return m
    raise KeyError(f"model key '{key}' not in {models_yaml}: {[m['key'] for m in cfg['models']]}")


def hub_reachable(host: str = "huggingface.co", timeout: float = 2.0) -> bool:
    """快速 TCP 探测 HF Hub 可达性（断网时避免 transformers/httpx 长超时）。"""
    import socket
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except Exception:
        return False
