import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from src.common import sha256_file

def run_dump(tokenizer_or_model: str, revision: str, out_dir: Path, meta: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_or_model, revision=revision
                                         if revision != "local" else None)
    ct = tok.chat_template or ""
    (out_dir / "chat_template.txt").write_text(ct, encoding="utf-8")
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(tok.special_tokens_map, ensure_ascii=False, indent=2), encoding="utf-8")
    added = {}
    for tid, t in tok.added_tokens_decoder.items():
        added[str(tid)] = str(t)
    (out_dir / "added_tokens_decoder.json").write_text(
        json.dumps(added, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "revisions.json").write_text(
        json.dumps({"model_key": meta["model_key"], "model_id": meta["model_id"],
                    "revision": revision}, ensure_ascii=False, indent=2), encoding="utf-8")
    import sys
    try:
        import torch, transformers
        versions = {"python": sys.version, "torch": torch.__version__,
                    "transformers": transformers.__version__}
    except Exception:
        versions = {}
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    files = {}
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            files[p.name] = sha256_file(p)
    return {"files": files, "template_sha256": files.get("chat_template.txt", "")}
