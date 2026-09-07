import json
from pathlib import Path
from src.dump_template import run_dump
from src.common import make_run_dir, sha256_file
from tests.fixtures.toy import make_toy_tokenizer, TOY_STYLES

def test_dump_toy(tmp_path):
    tok_dir = tmp_path / "tok"
    make_toy_tokenizer(tok_dir, TOY_STYLES["chatml"])
    run_dir = make_run_dir(tmp_path, "tag1")
    res = run_dump(str(tok_dir), "local", run_dir / "template",
                   {"model_key": "toy", "model_id": "toy/local"})
    td = run_dir / "template"
    assert (td / "chat_template.txt").exists()
    assert "im_start" in (td / "chat_template.txt").read_text()
    assert res["template_sha256"] == sha256_file(td / "chat_template.txt")
    rev = json.loads((td / "revisions.json").read_text())
    assert rev["model_key"] == "toy"
    # special tokens map 记录了 <|im_start|>
    stm = json.loads((td / "special_tokens_map.json").read_text())
    assert "<|im_start|>" in stm["bos_token"]
