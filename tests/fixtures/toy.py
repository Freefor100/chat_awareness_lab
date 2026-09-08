"""内存 toy tokenizer + 模板 fixture，单元测试不下载任何模型权重。"""
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, trainers
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

TOY_VOCAB_SIZE = 600

TOY_STYLES = {
    # Qwen3/3.5 ChatML 风格：<|im_start|>/<|im_end|>
    "chatml": (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    ),
    # Llama-2 风格：[INST] ...
    "llama_inst": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST] ' }}"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] + ' ' }}"
        "{% else %}{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% endif %}"
        "{% endfor %}"
    ),
    # 无 system 槽位、user-first 风格（system 消息直接拼 user 前缀）
    "no_system_user_first": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ '### SYSTEM:\\n' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'user' %}{{ '### USER:\\n' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ '### ASSISTANT:\\n' + message['content'] + '\\n' }}"
        "{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}{{ '### ASSISTANT:\\n' }}{% endif %}"
    ),
}

# 与 Qwen3 同风格的 ChatML 模板常量（toy 同款），供需要独立模板引用处使用
chatml_template = TOY_STYLES["chatml"]


def _byte_alphabet() -> list[str]:
    """ByteLevel(GPT2 风格)全字节表：BPE 只把语料里出现过的字符收进词表，
    而测试串含换行/大写/生僻字节，必须把全部 256 个字节映射字符都种进初始字母表，
    decode 才能逐字节无损还原。空格(0x20)在 pre_tokenize 输出里不单独成词，需特判。"""
    pt = ByteLevel(add_prefix_space=False)
    alpha = []
    for b in range(256):
        ch = chr(b)
        if ch == " ":
            alpha.append("\u0120")  # ByteLevel 用 U+0120(Ġ) 表示空格
        else:
            alpha.append(pt.pre_tokenize_str(ch)[0][0])
    return alpha


def _train_bpe() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    # add_prefix_space=False：前缀空格会在 decode 时残留为串首空格，破坏 round-trip 等式
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=TOY_VOCAB_SIZE, special_tokens=[], initial_alphabet=_byte_alphabet())
    tok.train_from_iterator([_corpus()], trainer=trainer)
    # ByteLevel pre_tokenizer 必须配 ByteLevel decoder，否则 decode 返回原始字节片
    tok.decoder = decoders.ByteLevel()
    return tok


def _corpus() -> list[str]:
    parts = []
    for i in range(120):
        parts.append(" ".join(f"tok{i}_{j}" for j in range(60)))
    parts.append("Copy the JSON below character-for-character exactly.")
    parts.append("protected marker SYS_CANARY_A1B2C3 BLOCKED nonce secret system user content")
    return parts


def _add_special_tokens(tok: Tokenizer) -> None:
    specials = ["<|im_start|>", "<|im_end|>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
                "### SYSTEM:", "### USER:", "### ASSISTANT:"]
    tok.add_special_tokens(specials)


def make_toy_tokenizer(dir: Path, chat_template: str, with_special_tokens: bool = True) -> PreTrainedTokenizerFast:
    """构造并落盘到 dir，返回可直接 from_pretrained 的 fast tokenizer。"""
    base = _train_bpe()
    if with_special_tokens:
        _add_special_tokens(base)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=base,
        bos_token="<|im_start|>", eos_token="<|im_end|>",
        unk_token="<|unk|>", pad_token="<|im_end|>",
    )
    fast.chat_template = chat_template
    fast.save_pretrained(str(dir))
    # 用落盘版本重载，保证与 AutoTokenizer.from_pretrained 行为一致
    return PreTrainedTokenizerFast.from_pretrained(str(dir))
