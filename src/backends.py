# src/backends.py — model/tokenizer load chain + Backend abstraction (Task 10)
#
# 加载链 (load_backend)：AutoTokenizer 必须先成功；tokenizer_only=True 到此为止
# (loaded=True, model=None)。需要权重时按序尝试（每档隔离：异常 → 记录 + 下一档）：
#   1. causal_bf16        AutoModelForCausalLM,  device_map="auto", bf16
#   2. multimodal_bf16    AutoModelForMultimodalLM, 同上（Qwen3.5 是 image-text-to-text）
#   3. causal_int4_bnb    AutoModelForCausalLM + BitsAndBytesConfig 4bit
#   4. multimodal_int4_bnb AutoModelForMultimodalLM + BitsAndBytesConfig 4bit
#   5. causal_int4_torchao AutoModelForCausalLM bf16 后 torchao int4（quantize_）
#   6. causal_bf16_cpu    AutoModelForCausalLM, device_map="cpu", bf16
#   7. causal_remote      trust_remote_code=True 重试 causal（config.auto_map 情形）
# 全失败 → 保留已加载 tokenizer：loaded=(tokenizer is not None)、failure_reason=
# "weights unsupported: ..."。transformers tokenizer 失败时先走 mistral-common 兜底。
# 每次路径都 append 到 load_attempts；成功时写 model_class/quant/device。
#
# 模块顶部只放 re/pathlib（无 transformers/torch import —— 单元测试免 GPU、
# 导入零副作用）；transformers/torch/hub/mistral-common 全部在方法内局部 import。
# 例外：AutoTokenizer 是模块级惰性钩子（见其注释），供 tests monkeypatch 用。
import re
from pathlib import Path

# 模块级惰性钩子：值为 None 表示尚未解析，Backend._load 首次使用时才局部导入
# transformers 的 AutoTokenizer（避免模块导入期依赖 transformers/torch）。
# tests/test_backends.py monkeypatch 这个名字以强制使用真实实现。
AutoTokenizer = None


def _is_hf_id(s: str) -> bool:
    """形如 owner/name 且不是本地路径 → 按 HF hub id 处理（附 revision）。"""
    return bool(re.match(r"^[\w.\-]+/[\w.\-]+$", s)) and not Path(s).exists()


class Backend:
    def __init__(self, cfg: dict, tokenizer_only: bool):
        self.model_id = cfg["id"]
        self.model_key = cfg["key"]
        self.revision = cfg.get("revision", "main")
        self.tokenizer_only = tokenizer_only
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.backend_name = None
        self.quant = None
        self.device = None
        self.model_class = None
        self.loaded = False
        self.failure_reason = None
        self.load_attempts = []
        self._load(cfg)

    # ---- tokenizer-level API ----
    def chat_template(self) -> str:
        src = self.processor or self.tokenizer
        return (src.chat_template or "")

    def apply_template(self, messages, add_generation_prompt=True, tokenize=False):
        src = self.processor or self.tokenizer
        if tokenize:
            out = src.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=add_generation_prompt,
                                          return_dict=True)
            return out
        return src.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=add_generation_prompt)

    def ids_from_messages(self, messages) -> dict:
        src = self.processor or self.tokenizer
        try:
            out = src.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True,
                                          return_dict=True, enable_thinking=False)
        except TypeError:
            # 模板不接受 enable_thinking（多数家族 / mistral 适配器）。
            # transformers 模板把未知 kwarg 传进 jinja context（不抛），此分支
            # 只对签名更严格的适配器 tokenizer 触发。
            out = src.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True,
                                          return_dict=True)
        return {"input_ids": list(out["input_ids"]),
                "attention_mask": list(out["attention_mask"])}

    def encode(self, text: str, add_special_tokens=False):
        return self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, skip_special_tokens=False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def special_ids(self) -> set:
        return {t for t in self.tokenizer.all_special_ids}

    def get_added_vocab(self) -> dict:
        return self.tokenizer.get_added_vocab()

    def release(self):
        import gc
        self.model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- generation API（权重不存在时抛 RuntimeError）----
    def _require_model(self):
        if self.model is None:
            raise RuntimeError(f"no weights for {self.model_id} ({self.failure_reason})")

    def generate_standard(self, input_ids, max_new_tokens, temperature, top_p, seed,
                          do_sample):
        self._require_model()
        import torch
        from transformers import GenerationConfig
        gc = GenerationConfig.from_model_config(self.model.config)
        gc.temperature = temperature
        gc.top_p = top_p
        gc.do_sample = do_sample
        gc.max_new_tokens = max_new_tokens
        if do_sample:
            torch.manual_seed(seed)
        inp = {"input_ids": torch.tensor([input_ids], device=self.model.device)}
        attn = torch.ones_like(inp["input_ids"])
        out = self.model.generate(**inp, attention_mask=attn, generation_config=gc)
        return {"output_ids": out[0].tolist(), "meta": {"mode": "standard",
                                                        "seed": seed, "quant": self.quant,
                                                        "device": self.device,
                                                        "backend": self.backend_name}}

    def forward_next_logits(self, input_ids):
        self._require_model()
        import torch
        inp = torch.tensor([input_ids], device=self.model.device)
        with torch.no_grad():
            return self.model(input_ids=inp).logits[0, -1]  # (vocab,)

    # ---- 加载主链：tokenizer(transformers → mistral-common 兜底) → 权重档位 ----
    def _load(self, cfg: dict):
        import os
        from src.common import hub_reachable
        # 断网时走本地快照：避免 transformers 对已缓存文件做网络 etag 检查而长超时
        if not hub_reachable():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        tok_cls = AutoTokenizer  # 模块级钩子（测试可 monkeypatch src.backends.AutoTokenizer）
        if tok_cls is None:
            from transformers import AutoTokenizer as _AT  # 惰性局部 import
            tok_cls = _AT
        # (a) tokenizer：transformers 优先（本地目录不带 revision）
        try:
            kw = {} if not _is_hf_id(self.model_id) else {"revision": self.revision}
            self.tokenizer = tok_cls.from_pretrained(self.model_id, **kw)
            self.backend_name = "transformers"
            self.loaded = True
        except Exception as e:
            self.load_attempts.append(f"tokenizer: {e!r}")
            # (b) mistral-common 兜底：下载 tokenizer 文件到临时目录后 from_file。
            # repo 不存在/gated 时第一个下载即 404 → 短路（后续文件与适配器必失败）。
            try:
                import tempfile
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
                from huggingface_hub import hf_hub_download
                try:
                    from huggingface_hub.errors import (GatedRepoError,
                                                        RepositoryNotFoundError)
                except ImportError:
                    # 老版本 hub：无 typed 异常 → 退化为逐文件记录、不短路
                    GatedRepoError = ()
                    RepositoryNotFoundError = ()
                tmp = tempfile.mkdtemp(prefix="mistral_tok_")
                repo_gone = False
                for fn in ("tokenizer.json", "tekken.json", "processor_config.json",
                           "tokenizer_config.json"):
                    if repo_gone:
                        break
                    try:
                        hf_hub_download(self.model_id, fn, revision=self.revision,
                                        local_dir=tmp)
                    except RepositoryNotFoundError:
                        repo_gone = True  # repo 不存在：mistral 文件也不可能存在
                    except GatedRepoError:
                        repo_gone = True  # gated：匿名下载 tokenizer 文件不可行
                    except Exception as fe:
                        self.load_attempts.append(f"mistral-dl {fn}: {fe!r}")
                if repo_gone:
                    raise RuntimeError("hub repo not found/gated: mistral-common "
                                       "tokenizer files unavailable")
                self.tokenizer = _MistralAdapter(MistralTokenizer.from_file(tmp))
                self.backend_name = "mistral-common"
                self.loaded = True
            except Exception as e2:
                self.load_attempts.append(f"mistral-common: {e2!r}")
                self.loaded = False
                self.failure_reason = "tokenizer: " + "; ".join(self.load_attempts)
                return
        if self.tokenizer_only:
            return
        # (c) 多模态仓库：AutoProcessor（chat template 可能在 processor 而非 tokenizer）
        if cfg.get("is_multimodal"):
            try:
                from transformers import AutoProcessor
                kw = {} if not _is_hf_id(self.model_id) else {"revision": self.revision}
                self.processor = AutoProcessor.from_pretrained(self.model_id, **kw)
            except Exception as e:
                self.load_attempts.append(f"processor: {e!r}")
        # (d) 权重档位链（闭包零参，见 _weight_attempts）：逐档隔离，异常记录后继续
        for t in self._weight_attempts():
            try:
                self.model = t.load()
                self.model_class = t.name
                self.quant = t.quant
                self.device = str(self.model.device)
                self.loaded = True
                self.load_attempts.append(f"ok:{t.name}:{self.quant}")
                return
            except Exception as e:
                self.load_attempts.append(f"{t.name}: {e!r}")
        self.failure_reason = "weights unsupported: " + "; ".join(self.load_attempts)
        # mistral-common 适配器只有 tokenizer：保留 loaded=True 但 model=None
        self.loaded = self.tokenizer is not None

    # ---- 权重尝试档位（每档 = 命名闭包 + quant 标签，异常由 _load 的 (d) 循环隔离）----
    def _weight_attempts(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM
        acts = []

        # 1. causal bf16，device_map="auto"（优先档：GPU 显存足够即命中）
        def causal():
            return AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, device_map="auto",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        acts.append(("causal_bf16", causal, None))

        # 2. multimodal（Qwen3.5 注册为 image-text-to-text，causal 类不认识其架构）
        def mm():
            return AutoModelForMultimodalLM.from_pretrained(
                self.model_id, revision=self.revision, device_map="auto",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        acts.append(("multimodal_bf16", mm, None))

        # 3. bitsandbytes 4bit（transformers>=5 只接受 BitsAndBytesConfig 对象，
        #    quantization_config={...} 旧 dict 写法已移除）
        def bnb4():
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            return AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, device_map="auto",
                quantization_config=qc, low_cpu_mem_usage=True)
        acts.append(("causal_int4_bnb", bnb4, "int4_bnb"))

        # 4. multimodal + bitsandbytes 4bit
        def mm_bnb4():
            from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            return AutoModelForMultimodalLM.from_pretrained(
                self.model_id, revision=self.revision, device_map="auto",
                quantization_config=qc, low_cpu_mem_usage=True)
        acts.append(("multimodal_int4_bnb", mm_bnb4, "int4_bnb"))

        # 5. torchao int4：bf16 载入后 quantize_。torchao API 版本探针：
        #    ≤0.9 有 quant_api.int4_weight_only()/weight_only(4)；
        #    >=0.18（实测 0.18.0）改为 config 对象 Int4WeightOnlyConfig()。
        #    探针全部失败 → 异常向上抛，由 (d) 记录该档失败。
        def torchao4():
            import torchao
            m = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, device_map="auto",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            from torchao.quantization import quantize_
            try:
                q = torchao.quantization.quant_api.int4_weight_only()
            except Exception:
                try:
                    q = torchao.quantization.quant_api.weight_only(4)
                except Exception:
                    from torchao.quantization.quant_api import Int4WeightOnlyConfig
                    q = Int4WeightOnlyConfig()
            quantize_(m, q)
            return m
        acts.append(("causal_int4_torchao", torchao4, "int4_torchao"))

        # 6. causal bf16 CPU（GPU 全失败后的兜底推理档）
        def cpu():
            return AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, device_map="cpu",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        acts.append(("causal_bf16_cpu", cpu, None))

        # 7. trust_remote_code 重试 causal（config.auto_map 的远程代码模型）
        def remote():
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True,
                device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        acts.append(("causal_remote", remote, None))

        class T:
            def __init__(self, n, f, q):
                self.name = n
                self.load = f
                self.quant = q

        return [T(n, f, q) for n, f, q in acts]


class _MistralAdapter:
    """把 MistralTokenizer 包成 Backend 需要的 tokenizer 形状（transformers 风格子集）。
    mistral-common 的 apply_chat_template/encode 返回结构随版本略异，
    适配器用探针兼容两种约定（.tokens 属性 vs tuple），探明结果记入 _probe。"""

    def __init__(self, mc):
        self._mc = mc
        self.chat_template = None
        self.all_special_ids = []
        # 探针：确认返回形状并记录（Task 19 对 Ministral 跑 E0 时核对）
        self._probe = {}
        try:
            r = self.apply_chat_template([{"role": "user", "content": "hi"}],
                                         tokenize=True)
            self._probe["apply_tokenize_shape"] = type(r).__name__
        except Exception as e:
            self._probe["apply_tokenize_error"] = repr(e)

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=True, return_dict=False):
        r = self._mc.apply_chat_template(messages, tokenize=tokenize,
                                         add_generation_prompt=add_generation_prompt)
        if tokenize:
            ids = getattr(r, "tokens", r[0] if isinstance(r, tuple) else r)
            if return_dict:
                return {"input_ids": list(ids),
                        "attention_mask": [1] * len(ids)}
            return list(ids)
        return r if isinstance(r, str) else r[1] if isinstance(r, tuple) else str(r)

    def encode(self, text, add_special_tokens=False):
        r = self._mc.encode(text)
        ids = getattr(r, "tokens", r[0] if isinstance(r, tuple) else r)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids), \
            f"unexpected mistral encode shape: {type(r)}"
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return self._mc.decode(list(ids))

    def __call__(self, text, add_special_tokens=False):
        return self.encode(text, add_special_tokens=add_special_tokens)


def load_backend(cfg: dict, tokenizer_only: bool = False) -> Backend:
    b = Backend(cfg, tokenizer_only)
    return b
