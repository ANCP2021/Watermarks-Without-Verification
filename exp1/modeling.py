
from __future__ import annotations

import torch

from .common import hf_token, log
from .settings import CHAT_TEMPLATE_DATE, SAMPLING

try: 
    from transformers import (LogitsProcessor, LogitsProcessorList,
                              TemperatureLogitsWarper, TopKLogitsWarper,
                              TopPLogitsWarper)
except ImportError:  # pragma: no cover
    from transformers.generation.logits_process import (  # type: ignore
        LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper,
        TopKLogitsWarper, TopPLogitsWarper)


class Float32LogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores.float()


def default_attn_impl(model_id: str) -> str:

    return "eager" if "gemma-2" in model_id.lower() else "sdpa"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[requested]


def load_tokenizer(model_id: str, token: str | None = None,
                   revision: str | None = None):
  
    from transformers import AutoTokenizer

    token = token or hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_id: str, device: str, dtype: torch.dtype,
                   attn_impl: str | None = None, token: str | None = None,
                   revision: str | None = None):
    from transformers import AutoModelForCausalLM

    attn_impl = attn_impl or default_attn_impl(model_id)
    token = token or hf_token()
    log(f"loading {model_id} (device={device}, dtype={dtype}, attn={attn_impl})")

    kwargs = {"torch_dtype": dtype, "token": token, "revision": revision}
    try:
        import accelerate 
        kwargs["low_cpu_mem_usage"] = True
    except ImportError:
        pass

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_impl, **kwargs)
    except (ValueError, TypeError) as exc:
        log(f"note: attn_implementation={attn_impl} rejected ({exc}); falling back to default")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        attn_impl = "default"

    model.to(device)
    model.eval()
    return model, attn_impl


def load_model_and_tokenizer(model_id: str, device: str, dtype: torch.dtype,
                             attn_impl: str | None = None, token: str | None = None,
                             revision: str | None = None):
    tokenizer = load_tokenizer(model_id, token, revision)
    model, attn_impl = load_causal_lm(model_id, device, dtype, attn_impl, token, revision)
    return model, tokenizer, attn_impl


def render_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            date_string=CHAT_TEMPLATE_DATE,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def encode_rendered(tokenizer, rendered: str, device: str):
   
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)


def build_processor_chain(synthid_processor=None) -> LogitsProcessorList:
    chain = [
        Float32LogitsProcessor(),
        TemperatureLogitsWarper(SAMPLING["temperature"]),
        TopKLogitsWarper(top_k=SAMPLING["top_k"], min_tokens_to_keep=1),
        TopPLogitsWarper(top_p=SAMPLING["top_p"], min_tokens_to_keep=1),
    ]
    if synthid_processor is not None:
        chain.append(synthid_processor)
    return LogitsProcessorList(chain)


def generation_kwargs(tokenizer, max_new_tokens: int) -> dict:
    return {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "min_length": 0,
        "repetition_penalty": SAMPLING["repetition_penalty"],
        "no_repeat_ngram_size": SAMPLING["no_repeat_ngram_size"],
        "pad_token_id": tokenizer.pad_token_id,
    }
