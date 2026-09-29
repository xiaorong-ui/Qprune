"""Load a standard LLaVA Llama checkpoint for QPrune inference."""

import os

import torch
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def load_pretrained_model(
    model_path, model_base, model_name, load_8bit=False, load_4bit=False,
    device_map="auto", device="cuda", use_flash_attn=False, visual_token_num=64,
    **kwargs,
):
    if model_base is not None:
        raise ValueError("Provide a merged LLaVA checkpoint with model_base=None")
    if load_8bit or load_4bit:
        raise ValueError("QPrune release expects an unquantized LLaVA checkpoint")
    if not 1 <= int(visual_token_num) <= 576:
        raise ValueError("visual_token_num must be between 1 and 576 per image view")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model_kwargs = dict(kwargs)
    model_kwargs.update({
        "low_cpu_mem_usage": os.environ.get("LLAVA_LOW_CPU_MEM_USAGE", "1") != "0",
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "visual_token_num": int(visual_token_num),
        "device_map": device_map if device == "cuda" else {"": device},
    })
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, **model_kwargs)

    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map if device == "cuda" else None)
    vision_tower.load_text_tower(device_map=device_map if device == "cuda" else None)
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=model.dtype)
    image_processor = vision_tower.image_processor
    context_length = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_length
