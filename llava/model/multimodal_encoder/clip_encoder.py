"""CLIP vision features, CLS attention, and text features for QPrune."""

import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizerFast,
    CLIPVisionConfig, CLIPVisionModel, CLIPVisionModelWithProjection,
)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(vision_tower)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def load_text_tower(self, device_map=None):
        CLIPVisionModelWithProjection._no_split_modules = ["CLIPEncoderLayer"]
        projected = CLIPVisionModelWithProjection.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.visual_projection = projected.visual_projection
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModelWithProjection.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.text_tower.requires_grad_(False)

    def encode_text_units(self, units):
        if not units:
            return None
        tokens = self.text_tokenizer(
            text=units, padding=True, truncation=True, return_tensors="pt"
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        return self.text_tower(**tokens).text_embeds

    def feature_select(self, output):
        features = output.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return features[:, 1:]
        if self.select_feature == "cls_patch":
            return features
        raise ValueError(f"Unexpected vision feature selection: {self.select_feature}")

    def _attention_layer(self, layer_spec):
        layers = self.vision_tower.vision_model.encoder.layers
        if layer_spec == "-2":
            return layers[-2]
        raise ValueError(f"Unsupported attention layer: {layer_spec}")

    @staticmethod
    def _reconstruct_cls_attention(layer, hidden_states):
        if hidden_states is None or hidden_states.ndim != 3 or hidden_states.shape[1] <= 1:
            return None
        hidden_states = layer.layer_norm1(hidden_states)
        attention = layer.self_attn
        batch_size, sequence_length, _ = hidden_states.shape
        heads, head_dim = attention.num_heads, attention.head_dim
        query = attention.q_proj(hidden_states[:, :1]) * attention.scale
        key = attention.k_proj(hidden_states)
        query = query.view(batch_size, 1, heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
        scores = query.float() @ key.float().transpose(-1, -2)
        return torch.softmax(scores, dim=-1)[:, :, 0, 1:]

    @torch.no_grad()
    def forward(self, images, texts=None, output_cls_attention=True, cls_attn_layer="-2"):
        if isinstance(images, list):
            images = torch.stack(images)
        pixel_values = images.to(device=self.device, dtype=self.dtype)
        layer = self._attention_layer(cls_attn_layer) if output_cls_attention else None
        captured = []
        hook = None
        if layer is not None:
            hook = layer.register_forward_pre_hook(
                lambda _module, args: captured.append(args[0]) if args else None
            )
        try:
            output = self.vision_tower(pixel_values, output_hidden_states=True)
        finally:
            if hook is not None:
                hook.remove()
        selected = self.feature_select(output)
        attention = self._reconstruct_cls_attention(layer, captured[0]) if captured else None
        projected = self.vision_tower.vision_model.post_layernorm(selected)
        image_embeds = self.vision_tower.visual_projection(projected.float())
        return selected.to(images.dtype), image_embeds, attention

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
