import torch
import torch.nn as nn

from transformers import (
    CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel, CLIPVisionModelWithProjection,
    CLIPTextConfig, CLIPTokenizerFast, CLIPTextModel, CLIPTextModelWithProjection,
)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # [CDPruner] Load text tower for CLIP model
    def load_text_tower(self, device_map=None):
        CLIPVisionModelWithProjection._no_split_modules = ['CLIPEncoderLayer']
        vision_tower_with_projection = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.visual_projection = vision_tower_with_projection.visual_projection

        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.text_tower.requires_grad_(False)

        self.max_position_embeddings = self.text_tower.config.max_position_embeddings

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def transition_prior_select(self, image_forward_outs, eps=1e-6):
        hidden_states = image_forward_outs.hidden_states
        curr_index = self.select_layer
        prev_index = curr_index - 1
        if hidden_states is None or abs(curr_index) >= len(hidden_states) or abs(prev_index) >= len(hidden_states):
            return None

        h_curr = hidden_states[curr_index]
        h_prev = hidden_states[prev_index]
        if self.select_feature == 'patch':
            h_curr = h_curr[:, 1:]
            h_prev = h_prev[:, 1:]
        elif self.select_feature != 'cls_patch':
            return None
        if h_curr.shape != h_prev.shape or h_curr.ndim != 3:
            return None

        h_curr = torch.nan_to_num(h_curr.float(), nan=0.0, posinf=0.0, neginf=0.0)
        h_prev = torch.nan_to_num(h_prev.float(), nan=0.0, posinf=0.0, neginf=0.0)
        safe_eps = max(float(eps), 1e-12)
        cosine = torch.nn.functional.cosine_similarity(h_prev, h_curr, dim=-1, eps=safe_eps).abs()
        direction = (1.0 - cosine).clamp_min(0.0)
        magnitude = h_curr.norm(dim=-1) / h_prev.norm(dim=-1).clamp_min(safe_eps)
        transition_raw = torch.softmax(direction, dim=-1) * magnitude.clamp_min(0.0)
        transition_raw = torch.nan_to_num(transition_raw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return transition_raw / transition_raw.sum(dim=-1, keepdim=True).clamp_min(safe_eps)

    def _resolve_cls_attention_layers(self, layer_spec):
        layers = self.vision_tower.vision_model.encoder.layers
        num_layers = len(layers)
        spec = str(layer_spec).strip().lower()
        if spec == "last":
            indices = [num_layers - 1]
        elif spec in {"-2", "-4"}:
            indices = [num_layers + int(spec)]
        elif spec == "last4mean":
            if num_layers < 4:
                return []
            indices = list(range(num_layers - 4, num_layers))
        else:
            return []
        if any(index < 0 or index >= num_layers for index in indices):
            return []
        return [(index, layers[index]) for index in indices]

    @staticmethod
    def _reconstruct_cls_attention(layer, hidden_states):
        if hidden_states is None or hidden_states.ndim != 3 or hidden_states.shape[1] <= 1:
            return None

        hidden_states = layer.layer_norm1(hidden_states)
        attention = layer.self_attn
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = attention.num_heads
        head_dim = attention.head_dim

        query = attention.q_proj(hidden_states[:, :1]) * attention.scale
        key = attention.k_proj(hidden_states)
        query = query.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        return torch.softmax(scores, dim=-1)[:, :, 0, 1:]

    def _forward_with_cls_attention(self, pixel_values, layer_spec="last"):
        """Run CLIP once and reconstruct only requested CLS-to-patch rows."""
        try:
            selected_layers = self._resolve_cls_attention_layers(layer_spec)
        except (AttributeError, IndexError, TypeError, ValueError):
            return self.vision_tower(pixel_values, output_hidden_states=True), None
        if not selected_layers:
            return self.vision_tower(pixel_values, output_hidden_states=True), None

        captured = {}
        handles = []
        for layer_index, layer in selected_layers:
            def capture_layer_input(module, args, index=layer_index):
                if args:
                    captured[index] = args[0]

            handles.append(layer.register_forward_pre_hook(capture_layer_input))
        try:
            outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        finally:
            for handle in handles:
                handle.remove()

        try:
            attentions = []
            for layer_index, layer in selected_layers:
                cls_attention = self._reconstruct_cls_attention(
                    layer,
                    captured.get(layer_index),
                )
                if cls_attention is None:
                    return outputs, None
                attentions.append(cls_attention)
            return outputs, torch.stack(attentions, dim=0).mean(dim=0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return outputs, None

    # [CDPruner] Get image and text embeds
    @torch.no_grad()
    def forward(self, images, texts=None, output_cls_attention=False, cls_attn_layer="last", output_transition_prior=False):
        if type(images) is list:
            image_features = []
            for image in images:
                pixel_values = image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                if output_cls_attention:
                    image_forward_out, _ = self._forward_with_cls_attention(
                        pixel_values,
                        layer_spec=cls_attn_layer,
                    )
                else:
                    image_forward_out = self.vision_tower(pixel_values, output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # [CDPruner] Get text embeds
            image_stream = torch.cuda.Stream()
            text_stream = torch.cuda.Stream()
            
            with torch.cuda.stream(image_stream):
                pixel_values = images.to(device=self.device, dtype=self.dtype)
                if output_cls_attention:
                    image_forward_outs, cls_attention = self._forward_with_cls_attention(
                        pixel_values,
                        layer_spec=cls_attn_layer,
                    )
                else:
                    image_forward_outs = self.vision_tower(pixel_values, output_hidden_states=True)
                    cls_attention = None
                image_outputs = self.feature_select(image_forward_outs)
                image_features = image_outputs.to(images.dtype)
                transition_prior = self.transition_prior_select(image_forward_outs) if output_transition_prior else None
            
            if texts is not None:
                with torch.cuda.stream(text_stream):
                    text_inputs = self.text_tokenizer(text=texts, return_tensors="pt")
                    text_segment = (text_inputs.input_ids.shape[1] - 1) // self.max_position_embeddings + 1
                    text_padding = self.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
                    text_inputs = {
                        k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
                                     dim=1).reshape(-1, self.max_position_embeddings).to(device=self.device)
                        for k, v in text_inputs.items()
                    }
                    text_embeds = self.text_tower(**text_inputs).text_embeds
            
            torch.cuda.synchronize()

            if texts is not None:
                image_embeds = self.vision_tower.vision_model.post_layernorm(image_outputs)
                image_embeds = self.vision_tower.visual_projection(image_embeds.float())
                if output_cls_attention and output_transition_prior:
                    image_features = (image_features, image_embeds, text_embeds, cls_attention, transition_prior)
                elif output_cls_attention:
                    image_features = (image_features, image_embeds, text_embeds, cls_attention)
                elif output_transition_prior:
                    image_features = (image_features, image_embeds, text_embeds, transition_prior)
                else:
                    image_features = (image_features, image_embeds, text_embeds)

        return image_features

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
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
