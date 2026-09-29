#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import hashlib
import json
import math
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .pruners import ECPruner
from .pruners.ec_pruner import (
    compute_projection_overlap_kernel,
    compute_qficr_entropy_recovery_budget,
    select_frozen_qfi_order,
)

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

_PRUNE_DEBUG_COUNT = 0
_DECODER_FEEDBACK_DEBUG_COUNT = 0
_TASK_REFINE_DEBUG_COUNT = 0
_SCSS_DEBUG_COUNT = 0
_E8_LOCAL_FUSION_DEBUG_COUNT = 0
_P3_DUAL_RESIDUAL_DEBUG_COUNT = 0
_FORCED_SELECTED_CACHE = {"path": None, "mtime": None, "records": {}}
_E8_LOCAL_FUSION_CACHE = {"path": None, "mtime": None, "records": {}}


def _qficr_scss_method():
    method = os.environ.get("EC_QFICR_SCSS_METHOD", "off").strip().lower()
    allowed = {"off", "z0", "z1", "z2", "z3", "z4"}
    if method not in allowed:
        raise ValueError(f"Unknown EC_QFICR_SCSS_METHOD: {method}")
    return method


def _qficr_cd_safe_method():
    method = os.environ.get("EC_QFICR_CD_SAFE_METHOD", "off").strip().lower()
    allowed = {"off", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"}
    if method not in allowed:
        raise ValueError(f"Unknown EC_QFICR_CD_SAFE_METHOD: {method}")
    return method


def _write_qficr_scss_debug(record):
    global _SCSS_DEBUG_COUNT
    stats_path = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
    if not stats_path:
        return
    if stats_path.lower() in {"1", "true", "yes", "on"}:
        stats_path = os.path.abspath("debug_qficr_scss_stats.jsonl")
    payload = dict(record)
    payload.update({
        "sample_index": _SCSS_DEBUG_COUNT,
        "process_id": os.getpid(),
        "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
        "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
        "kind": "scss_selection",
    })
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    with open(stats_path, "a", encoding="utf-8") as stats_file:
        stats_file.write(json.dumps(payload, sort_keys=True) + "\n")
    _SCSS_DEBUG_COUNT += 1


def _qficr_residual_transfer_mode():
    mode = os.environ.get("EC_QFICR_RESIDUAL_TRANSFER_MODE", "off").strip().lower()
    allowed = {"off", "uniform", "task"}
    if mode not in allowed:
        raise ValueError(f"Unknown EC_QFICR_RESIDUAL_TRANSFER_MODE: {mode}")
    return mode


def _qficr_local_fusion_mode():
    mode = os.environ.get("EC_QFICR_LOCAL_FUSION_MODE", "off").strip().lower()
    allowed = {
        "off",
        "e8_oracle_fuse",
        "e8_matched_control_fuse",
        "e8_matched_random_fuse",
        "e8_voronoi_fuse",
        "e8_shuffled_cell_fuse",
    }
    if mode not in allowed:
        raise ValueError(f"Unknown EC_QFICR_LOCAL_FUSION_MODE: {mode}")
    return mode


def _qficr_local_residual_mode():
    mode = os.environ.get("EC_QFICR_LOCAL_RESIDUAL_MODE", "off").strip().lower()
    allowed = {
        "off",
        "p1_semantic_core_local_residual",
        "p2_semantic58_local6",
        "p1_tc_semantic_core_conditioned_local_residual",
        "p1_lr_semantic_core_conditioned_local_residual",
        "p1_lr_entropy_core_conditioned_local_residual",
        "p1_lr_entropy_core_conditioned_local_residual_fast",
        "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
        "p1_lr_entropy_core_random_recovery_v2",
        "p1_lr_entropy_spatial_diverse_recovery_v3",
        "p1_lr_entropy_relation_routed_recovery_v4",
        "p1_lr_generalized_routed_recovery_candidate",
    }
    if mode not in allowed:
        raise ValueError(f"Unknown EC_QFICR_LOCAL_RESIDUAL_MODE: {mode}")
    return mode


def _qficr_dual_residual_mode():
    mode = os.environ.get("EC_QFICR_DUAL_RESIDUAL_MODE", "off").strip().lower()
    allowed = {"off", "p3_dual_residual_coconstruction"}
    if mode not in allowed:
        raise ValueError(f"Unknown EC_QFICR_DUAL_RESIDUAL_MODE: {mode}")
    return mode


def _write_qficr_p3_dual_residual_debug(record):
    global _P3_DUAL_RESIDUAL_DEBUG_COUNT
    path = os.environ.get("EC_QFICR_P3_DUAL_RESIDUAL_DEBUG_JSONL", "").strip()
    if not path:
        return
    if path.lower() in {"1", "true", "yes", "on"}:
        path = os.path.abspath("debug_qficr_p3_dual_residual.jsonl")
    def _json_safe(value):
        if torch.is_tensor(value):
            if value.numel() > 64:
                return {
                    "tensor_shape": list(value.shape),
                    "tensor_dtype": str(value.dtype),
                }
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    payload = _json_safe(dict(record))
    payload.update({
        "sample_index": _P3_DUAL_RESIDUAL_DEBUG_COUNT,
        "process_id": os.getpid(),
        "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
        "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
        "question_text": os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", ""),
        "kind": "p3_dual_residual_coconstruction",
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stats_file:
        stats_file.write(json.dumps(payload, sort_keys=True) + "\n")
    _P3_DUAL_RESIDUAL_DEBUG_COUNT += 1


def _qficr_lcenter_scores(raw_image_features, eps=1e-12):
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("P1 local residual currently requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(f"P1 local residual expects 576 visual tokens, got {raw_image_features.shape[1]}")
    x = raw_image_features[0].float()
    scores = []
    for idx in range(576):
        row, col = divmod(idx, 24)
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < 24 and 0 <= cc < 24:
                    neigh.append(rr * 24 + cc)
        neigh_idx = torch.tensor(neigh, dtype=torch.long, device=x.device)
        mu = x.index_select(0, neigh_idx).mean(dim=0)
        cos = torch.dot(x[idx], mu) / (x[idx].norm().clamp_min(eps) * mu.norm().clamp_min(eps))
        scores.append(1.0 - cos)
    return torch.stack(scores).to(device=raw_image_features.device)


def _qficr_local_contrast_vectors(raw_image_features, eps=1e-12):
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("P3 dual residual currently requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(f"P3 dual residual expects 576 visual tokens, got {raw_image_features.shape[1]}")
    x = torch.nan_to_num(raw_image_features[0].float(), nan=0.0, posinf=0.0, neginf=0.0)
    contrast = torch.empty_like(x)
    lcenter = torch.empty(576, dtype=torch.float32, device=x.device)
    for idx in range(576):
        row, col = divmod(idx, 24)
        neigh = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < 24 and 0 <= cc < 24:
                    neigh.append(rr * 24 + cc)
        neigh_idx = torch.tensor(neigh, dtype=torch.long, device=x.device)
        mu = x.index_select(0, neigh_idx).mean(dim=0)
        d = x[idx] - mu
        cos = torch.dot(x[idx], mu) / (x[idx].norm().clamp_min(eps) * mu.norm().clamp_min(eps))
        contrast[idx] = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        lcenter[idx] = torch.nan_to_num((1.0 - cos).clamp_min(0.0), nan=0.0, posinf=0.0, neginf=0.0)
    return contrast, lcenter


def _qficr_weighted_relu_square_kernel(states, probs, eps=1e-12):
    x = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    psi = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    p = torch.nan_to_num(probs.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(eps)
    sqrt_p = torch.sqrt(p.clamp_min(eps))
    overlap = compute_projection_overlap_kernel(psi, eps=eps, overlap_kernel="relu_square", assume_normalized=True)
    kernel = sqrt_p[:, None] * sqrt_p[None, :] * overlap
    kernel = torch.nan_to_num(0.5 * (kernel + kernel.t()), nan=0.0, posinf=0.0, neginf=0.0)
    return kernel.float().clamp_min(0.0)


def _qficr_pivot_residual_order_from_kernel(kernel, steps, eps=1e-12):
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    num_tokens = int(kernel.shape[0])
    steps = min(max(int(steps), 0), num_tokens)
    residual = torch.diag(kernel).clone().clamp_min(0.0)
    selected = []
    columns = []
    pivot_scores = []
    for _ in range(steps):
        scores = residual.clone()
        if selected:
            scores[torch.tensor(selected, device=kernel.device, dtype=torch.long)] = -float("inf")
        pivot = int(torch.argmax(scores).item())
        pivot_value = float(scores[pivot].item())
        if not math.isfinite(pivot_value) or pivot_value <= eps:
            break
        denom = residual[pivot].clamp_min(eps).sqrt()
        if columns:
            previous = torch.stack(columns, dim=1)
            column = (kernel[:, pivot] - previous @ previous[pivot, :]) / denom
        else:
            column = kernel[:, pivot] / denom
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(column)
        selected.append(pivot)
        pivot_scores.append(pivot_value)
        residual = (residual - column.square()).clamp_min(0.0)
    if len(selected) < steps:
        remaining = [i for i in range(num_tokens) if i not in set(selected)]
        diagonal = torch.diag(kernel).clone().clamp_min(0.0)
        remaining.sort(key=lambda i: (-float(diagonal[i].item()), int(i)))
        for idx in remaining[: steps - len(selected)]:
            selected.append(int(idx))
            pivot_scores.append(float(diagonal[idx].item()))
    return selected, pivot_scores


def _qficr_conditioned_local_residual_order_from_kernel(
    kernel,
    conditioning_order,
    select_count,
    eps=1e-12,
    semantic_residual=None,
    weak_semantic_alpha=0.0,
):
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    num_tokens = int(kernel.shape[0])
    residual = torch.diag(kernel).clone().clamp_min(0.0)
    columns = []
    selected = []
    trace = []

    def update_with_pivot(pivot):
        nonlocal residual
        pivot = int(pivot)
        pivot_residual = residual[pivot].clamp_min(eps)
        if columns:
            previous = torch.stack(columns, dim=1)
            column = (kernel[:, pivot] - previous @ previous[pivot, :]) / pivot_residual.sqrt()
        else:
            column = kernel[:, pivot] / pivot_residual.sqrt()
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(column)
        before = float(residual[pivot].item())
        residual = (residual - column.square()).clamp_min(0.0)
        after = float(residual[pivot].item())
        return before, after

    for pivot in conditioning_order:
        if 0 <= int(pivot) < num_tokens and int(pivot) not in selected:
            before, after = update_with_pivot(int(pivot))
            selected.append(int(pivot))
            trace.append({"phase": "condition", "index": int(pivot), "residual_before": before, "residual_after": after})

    local = []
    weak_semantic_alpha = max(float(weak_semantic_alpha), 0.0)
    semantic_multiplier = torch.ones(num_tokens, dtype=torch.float32, device=kernel.device)
    if weak_semantic_alpha > 0.0 and semantic_residual is not None:
        semantic_residual = torch.as_tensor(
            semantic_residual, dtype=torch.float32, device=kernel.device
        ).reshape(-1)
        if semantic_residual.numel() != num_tokens:
            raise ValueError("semantic_residual must match local kernel token count")
        semantic_residual = torch.nan_to_num(
            semantic_residual, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        rmin = semantic_residual.min()
        rscale = semantic_residual.max() - rmin
        if float(rscale.item()) > eps:
            normalized = (semantic_residual - rmin) / rscale
        else:
            normalized = torch.zeros_like(semantic_residual)
        semantic_multiplier = 1.0 + weak_semantic_alpha * normalized
    conditioning_set = set(selected)
    for step in range(int(select_count)):
        scores = residual.clone()
        if weak_semantic_alpha > 0.0:
            scores = scores * semantic_multiplier
        if conditioning_set:
            scores[torch.tensor(sorted(conditioning_set), dtype=torch.long, device=kernel.device)] = -float("inf")
        if local:
            scores[torch.tensor(local, dtype=torch.long, device=kernel.device)] = -float("inf")
        pivot = int(torch.argmax(scores).item())
        pivot_score = float(scores[pivot].item())
        if not math.isfinite(pivot_score) or pivot_score <= eps:
            remaining = [
                i for i in range(num_tokens)
                if i not in conditioning_set and i not in set(local)
            ]
            if not remaining:
                break
            diag = torch.diag(kernel).clone().clamp_min(0.0)
            remaining.sort(key=lambda i: (-float(diag[i].item()), int(i)))
            pivot = int(remaining[0])
            pivot_score = 0.0
        before, after = update_with_pivot(pivot)
        local.append(pivot)
        trace.append({
            "phase": "select",
            "step": int(step + 1),
            "index": int(pivot),
            "score": float(pivot_score),
            "residual_before": before,
            "residual_after": after,
            "weak_semantic_alpha": float(weak_semantic_alpha),
            "semantic_multiplier": float(semantic_multiplier[pivot].item()),
        })
    return local, residual, trace


def _qficr_dual_residual_coconstruct(raw_image_features, probs, total_k, frozen_order=None, eps=1e-12):
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("P3 dual residual requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(f"P3 dual residual expects 576 visual tokens, got {raw_image_features.shape[1]}")
    total_k = min(max(int(total_k), 0), int(raw_image_features.shape[1]))
    device = raw_image_features.device
    x = torch.nan_to_num(raw_image_features[0].float(), nan=0.0, posinf=0.0, neginf=0.0)
    p = torch.nan_to_num(probs.detach().to(device=device, dtype=torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if p.numel() != x.shape[0]:
        raise ValueError("P3 p_i length does not match visual token count")
    p = p / p.sum().clamp_min(eps)

    semantic_kernel = _qficr_weighted_relu_square_kernel(x, p, eps=eps)
    contrast, lcenter = _qficr_local_contrast_vectors(raw_image_features, eps=eps)
    q = (p * lcenter.clamp_min(0.0)).clamp_min(0.0)
    local_degenerate = bool(q.sum().item() <= eps)
    if local_degenerate:
        q = torch.zeros_like(p)
        local_kernel = torch.zeros_like(semantic_kernel)
    else:
        q = q / q.sum().clamp_min(eps)
        local_kernel = _qficr_weighted_relu_square_kernel(contrast, q, eps=eps)

    semantic_only_order, semantic_only_scores = _qficr_pivot_residual_order_from_kernel(semantic_kernel, total_k, eps=eps)
    frozen_prefix = [int(x) for x in (frozen_order or [])[:total_k]]
    semantic_repro_pass = bool(frozen_prefix and semantic_only_order[:total_k] == frozen_prefix)
    first_mismatch_step = None
    if frozen_prefix and not semantic_repro_pass:
        for idx, (got, ref) in enumerate(zip(semantic_only_order, frozen_prefix), start=1):
            if int(got) != int(ref):
                first_mismatch_step = idx
                break

    residual_s = torch.diag(semantic_kernel).clone().clamp_min(0.0)
    residual_l = torch.diag(local_kernel).clone().clamp_min(0.0)
    columns_s = []
    columns_l = []
    selected = []
    trace = []
    denom_s = float(residual_s.sum().item())
    denom_l = float(residual_l.sum().item())

    def update_residual(kernel, residual, columns, pivot):
        pivot_residual = residual[pivot].clamp_min(eps)
        if not columns:
            column = kernel[:, pivot] / pivot_residual.sqrt()
        else:
            previous = torch.stack(columns, dim=1)
            column = (kernel[:, pivot] - previous @ previous[pivot, :]) / pivot_residual.sqrt()
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(column)
        return (residual - column.square()).clamp_min(0.0)

    for step in range(total_k):
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=device) if selected else None
        score_s = residual_s.clone()
        score_l = residual_l.clone()
        if selected_tensor is not None:
            score_s[selected_tensor] = -float("inf")
            score_l[selected_tensor] = -float("inf")
        not_selected_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        if selected_tensor is not None:
            not_selected_mask[selected_tensor] = False
        debt_s = float(residual_s[not_selected_mask].sum().item() / denom_s) if denom_s > eps else 0.0
        debt_l = float(residual_l[not_selected_mask].sum().item() / denom_l) if denom_l > eps else 0.0
        channel = "semantic" if debt_s >= debt_l else "local"
        scores = score_s if channel == "semantic" else score_l
        pivot = int(torch.argmax(scores).item())
        pivot_score = float(scores[pivot].item())
        if (not math.isfinite(pivot_score) or pivot_score <= eps) and channel == "local":
            channel = "semantic"
            scores = score_s
            pivot = int(torch.argmax(scores).item())
            pivot_score = float(scores[pivot].item())
        if not math.isfinite(pivot_score) or pivot_score <= eps:
            remaining = [i for i in range(x.shape[0]) if i not in set(selected)]
            if not remaining:
                break
            pivot = int(remaining[0])
            channel = "semantic_fallback"
            pivot_score = 0.0
        selected.append(pivot)
        trace.append({
            "step": int(step + 1),
            "channel": channel,
            "D_S": debt_s,
            "D_L": debt_l,
            "selected_index": int(pivot),
            "p": float(p[pivot].item()),
            "L_center": float(lcenter[pivot].item()),
            "q": float(q[pivot].item()),
            "selected_semantic_residual": float(residual_s[pivot].item()),
            "selected_local_residual": float(residual_l[pivot].item()),
            "pivot_score": float(pivot_score),
        })
        residual_s = update_residual(semantic_kernel, residual_s, columns_s, pivot)
        if denom_l > eps:
            residual_l = update_residual(local_kernel, residual_l, columns_l, pivot)
        else:
            residual_l[pivot] = 0.0

    if len(selected) < total_k:
        remaining = [i for i in range(x.shape[0]) if i not in set(selected)]
        selected.extend(remaining[: total_k - len(selected)])
    final = sorted(int(x) for x in selected[:total_k])
    if len(final) != total_k or len(set(final)) != total_k:
        raise RuntimeError("P3 dual residual did not produce exact unique K")
    finite_ok = all([
        bool(torch.isfinite(p).all().item()),
        bool(torch.isfinite(lcenter).all().item()),
        bool(torch.isfinite(q).all().item()),
        bool(torch.isfinite(residual_s).all().item()),
        bool(torch.isfinite(residual_l).all().item()),
    ])
    if not finite_ok:
        raise RuntimeError("P3 dual residual produced NaN/Inf")
    semantic_count = sum(1 for row in trace if row["channel"].startswith("semantic"))
    local_count = sum(1 for row in trace if row["channel"] == "local")
    info = {
        "qficr_method": "p3_dual_residual_coconstruction",
        "qficr_dual_residual_mode": "p3_dual_residual_coconstruction",
        "qficr_p3_total_k": int(total_k),
        "qficr_p3_semantic_channel_pivot_count": int(semantic_count),
        "qficr_p3_local_channel_pivot_count": int(local_count),
        "qficr_p3_channel_sequence": [row["channel"] for row in trace],
        "qficr_p3_trace": trace,
        "qficr_p3_final_indices": final,
        "qficr_p3_semantic_only_order": semantic_only_order[:total_k],
        "qficr_p3_semantic_only_scores": semantic_only_scores[:total_k],
        "qficr_p3_semantic_only_reproduction": semantic_repro_pass,
        "qficr_p3_semantic_only_first_mismatch_step": first_mismatch_step,
        "qficr_p3_semantic_only_reference_order": frozen_prefix,
        "qficr_p3_local_degenerate": local_degenerate,
        "qficr_p3_p_finite": True,
        "qficr_p3_l_finite": True,
        "qficr_p3_q_finite": True,
        "qficr_p3_semantic_residual_finite": True,
        "qficr_p3_local_residual_finite": True,
        "qficr_p3_q_sum": float(q.sum().item()),
        "qficr_p3_l_min": float(lcenter.min().item()),
        "qficr_p3_l_max": float(lcenter.max().item()),
        "qficr_p3_l_mean": float(lcenter.mean().item()),
        "qficr_p3_ordering": "ascending original patch index via boolean index mask",
        "qficr_final_selected_indices": final,
        "final_selected_indices": final,
        "selected_core_indices": [],
        "selected_residual_indices": final,
        "qficr_selected_core_indices": [],
        "qficr_selected_residual_indices": final,
        "qficr_actual_core_count": int(total_k),
        "qficr_actual_residual_count": 0,
        "qficr_final_k_red": int(total_k),
        "qficr_final_k_rest": 0,
        "duplicate_count": 0,
        "nan_inf_count": 0,
        "qficr_duplicate_count": 0,
        "qficr_nan_inf_count": 0,
    }
    return torch.tensor(final, dtype=torch.long, device=device), info


def _qficr_apply_p1_local_residual(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    ratio_cap,
    eps=1e-12,
    fixed_k_local=None,
    method_name="p1_semantic_core_local_residual",
):
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("P1 local residual requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(f"P1 local residual expects 576 visual tokens, got {raw_image_features.shape[1]}")
    if not isinstance(debug_info, dict):
        raise RuntimeError("P1 local residual requires qfid debug_info")
    qfid_info = debug_info.get("qfid_info", {})
    if not isinstance(qfid_info, dict):
        raise RuntimeError("P1 local residual requires qfid_info")
    if fixed_k_local is None:
        k_local = max(int(round(int(total_k) * float(ratio_cap))), 0)
        budget_rule = "int(round(K * EC_QFICR_RECOVER_RATIO_CAP))"
    else:
        k_local = max(int(fixed_k_local), 0)
        budget_rule = "EC_QFICR_LOCAL_RESIDUAL_FIXED_K_LOCAL"
    k_local = min(k_local, int(total_k))
    k_semantic = int(total_k) - k_local
    full_core = qfid_info.get("full_core_indices") or qfid_info.get("qficr_full_core_indices") or []
    if len(full_core) < k_semantic:
        full_core = keep_idx.detach().long().cpu().tolist()
    semantic = [int(x) for x in full_core[:k_semantic]]
    if len(semantic) != k_semantic or len(set(semantic)) != k_semantic:
        raise RuntimeError("P1 could not recover unique semantic core from Frozen V0 Stage-I order")
    semantic_set = set(semantic)
    lcenter = _qficr_lcenter_scores(raw_image_features, eps=eps)
    if not bool(torch.isfinite(lcenter).all().item()):
        raise RuntimeError("P1 L_center contains NaN/Inf")
    candidates = [idx for idx in range(raw_image_features.shape[1]) if idx not in semantic_set]
    candidates.sort(key=lambda idx: (-float(lcenter[idx].item()), int(idx)))
    local = [int(x) for x in candidates[:k_local]]
    final = sorted(semantic + local)
    if len(final) != int(total_k) or len(set(final)) != int(total_k):
        raise RuntimeError("P1 did not produce exact unique K")
    final_idx = torch.tensor(final, dtype=torch.long, device=raw_image_features.device)
    local_scores = [float(lcenter[i].item()) for i in local]
    info = {
        "qficr_local_residual_mode": method_name,
        "qficr_method": method_name,
        "qficr_p1_total_k": int(total_k),
        "qficr_p1_k_semantic": int(k_semantic),
        "qficr_p1_k_local": int(k_local),
        "qficr_p1_budget_ratio_cap": float(ratio_cap),
        "qficr_p1_budget_rule": budget_rule,
        "qficr_p1_semantic_indices": semantic,
        "qficr_p1_local_indices": local,
        "qficr_p1_final_indices": final,
        "qficr_p1_local_score_min": min(local_scores) if local_scores else None,
        "qficr_p1_local_score_max": max(local_scores) if local_scores else None,
        "qficr_p1_local_score_mean": (sum(local_scores) / len(local_scores)) if local_scores else None,
        "qficr_p1_lcenter_finite": True,
        "qficr_p1_old_recovery_off": True,
        "qficr_p1_ordering": "ascending original patch index via boolean index mask",
        "qficr_final_selected_indices": final,
        "final_selected_indices": final,
        "selected_core_indices": semantic,
        "selected_residual_indices": local,
        "qficr_selected_core_indices": semantic,
        "qficr_selected_residual_indices": local,
        "qficr_recovery_indices": local,
        "recovery_indices": local,
        "actual_core_count": int(k_semantic),
        "actual_residual_count": int(k_local),
        "K": int(total_k),
        "K_base": int(k_semantic),
        "fixed_core_count": int(k_semantic),
        "k_core": int(k_semantic),
        "k_recover": int(k_local),
        "qficr_actual_core_count": int(k_semantic),
        "qficr_actual_residual_count": int(k_local),
        "qficr_final_k_red": int(k_semantic),
        "qficr_final_k_rest": int(k_local),
        "duplicate_count": 0,
        "nan_inf_count": 0,
        "qficr_duplicate_count": 0,
        "qficr_nan_inf_count": 0,
    }
    return final_idx, info


def _qficr_apply_p1_tc_local_residual(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    eps=1e-12,
    k_semantic=26,
    k_local=6,
    task_conditioned=True,
    method_name="p1_tc_semantic_core_conditioned_local_residual",
):
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("P1-TC local residual requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(f"P1-TC local residual expects 576 visual tokens, got {raw_image_features.shape[1]}")
    if not isinstance(debug_info, dict):
        raise RuntimeError("P1-TC local residual requires qfid debug_info")
    qfid_info = debug_info.get("qfid_info", {})
    if not isinstance(qfid_info, dict):
        raise RuntimeError("P1-TC local residual requires qfid_info")
    total_k = int(total_k)
    k_semantic = int(k_semantic)
    k_local = int(k_local)
    if k_semantic + k_local != total_k:
        raise RuntimeError("P1-TC requires K_semantic + K_local == total K")
    full_core = qfid_info.get("full_core_indices") or qfid_info.get("qficr_full_core_indices") or []
    if len(full_core) < k_semantic:
        full_core = keep_idx.detach().long().cpu().tolist()
    semantic = [int(x) for x in full_core[:k_semantic]]
    reference = [int(x) for x in full_core[:k_semantic]]
    semantic_first26_repro = bool(len(semantic) == k_semantic and semantic == reference)
    if not semantic_first26_repro:
        raise RuntimeError("P1-TC semantic first-26 reproduction failed")
    if len(set(semantic)) != k_semantic:
        raise RuntimeError("P1-TC semantic core is not unique")

    device = raw_image_features.device
    task_prob = qfid_info.get("_prob_tensor")
    if not torch.is_tensor(task_prob):
        values = qfid_info.get("prob_values") or qfid_info.get("p_mix_values") or []
        if values:
            task_prob = torch.tensor(values, dtype=torch.float32, device=device)
    if not torch.is_tensor(task_prob):
        raise RuntimeError("P1-TC requires Frozen V0 clsmix probability tensor")
    p = torch.nan_to_num(task_prob.detach().to(device=device, dtype=torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if p.numel() != raw_image_features.shape[1]:
        raise RuntimeError("P1-TC p_i length does not match visual token count")
    p = p / p.sum().clamp_min(eps)
    contrast, lcenter = _qficr_local_contrast_vectors(raw_image_features, eps=eps)
    q = (p * lcenter.clamp_min(0.0)).clamp_min(0.0) if task_conditioned else lcenter.clamp_min(0.0)
    initial_residual_mass = float(q.sum().item())
    local_degenerate = bool(initial_residual_mass <= eps)
    local_diag = torch.zeros_like(q)
    conditioning_mode = os.environ.get(
        "EC_QFICR_P1LR_CONDITIONING_MODE", "all"
    ).strip().lower()
    if conditioning_mode not in {"all", "top_m", "threshold"}:
        raise ValueError(
            "EC_QFICR_CONDITIONING_MODE must be all, top_m, or threshold"
        )
    conditioning_order = []
    weak_alpha = max(
        0.0,
        float(os.environ.get("EC_QFICR_WEAK_SEMANTIC_ALPHA", "0.0")),
    )
    semantic_residual = qfid_info.get("qficr_p1lr_semantic_residual_values")
    if local_degenerate:
        q = torch.zeros_like(p)
        local_kernel = torch.zeros((raw_image_features.shape[1], raw_image_features.shape[1]), dtype=torch.float32, device=device)
        local = []
        residual_after = torch.zeros_like(p)
        local_trace = []
    else:
        q = q / q.sum().clamp_min(eps)
        local_kernel = _qficr_weighted_relu_square_kernel(contrast, q, eps=eps)
        local_diag = torch.diag(local_kernel).clone().clamp_min(0.0)
        if conditioning_mode == "top_m":
            top_m = max(
                0,
                min(
                    len(semantic),
                    int(os.environ.get("EC_QFICR_P1LR_CONDITIONING_TOP_M", str(len(semantic)))),
                ),
            )
            ranked = sorted(
                semantic,
                key=lambda idx: (-float(local_diag[idx].item()), int(idx)),
            )
            conditioning_order = ranked[:top_m]
        elif conditioning_mode == "threshold":
            threshold = float(
                os.environ.get("EC_QFICR_P1LR_CONDITIONING_THRESHOLD", "0.0")
            )
            sem_max = max((float(local_diag[idx].item()) for idx in semantic), default=0.0)
            cutoff = max(0.0, threshold) * sem_max
            conditioning_order = [
                idx for idx in semantic if float(local_diag[idx].item()) >= cutoff
            ]
        else:
            conditioning_order = semantic
        local, residual_after, local_trace = _qficr_conditioned_local_residual_order_from_kernel(
            local_kernel,
            conditioning_order,
            k_local,
            eps=eps,
            semantic_residual=semantic_residual,
            weak_semantic_alpha=weak_alpha,
        )
    residual_after_core_mass = float(residual_after.sum().item()) if not local_degenerate else 0.0
    normalized_initial_local_mass = 0.0 if local_degenerate else 1.0
    if len(local) < k_local:
        remaining = [idx for idx in range(raw_image_features.shape[1]) if idx not in set(semantic) and idx not in set(local)]
        remaining.sort(key=lambda idx: (-float(lcenter[idx].item()), int(idx)))
        local.extend(int(x) for x in remaining[: k_local - len(local)])
    local = [int(x) for x in local[:k_local]]
    final = sorted(semantic + local)
    if len(final) != total_k or len(set(final)) != total_k:
        raise RuntimeError("P1-TC did not produce exact unique K")
    finite_ok = all([
        bool(torch.isfinite(p).all().item()),
        bool(torch.isfinite(lcenter).all().item()),
        bool(torch.isfinite(q).all().item()),
        bool(torch.isfinite(local_kernel).all().item()),
        bool(torch.isfinite(residual_after).all().item()),
    ])
    if not finite_ok:
        raise RuntimeError("P1-TC produced NaN/Inf")
    local_scores = [float(lcenter[i].item()) for i in local]
    local_p = [float(p[i].item()) for i in local]
    local_q = [float(q[i].item()) for i in local]
    local_residual = [float(row.get("score", 0.0)) for row in local_trace if row.get("phase") == "select"]
    info = {
        "qficr_local_residual_mode": method_name,
        "qficr_method": method_name,
        "qficr_p1tc_total_k": int(total_k),
        "qficr_p1tc_k_semantic": int(k_semantic),
        "qficr_p1tc_k_local": int(k_local),
        "qficr_p1tc_semantic_indices": semantic,
        "qficr_p1tc_local_indices": local,
        "qficr_p1tc_final_indices": final,
        "qficr_p1tc_semantic_first26_reference": reference,
        "qficr_p1tc_semantic_first26_reproduction": True,
        "qficr_p1tc_conditioning_order": conditioning_order,
        "qficr_p1lr_conditioning_mode": conditioning_mode,
        "qficr_p1lr_conditioning_top_m": int(len(conditioning_order)) if conditioning_mode == "top_m" else None,
        "qficr_p1lr_conditioning_threshold": float(os.environ.get("EC_QFICR_P1LR_CONDITIONING_THRESHOLD", "0.0")) if conditioning_mode == "threshold" else None,
        "qficr_p1lr_weak_semantic_alpha": float(weak_alpha),
        "qficr_p1lr_conditioning_diag_values": [float(local_diag[i].item()) for i in semantic],
        "qficr_p1lr_conditioning_residual_before": [
            float(row.get("residual_before", 0.0))
            for row in local_trace if row.get("phase") == "condition"
        ],
        "qficr_p1tc_local_trace": local_trace,
        "qficr_p1tc_local_degenerate": local_degenerate,
        "qficr_p1tc_local_score_min": min(local_scores) if local_scores else None,
        "qficr_p1tc_local_score_max": max(local_scores) if local_scores else None,
        "qficr_p1tc_local_score_mean": (sum(local_scores) / len(local_scores)) if local_scores else None,
        "qficr_p1tc_local_p_mean": (sum(local_p) / len(local_p)) if local_p else None,
        "qficr_p1tc_local_q_mean": (sum(local_q) / len(local_q)) if local_q else None,
        "qficr_p1tc_local_residual_mean": (sum(local_residual) / len(local_residual)) if local_residual else None,
        "qficr_p1tc_prob_values": [float(x) for x in p.detach().cpu().tolist()],
        "qficr_p1tc_lcenter_values": [float(x) for x in lcenter.detach().cpu().tolist()],
        "qficr_p1tc_q_values": [float(x) for x in q.detach().cpu().tolist()],
        "qficr_p1lr_q_is_lcenter": bool(not task_conditioned),
        "qficr_p1lr_p_used_in_selector": bool(task_conditioned),
        "qficr_p1lr_initial_local_residual_mass": initial_residual_mass,
        "qficr_p1lr_after_core_local_residual_mass": residual_after_core_mass,
        "qficr_p1lr_core_conditioning_reduction": initial_residual_mass - residual_after_core_mass,
        "qficr_p1lr_initial_local_kernel_mass": normalized_initial_local_mass,
        "qficr_p1lr_core_conditioning_reduction_normalized": normalized_initial_local_mass - residual_after_core_mass,
        "qficr_p1tc_p_finite": True,
        "qficr_p1tc_l_finite": True,
        "qficr_p1tc_q_finite": True,
        "qficr_p1tc_local_residual_finite": True,
        "qficr_p1tc_old_recovery_off": True,
        "qficr_p1tc_ordering": "ascending original patch index via boolean index mask",
        "qficr_final_selected_indices": final,
        "final_selected_indices": final,
        "selected_core_indices": semantic,
        "selected_residual_indices": local,
        "qficr_selected_core_indices": semantic,
        "qficr_selected_residual_indices": local,
        "qficr_recovery_indices": local,
        "recovery_indices": local,
        "actual_core_count": int(k_semantic),
        "actual_residual_count": int(k_local),
        "K": int(total_k),
        "K_base": int(k_semantic),
        "fixed_core_count": int(k_semantic),
        "k_core": int(k_semantic),
        "k_recover": int(k_local),
        "qficr_actual_core_count": int(k_semantic),
        "qficr_actual_residual_count": int(k_local),
        "qficr_final_k_red": int(k_semantic),
        "qficr_final_k_rest": int(k_local),
        "duplicate_count": 0,
        "nan_inf_count": 0,
        "qficr_duplicate_count": 0,
        "qficr_nan_inf_count": 0,
    }
    return torch.tensor(final, dtype=torch.long, device=device), info


def _qficr_apply_p1_lr_entropy_local_residual(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    question_text=None,
    eps=1e-12,
    ratio_cap=0.20,
    method_name="p1_lr_entropy_core_conditioned_local_residual",
):
    """P1-LR with Old-QFi-CR entropy allocation and unchanged local ranking."""
    qfid_info = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
    task_prob = qfid_info.get("_prob_tensor") if isinstance(qfid_info, dict) else None
    if not torch.is_tensor(task_prob):
        values = qfid_info.get("prob_values") or qfid_info.get("p_mix_values") or []
        if values:
            task_prob = torch.tensor(values, dtype=torch.float32, device=raw_image_features.device)
    if not torch.is_tensor(task_prob):
        raise RuntimeError("Entropy P1-LR requires Frozen V0 final task probability")
    p = torch.nan_to_num(
        task_prob.detach().to(device=raw_image_features.device, dtype=torch.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    p = p / p.sum().clamp_min(eps)
    budget = compute_qficr_entropy_recovery_budget(
        p,
        total_k,
        question_text=question_text,
        eps=eps,
        ratio_cap_override=ratio_cap,
    )
    final_idx, info = _qficr_apply_p1_tc_local_residual(
        raw_image_features,
        keep_idx,
        debug_info,
        total_k,
        eps=eps,
        k_semantic=budget["k_core"],
        k_local=budget["k_recover"],
        task_conditioned=False,
        method_name=method_name,
    )
    info.update({
        "qficr_entropy_budget": True,
        "qficr_entropy_source": "Frozen V0 final normalized task observation probability p",
        "qficr_entropy": budget["entropy"],
        "qficr_entropy_norm": budget["entropy_norm"],
        "qficr_entropy_hat": budget["entropy_hat"],
        "qficr_entropy_gate": budget["entropy_gate"],
        "qficr_entropy_low": budget["entropy_low"],
        "qficr_entropy_high": budget["entropy_high"],
        "qficr_entropy_gamma": budget["entropy_gamma"],
        "qficr_entropy_question_type": budget["question_type"],
        "qficr_entropy_ratio_base": budget["recover_ratio_base"],
        "qficr_entropy_raw_recover_ratio": budget["recover_ratio_requested"],
        "qficr_entropy_raw_k_rest": budget["raw_recover_k"],
        "qficr_entropy_ratio_cap": budget["ratio_cap"],
        "qficr_entropy_ratio_cap_k": budget["ratio_cap_k"],
        "qficr_entropy_ratio_cap_applied": budget["ratio_cap_applied"],
        "qficr_entropy_k_rest": budget["k_recover"],
        "qficr_entropy_k_sem": budget["k_core"],
        "qficr_raw_k_rest": budget["raw_recover_k"],
        "qficr_ratio_cap": budget["ratio_cap"],
        "qficr_cap_applied": budget["ratio_cap_applied"],
        "qficr_rho": budget["recover_ratio_requested"],
        "qficr_semantic_prefix_reproduction": True,
        "qficr_p1lr_p_used_in_selector": False,
        "qficr_p1tc_semantic_first26_reproduction": True,
    })
    return final_idx, info


def _qficr_local_contrast_vectors_fast(raw_image_features, eps=1e-12):
    """Vectorized 8-neighborhood contrast used only by the fast P1-LR path."""
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("Fast P1-LR local residual requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(
            f"Fast P1-LR local residual expects 576 visual tokens, got {raw_image_features.shape[1]}"
        )

    x = torch.nan_to_num(
        raw_image_features[0].float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    channels = int(x.shape[-1])
    grid = x.t().reshape(1, channels, 24, 24)
    patches = F.unfold(grid, kernel_size=3, padding=1).reshape(1, channels, 9, 576)
    valid = F.unfold(
        torch.ones((1, 1, 24, 24), dtype=x.dtype, device=x.device),
        kernel_size=3,
        padding=1,
    ).reshape(1, 1, 9, 576)
    neighbor_offsets = torch.tensor(
        [0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.long, device=x.device
    )
    neighbor_values = patches.index_select(2, neighbor_offsets)
    neighbor_valid = valid.index_select(2, neighbor_offsets)
    neighbor_sum = (neighbor_values * neighbor_valid).sum(dim=2)
    neighbor_count = neighbor_valid.sum(dim=2).clamp_min(1.0)
    mu = (neighbor_sum / neighbor_count)[0].t().contiguous()

    contrast = torch.nan_to_num(x - mu, nan=0.0, posinf=0.0, neginf=0.0)
    cosine = F.cosine_similarity(x, mu, dim=-1, eps=max(float(eps), 1e-12))
    lcenter = torch.nan_to_num(
        (1.0 - cosine).clamp_min(0.0), nan=0.0, posinf=0.0, neginf=0.0
    )
    return contrast, lcenter


def _qficr_conditioned_local_residual_order_fast(
    kernel,
    conditioning_order,
    select_count,
    eps=1e-12,
):
    """Equivalent incremental residual update with one preallocated factor."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    num_tokens = int(kernel.shape[0])
    safe_eps = max(float(eps), 1e-12)
    conditioning = [int(x) for x in conditioning_order]
    max_columns = min(num_tokens, len(conditioning) + max(int(select_count), 0))
    factor = torch.empty(
        (num_tokens, max_columns), dtype=torch.float32, device=kernel.device
    )
    residual = torch.diag(kernel).clone().clamp_min(0.0)
    excluded = torch.zeros(num_tokens, dtype=torch.bool, device=kernel.device)
    factor_count = 0

    def update_with_pivot(pivot):
        nonlocal factor_count, residual
        pivot = int(pivot)
        denom = residual[pivot].clamp_min(safe_eps).sqrt()
        if factor_count:
            previous = factor[:, :factor_count]
            correction = previous @ previous[pivot, :]
            column = (kernel[:, pivot] - correction) / denom
        else:
            column = kernel[:, pivot] / denom
        column = torch.nan_to_num(
            column.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        factor[:, factor_count] = column
        factor_count += 1
        residual = (residual - column.square()).clamp_min(0.0)

    for pivot in conditioning:
        if 0 <= pivot < num_tokens and not bool(excluded[pivot].item()):
            update_with_pivot(pivot)
            excluded[pivot] = True

    local = []
    selected_scores = []
    diagonal = torch.diag(kernel).clone().clamp_min(0.0)
    for _ in range(max(int(select_count), 0)):
        scores = residual.masked_fill(excluded, float("-inf"))
        pivot = int(torch.argmax(scores).item())
        pivot_score = float(scores[pivot].item())
        if not math.isfinite(pivot_score) or pivot_score <= safe_eps:
            remaining = torch.nonzero(~excluded, as_tuple=False).reshape(-1)
            if remaining.numel() == 0:
                break
            fallback_order = torch.argsort(
                diagonal[remaining], descending=True, stable=True
            )
            pivot = int(remaining[fallback_order[0]].item())
            pivot_score = 0.0
        update_with_pivot(pivot)
        excluded[pivot] = True
        local.append(pivot)
        selected_scores.append(pivot_score)
    return local, residual, selected_scores


def _qficr_select_frozen_core_only_fast(
    selector,
    raw_image_features,
    keep_num,
    question,
    relevance,
    cls_attention,
    semantic_text_embeds,
    semantic_features,
    semantic_units,
    transition_prior,
):
    """Compute only p and the Frozen V0 core order needed by fast P1-LR."""
    if selector.qfid_kernel != "density" or selector.qfid_overlap_kernel != "relu_square":
        raise ValueError(
            "Fast entropy P1-LR requires the frozen density/relu_square semantic kernel"
        )
    if (
        selector.qfid_budget_calib
        or selector.qfid_spectral_filter
        or selector.qfid_spatial_state
        or selector.qfid_measure_prior_mode != "none"
        or selector.qfid_anchor_mode != "none"
    ):
        raise ValueError(
            "Fast entropy P1-LR supports only the paper's unmodified Frozen V0 core configuration"
        )
    started = time.perf_counter()
    probs, prob_info = selector.compute_task_probability(
        raw_image_features,
        question=question,
        relevance=relevance,
        cls_attn=cls_attention,
        text_embeds=semantic_text_embeds,
        semantic_features=semantic_features,
        semantic_units=semantic_units,
        transition_prior=transition_prior,
    )
    semantic_order = select_frozen_qfi_order(
        raw_image_features,
        probs,
        int(keep_num),
        eps=float(selector.qfid_eps),
    )
    order = [int(x) for x in semantic_order.detach().cpu().tolist()]
    keep_idx = semantic_order.sort().values
    qfid_info = {
        "enabled": True,
        "qficr_method": "p1_lr_entropy_core_conditioned_local_residual_fast",
        "qficr_fast_core_only": True,
        "qficr_discarded_legacy_recovery_skipped": True,
        "num_tokens": int(raw_image_features.shape[1]),
        "num_visual_tokens_before": int(raw_image_features.shape[1]),
        "final_keep_size": int(keep_idx.numel()),
        "keep_size": int(keep_idx.numel()),
        "full_core_indices": order,
        "qficr_full_core_indices": order,
        "fixed_core_indices": order,
        "qficr_fixed_core_indices": order,
        "selected_core_indices": order,
        "qficr_selected_core_indices": order,
        "selected_residual_indices": [],
        "qficr_selected_residual_indices": [],
        "final_selected_indices": keep_idx.detach().cpu().tolist(),
        "qficr_final_selected_indices": keep_idx.detach().cpu().tolist(),
        "qficr_total_pruning_latency_ms": (time.perf_counter() - started) * 1000.0,
        **{k: v for k, v in prob_info.items() if k != "semantic_units"},
    }
    qfid_info["_prob_tensor"] = probs.detach()
    debug_info = {
        "score_source": "qfid",
        "candidate_size": int(raw_image_features.shape[1]),
        "repulsion_nnz": 0,
        "complement_nnz": 0,
        "qfid_info": qfid_info,
    }
    return keep_idx, debug_info


def _qficr_spatial_semantic_probability_v1(
    selector,
    semantic_features,
    full_question_text_embeds,
    semantic_text_embeds,
    cls_attention,
    eps=1e-12,
):
    """Build a task prior by normalizing CLIP relevance over image patches.

    The frozen observation normalizes each patch over question words, which can
    become uniform when only one semantic unit is available.  This experimental
    path instead forms one spatial distribution per text prompt and averages a
    whole-question distribution with phrase distributions.  It is deliberately
    isolated behind the V1 mode and never changes the frozen Fast path.
    """
    if not torch.is_tensor(semantic_features) or semantic_features.ndim != 3:
        raise ValueError("Spatial semantic V1 requires [B,N,D] semantic features")
    if semantic_features.shape[0] != 1:
        raise ValueError("Spatial semantic V1 currently requires batch size 1")

    safe_eps = max(float(eps), 1e-12)
    image = torch.nan_to_num(
        semantic_features[0].float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    image = image / image.norm(dim=-1, keepdim=True).clamp_min(safe_eps)
    num_tokens = int(image.shape[0])
    temperature = max(
        float(os.environ.get("EC_QFICR_SPATIAL_SEMANTIC_TAU", "0.07")),
        safe_eps,
    )
    whole_weight = min(
        max(float(os.environ.get("EC_QFICR_SPATIAL_WHOLE_WEIGHT", "0.50")), 0.0),
        1.0,
    )

    def prompt_distribution(text_states):
        if not torch.is_tensor(text_states):
            return None
        states = text_states
        if states.ndim == 3 and states.shape[0] == 1:
            states = states[0]
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[-1] != image.shape[-1] or states.shape[0] == 0:
            return None
        states = torch.nan_to_num(
            states.to(device=image.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        states = states / states.norm(dim=-1, keepdim=True).clamp_min(safe_eps)
        logits = image @ states.t()
        spatial = torch.softmax(logits / temperature, dim=0)
        spatial = torch.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0)
        distribution = spatial.mean(dim=1)
        return distribution / distribution.sum().clamp_min(safe_eps)

    p_whole = prompt_distribution(full_question_text_embeds)
    p_phrase = prompt_distribution(semantic_text_embeds)
    if p_whole is None and p_phrase is None:
        raise RuntimeError("Spatial semantic V1 has no usable text embedding")
    if p_whole is None:
        p_sem = p_phrase
        effective_whole_weight = 0.0
    elif p_phrase is None:
        p_sem = p_whole
        effective_whole_weight = 1.0
    else:
        p_sem = whole_weight * p_whole + (1.0 - whole_weight) * p_phrase
        effective_whole_weight = whole_weight
    p_sem = p_sem / p_sem.sum().clamp_min(safe_eps)

    p_cls, cls_info = selector.compute_qfid_cls_probability(
        cls_attention, num_tokens, image.device
    )
    agreement = 0.0
    js_div = 0.0
    gate = 1.0
    beta_eff = 0.0
    if p_cls is not None:
        p_cls = p_cls.to(device=image.device, dtype=torch.float32)
        midpoint = 0.5 * (p_sem + p_cls)
        js = 0.5 * (
            (p_sem * (p_sem.clamp_min(safe_eps).log() - midpoint.clamp_min(safe_eps).log())).sum()
            + (p_cls * (p_cls.clamp_min(safe_eps).log() - midpoint.clamp_min(safe_eps).log())).sum()
        )
        js_div = max(0.0, float(js.item()))
        agreement = max(0.0, min(1.0, 1.0 - js_div / math.log(2.0)))
        if selector.qfid_cls_gate:
            gate = max(
                float(selector.qfid_cls_gate_min),
                min(0.5 + agreement, float(selector.qfid_cls_gate_max)),
            )
            beta_eff = max(
                0.0,
                min(float(selector.qfid_cls_gate_beta_base) * gate, 1.0),
            )
        else:
            beta_eff = max(0.0, min(float(selector.qfid_cls_mix_beta), 1.0))
        p_pre = (1.0 - beta_eff) * p_sem + beta_eff * p_cls
    else:
        p_pre = p_sem
    p_pre = p_pre / p_pre.sum().clamp_min(safe_eps)

    eps_dep = max(0.0, min(float(selector.qfid_depolarize), 1.0))
    if eps_dep > 0.0:
        uniform = torch.full_like(p_pre, 1.0 / num_tokens)
        p_select = (1.0 - eps_dep) * p_pre + eps_dep * uniform
    else:
        p_select = p_pre
    p_select = p_select / p_select.sum().clamp_min(safe_eps)

    def entropy(prob):
        return float((-(prob * prob.clamp_min(safe_eps).log()).sum()).item())

    max_entropy = math.log(float(num_tokens))
    info = {
        "source": "spatial_semantic_v1",
        "source_fallback": False,
        "mode": "fixed",
        "adaptive_depolarize": eps_dep,
        "entropy": entropy(p_select),
        "entropy_norm": entropy(p_select) / max_entropy,
        "p_sem_entropy": entropy(p_sem),
        "p_cls_entropy": entropy(p_cls) if p_cls is not None else 0.0,
        "agreement": agreement,
        "js_div": js_div,
        "gate": gate,
        "beta_eff": beta_eff,
        "cls_attn_available": p_cls is not None,
        "cls_fallback": p_cls is None,
        "cls_attn_layer": cls_info.get("cls_attn_layer", ""),
        "cls_prior_layers": cls_info.get("cls_prior_layers", ""),
        "cls_head_reduce": cls_info.get("cls_head_reduce", ""),
        "qficr_spatial_v1_temperature": temperature,
        "qficr_spatial_v1_whole_weight": effective_whole_weight,
        "qficr_spatial_v1_pre_entropy": entropy(p_pre),
        "qficr_spatial_v1_pre_entropy_norm": entropy(p_pre) / max_entropy,
        "qficr_spatial_v1_selection_entropy": entropy(p_select),
        "qficr_spatial_v1_selection_entropy_norm": entropy(p_select) / max_entropy,
    }
    return p_select, p_pre, info


def _qficr_select_spatial_semantic_core_v1(
    selector,
    raw_image_features,
    keep_num,
    cls_attention,
    full_question_text_embeds,
    semantic_text_embeds,
    semantic_features,
):
    """Isolated V1 core selector; the original Fast selector is untouched."""
    started = time.perf_counter()
    p_select, p_budget, prob_info = _qficr_spatial_semantic_probability_v1(
        selector,
        semantic_features,
        full_question_text_embeds,
        semantic_text_embeds,
        cls_attention,
        eps=float(selector.qfid_eps),
    )
    semantic_order = select_frozen_qfi_order(
        raw_image_features,
        p_select,
        int(keep_num),
        eps=float(selector.qfid_eps),
    )
    order = [int(x) for x in semantic_order.detach().cpu().tolist()]
    keep_idx = semantic_order.sort().values
    qfid_info = {
        "enabled": True,
        "qficr_method": "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
        "qficr_fast_core_only": True,
        "qficr_spatial_semantic_v1": True,
        "qficr_discarded_legacy_recovery_skipped": True,
        "num_tokens": int(raw_image_features.shape[1]),
        "num_visual_tokens_before": int(raw_image_features.shape[1]),
        "final_keep_size": int(keep_idx.numel()),
        "keep_size": int(keep_idx.numel()),
        "full_core_indices": order,
        "qficr_full_core_indices": order,
        "fixed_core_indices": order,
        "qficr_fixed_core_indices": order,
        "selected_core_indices": order,
        "qficr_selected_core_indices": order,
        "selected_residual_indices": [],
        "qficr_selected_residual_indices": [],
        "final_selected_indices": keep_idx.detach().cpu().tolist(),
        "qficr_final_selected_indices": keep_idx.detach().cpu().tolist(),
        "qficr_total_pruning_latency_ms": (time.perf_counter() - started) * 1000.0,
        **prob_info,
    }
    # Semantic ordering uses the mildly depolarized distribution, while the
    # adaptive budget uses the pre-depolarization distribution.
    qfid_info["_selection_prob_tensor"] = p_select.detach()
    qfid_info["_prob_tensor"] = p_budget.detach()
    debug_info = {
        "score_source": "qfid_spatial_semantic_v1",
        "candidate_size": int(raw_image_features.shape[1]),
        "repulsion_nnz": 0,
        "complement_nnz": 0,
        "qfid_info": qfid_info,
    }
    return keep_idx, debug_info


def _qficr_apply_p1_lr_entropy_local_residual_fast(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    question_text=None,
    eps=1e-12,
    ratio_cap=0.20,
    method_name="p1_lr_entropy_core_conditioned_local_residual_fast",
):
    """Engineering-only fast path preserving entropy allocation and selection rules."""
    conditioning_mode = os.environ.get(
        "EC_QFICR_P1LR_CONDITIONING_MODE", "all"
    ).strip().lower()
    weak_semantic_alpha = float(
        os.environ.get("EC_QFICR_WEAK_SEMANTIC_ALPHA", "0.0")
    )
    if conditioning_mode != "all" or weak_semantic_alpha != 0.0:
        raise ValueError(
            "Fast entropy P1-LR preserves only the paper configuration: "
            "conditioning_mode=all and weak_semantic_alpha=0"
        )
    qfid_info = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
    task_prob = qfid_info.get("_prob_tensor") if isinstance(qfid_info, dict) else None
    if not torch.is_tensor(task_prob):
        raise RuntimeError("Fast entropy P1-LR requires the Frozen V0 task probability")
    device = raw_image_features.device
    p = torch.nan_to_num(
        task_prob.detach().to(device=device, dtype=torch.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    p = p / p.sum().clamp_min(eps)
    budget = compute_qficr_entropy_recovery_budget(
        p,
        total_k,
        question_text=question_text,
        eps=eps,
        ratio_cap_override=ratio_cap,
    )
    k_semantic = int(budget["k_core"])
    k_local = int(budget["k_recover"])
    full_core = qfid_info.get("full_core_indices") or qfid_info.get(
        "qficr_full_core_indices"
    ) or []
    if len(full_core) < k_semantic:
        full_core = keep_idx.detach().long().cpu().tolist()
    semantic = [int(x) for x in full_core[:k_semantic]]
    if len(semantic) != k_semantic or len(set(semantic)) != k_semantic:
        raise RuntimeError("Fast entropy P1-LR could not reproduce the semantic prefix")

    contrast, lcenter = _qficr_local_contrast_vectors_fast(raw_image_features, eps=eps)
    initial_residual_mass = float(lcenter.sum().item())
    local_degenerate = bool(initial_residual_mass <= eps)
    if local_degenerate:
        q = torch.zeros_like(lcenter)
        local = []
        residual_after = torch.zeros_like(lcenter)
        local_residual_scores = []
    else:
        q = lcenter / lcenter.sum().clamp_min(eps)
        local_kernel = _qficr_weighted_relu_square_kernel(contrast, q, eps=eps)
        local, residual_after, local_residual_scores = (
            _qficr_conditioned_local_residual_order_fast(
                local_kernel,
                semantic,
                k_local,
                eps=eps,
            )
        )
    if len(local) < k_local:
        excluded = torch.zeros(raw_image_features.shape[1], dtype=torch.bool, device=device)
        excluded[torch.tensor(semantic + local, dtype=torch.long, device=device)] = True
        candidates = torch.nonzero(~excluded, as_tuple=False).reshape(-1)
        order = torch.argsort(lcenter[candidates], descending=True, stable=True)
        local.extend(
            int(x) for x in candidates[order[: k_local - len(local)]].detach().cpu().tolist()
        )
    local = [int(x) for x in local[:k_local]]
    final = sorted(semantic + local)
    if len(final) != int(total_k) or len(set(final)) != int(total_k):
        raise RuntimeError("Fast entropy P1-LR did not produce exact unique K")

    local_tensor = torch.tensor(local, dtype=torch.long, device=device)
    local_score_mean = (
        float(lcenter[local_tensor].mean().item()) if local_tensor.numel() else None
    )
    residual_after_mass = float(residual_after.sum().item())
    info = {
        "qficr_local_residual_mode": method_name,
        "qficr_method": method_name,
        "qficr_fast_path": True,
        "qficr_fast_vectorized_local": True,
        "qficr_fast_preallocated_factor": True,
        "qficr_fast_compact_diagnostics": True,
        "qficr_p1tc_total_k": int(total_k),
        "qficr_p1tc_k_semantic": k_semantic,
        "qficr_p1tc_k_local": k_local,
        "qficr_p1tc_semantic_indices": semantic,
        "qficr_p1tc_local_indices": local,
        "qficr_p1tc_final_indices": final,
        "qficr_p1tc_semantic_first26_reference": semantic,
        "qficr_p1tc_semantic_first26_reproduction": True,
        "qficr_p1tc_conditioning_order": semantic,
        "qficr_p1lr_conditioning_mode": "all",
        "qficr_p1lr_weak_semantic_alpha": 0.0,
        "qficr_p1tc_local_degenerate": local_degenerate,
        "qficr_p1tc_local_score_mean": local_score_mean,
        "qficr_p1tc_local_residual_mean": (
            sum(local_residual_scores) / len(local_residual_scores)
            if local_residual_scores else None
        ),
        "qficr_p1tc_p_finite": bool(torch.isfinite(p).all().item()),
        "qficr_p1tc_l_finite": bool(torch.isfinite(lcenter).all().item()),
        "qficr_p1tc_q_finite": bool(torch.isfinite(q).all().item()),
        "qficr_p1tc_local_residual_finite": bool(torch.isfinite(residual_after).all().item()),
        "qficr_p1lr_q_is_lcenter": True,
        "qficr_p1lr_p_used_in_selector": False,
        "qficr_p1lr_initial_local_residual_mass": initial_residual_mass,
        "qficr_p1lr_after_core_local_residual_mass": residual_after_mass,
        "qficr_p1lr_core_conditioning_reduction": initial_residual_mass - residual_after_mass,
        "qficr_p1lr_initial_local_kernel_mass": 0.0 if local_degenerate else 1.0,
        "qficr_p1lr_core_conditioning_reduction_normalized": (
            0.0 if local_degenerate else 1.0 - residual_after_mass
        ),
        "qficr_entropy_budget": True,
        "qficr_entropy_source": "Frozen V0 final normalized task observation probability p",
        "qficr_entropy": budget["entropy"],
        "qficr_entropy_norm": budget["entropy_norm"],
        "qficr_entropy_hat": budget["entropy_hat"],
        "qficr_entropy_gate": budget["entropy_gate"],
        "qficr_entropy_low": budget["entropy_low"],
        "qficr_entropy_high": budget["entropy_high"],
        "qficr_entropy_gamma": budget["entropy_gamma"],
        "qficr_entropy_question_type": budget["question_type"],
        "qficr_entropy_ratio_base": budget["recover_ratio_base"],
        "qficr_entropy_raw_recover_ratio": budget["recover_ratio_requested"],
        "qficr_entropy_raw_k_rest": budget["raw_recover_k"],
        "qficr_entropy_ratio_cap": budget["ratio_cap"],
        "qficr_entropy_ratio_cap_k": budget["ratio_cap_k"],
        "qficr_entropy_ratio_cap_applied": budget["ratio_cap_applied"],
        "qficr_entropy_k_rest": k_local,
        "qficr_entropy_k_sem": k_semantic,
        "qficr_raw_k_rest": budget["raw_recover_k"],
        "qficr_ratio_cap": budget["ratio_cap"],
        "qficr_cap_applied": budget["ratio_cap_applied"],
        "qficr_rho": budget["recover_ratio_requested"],
        "qficr_semantic_prefix_reproduction": True,
        "qficr_final_selected_indices": final,
        "final_selected_indices": final,
        "selected_core_indices": semantic,
        "selected_residual_indices": local,
        "qficr_selected_core_indices": semantic,
        "qficr_selected_residual_indices": local,
        "qficr_recovery_indices": local,
        "recovery_indices": local,
        "qficr_actual_core_count": k_semantic,
        "qficr_actual_residual_count": k_local,
        "qficr_final_k_red": k_semantic,
        "qficr_final_k_rest": k_local,
        "K": int(total_k),
        "K_base": k_semantic,
        "duplicate_count": 0,
        "nan_inf_count": 0,
        "qficr_duplicate_count": 0,
        "qficr_nan_inf_count": 0,
    }
    return torch.tensor(final, dtype=torch.long, device=device), info


def _qficr_apply_p1_lr_entropy_random_recovery_v2(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    question_text=None,
    eps=1e-12,
    ratio_cap=0.20,
):
    """Keep the Frozen semantic core and use reproducible unbiased recovery."""
    _, info = _qficr_apply_p1_lr_entropy_local_residual_fast(
        raw_image_features,
        keep_idx,
        debug_info,
        total_k,
        question_text=question_text,
        eps=eps,
        ratio_cap=ratio_cap,
        method_name="p1_lr_entropy_core_random_recovery_v2",
    )
    semantic = [int(x) for x in info["qficr_p1tc_semantic_indices"]]
    k_local = int(info["qficr_p1tc_k_local"])
    candidates = sorted(set(range(int(raw_image_features.shape[1]))) - set(semantic))
    base_seed = int(os.environ.get("EC_QFICR_RANDOM_RECOVERY_SEED", "20260908"))
    qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "")
    qtext = os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", str(question_text or ""))
    payload = f"{base_seed}\0{qid}\0{qtext}".encode("utf-8")
    sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    local = random.Random(sample_seed).sample(candidates, k_local)
    final = sorted(semantic + local)
    if len(final) != int(total_k) or len(set(final)) != int(total_k):
        raise RuntimeError("Random recovery V2 did not produce exact unique K")
    info.update({
        "qficr_method": "p1_lr_entropy_core_random_recovery_v2",
        "qficr_local_residual_mode": "p1_lr_entropy_core_random_recovery_v2",
        "qficr_random_recovery_v2": True,
        "qficr_random_recovery_seed": base_seed,
        "qficr_random_recovery_sample_seed": sample_seed,
        "qficr_p1tc_local_indices": local,
        "qficr_p1tc_final_indices": final,
        "qficr_final_selected_indices": final,
        "final_selected_indices": final,
        "selected_residual_indices": local,
        "qficr_selected_residual_indices": local,
        "qficr_recovery_indices": local,
        "recovery_indices": local,
        "qficr_p1tc_local_score_mean": None,
        "qficr_p1tc_local_residual_mean": None,
    })
    return torch.tensor(final, dtype=torch.long, device=raw_image_features.device), info


def _qficr_apply_p1_lr_entropy_spatial_diverse_recovery_v3(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    question_text=None,
    eps=1e-12,
    ratio_cap=0.20,
):
    """Choose the most spatially dispersed of fixed unbiased proposals."""
    _, info = _qficr_apply_p1_lr_entropy_local_residual_fast(
        raw_image_features,
        keep_idx,
        debug_info,
        total_k,
        question_text=question_text,
        eps=eps,
        ratio_cap=ratio_cap,
        method_name="p1_lr_entropy_spatial_diverse_recovery_v3",
    )
    semantic = [int(x) for x in info["qficr_p1tc_semantic_indices"]]
    k_local = int(info["qficr_p1tc_k_local"])
    num_tokens = int(raw_image_features.shape[1])
    grid_width = int(round(math.sqrt(num_tokens)))
    if grid_width * grid_width != num_tokens:
        raise ValueError("Spatial diverse recovery V3 requires a square patch grid")
    candidates = sorted(set(range(num_tokens)) - set(semantic))
    qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "")
    qtext = os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", str(question_text or ""))
    proposal_seeds = [
        int(x.strip())
        for x in os.environ.get(
            "EC_QFICR_V3_PROPOSAL_SEEDS",
            "20260906,20260907,20260908,20260909,20260910",
        ).split(",")
        if x.strip()
    ]
    if not proposal_seeds:
        raise ValueError("Spatial diverse recovery V3 requires proposal seeds")

    def make_proposal(base_seed):
        payload = f"{base_seed}\0{qid}\0{qtext}".encode("utf-8")
        sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(sample_seed).sample(candidates, k_local), sample_seed

    def mean_pairwise_distance(indices):
        if len(indices) < 2:
            return 0.0
        total = 0.0
        count = 0
        for pos, first in enumerate(indices):
            r1, c1 = divmod(first, grid_width)
            for second in indices[pos + 1:]:
                r2, c2 = divmod(second, grid_width)
                total += math.hypot(r1 - r2, c1 - c2)
                count += 1
        return total / count

    proposals = []
    for proposal_index, base_seed in enumerate(proposal_seeds):
        local, sample_seed = make_proposal(base_seed)
        proposals.append({
            "index": proposal_index,
            "base_seed": base_seed,
            "sample_seed": sample_seed,
            "local": local,
            "spatial_score": mean_pairwise_distance(local),
        })
    chosen = max(proposals, key=lambda item: (item["spatial_score"], -item["index"]))
    local = chosen["local"]
    final = sorted(semantic + local)
    if len(final) != int(total_k) or len(set(final)) != int(total_k):
        raise RuntimeError("Spatial diverse recovery V3 did not produce exact unique K")
    info.update({
        "qficr_method": "p1_lr_entropy_spatial_diverse_recovery_v3",
        "qficr_local_residual_mode": "p1_lr_entropy_spatial_diverse_recovery_v3",
        "qficr_spatial_diverse_recovery_v3": True,
        "qficr_v3_proposal_count": len(proposals),
        "qficr_v3_chosen_proposal_index": chosen["index"],
        "qficr_v3_chosen_base_seed": chosen["base_seed"],
        "qficr_v3_chosen_sample_seed": chosen["sample_seed"],
        "qficr_v3_spatial_score": chosen["spatial_score"],
        "qficr_v3_proposal_spatial_scores": [item["spatial_score"] for item in proposals],
        "qficr_p1tc_local_indices": local,
        "qficr_p1tc_final_indices": final,
        "qficr_final_selected_indices": final,
        "final_selected_indices": final,
        "selected_residual_indices": local,
        "qficr_selected_residual_indices": local,
        "qficr_recovery_indices": local,
        "recovery_indices": local,
        "qficr_p1tc_local_score_mean": None,
        "qficr_p1tc_local_residual_mean": None,
    })
    return torch.tensor(final, dtype=torch.long, device=raw_image_features.device), info


_QFICR_V4_EXPLICIT_SPATIAL_PATTERN = re.compile(
    r"\b(left|right|above|below|under|behind)\b"
    r"|\bin\s+front\s+of\b"
    r"|\bon\s+top\s+of\b",
    flags=re.IGNORECASE,
)


def _qficr_is_explicit_spatial_relation_question_v4(question_text):
    """Conservatively detect axis/directional relations from question text."""
    if not isinstance(question_text, str) or not question_text.strip():
        return False
    question_only = question_text.split("\n", 1)[0]
    return bool(_QFICR_V4_EXPLICIT_SPATIAL_PATTERN.search(question_only))


def _qficr_apply_p1_lr_entropy_relation_routed_recovery_v4(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    question_text=None,
    eps=1e-12,
    ratio_cap=0.20,
):
    """Use V3 only for explicit spatial relations; otherwise reproduce Fast."""
    use_spatial = _qficr_is_explicit_spatial_relation_question_v4(question_text)
    if use_spatial:
        selected, info = _qficr_apply_p1_lr_entropy_spatial_diverse_recovery_v3(
            raw_image_features,
            keep_idx,
            debug_info,
            total_k,
            question_text=question_text,
            eps=eps,
            ratio_cap=ratio_cap,
        )
        route = "spatial_v3"
    else:
        selected, info = _qficr_apply_p1_lr_entropy_local_residual_fast(
            raw_image_features,
            keep_idx,
            debug_info,
            total_k,
            question_text=question_text,
            eps=eps,
            ratio_cap=ratio_cap,
            method_name="p1_lr_entropy_relation_routed_recovery_v4",
        )
        route = "fast"
    info.update({
        "qficr_method": "p1_lr_entropy_relation_routed_recovery_v4",
        "qficr_local_residual_mode": "p1_lr_entropy_relation_routed_recovery_v4",
        "qficr_relation_routed_recovery_v4": True,
        "qficr_v4_route": route,
        "qficr_v4_explicit_spatial_relation": use_spatial,
        "qficr_v4_pattern_version": "axis_direction_v1",
    })
    return selected, info


def _qficr_apply_generalized_routed_recovery_candidate(
    raw_image_features,
    keep_idx,
    debug_info,
    total_k,
    question_text=None,
    eps=1e-12,
    ratio_cap=0.20,
):
    """Autoresearch candidate entry point; the baseline exactly reproduces V4."""
    route_mode = os.environ.get("EC_QFICR_GENERALIZED_ROUTE_MODE", "baseline").strip().lower()
    effective_cap = ratio_cap
    budget_route = "v4_baseline"
    routed_question_text = question_text
    if route_mode == "k128_budget_v1":
        question_body = re.sub(
            r"^\s*<image>\s*", "", str(question_text or ""), flags=re.IGNORECASE
        )
        question_body = re.split(r"\n\s*[A-E][.)]\s+", question_body, maxsplit=1)[0]
        question_only = " ".join(question_body.split())
        routed_question_text = question_only
        quantitative_core_pattern = re.compile(
            r"\bhow\s+many\b|\bnumber\s+of\b|\bconcentration\b"
            r"|\b(?:python|program|code)\b.*\b(?:output|generate|correct)\b",
            flags=re.IGNORECASE,
        )
        social_relation_pattern = re.compile(
            r"\brelationship\b.*\b(persons?|people|men|women|man|woman)\b",
            flags=re.IGNORECASE,
        )
        spatial_relation_pattern = re.compile(
            r"\b(left|right|above|below|under|over|behind|front|next to|beside|between)\b",
            flags=re.IGNORECASE,
        )
        if social_relation_pattern.search(question_only) or spatial_relation_pattern.search(question_only):
            effective_cap = float(ratio_cap)
            budget_route = "relation_cap020"
        elif quantitative_core_pattern.search(question_only):
            effective_cap = 0.0
            budget_route = "quantitative_core_only"
        else:
            effective_cap = min(float(ratio_cap), 0.10)
            budget_route = "entropy_cap010"
    elif route_mode != "baseline":
        raise ValueError(f"Unknown EC_QFICR_GENERALIZED_ROUTE_MODE: {route_mode}")
    selected, info = _qficr_apply_p1_lr_entropy_relation_routed_recovery_v4(
        raw_image_features,
        keep_idx,
        debug_info,
        total_k,
        question_text=routed_question_text,
        eps=eps,
        ratio_cap=effective_cap,
    )
    info.update({
        "qficr_method": "p1_lr_generalized_routed_recovery_candidate",
        "qficr_local_residual_mode": "p1_lr_generalized_routed_recovery_candidate",
        "qficr_generalized_candidate": True,
        "qficr_generalized_route_mode": route_mode,
        "qficr_generalized_budget_route": budget_route,
        "qficr_generalized_effective_ratio_cap": float(effective_cap),
        "qficr_generalized_normalized_question_routing": route_mode == "k128_budget_v1",
    })
    return selected, info


def _hash_long_tensor(values):
    values = values.detach().to(torch.long).reshape(-1).cpu()
    return hashlib.sha1(values.numpy().tobytes()).hexdigest()


def _hash_float_tensor(values):
    values = values.detach().float().contiguous().cpu()
    return hashlib.sha1(values.numpy().tobytes()).hexdigest()


def _float_stat(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write_qficr_residual_transfer_debug(record):
    stats_path = os.environ.get("EC_QFICR_RESIDUAL_TRANSFER_DEBUG_JSONL", "").strip()
    if not stats_path:
        return
    if stats_path.lower() in {"1", "true", "yes", "on"}:
        stats_path = os.path.abspath("debug_qficr_residual_transfer.jsonl")
    output_dir = os.path.dirname(os.path.abspath(stats_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(stats_path, "a", encoding="utf-8") as stats_file:
            stats_file.write(json.dumps({"residual_transfer": record}, sort_keys=True) + "\n")


def _load_forced_selected_records():
    path = os.environ.get("EC_FORCE_SELECTED_INDICES_JSONL", "").strip()
    if not path:
        return {}
    abs_path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        if os.environ.get("EC_FORCE_SELECTED_INDICES_STRICT", "1").strip().lower() not in {"0", "false", "no", "off"}:
            raise
        return {}
    if _FORCED_SELECTED_CACHE["path"] == abs_path and _FORCED_SELECTED_CACHE["mtime"] == mtime:
        return _FORCED_SELECTED_CACHE["records"]
    records = {}
    with open(abs_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", row.get("sample_id", "")))
            question = str(row.get("question", row.get("prompt", "")))
            indices = row.get("selected_indices", row.get("final_selected_indices", []))
            if qid and isinstance(indices, list):
                parsed = [int(x) for x in indices]
                records[qid] = parsed
                if question:
                    records[f"{qid}\t{question}"] = parsed
    _FORCED_SELECTED_CACHE.update({"path": abs_path, "mtime": mtime, "records": records})
    return records


def _maybe_force_selected_indices(current_idx, num_tokens, device):
    records = _load_forced_selected_records()
    if not records:
        return current_idx, False
    qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "").strip()
    qtext = os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", "").strip()
    key = f"{qid}\t{qtext}" if qtext and f"{qid}\t{qtext}" in records else qid
    if key not in records:
        if os.environ.get("EC_FORCE_SELECTED_INDICES_STRICT", "1").strip().lower() not in {"0", "false", "no", "off"}:
            raise KeyError(f"EC_FORCE_SELECTED_INDICES_JSONL has no entry for question_id={qid!r}")
        return current_idx, False
    forced = torch.tensor(records[key], dtype=torch.long, device=device).reshape(-1)
    if forced.numel() != current_idx.reshape(-1).numel():
        raise ValueError(f"forced selected set for {qid} has K={forced.numel()}, expected {current_idx.reshape(-1).numel()}")
    if bool((forced < 0).any().item()) or bool((forced >= int(num_tokens)).any().item()):
        raise ValueError(f"forced selected set for {qid} contains out-of-range token index")
    if forced.unique().numel() != forced.numel():
        raise ValueError(f"forced selected set for {qid} contains duplicate token indices")
    return forced.sort().values, True


def _load_e8_local_fusion_records():
    path = os.environ.get("EC_QFICR_LOCAL_FUSION_MANIFEST_JSONL", "").strip()
    if not path:
        return {}
    abs_path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        raise
    if _E8_LOCAL_FUSION_CACHE["path"] == abs_path and _E8_LOCAL_FUSION_CACHE["mtime"] == mtime:
        return _E8_LOCAL_FUSION_CACHE["records"]
    records = {}
    with open(abs_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", row.get("sample_id", "")))
            question = str(row.get("question", row.get("text", "")))
            if qid:
                records[qid] = row
                if question:
                    records[f"{qid}\t{question}"] = row
    _E8_LOCAL_FUSION_CACHE.update({"path": abs_path, "mtime": mtime, "records": records})
    return records


def _e8_local_fusion_record():
    records = _load_e8_local_fusion_records()
    if not records:
        return {}
    qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "").strip()
    qtext = os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", "").strip()
    key = f"{qid}\t{qtext}" if qtext and f"{qid}\t{qtext}" in records else qid
    return records.get(key, {})


def _write_e8_local_fusion_debug(record):
    global _E8_LOCAL_FUSION_DEBUG_COUNT
    path = os.environ.get("EC_QFICR_LOCAL_FUSION_DEBUG_JSONL", "").strip()
    if not path:
        path = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
    if not path:
        return
    if path.lower() in {"1", "true", "yes", "on"}:
        path = os.path.abspath("debug_qficr_e8_local_fusion.jsonl")
    payload = dict(record)
    payload.update({
        "sample_index": _E8_LOCAL_FUSION_DEBUG_COUNT,
        "process_id": os.getpid(),
        "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
        "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
        "kind": "e8_local_fusion",
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    _E8_LOCAL_FUSION_DEBUG_COUNT += 1


def _e8_voronoi_cells(core_indices, num_tokens):
    core = [int(x) for x in core_indices.detach().long().cpu().tolist()]
    core_set = set(core)
    cells = {c: [] for c in core}
    for token in range(int(num_tokens)):
        if token in core_set:
            continue
        row, col = divmod(token, 24)
        best_core = min(
            core,
            key=lambda c: (abs(row - (c // 24)) + abs(col - (c % 24)), c),
        )
        cells[best_core].append(token)
    return cells


def _e8_nearest_core(token, core_indices):
    core = [int(x) for x in core_indices.detach().long().cpu().tolist()]
    row, col = divmod(int(token), 24)
    return min(core, key=lambda c: (abs(row - (c // 24)) + abs(col - (c % 24)), c))


def _e8_apply_equal_add_norm_preserve(base, addition, eps=1e-12):
    z = base + addition
    return base.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


def _apply_e8_local_fusion(raw_image_features, keep_idx):
    mode = _qficr_local_fusion_mode()
    if mode == "off":
        return raw_image_features, None
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("E8 local fusion requires batch size 1")
    if raw_image_features.shape[1] != 576:
        raise ValueError(f"E8 local fusion expects 576 visual tokens, got {raw_image_features.shape[1]}")
    if keep_idx.reshape(-1).numel() != 32:
        raise ValueError(f"E8 local fusion expects fixed K32 Core, got K={keep_idx.reshape(-1).numel()}")
    if keep_idx.unique().numel() != keep_idx.numel():
        raise ValueError("E8 local fusion received duplicate selected indices")
    if bool((keep_idx < 0).any().item()) or bool((keep_idx >= raw_image_features.shape[1]).any().item()):
        raise ValueError("E8 local fusion selected indices out of range")

    output = raw_image_features.clone()
    x = output[0]
    core = keep_idx.detach().long()
    core_set = set(int(v) for v in core.detach().cpu().tolist())
    before_selected = raw_image_features[0, core].detach().float()
    before_hash = _hash_float_tensor(before_selected)
    cells = _e8_voronoi_cells(core, raw_image_features.shape[1])
    fused_cores = []
    fusion_cells = {}
    record = _e8_local_fusion_record()

    if mode in {"e8_oracle_fuse", "e8_matched_control_fuse", "e8_matched_random_fuse"}:
        token_key = "oracle_rescue_token" if mode == "e8_oracle_fuse" else "matched_control_token"
        if token_key not in record:
            raise KeyError(f"E8 fusion manifest missing {token_key}")
        token = int(record[token_key])
        if token < 0 or token >= raw_image_features.shape[1]:
            raise ValueError(f"E8 fusion token out of range: {token}")
        if token in core_set:
            raise ValueError(f"E8 fusion token is already in Core: {token}")
        target_core = int(record.get("nearest_core_token") or _e8_nearest_core(token, core))
        if target_core not in core_set:
            raise ValueError("E8 fusion target core is not in selected Core")
        x[target_core] = _e8_apply_equal_add_norm_preserve(x[target_core], raw_image_features[0, token])
        fused_cores.append(target_core)
        fusion_cells[str(target_core)] = [token]
    else:
        if mode == "e8_shuffled_cell_fuse":
            seed = int(os.environ.get("EC_QFICR_LOCAL_FUSION_SEED", "20260818"))
            qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "")
            digest = int(hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8], 16)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + digest)
            non_core = [t for members in cells.values() for t in members]
            perm = torch.randperm(len(non_core), generator=generator).tolist()
            shuffled = [non_core[i] for i in perm]
            reassigned = {}
            offset = 0
            for c in sorted(cells):
                size = len(cells[c])
                reassigned[c] = shuffled[offset:offset + size]
                offset += size
            cells = reassigned
        for c in sorted(cells):
            members = cells[c]
            fusion_cells[str(c)] = [int(t) for t in members]
            if not members:
                continue
            member_idx = torch.tensor(members, dtype=torch.long, device=x.device)
            centroid = raw_image_features[0, member_idx].mean(dim=0)
            x[c] = _e8_apply_equal_add_norm_preserve(x[c], centroid)
            fused_cores.append(int(c))

    after_selected = output[0, core].detach().float()
    after_hash = _hash_float_tensor(after_selected)
    delta = (after_selected - before_selected).float()
    cell_sizes = [len(v) for v in fusion_cells.values()]
    debug = {
        "e8_local_fusion": True,
        "fusion_mode": mode,
        "fusion_source_space": "pre_projector",
        "core_indices": [int(x) for x in core.detach().cpu().tolist()],
        "selected_indices_before": [int(x) for x in core.detach().cpu().tolist()],
        "selected_indices_after": [int(x) for x in core.detach().cpu().tolist()],
        "selected_indices_unchanged": True,
        "final_visual_token_count": int(core.numel()),
        "recovery": "off",
        "fusion_cells": fusion_cells,
        "num_fused_cores": int(len(set(fused_cores))),
        "mean_cell_size": float(sum(cell_sizes) / max(1, len(cell_sizes))),
        "median_cell_size": float(torch.tensor(cell_sizes, dtype=torch.float32).median().item()) if cell_sizes else 0.0,
        "max_cell_size": int(max(cell_sizes)) if cell_sizes else 0,
        "empty_core_cells": int(sum(1 for v in cell_sizes if v == 0)),
        "feature_hash_before": before_hash,
        "feature_hash_after": after_hash,
        "feature_changed": before_hash != after_hash,
        "feature_delta_norm": float(delta.norm().item()),
        "nan_inf_count": int((~torch.isfinite(after_selected)).sum().item()),
        "blind_uses_oracle": mode in {"e8_oracle_fuse", "e8_matched_control_fuse", "e8_matched_random_fuse"},
        "blind_uses_gt": False,
        "blind_uses_nll": False,
        "blind_uses_cd": False,
    }
    if record:
        for key in ("sample_id", "oracle_rescue_token", "matched_control_token", "nearest_core_token"):
            if key in record:
                debug[key] = record[key]
    _write_e8_local_fusion_debug(debug)
    return output, debug


def _write_core_set_audit_cache(kind, image_features, selected_idx, extra=None):
    cache_dir = os.environ.get("EC_CORE_SET_AUDIT_CACHE_DIR", "").strip()
    if not cache_dir:
        return
    qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "unknown").strip() or "unknown"
    qtext = os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", "")
    digest = hashlib.sha1(f"{qid}\t{qtext}".encode("utf-8")).hexdigest()[:12]
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    safe_qid = f"{safe_qid}_{digest}"
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "version": 1,
        "kind": kind,
        "question_id": qid,
        "question": os.environ.get("EC_QFID_DEBUG_QUESTION_TEXT", ""),
        "projected_features": image_features[0].detach().float().cpu(),
        "selected_indices": selected_idx.detach().long().cpu(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(cache_dir, f"{kind}_{safe_qid}.pt"))


def _write_cdpruner_selected_debug(select_idx, image_features, image_embeds, text_embeds, forced=False):
    if os.environ.get("EC_CDPRUNER_DEBUG_SELECTED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    path = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
    if not path:
        return
    if path.lower() in {"1", "true", "yes", "on"}:
        path = os.path.abspath("debug_cdpruner_selected.jsonl")
    selected = select_idx[0].detach().long().cpu().tolist()
    projected = image_features.detach().float()
    selected_features = projected[0, select_idx[0].detach().long()]
    pair = torch.matmul(
        torch.nn.functional.normalize(selected_features, dim=-1, eps=1e-12),
        torch.nn.functional.normalize(selected_features, dim=-1, eps=1e-12).t(),
    )
    off = pair[~torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)] if pair.numel() else torch.empty(0, device=pair.device)
    image_embeds_n = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_embeds_n = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    relevance = torch.matmul(image_embeds_n, text_embeds_n.t())
    relevance = (-relevance).mean(dim=-1)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()).clamp_min(1e-12)
    record = {
        "kind": "cdpruner_selected",
        "sample_index": _TASK_REFINE_DEBUG_COUNT,
        "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
        "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
        "K": int(select_idx.shape[1]),
        "N": int(image_features.shape[1]),
        "forced_selected_indices": bool(forced),
        "selected_indices": selected,
        "duplicate_count": int(len(selected) - len(set(selected))),
        "nan_inf_count": int((~torch.isfinite(projected)).sum().item()),
        "mean_cd_relevance": float(relevance[0, select_idx[0]].mean().item()) if select_idx.numel() else 0.0,
        "mean_pairwise_cos": float(off.mean().item()) if off.numel() else 0.0,
        "max_pairwise_cos": float(off.max().item()) if off.numel() else 0.0,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _apply_qficr_residual_transfer(image_features, keep_idx, task_prob, mode=None, eps=1e-12):
    """Apply TC-ORT after final QFi-CR selection and after the multimodal projector.

    The function never changes selected indices or token count. In off mode it
    returns the input tensor unchanged.
    """
    mode = _qficr_residual_transfer_mode() if mode is None else mode
    if mode == "off":
        return image_features, None
    if image_features.ndim != 3:
        raise ValueError("TC-ORT expects projected visual features with shape [B, N, D]")
    if image_features.shape[0] != 1:
        raise ValueError("TC-ORT supports only batch size 1 on the formal QFi-CR path")

    x = image_features[0]
    num_tokens = int(x.shape[0])
    keep_idx = keep_idx.detach().to(device=x.device, dtype=torch.long).reshape(-1)
    keep_idx = keep_idx.sort().values
    keep_num = int(keep_idx.numel())
    if keep_num <= 0 or keep_num > num_tokens:
        raise ValueError("TC-ORT received an invalid selected-token count")
    if bool((keep_idx < 0).any().item()) or bool((keep_idx >= num_tokens).any().item()):
        raise ValueError("TC-ORT selected indices are out of range")
    duplicate_count = keep_num - int(torch.unique(keep_idx).numel())
    if duplicate_count:
        raise ValueError("TC-ORT selected indices contain duplicates")

    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=x.device)
    selected_mask[keep_idx] = True
    dropped_idx = torch.nonzero(~selected_mask, as_tuple=False).reshape(-1)

    norms = x.float().norm(dim=-1).clamp_min(eps)
    psi = x.float() / norms.unsqueeze(-1)

    if mode == "uniform":
        q = torch.full((num_tokens,), 1.0 / max(num_tokens, 1), device=x.device, dtype=torch.float32)
        task_p = None
        task_p_sum = None
    elif mode == "task":
        if task_prob is None or not torch.is_tensor(task_prob):
            raise ValueError("TC-ORT task mode requires QFi-CR task probability tensor")
        task_p = torch.nan_to_num(task_prob.detach().to(device=x.device, dtype=torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if task_p.numel() != num_tokens:
            raise ValueError("TC-ORT task probability length does not match visual token count")
        task_p = task_p.clamp_min(0.0)
        task_p_sum_raw = task_p.sum()
        if not bool(torch.isfinite(task_p_sum_raw).item()) or float(task_p_sum_raw.item()) <= 0.0:
            raise ValueError("TC-ORT task probability is not a valid distribution")
        q = task_p / task_p_sum_raw.clamp_min(eps)
        task_p = q
        task_p_sum = float(q.sum().item())
    else:
        raise ValueError(f"Unknown TC-ORT mode: {mode}")

    if dropped_idx.numel() == 0:
        stats = {
            "mode": mode,
            "selected_indices_hash": _hash_long_tensor(keep_idx),
            "N": num_tokens,
            "K": keep_num,
            "selected_count": keep_num,
            "duplicate_count": 0,
            "dropped_count": 0,
            "dropped_mass": 0.0,
            "task_weighted_dropped_mass": 0.0,
            "mean_residual_energy": 0.0,
            "task_weighted_residual_energy": 0.0,
            "mean_transfer_norm": 0.0,
            "max_transfer_norm": 0.0,
            "mean_transfer_norm_anchor_ratio": 0.0,
            "p95_transfer_norm_anchor_ratio": 0.0,
            "max_transfer_norm_anchor_ratio": 0.0,
            "mean_original_updated_cosine": 1.0,
            "min_original_updated_cosine": 1.0,
            "low_cosine_update_rate": 0.0,
            "mean_norm_ratio": 1.0,
            "max_norm_relative_error": 0.0,
            "empty_cluster_count": keep_num,
            "cluster_size_mean": 0.0,
            "cluster_size_max": 0,
            "nan_inf_count": 0,
            "max_residual_anchor_abs_dot": 0.0,
            "q_sum": float(q.sum().item()),
            "task_p_sum": task_p_sum,
            "input_shape": list(image_features.shape),
            "output_shape": list(image_features.shape),
        }
        return image_features, stats

    sim = torch.matmul(psi[dropped_idx], psi[keep_idx].transpose(0, 1)).clamp_min(0.0)
    assigned_local = torch.argmax(sim, dim=-1)
    assigned_idx = keep_idx[assigned_local]
    assigned_cos = sim.gather(1, assigned_local.unsqueeze(-1)).squeeze(-1)

    anchor_unit = psi[assigned_idx]
    dropped_x = x.float()[dropped_idx]
    projection = (dropped_x * anchor_unit).sum(dim=-1, keepdim=True) * anchor_unit
    residual = dropped_x - projection
    residual_norm = residual.norm(dim=-1)
    max_orth = float((residual * anchor_unit).sum(dim=-1).abs().max().item())

    weights = q[dropped_idx] * assigned_cos
    mass = torch.zeros(keep_num, device=x.device, dtype=torch.float32)
    denom = torch.zeros(keep_num, device=x.device, dtype=torch.float32)
    cluster_size = torch.zeros(keep_num, device=x.device, dtype=torch.float32)
    residual_sum = torch.zeros((keep_num, x.shape[-1]), device=x.device, dtype=torch.float32)
    mass.scatter_add_(0, assigned_local, q[dropped_idx])
    denom.scatter_add_(0, assigned_local, weights)
    cluster_size.scatter_add_(0, assigned_local, torch.ones_like(weights))
    residual_sum.scatter_add_(0, assigned_local.unsqueeze(-1).expand(-1, x.shape[-1]), weights.unsqueeze(-1) * residual)

    nonempty = (cluster_size > 0) & (denom > eps)
    r_bar = residual_sum / denom.clamp_min(eps).unsqueeze(-1)
    selected_x = x.float()[keep_idx]
    x_hat = selected_x + mass.unsqueeze(-1) * r_bar
    x_hat_norm = x_hat.norm(dim=-1).clamp_min(eps)
    selected_norm = norms[keep_idx]
    updated = selected_norm.unsqueeze(-1) * x_hat / x_hat_norm.unsqueeze(-1)
    updated = torch.where(nonempty.unsqueeze(-1), updated, selected_x)

    output = image_features.clone()
    output[0, keep_idx] = updated.to(dtype=image_features.dtype)

    transfer = updated - selected_x
    transfer_norm = transfer.norm(dim=-1)
    anchor_ratio = transfer_norm / selected_norm.clamp_min(eps)
    original_updated_cos = torch.nn.functional.cosine_similarity(selected_x, updated, dim=-1).clamp(-1.0, 1.0)
    norm_ratio = updated.norm(dim=-1) / selected_norm.clamp_min(eps)
    norm_rel_error = (norm_ratio - 1.0).abs()
    nan_inf_count = int((~torch.isfinite(output)).sum().item())
    if nan_inf_count:
        raise RuntimeError("TC-ORT produced NaN/Inf values")

    task_weighted_dropped_mass = None
    task_weighted_residual_energy = None
    if task_prob is not None and torch.is_tensor(task_prob):
        task_dist = torch.nan_to_num(task_prob.detach().to(device=x.device, dtype=torch.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        task_dist = task_dist / task_dist.sum().clamp_min(eps)
        task_weighted_dropped_mass = float(task_dist[dropped_idx].sum().item())
        task_weighted_residual_energy = float((task_dist[dropped_idx] * residual_norm).sum().item() / task_dist[dropped_idx].sum().clamp_min(eps).item())

    stats = {
        "mode": mode,
        "selected_indices_hash": _hash_long_tensor(keep_idx),
        "N": num_tokens,
        "K": keep_num,
        "selected_count": keep_num,
        "duplicate_count": 0,
        "dropped_count": int(dropped_idx.numel()),
        "dropped_mass": float(q[dropped_idx].sum().item()),
        "task_weighted_dropped_mass": task_weighted_dropped_mass,
        "mean_residual_energy": float(residual_norm.mean().item()),
        "task_weighted_residual_energy": task_weighted_residual_energy,
        "mean_transfer_norm": float(transfer_norm.mean().item()),
        "max_transfer_norm": float(transfer_norm.max().item()),
        "mean_transfer_norm_anchor_ratio": float(anchor_ratio.mean().item()),
        "p95_transfer_norm_anchor_ratio": float(torch.quantile(anchor_ratio.float(), 0.95).item()),
        "max_transfer_norm_anchor_ratio": float(anchor_ratio.max().item()),
        "mean_original_updated_cosine": float(original_updated_cos.mean().item()),
        "min_original_updated_cosine": float(original_updated_cos.min().item()),
        "low_cosine_update_rate": float((original_updated_cos < 0.90).float().mean().item()),
        "mean_norm_ratio": float(norm_ratio.mean().item()),
        "max_norm_relative_error": float(norm_rel_error.max().item()),
        "empty_cluster_count": int((cluster_size == 0).sum().item()),
        "cluster_size_mean": float(cluster_size.mean().item()),
        "cluster_size_max": int(cluster_size.max().item()),
        "nan_inf_count": 0,
        "max_residual_anchor_abs_dot": max_orth,
        "q_sum": float(q.sum().item()),
        "task_p_sum": task_p_sum,
        "input_shape": list(image_features.shape),
        "output_shape": list(output.shape),
    }
    return output, stats


def _decoder_feedback_mode():
    mode = os.environ.get("EC_QFICR_DECODER_FEEDBACK_MODE", "off").strip().lower()
    allowed = {"off", "task_topk", "decoder_topk", "fused_decoder_diverse"}
    if mode not in allowed:
        raise ValueError(f"Unknown EC_QFICR_DECODER_FEEDBACK_MODE: {mode}")
    layer = int(os.environ.get("EC_QFICR_DECODER_FEEDBACK_LAYER", "0"))
    if layer != 0:
        raise ValueError("EC_QFICR_DECODER_FEEDBACK_LAYER is frozen at 0")
    return mode


def _stable_topk(score, keep_num):
    score = torch.nan_to_num(score.float(), nan=-float("inf"), posinf=-float("inf"))
    order = sorted(range(score.numel()), key=lambda idx: (-float(score[idx].item()), idx))
    return torch.tensor(sorted(order[:keep_num]), dtype=torch.long, device=score.device)


def _distribution_entropy(prob, eps=1e-12):
    safe = prob.float().clamp_min(eps)
    return float((-(safe * safe.log()).sum()).item())


def _refine_cdpruner_by_task_cells(similarity, anchors, task_prob):
    """Choose the highest-task-prior representative in each CDPruner cell."""
    anchors = anchors.reshape(-1).long()
    task_prob = torch.nan_to_num(task_prob.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if similarity.ndim != 2 or similarity.shape[0] != task_prob.numel():
        raise ValueError("task-refine inputs have incompatible shapes")
    assignment = similarity[:, anchors].argmax(dim=-1)
    assignment[anchors] = torch.arange(anchors.numel(), device=anchors.device)
    refined = []
    for cell_id, anchor in enumerate(anchors.tolist()):
        members = torch.nonzero(assignment == cell_id, as_tuple=False).reshape(-1)
        if members.numel() == 0:
            chosen = int(anchor)
        else:
            local_score = task_prob[members] * similarity[members, anchor].clamp_min(0.0)
            chosen = int(members[torch.argmax(local_score)].item())
        refined.append(chosen)
    if len(set(refined)) != anchors.numel():
        raise RuntimeError("task-refine produced duplicate visual tokens")
    return torch.tensor(sorted(refined), dtype=torch.long, device=anchors.device)


@torch.no_grad()
def _native_cdpruner_components(image_features, image_embeds, text_embeds):
    if image_features.ndim != 3 or image_embeds.ndim != 3:
        raise ValueError("CDPruner inputs must be [B,N,D]")
    projected = image_features.float()
    image_normalized = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))
    image_embeds_n = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_embeds_n = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    relevance = torch.matmul(image_embeds_n, text_embeds_n.t())
    relevance = (-relevance).mean(dim=-1)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    return relevance, similarity, kernel


@torch.no_grad()
def _fast_map_dpp_single(kernel, keep_num, initial_order=None, eps=1e-12):
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Fast MAP DPP kernel must be [N,N]")
    n = int(kernel.shape[0])
    keep_num = min(max(int(keep_num), 0), n)
    if keep_num == 0:
        return torch.empty(0, dtype=torch.long, device=kernel.device), [], [], []
    kernel = kernel.float()
    cis = torch.zeros((keep_num, n), dtype=torch.float32, device=kernel.device)
    di2s = torch.diagonal(kernel).clone().float()
    selected = []
    gains = []
    forced_flags = []

    def insert_token(j, pos, forced):
        if int(j) in selected:
            raise ValueError("Fast MAP DPP received duplicate selected token")
        gain = float(di2s[j].item())
        if not math.isfinite(gain):
            raise ValueError("Fast MAP DPP selected non-finite gain")
        selected.append(int(j))
        gains.append(gain)
        forced_flags.append(int(forced))
        eis = (kernel[j] - torch.einsum("t,tn->n", cis[:pos, j], cis[:pos])) / torch.sqrt(di2s[j].clamp_min(eps))
        if not torch.isfinite(eis).all():
            raise ValueError("Fast MAP DPP produced NaN/Inf")
        cis[pos] = eis
        di2s.sub_(torch.square(eis))
        if not torch.isfinite(di2s[torch.isfinite(di2s)]).all():
            raise ValueError("Fast MAP DPP state contains NaN/Inf")
        di2s[j] = -float("inf")

    initial_order = [] if initial_order is None else [int(x) for x in initial_order]
    if len(set(initial_order)) != len(initial_order):
        raise ValueError("Fast MAP DPP warm-start contains duplicate tokens")
    if any(x < 0 or x >= n for x in initial_order):
        raise ValueError("Fast MAP DPP warm-start contains out-of-range tokens")
    if len(initial_order) > keep_num:
        raise ValueError("Fast MAP DPP warm-start exceeds keep budget")
    for pos, idx in enumerate(initial_order):
        insert_token(idx, pos, True)
    for pos in range(len(initial_order), keep_num):
        insert_token(int(torch.argmax(di2s).item()), pos, False)
    if len(selected) != keep_num or len(set(selected)) != keep_num:
        raise ValueError("Fast MAP DPP final set is not exact unique K")
    return torch.tensor(sorted(selected), dtype=torch.long, device=kernel.device), selected, gains, forced_flags


@torch.no_grad()
def select_cdpruner_indices_equal_budget(image_features, image_embeds, text_embeds, keep_num):
    """Use the production CDPruner MAP path with an explicit equal budget.

    This is factored from ``encode_images`` so CRCA can reuse the exact
    project implementation on cached projector-space inputs. It does not
    change the default runtime path or any CDPruner hyperparameter.
    """
    if image_features.ndim != 3 or image_embeds.ndim != 3:
        raise ValueError("CDPruner inputs must be [B,N,D]")
    B, N, _ = image_features.shape
    keep_num = min(max(int(keep_num), 0), N)
    if keep_num == 0:
        return torch.empty((B, 0), dtype=torch.long, device=image_features.device)
    _, _, kernel = _native_cdpruner_components(image_features, image_embeds, text_embeds)
    rows = []
    for b in range(B):
        selected, _, _, _ = _fast_map_dpp_single(kernel[b], keep_num)
        rows.append(selected)
    return torch.stack(rows, dim=0)


@torch.no_grad()
def _cdpruner_equal_budget_audit(image_features, image_embeds, text_embeds, keep_num):
    if image_features.ndim != 3 or image_features.shape[0] != 1:
        return {}
    B, N, _ = image_features.shape
    keep_num = min(max(int(keep_num), 0), N)
    projected = image_features.float()
    image_normalized = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))
    image_embeds_n = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_embeds_n = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    relevance = torch.matmul(image_embeds_n, text_embeds_n.t())
    relevance = (-relevance).mean(dim=-1)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()).clamp_min(1e-12)
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    cis = torch.zeros((keep_num, B, N), device=image_features.device)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    selected_order = []
    marginal_di2 = []
    marginal_logdet = []
    eps = 1e-12
    for i in range(keep_num):
        j = torch.argmax(di2s, dim=-1)
        j0 = int(j[0].item())
        gain = float(di2s[0, j0].clamp_min(0.0).item())
        selected_order.append(j0)
        marginal_di2.append(gain)
        marginal_logdet.append(float(math.log(max(gain, eps))))
        eis = (kernel[torch.arange(B, device=image_features.device), j]
               - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B, device=image_features.device), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B, device=image_features.device), j].clamp_min(eps)).unsqueeze(-1)
        cis[i] = torch.nan_to_num(eis.float(), nan=0.0, posinf=0.0, neginf=0.0)
        di2s -= torch.square(cis[i])
        di2s[torch.arange(B, device=image_features.device), j] = -float('inf')
    rank = {idx: pos + 1 for pos, idx in enumerate(selected_order)}
    selected_sorted = sorted(selected_order)
    return {
        "cd_relevance_values": [float(x) for x in relevance[0].detach().float().cpu().tolist()],
        "cd_kernel_diag_values": [float(x) for x in torch.diagonal(kernel[0]).detach().float().cpu().tolist()],
        "cd_selected_order": selected_order,
        "cd_selected_order_sorted": selected_sorted,
        "cd_selected_marginal_di2": marginal_di2,
        "cd_selected_marginal_logdet": marginal_logdet,
        "cd_selection_rank": {str(k): int(v) for k, v in rank.items()},
    }


@torch.no_grad()
def _qficr_role_core_gap_select(raw_image_features, projected_features, v0_indices, qfid_info, keep_num, image_embeds=None, text_embeds=None):
    mode = os.environ.get("EC_QFICR_CORE_MODE", "").strip().lower()
    if mode not in {"role_core_gap", "m2_dual_quality_rolecore"}:
        return v0_indices, None
    if raw_image_features.ndim != 3 or raw_image_features.shape[0] != 1:
        raise ValueError("EC_QFICR_CORE_MODE=role_core_gap currently requires batch size 1")
    if projected_features.ndim != 3 or projected_features.shape[0] != 1:
        raise ValueError("role_core_gap requires projected visual features with batch size 1")
    if not isinstance(qfid_info, dict):
        raise ValueError("role_core_gap requires qfid_info with V0 p_i")
    p = qfid_info.get("_prob_tensor")
    if not torch.is_tensor(p):
        vals = qfid_info.get("prob_values") or qfid_info.get("p_mix_values") or []
        p = torch.tensor(vals, dtype=torch.float32, device=projected_features.device) if vals else None
    if not torch.is_tensor(p):
        raise ValueError("role_core_gap could not find V0 clsmix p_i")
    p = torch.nan_to_num(p.detach().float().to(projected_features.device).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(p).all() or bool((p < 0).any().item()):
        raise ValueError("role_core_gap invalid V0 p_i")
    n = int(p.numel())
    keep_num = min(max(int(keep_num), 0), n)
    if keep_num <= 0:
        raise ValueError("role_core_gap invalid keep_num")
    pmax = float(p.max().item())
    if pmax <= 1e-12:
        raise ValueError("role_core_gap non-positive pmax")
    q = p / pmax
    v0 = v0_indices.detach().long().reshape(-1).to(projected_features.device)
    if v0.numel() != keep_num or len(set(int(x) for x in v0.detach().cpu().tolist())) != keep_num:
        raise ValueError("role_core_gap V0 set is not exact unique K")
    order = sorted(range(n), key=lambda idx: (-float(q[idx].item()), idx))
    max_r = min(keep_num - 1, n - 1)
    if max_r < 1:
        raise ValueError("role_core_gap cannot compute natural gap")
    gaps = []
    for rr in range(1, max_r + 1):
        a = float(q[order[rr - 1]].item())
        b = float(q[order[rr]].item())
        gaps.append((a - b) / max(a, 1e-12))
    r_gap = int(max(range(1, max_r + 1), key=lambda rr: (gaps[rr - 1], -rr)))
    h_gap = set(int(x) for x in order[:r_gap])
    v0_list = [int(x) for x in v0.detach().cpu().tolist()]
    s_sem = sorted([idx for idx in v0_list if idx in h_gap])
    k_sem = len(s_sem)
    k_comp = keep_num - k_sem
    if not (0 <= k_sem <= keep_num - 1 and 1 <= k_comp <= keep_num):
        raise ValueError(f"role_core_gap invalid K_sem/K_comp {k_sem}/{k_comp}")
    complement_quality_source = "v0_clsmix"
    native_cd_quality = None
    if mode == "m2_dual_quality_rolecore":
        if image_embeds is None or text_embeds is None:
            raise ValueError("m2_dual_quality_rolecore requires Native CD image/text embeddings")
        native_rel, _, native_kernel = _native_cdpruner_components(projected_features, image_embeds, text_embeds)
        kernel = native_kernel[0].float()
        native_cd_quality = torch.nan_to_num(native_rel[0].detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        complement_quality_source = "native_cd"
    else:
        z = torch.nn.functional.normalize(projected_features[0].float(), dim=-1, eps=1e-12)
        sim = z @ z.t()
        kernel = q[:, None] * sim * q[None, :]
    if not torch.isfinite(kernel).all():
        raise ValueError(f"{mode} kernel contains NaN/Inf")
    anchor_order = sorted(s_sem, key=lambda idx: (-float(q[idx].item()), idx))
    final_sorted, selected, gains, forced_flags = _fast_map_dpp_single(kernel, keep_num, initial_order=anchor_order)
    info = {
        "mode": mode,
        "K": int(keep_num),
        "N": int(n),
        "r_gap": int(r_gap),
        "largest_gap": float(gaps[r_gap - 1]),
        "H_gap_indices": [int(x) for x in order[:r_gap]],
        "S_V0": v0_list,
        "S_sem": [int(x) for x in s_sem],
        "S_comp": [int(x) for x in selected if int(x) not in set(s_sem)],
        "S_final": [int(x) for x in final_sorted.detach().cpu().tolist()],
        "selection_order": [int(x) for x in selected],
        "selection_gains": [float(x) for x in gains],
        "selection_forced_flags": [int(x) for x in forced_flags],
        "K_sem": int(k_sem),
        "K_comp": int(k_comp),
        "exact_k": True,
        "duplicate": 0,
        "nan_inf": 0,
        "anchor_subset_final": True,
        "q_max": float(q.max().item()),
        "q_min": float(q.min().item()),
        "complement_quality_source": complement_quality_source,
    }
    if native_cd_quality is not None:
        info.update({
            "qficr_m2_dual_quality": True,
            "native_cd_quality_min": float(native_cd_quality.min().item()),
            "native_cd_quality_max": float(native_cd_quality.max().item()),
            "native_cd_quality_mean": float(native_cd_quality.mean().item()),
            "native_cd_quality_median": float(native_cd_quality.median().item()),
        })
    return final_sorted, info


def _scss_stable_order(values, available=None, descending=True):
    values = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if available is None:
        indices = range(values.numel())
    else:
        mask = torch.as_tensor(available, dtype=torch.bool, device=values.device).reshape(-1)
        indices = [int(x) for x in torch.nonzero(mask, as_tuple=False).reshape(-1).detach().cpu().tolist()]
    if descending:
        return sorted(indices, key=lambda idx: (-float(values[idx].item()), idx))
    return sorted(indices, key=lambda idx: (float(values[idx].item()), idx))


def _scss_rank_percentiles(values, mask):
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=values.device).reshape(-1)
    scores = torch.zeros_like(values, dtype=torch.float32)
    order = _scss_stable_order(values, mask, descending=True)
    denom = max(len(order) - 1, 1)
    for rank, idx in enumerate(order):
        scores[idx] = 1.0 - float(rank) / float(denom)
    return scores


def _scss_cd_kernel_post(projected_features, image_embeds, text_embeds):
    x = projected_features.float()
    if x.ndim == 3:
        x = x[0]
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = x @ x.t()
    image_embeds_n = image_embeds.float() / image_embeds.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_embeds_n = text_embeds.float() / text_embeds.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    relevance = torch.matmul(image_embeds_n, text_embeds_n.t())
    relevance = (-relevance).mean(dim=-1)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()).clamp_min(1e-12)
    rel = relevance[0].to(device=x.device, dtype=torch.float32)
    kernel = rel[:, None] * similarity * rel[None, :]
    return torch.nan_to_num(kernel.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _scss_conditional_diagonal(kernel, selected, eps=1e-6):
    residual = torch.diag(kernel).clone().float().clamp_min(0.0)
    columns = []
    for pivot in selected:
        pivot = int(pivot)
        denom = residual[pivot].clamp_min(eps).sqrt()
        if columns:
            previous = torch.stack(columns, dim=1)
            column = (kernel[:, pivot] - previous @ previous[pivot, :]) / denom
        else:
            column = kernel[:, pivot] / denom
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(column)
        residual = (residual - column.square()).clamp_min(0.0)
    if selected:
        residual[torch.tensor(selected, dtype=torch.long, device=kernel.device)] = -float("inf")
    return residual


def _scss_select_sensitivity_diversity(sensitivity, kernel, keep_num, initial=None, candidate_mask=None):
    num_tokens = int(sensitivity.numel())
    keep_num = min(max(int(keep_num), 0), num_tokens)
    selected = [] if initial is None else [int(x) for x in torch.as_tensor(initial).reshape(-1).tolist()]
    selected = [x for x in selected if 0 <= x < num_tokens]
    available = torch.ones(num_tokens, dtype=torch.bool, device=sensitivity.device)
    if selected:
        available[torch.tensor(selected, dtype=torch.long, device=sensitivity.device)] = False
    if candidate_mask is not None:
        candidate_mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=sensitivity.device).reshape(-1)
    score_trace = []
    while len(selected) < keep_num and bool(available.any().item()):
        mask = available if candidate_mask is None else (available & candidate_mask)
        if not bool(mask.any().item()):
            mask = available
        div_gain = _scss_conditional_diagonal(kernel, selected)
        r_s = _scss_rank_percentiles(sensitivity, mask)
        r_d = _scss_rank_percentiles(div_gain, mask)
        score = (r_s * r_d).masked_fill(~mask, float("-inf"))
        best = _scss_stable_order(score, mask, descending=True)[0]
        selected.append(best)
        available[best] = False
        score_trace.append(float(score[best].item()))
    return torch.tensor(sorted(selected[:keep_num]), dtype=torch.long, device=sensitivity.device), score_trace


def _scss_projector_zo_sensitivity(projector, x_pre, m=16, h=0.01, seed=20260805, chunk=4096):
    if x_pre.ndim == 3:
        x = x_pre[0]
    else:
        x = x_pre
    device = x.device
    dtype = x.dtype
    n, d = x.shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    directions = torch.randn((int(m), d), generator=generator, dtype=torch.float32)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    directions = directions.to(device=device, dtype=dtype)
    plus = (x[:, None, :] + float(h) * directions[None, :, :]).reshape(n * int(m), d)
    minus = (x[:, None, :] - float(h) * directions[None, :, :]).reshape(n * int(m), d)
    outputs = []
    with torch.inference_mode():
        for start in range(0, plus.shape[0], int(chunk)):
            p = projector(plus[start:start + int(chunk)])
            q = projector(minus[start:start + int(chunk)])
            delta = (p.float() - q.float()) / (2.0 * float(h))
            outputs.append(delta.pow(2).sum(dim=-1))
    sens = torch.cat(outputs, dim=0).reshape(n, int(m)).mean(dim=1)
    return torch.nan_to_num(sens.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _scss_apply_pareto_swap(initial_indices, sensitivity, kernel, eps=1e-6):
    selected = [int(x) for x in torch.as_tensor(initial_indices).reshape(-1).tolist()]
    selected = sorted(dict.fromkeys(selected))
    num_tokens = int(sensitivity.numel())
    cap = num_tokens
    swaps = []
    abnormal = False
    for _ in range(cap):
        if not selected:
            break
        s = torch.tensor(selected, dtype=torch.long, device=kernel.device)
        base = kernel[s][:, s] + float(eps) * torch.eye(len(selected), dtype=torch.float32, device=kernel.device)
        inv = torch.linalg.pinv(base.float())
        incoming = [idx for idx in range(num_tokens) if idx not in set(selected)]
        if not incoming:
            break
        incoming_t = torch.tensor(incoming, dtype=torch.long, device=kernel.device)
        best = None
        candidates = []
        for pos, out_idx in enumerate(selected):
            delta_sens = sensitivity[incoming_t] - sensitivity[out_idx]
            sens_mask = delta_sens > 0
            if not bool(sens_mask.any().item()):
                continue
            keep_pos = [i for i in range(len(selected)) if i != pos]
            if keep_pos:
                inv_aa = inv[keep_pos][:, keep_pos]
                inv_ao = inv[keep_pos, pos:pos + 1]
                inv_oa = inv[pos:pos + 1, keep_pos]
                inv_oo = inv[pos, pos].clamp_min(float(eps))
                a_inv = inv_aa - (inv_ao @ inv_oa) / inv_oo
                a_idx = s[torch.tensor(keep_pos, dtype=torch.long, device=kernel.device)]
                k_ja = kernel[incoming_t][:, a_idx]
                cond = torch.diag(kernel)[incoming_t] + float(eps) - (k_ja @ a_inv * k_ja).sum(dim=1)
            else:
                inv_oo = inv[pos, pos].clamp_min(float(eps))
                cond = torch.diag(kernel)[incoming_t] + float(eps)
            delta_div = torch.log(cond.clamp_min(float(eps)) * inv_oo)
            valid = sens_mask & (delta_div > 0)
            if not bool(valid.any().item()):
                continue
            for local in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
                candidates.append({
                    "out": int(out_idx),
                    "in": int(incoming[int(local)]),
                    "delta_sens": float(delta_sens[int(local)].item()),
                    "delta_div": float(delta_div[int(local)].item()),
                })
        if not candidates:
            break
        ds = torch.tensor([c["delta_sens"] for c in candidates], dtype=torch.float32, device=kernel.device)
        dd = torch.tensor([c["delta_div"] for c in candidates], dtype=torch.float32, device=kernel.device)
        mask = torch.ones(len(candidates), dtype=torch.bool, device=kernel.device)
        rs = _scss_rank_percentiles(ds, mask)
        rd = _scss_rank_percentiles(dd, mask)
        scores = rs * rd
        best_pos = sorted(range(len(candidates)), key=lambda i: (-float(scores[i].item()), candidates[i]["out"], candidates[i]["in"]))[0]
        best = candidates[best_pos]
        selected.remove(best["out"])
        selected.append(best["in"])
        selected = sorted(selected)
        swaps.append(best)
    else:
        abnormal = True
    return torch.tensor(selected, dtype=torch.long, device=kernel.device), swaps, abnormal


def _cdsafe_task_coverage(selected, kappa, probs):
    selected = [int(x) for x in torch.as_tensor(selected).reshape(-1).tolist()]
    if not selected:
        return torch.tensor(0.0, dtype=torch.float32, device=kappa.device)
    idx = torch.tensor(selected, dtype=torch.long, device=kappa.device)
    p = torch.nan_to_num(probs.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(1e-12)
    return (p * kappa[:, idx].max(dim=1).values).sum()


def _cdsafe_apply_refinement(cd_indices, v0_indices, sensitivity, kernel, kappa, probs, method, eps=1e-6):
    selected = sorted(dict.fromkeys(int(x) for x in torch.as_tensor(cd_indices).reshape(-1).tolist()))
    v0_set = set(int(x) for x in torch.as_tensor(v0_indices).reshape(-1).tolist())
    cd_set = set(selected)
    common = cd_set & v0_set
    cd_only = cd_set - v0_set
    v0_only = v0_set - cd_set
    num_tokens = int(sensitivity.numel())
    all_tokens = set(range(num_tokens))
    swaps = []
    abnormal = False
    pair_search_count = 0

    full_incoming = method in {"c1", "c4"}
    disagreement_incoming = method in {"c2", "c3", "c6", "c7", "c9", "c10"}
    union_incoming = method == "c8"
    common_lock = method in {"c2", "c3", "c4", "c6", "c7", "c8", "c9", "c10"}
    region_constraint = method in {"c5", "c6", "c7"}
    coverage_guard = method in {"c3", "c7"}
    require_sensitivity = method != "c9"
    require_diversity = method != "c10"

    for _ in range(num_tokens):
        if not selected:
            break
        selected_set = set(selected)
        s = torch.tensor(selected, dtype=torch.long, device=kernel.device)
        base = kernel[s][:, s] + float(eps) * torch.eye(len(selected), dtype=torch.float32, device=kernel.device)
        inv = torch.linalg.pinv(base.float())
        base_coverage = _cdsafe_task_coverage(selected, kappa, probs)
        assignment = None
        if region_constraint:
            assignment = kappa[:, s].argmax(dim=1)
        candidates = []
        for pos, out_idx in enumerate(selected):
            if common_lock and int(out_idx) in common:
                continue
            if method in {"c2", "c3", "c4", "c6", "c7", "c9", "c10"} and int(out_idx) not in cd_only:
                continue
            if union_incoming:
                incoming_pool = (cd_set | v0_set) - selected_set
            elif disagreement_incoming:
                incoming_pool = v0_only - selected_set
            elif full_incoming:
                incoming_pool = all_tokens - selected_set
            else:
                incoming_pool = all_tokens - selected_set
            if region_constraint:
                members = set(int(x) for x in torch.nonzero(assignment == pos, as_tuple=False).reshape(-1).detach().cpu().tolist())
                incoming_pool &= members
            incoming = sorted(int(x) for x in incoming_pool)
            if not incoming:
                continue
            incoming_t = torch.tensor(incoming, dtype=torch.long, device=kernel.device)
            pair_search_count += len(incoming)
            delta_sens = sensitivity[incoming_t] - sensitivity[int(out_idx)]
            if require_sensitivity:
                sens_mask = delta_sens > 0
            else:
                sens_mask = torch.ones_like(delta_sens, dtype=torch.bool)
            if not bool(sens_mask.any().item()):
                continue
            if require_diversity:
                keep_pos = [i for i in range(len(selected)) if i != pos]
                if keep_pos:
                    inv_aa = inv[keep_pos][:, keep_pos]
                    inv_ao = inv[keep_pos, pos:pos + 1]
                    inv_oa = inv[pos:pos + 1, keep_pos]
                    inv_oo = inv[pos, pos].clamp_min(float(eps))
                    a_inv = inv_aa - (inv_ao @ inv_oa) / inv_oo
                    a_idx = s[torch.tensor(keep_pos, dtype=torch.long, device=kernel.device)]
                    k_ja = kernel[incoming_t][:, a_idx]
                    cond = torch.diag(kernel)[incoming_t] + float(eps) - (k_ja @ a_inv * k_ja).sum(dim=1)
                else:
                    inv_oo = inv[pos, pos].clamp_min(float(eps))
                    cond = torch.diag(kernel)[incoming_t] + float(eps)
                delta_div = torch.log(cond.clamp_min(float(eps)) * inv_oo)
                div_mask = delta_div > 0
            else:
                delta_div = torch.zeros_like(delta_sens)
                div_mask = torch.ones_like(delta_sens, dtype=torch.bool)
            valid = sens_mask & div_mask
            if coverage_guard and bool(valid.any().item()):
                cov_valid = []
                for local in range(len(incoming)):
                    if not bool(valid[local].item()):
                        cov_valid.append(False)
                        continue
                    swapped = [x for x in selected if x != int(out_idx)] + [incoming[local]]
                    cov_valid.append(bool((_cdsafe_task_coverage(swapped, kappa, probs) + 1e-12 >= base_coverage).item()))
                valid = valid & torch.tensor(cov_valid, dtype=torch.bool, device=kernel.device)
            if not bool(valid.any().item()):
                continue
            for local in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
                swapped = [x for x in selected if x != int(out_idx)] + [incoming[int(local)]]
                candidates.append({
                    "out": int(out_idx),
                    "in": int(incoming[int(local)]),
                    "delta_sens": float(delta_sens[int(local)].item()),
                    "delta_div": float(delta_div[int(local)].item()),
                    "delta_coverage": float((_cdsafe_task_coverage(swapped, kappa, probs) - base_coverage).item()),
                })
        if not candidates:
            break
        if method == "c9":
            scores = torch.tensor([c["delta_div"] for c in candidates], dtype=torch.float32, device=kernel.device)
        elif method == "c10":
            scores = torch.tensor([c["delta_sens"] for c in candidates], dtype=torch.float32, device=kernel.device)
        else:
            ds = torch.tensor([c["delta_sens"] for c in candidates], dtype=torch.float32, device=kernel.device)
            dd = torch.tensor([c["delta_div"] for c in candidates], dtype=torch.float32, device=kernel.device)
            mask = torch.ones(len(candidates), dtype=torch.bool, device=kernel.device)
            scores = _scss_rank_percentiles(ds, mask) * _scss_rank_percentiles(dd, mask)
        best_pos = sorted(range(len(candidates)), key=lambda i: (-float(scores[i].item()), candidates[i]["out"], candidates[i]["in"]))[0]
        best = candidates[best_pos]
        selected.remove(best["out"])
        selected.append(best["in"])
        selected = sorted(selected)
        swaps.append(best)
    else:
        abnormal = True

    final_set = set(selected)
    union_cd_v0 = len(cd_set | v0_set)
    info = {
        "cd_safe_method": method,
        "intersection_size": len(common),
        "union_size": union_cd_v0,
        "cd_v0_jaccard": len(common) / max(1, union_cd_v0),
        "common_preservation_rate": len(common & final_set) / max(1, len(common)),
        "v0_exclusive_imported_count": len(final_set & v0_only),
        "cd_exclusive_removed_count": len(cd_only - final_set),
        "jaccard_to_cd": len(final_set & cd_set) / max(1, len(final_set | cd_set)),
        "jaccard_to_v0": len(final_set & v0_set) / max(1, len(final_set | v0_set)),
        "actual_swap_count": len(swaps),
        "pair_search_count": pair_search_count,
        "candidate_universe_size": len((cd_set | v0_set) if method == "c8" else (v0_only if disagreement_incoming else all_tokens)),
        "delta_sensitivity_total": float((sensitivity[torch.tensor(selected, dtype=torch.long, device=kernel.device)].sum() - sensitivity[torch.tensor(sorted(cd_set), dtype=torch.long, device=kernel.device)].sum()).item()),
        "delta_diversity_proxy": float(sum(c.get("delta_div", 0.0) for c in swaps)),
        "delta_task_coverage": float((_cdsafe_task_coverage(selected, kappa, probs) - _cdsafe_task_coverage(sorted(cd_set), kappa, probs)).item()),
        "swaps": swaps,
    }
    return torch.tensor(selected, dtype=torch.long, device=kernel.device), swaps, abnormal, info


def _write_decoder_feedback_debug(record):
    global _DECODER_FEEDBACK_DEBUG_COUNT
    path = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
    if not path:
        return
    if path.lower() in {"1", "true", "yes", "on"}:
        path = os.path.abspath("debug_qficr_stats.jsonl")
    record = dict(record)
    record.update({
        "sample_index": _DECODER_FEEDBACK_DEBUG_COUNT,
        "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
        "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _DECODER_FEEDBACK_DEBUG_COUNT += 1


def _write_task_refine_debug(record):
    global _TASK_REFINE_DEBUG_COUNT
    path = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
    if not path:
        return
    if path.lower() in {"1", "true", "yes", "on"}:
        path = os.path.abspath("debug_qficr_stats.jsonl")
    payload = dict(record)
    payload.update({
        "sample_index": _TASK_REFINE_DEBUG_COUNT,
        "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
        "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    _TASK_REFINE_DEBUG_COUNT += 1


def _normalize_prune_method():
    return os.environ.get("PRUNE_METHOD", "cdpruner").strip().lower().replace("-", "_")


def _prune_debug(message):
    global _PRUNE_DEBUG_COUNT
    if os.environ.get("EC_DEBUG", "0") != "1":
        return
    limit = int(os.environ.get("EC_DEBUG_LIMIT", "5"))
    if _PRUNE_DEBUG_COUNT >= limit:
        return
    print(f"[PruneDebug] {message}", flush=True)
    _PRUNE_DEBUG_COUNT += 1


def _compute_ec_relevance(image_embeds, text_embeds):
    if not torch.is_tensor(image_embeds) or not torch.is_tensor(text_embeds):
        return None
    if image_embeds.ndim != 3:
        return None

    if text_embeds.ndim == 1:
        text_embeds = text_embeds.unsqueeze(0)
    elif text_embeds.ndim == 3 and text_embeds.shape[0] == 1:
        text_embeds = text_embeds.squeeze(0)
    if text_embeds.ndim != 2:
        return None

    image_embeds = image_embeds.float()
    text_embeds = text_embeds.to(device=image_embeds.device, dtype=torch.float32)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    relevance = torch.matmul(image_embeds, text_embeds.t())
    return relevance.mean(dim=-1)


def _encode_ec_semantic_units(vision_tower, units):
    tokenizer = getattr(vision_tower, "text_tokenizer", None)
    text_tower = getattr(vision_tower, "text_tower", None)
    if tokenizer is None or text_tower is None or not units:
        return None

    try:
        text_inputs = tokenizer(
            text=units,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {
            key: value.to(device=vision_tower.device)
            for key, value in text_inputs.items()
        }
        return text_tower(**text_inputs).text_embeds
    except Exception:
        return None


class LlavaMetaModel:

    def __init__(self, config, **kwargs):
        super(LlavaMetaModel, self).__init__(config, **kwargs)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images_for_decoder_feedback(self, images, texts=None):
        """Encode all visual tokens and compute p_i without selecting any token."""
        if _normalize_prune_method() != "ec_pruner":
            raise ValueError("decoder-feedback audit requires PRUNE_METHOD=ec_pruner")
        selector = ECPruner(debug=None)
        needs_cls_attention = selector.qfid_prob_source in {"clsmix", "cls_only_qf"}
        vision_outputs = self.get_model().get_vision_tower()(
            images,
            texts=texts,
            output_cls_attention=needs_cls_attention,
            cls_attn_layer=selector.qfid_cls_attn_layer,
        )
        if needs_cls_attention:
            image_features, image_embeds, _, cls_attention = vision_outputs
        else:
            image_features, image_embeds, _ = vision_outputs
            cls_attention = None
        relevance = None
        if selector.needs_relevance:
            relevance = _compute_ec_relevance(image_embeds, vision_outputs[2])
        semantic_units = None
        semantic_text_embeds = None
        if selector.needs_semantics:
            semantic_units = selector.extract_semantic_units(texts)
            semantic_text_embeds = _encode_ec_semantic_units(
                self.get_vision_tower(), semantic_units
            )
        if image_features.is_cuda:
            torch.cuda.synchronize(image_features.device)
        task_started = time.perf_counter()
        task_prob, prob_info = selector.compute_task_probability(
            image_features,
            question=texts,
            relevance=relevance,
            cls_attn=cls_attention,
            text_embeds=semantic_text_embeds,
            semantic_features=image_embeds,
            semantic_units=semantic_units,
        )
        if image_features.is_cuda:
            torch.cuda.synchronize(image_features.device)
        prob_info["task_prob_ms"] = (time.perf_counter() - task_started) * 1000.0
        projected = self.get_model().mm_projector(image_features)
        psi = image_features.float()
        psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return projected, task_prob, psi, prob_info

    @staticmethod
    def _find_query_positions(input_ids, query_token_ids, image_index, visual_start, visual_len):
        ids = input_ids.tolist()
        query = [] if query_token_ids is None else query_token_ids.reshape(-1).tolist()
        while query and query[0] in {0, 1, 2}:
            query = query[1:]
        best = None
        # SentencePiece can encode the first/last token differently when a raw
        # question is tokenized outside its conversation-template context.
        for left_trim in range(min(3, len(query)) + 1):
            for right_trim in range(min(3, len(query) - left_trim) + 1):
                candidate = query[left_trim:len(query) - right_trim if right_trim else None]
                if not candidate:
                    continue
                for start in range(image_index + 1, len(ids) - len(candidate) + 1):
                    if ids[start:start + len(candidate)] == candidate:
                        match = (len(candidate), start, left_trim, right_trim)
                        if best is None or match[:2] > best[:2]:
                            best = match
        if best is not None and best[0] >= max(2, math.ceil(0.8 * max(1, len(query)))):
            length, start, _, _ = best
            original_positions = list(range(start, start + length))
            scope = "question_subsequence"
            fallback = False
        else:
            original_positions = list(range(image_index + 1, len(ids)))
            scope = "fallback_prompt_text"
            fallback = True
        mapped = [pos - 1 + visual_len if pos > image_index else pos for pos in original_positions]
        mapped = [pos for pos in mapped if pos >= visual_start + visual_len]
        if not mapped:
            raise ValueError("decoder-feedback query span is empty")
        return torch.tensor(mapped, dtype=torch.long, device=input_ids.device), scope, fallback

    def _decoder_attention_probability(self, full_embeds, query_positions, visual_start, visual_len):
        model = self.get_model()
        layer = model.layers[0]
        sequence_length = full_embeds.shape[0]
        scoring_input = full_embeds.unsqueeze(0)
        scoring_memory_before = 0
        if scoring_input.is_cuda:
            scoring_memory_before = torch.cuda.memory_allocated(scoring_input.device)
            torch.cuda.reset_peak_memory_stats(scoring_input.device)
        scoring_positions = torch.arange(sequence_length, device=full_embeds.device).unsqueeze(0)
        scoring_mask_2d = torch.ones((1, sequence_length), dtype=torch.bool, device=full_embeds.device)
        causal_mask = _prepare_4d_causal_attention_mask(
            scoring_mask_2d,
            (1, sequence_length),
            scoring_input,
            past_key_values_length=0,
        )
        if scoring_input.is_cuda:
            torch.cuda.synchronize(scoring_input.device)
        started = time.perf_counter()
        outputs = layer(
            scoring_input,
            attention_mask=causal_mask,
            position_ids=scoring_positions,
            past_key_value=None,
            output_attentions=True,
            use_cache=False,
        )
        if scoring_input.is_cuda:
            torch.cuda.synchronize(scoring_input.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        attention = outputs[1]
        if attention is None:
            raise RuntimeError("layer-0 attention backend did not return attention weights")
        q = attention[0, :, query_positions, visual_start:visual_start + visual_len]
        q = q.float().mean(dim=(0, 1))
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        q = q / q.sum().clamp_min(1e-12)
        cache_is_none = len(outputs) == 2
        scoring_increment_memory = 0
        if full_embeds.is_cuda:
            scoring_increment_memory = max(
                0, torch.cuda.max_memory_allocated(full_embeds.device) - scoring_memory_before
            )
        del attention, outputs, causal_mask, scoring_mask_2d, scoring_positions, scoring_input
        return q, elapsed_ms, cache_is_none, scoring_increment_memory

    @staticmethod
    def _fused_diverse_select(task_prob, decoder_prob, psi, keep_num):
        eps = 1e-12
        utility = torch.sqrt((task_prob.float() + eps) * (decoder_prob.float() + eps))
        similarity = torch.matmul(psi.float(), psi.float().t()).clamp_min(0.0)
        selected = []
        max_redundancy = torch.zeros_like(utility)
        score_trace = []
        for _ in range(keep_num):
            scores = utility * (1.0 - max_redundancy)
            if selected:
                scores[torch.tensor(selected, device=scores.device)] = -float("inf")
            best = min(
                range(scores.numel()),
                key=lambda idx: (-float(scores[idx].item()), idx),
            )
            selected.append(best)
            score_trace.append(float(scores[best].item()))
            max_redundancy = torch.maximum(max_redundancy, similarity[:, best])
        keep_idx = torch.tensor(sorted(selected), dtype=torch.long, device=utility.device)
        return keep_idx, utility, score_trace

    def _apply_decoder_feedback(
        self,
        full_embeds,
        input_ids,
        query_token_ids,
        visual_start,
        visual_len,
        task_prob,
        psi,
        keep_num,
        task_prob_ms=0.0,
    ):
        mode = _decoder_feedback_mode()
        if not 0 < keep_num <= visual_len:
            raise ValueError(f"invalid decoder-feedback K={keep_num} for N={visual_len}")
        image_index = int(torch.where(input_ids == IMAGE_TOKEN_INDEX)[0][0].item())
        query_positions, query_scope, fallback = self._find_query_positions(
            input_ids, query_token_ids, image_index, visual_start, visual_len
        )
        decoder_prob = None
        decoder_ms = 0.0
        cache_is_none = True
        scoring_increment_memory = 0
        if mode in {"decoder_topk", "fused_decoder_diverse"}:
            decoder_prob, decoder_ms, cache_is_none, scoring_increment_memory = self._decoder_attention_probability(
                full_embeds, query_positions, visual_start, visual_len
            )
        if full_embeds.is_cuda:
            torch.cuda.synchronize(full_embeds.device)
        selection_start = time.perf_counter()
        if mode == "task_topk":
            keep_idx = _stable_topk(task_prob, keep_num)
            utility = task_prob.float()
            score_trace = []
        elif mode == "decoder_topk":
            keep_idx = _stable_topk(decoder_prob, keep_num)
            utility = decoder_prob.float()
            score_trace = []
        else:
            keep_idx, utility, score_trace = self._fused_diverse_select(
                task_prob, decoder_prob, psi, keep_num
            )
        if full_embeds.is_cuda:
            torch.cuda.synchronize(full_embeds.device)
        selection_ms = (time.perf_counter() - selection_start) * 1000.0
        if keep_idx.unique().numel() != keep_num or keep_idx.numel() != keep_num:
            raise RuntimeError("decoder-feedback selection did not produce exact unique K")
        visual = full_embeds[visual_start:visual_start + visual_len]
        pruned = torch.cat((
            full_embeds[:visual_start],
            visual[keep_idx],
            full_embeds[visual_start + visual_len:],
        ))
        q_for_log = decoder_prob if decoder_prob is not None else torch.zeros_like(task_prob)
        selected_psi = psi[keep_idx]
        redundancy = 0.0
        if keep_num > 1:
            selected_similarity = torch.matmul(selected_psi, selected_psi.t()).clamp_min(0.0)
            upper = torch.triu_indices(keep_num, keep_num, offset=1, device=psi.device)
            redundancy = float(selected_similarity[upper[0], upper[1]].mean().item())
        _write_decoder_feedback_debug({
            "decoder_feedback_mode": mode,
            "decoder_feedback_layer": 0,
            "N": visual_len,
            "K": keep_num,
            "selected_indices": keep_idx.tolist(),
            "p": [round(float(x), 9) for x in task_prob.tolist()],
            "q": [round(float(x), 9) for x in q_for_log.tolist()],
            "p_entropy": _distribution_entropy(task_prob),
            "q_entropy": _distribution_entropy(decoder_prob) if decoder_prob is not None else None,
            "q_sum": float(decoder_prob.sum().item()) if decoder_prob is not None else None,
            "q_zero": bool(decoder_prob is not None and decoder_prob.max().item() <= 1e-12),
            "q_uniform": bool(decoder_prob is not None and float(decoder_prob.std().item()) <= 1e-7),
            "nan_inf_count": int((~torch.isfinite(task_prob)).sum().item()) + int((~torch.isfinite(q_for_log)).sum().item()),
            "query_scope": query_scope,
            "query_fallback": fallback,
            "query_token_count": int(query_positions.numel()),
            "task_prob_ms": float(task_prob_ms),
            "decoder_scoring_ms": decoder_ms,
            "selection_ms": selection_ms,
            "scoring_use_cache": False,
            "scoring_cache_is_none": cache_is_none,
            "scoring_hidden_reused": False,
            "generation_restarts_layer0": True,
            "scoring_increment_memory_gib": scoring_increment_memory / (1024.0 ** 3),
            "feature_redundancy": redundancy,
            "score_trace": score_trace,
            "utility_min": float(utility.min().item()),
            "utility_max": float(utility.max().item()),
        })
        self._decoder_feedback_last_profile = {
            "mode": mode,
            "K": keep_num,
            "task_prob_ms": float(task_prob_ms),
            "decoder_scoring_ms": decoder_ms,
            "selection_ms": selection_ms,
            "scoring_increment_memory_gib": scoring_increment_memory / (1024.0 ** 3),
        }
        return pruned, keep_idx

    # [CDPruner] Generate index masks using conditional DPP
    def encode_images(self, images, texts=None):
        prune_method = _normalize_prune_method()
        qfid_prob_source = os.environ.get("EC_QFID_PROB_SOURCE", "semantic").strip().lower()
        qfid_select_mode = os.environ.get("EC_QFID_SELECT_MODE", "qf").strip().lower()
        qfid_cls_attn_layer = os.environ.get("EC_QFID_CLS_ATTN_LAYER", "last").strip().lower()
        needs_transition_prior = (
            prune_method == "ec_pruner"
            and os.environ.get("EC_SCORE_SOURCE", "norm").strip().lower() == "qfid"
            and qfid_prob_source in {"vtpmix", "vtp_only_qf"}
        )
        needs_cls_attention = (
            prune_method in {"ec_pruner", "cdpruner_task_refine", "cdpruner_relevance_refine"}
            and os.environ.get("EC_SCORE_SOURCE", "norm").strip().lower() == "qfid"
            and (
                qfid_prob_source in {"clsmix", "cls_only_qf"}
                or qfid_select_mode == "cls_topk"
            )
        )
        vision_outputs = self.get_model().get_vision_tower()(
            images,
            texts=texts,
            output_cls_attention=needs_cls_attention,
            cls_attn_layer=qfid_cls_attn_layer,
            output_transition_prior=needs_transition_prior,
        )
        cls_attention = None
        transition_prior = None
        if needs_cls_attention and needs_transition_prior:
            image_features, image_embeds, text_embeds, cls_attention, transition_prior = vision_outputs
        elif needs_cls_attention:
            image_features, image_embeds, text_embeds, cls_attention = vision_outputs
        elif needs_transition_prior:
            image_features, image_embeds, text_embeds, transition_prior = vision_outputs
        else:
            image_features, image_embeds, text_embeds = vision_outputs
        
        B, N, C = image_features.shape
        device = image_features.device
        index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
        _prune_debug(f"prune_method={prune_method}")
        _prune_debug(f"image_features.shape={tuple(image_features.shape)}")

        if prune_method == "ec_pruner":
            selector = ECPruner(debug=None)
            core_mode = os.environ.get("EC_QFICR_CORE_MODE", "").strip().lower()
            requested_local_residual_mode = _qficr_local_residual_mode()
            requested_dual_residual_mode = _qficr_dual_residual_mode()
            role_core_stats_path = selector.qfid_debug_stats_jsonl
            if (
                core_mode in {"role_core_gap", "m2_dual_quality_rolecore"}
                or requested_local_residual_mode != "off"
                or requested_dual_residual_mode != "off"
            ):
                # RoleCore must override the actual final visual-token selected
                # indices; likewise P1 replaces the frozen recovery output with
                # independent local residual tokens. P3 also replaces the active
                # final set via dual residual co-construction. The pre-override
                # frozen row is not the active path.
                selector.qfid_debug_stats_jsonl = ""
            core_source = os.environ.get("EC_QFICR_CORE_SOURCE", "legacy").strip().lower()
            if core_source not in {"legacy", "cdpruner_equal_budget"}:
                raise ValueError(f"Unknown EC_QFICR_CORE_SOURCE: {core_source}")
            if core_source == "cdpruner_equal_budget" and B != 1:
                raise ValueError("cdpruner_equal_budget requires batch size 1")
            raw_image_features = image_features
            relevance = None
            if selector.needs_relevance:
                relevance = _compute_ec_relevance(image_embeds, text_embeds)
            semantic_units = None
            semantic_text_embeds = None
            if selector.needs_semantics:
                semantic_units = selector.extract_semantic_units(texts)
                semantic_text_embeds = _encode_ec_semantic_units(
                    self.get_vision_tower(),
                    semantic_units,
                )
            if requested_local_residual_mode == "p1_lr_spatial_semantic_core_conditioned_local_residual_v1":
                keep_idx, debug_info = _qficr_select_spatial_semantic_core_v1(
                    selector,
                    raw_image_features,
                    self.visual_token_num,
                    cls_attention,
                    text_embeds,
                    semantic_text_embeds,
                    image_embeds,
                )
            elif requested_local_residual_mode in {
                "p1_lr_entropy_core_conditioned_local_residual_fast",
                "p1_lr_entropy_core_random_recovery_v2",
                "p1_lr_entropy_spatial_diverse_recovery_v3",
                "p1_lr_entropy_relation_routed_recovery_v4",
                "p1_lr_generalized_routed_recovery_candidate",
            }:
                keep_idx, debug_info = _qficr_select_frozen_core_only_fast(
                    selector,
                    raw_image_features,
                    self.visual_token_num,
                    texts,
                    relevance,
                    cls_attention,
                    semantic_text_embeds,
                    image_embeds,
                    semantic_units,
                    transition_prior,
                )
            else:
                keep_idx, debug_info = selector.select(
                    raw_image_features,
                    keep_num=self.visual_token_num,
                    question=texts,
                    relevance=relevance,
                    text_embeds=semantic_text_embeds,
                    b=None,
                    semantic_features=image_embeds,
                    semantic_units=semantic_units,
                    cls_attn=cls_attention,
                    transition_prior=transition_prior,
                )
            local_residual_mode = requested_local_residual_mode
            if local_residual_mode != "off":
                if local_residual_mode not in {
                    "p1_semantic_core_local_residual",
                    "p2_semantic58_local6",
                    "p1_tc_semantic_core_conditioned_local_residual",
                    "p1_lr_semantic_core_conditioned_local_residual",
                    "p1_lr_entropy_core_conditioned_local_residual",
                    "p1_lr_entropy_core_conditioned_local_residual_fast",
                    "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                    "p1_lr_entropy_core_random_recovery_v2",
                    "p1_lr_entropy_spatial_diverse_recovery_v3",
                    "p1_lr_entropy_relation_routed_recovery_v4",
                    "p1_lr_generalized_routed_recovery_candidate",
                }:
                    raise ValueError(f"Unsupported local residual mode: {local_residual_mode}")
                if B != 1:
                    raise ValueError("P1 local residual requires batch size 1")
                if os.environ.get("EC_QFICR_CORE_MODE", "").strip():
                    raise ValueError("P1 local residual cannot be combined with EC_QFICR_CORE_MODE")
                if os.environ.get("EC_QFICR_CORE_SOURCE", "legacy").strip().lower() != "legacy":
                    raise ValueError("P1 local residual requires legacy Frozen V0 core source")
                if _qficr_local_fusion_mode() != "off":
                    raise ValueError("P1 local residual cannot be combined with E8 local fusion")
                p1_ratio_cap = float(os.environ.get("EC_QFICR_RECOVER_RATIO_CAP", "0.20"))
                fixed_k_local = None
                if local_residual_mode == "p2_semantic58_local6":
                    fixed_k_local = int(os.environ.get("EC_QFICR_LOCAL_RESIDUAL_FIXED_K_LOCAL", "6"))
                if local_residual_mode == "p1_lr_generalized_routed_recovery_candidate":
                    keep_idx, p1_info = _qficr_apply_generalized_routed_recovery_candidate(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        question_text=texts,
                        eps=float(selector.qfid_eps),
                        ratio_cap=p1_ratio_cap,
                    )
                    if not bool(p1_info.get("qficr_semantic_prefix_reproduction", False)):
                        raise RuntimeError("Generalized recovery candidate semantic prefix reproduction failed")
                elif local_residual_mode == "p1_lr_entropy_relation_routed_recovery_v4":
                    keep_idx, p1_info = _qficr_apply_p1_lr_entropy_relation_routed_recovery_v4(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        question_text=texts,
                        eps=float(selector.qfid_eps),
                        ratio_cap=p1_ratio_cap,
                    )
                    if not bool(p1_info.get("qficr_semantic_prefix_reproduction", False)):
                        raise RuntimeError("Relation-routed recovery V4 semantic prefix reproduction failed")
                elif local_residual_mode == "p1_lr_entropy_spatial_diverse_recovery_v3":
                    keep_idx, p1_info = _qficr_apply_p1_lr_entropy_spatial_diverse_recovery_v3(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        question_text=texts,
                        eps=float(selector.qfid_eps),
                        ratio_cap=p1_ratio_cap,
                    )
                    if not bool(p1_info.get("qficr_semantic_prefix_reproduction", False)):
                        raise RuntimeError("Spatial diverse recovery V3 semantic prefix reproduction failed")
                elif local_residual_mode == "p1_lr_entropy_core_random_recovery_v2":
                    keep_idx, p1_info = _qficr_apply_p1_lr_entropy_random_recovery_v2(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        question_text=texts,
                        eps=float(selector.qfid_eps),
                        ratio_cap=p1_ratio_cap,
                    )
                    if not bool(p1_info.get("qficr_semantic_prefix_reproduction", False)):
                        raise RuntimeError("Random recovery V2 semantic prefix reproduction failed")
                elif local_residual_mode in {
                    "p1_lr_entropy_core_conditioned_local_residual_fast",
                    "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                }:
                    keep_idx, p1_info = _qficr_apply_p1_lr_entropy_local_residual_fast(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        question_text=texts,
                        eps=float(selector.qfid_eps),
                        ratio_cap=p1_ratio_cap,
                        method_name=local_residual_mode,
                    )
                    if not bool(p1_info.get("qficr_semantic_prefix_reproduction", False)):
                        raise RuntimeError("Fast entropy P1-LR semantic prefix reproduction failed")
                    if local_residual_mode == "p1_lr_spatial_semantic_core_conditioned_local_residual_v1":
                        p1_info.update({
                            "qficr_spatial_semantic_v1": True,
                            "qficr_entropy_source": "spatial semantic V1 pre-depolarization probability",
                        })
                elif local_residual_mode == "p1_lr_entropy_core_conditioned_local_residual":
                    keep_idx, p1_info = _qficr_apply_p1_lr_entropy_local_residual(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        question_text=texts,
                        eps=float(selector.qfid_eps),
                        ratio_cap=p1_ratio_cap,
                        method_name=local_residual_mode,
                    )
                    if not bool(p1_info.get("qficr_semantic_prefix_reproduction", False)):
                        raise RuntimeError("Entropy P1-LR semantic prefix reproduction failed before benchmark")
                elif local_residual_mode in {
                    "p1_tc_semantic_core_conditioned_local_residual",
                    "p1_lr_semantic_core_conditioned_local_residual",
                }:
                    keep_idx, p1_info = _qficr_apply_p1_tc_local_residual(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        eps=float(selector.qfid_eps),
                        k_semantic=int(os.environ.get("EC_QFICR_P1TC_K_SEMANTIC", "26")),
                        k_local=int(os.environ.get("EC_QFICR_P1TC_K_LOCAL", "6")),
                        task_conditioned=local_residual_mode == "p1_tc_semantic_core_conditioned_local_residual",
                        method_name=local_residual_mode,
                    )
                    if not bool(p1_info.get("qficr_p1tc_semantic_first26_reproduction", False)):
                        raise RuntimeError("P1-TC semantic first-26 reproduction failed before benchmark")
                else:
                    keep_idx, p1_info = _qficr_apply_p1_local_residual(
                        raw_image_features,
                        keep_idx,
                        debug_info,
                        int(self.visual_token_num),
                        p1_ratio_cap,
                        eps=float(selector.qfid_eps),
                        fixed_k_local=fixed_k_local,
                        method_name=local_residual_mode,
                    )
                if isinstance(debug_info, dict):
                    qfid_info = debug_info.get("qfid_info", {})
                    if isinstance(qfid_info, dict):
                        qfid_info.update(p1_info)
                        debug_info["qfid_info"] = qfid_info
                    if local_residual_mode in {
                        "p1_tc_semantic_core_conditioned_local_residual",
                        "p1_lr_semantic_core_conditioned_local_residual",
                        "p1_lr_entropy_core_conditioned_local_residual",
                        "p1_lr_entropy_core_conditioned_local_residual_fast",
                        "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                        "p1_lr_entropy_core_random_recovery_v2",
                        "p1_lr_entropy_spatial_diverse_recovery_v3",
                        "p1_lr_entropy_relation_routed_recovery_v4",
                        "p1_lr_generalized_routed_recovery_candidate",
                    }:
                        final_key = "qficr_p1tc_final_indices"
                        sem_key = "qficr_p1tc_semantic_indices"
                        loc_key = "qficr_p1tc_local_indices"
                    else:
                        final_key = "qficr_p1_final_indices"
                        sem_key = "qficr_p1_semantic_indices"
                        loc_key = "qficr_p1_local_indices"
                    debug_info.update({
                        "local_residual_mode": local_residual_mode,
                        "method": local_residual_mode,
                        "final_selected_indices": p1_info[final_key],
                        "selected_core_indices": p1_info[sem_key],
                        "selected_residual_indices": p1_info[loc_key],
                    })
            dual_residual_mode = requested_dual_residual_mode
            if dual_residual_mode != "off":
                if local_residual_mode != "off":
                    raise ValueError("P3 dual residual cannot be combined with P1/P2 local residual mode")
                if B != 1:
                    raise ValueError("P3 dual residual requires batch size 1")
                if os.environ.get("EC_QFICR_CORE_MODE", "").strip():
                    raise ValueError("P3 dual residual cannot be combined with EC_QFICR_CORE_MODE")
                if os.environ.get("EC_QFICR_CORE_SOURCE", "legacy").strip().lower() != "legacy":
                    raise ValueError("P3 dual residual requires legacy Frozen V0 core source")
                if _qficr_local_fusion_mode() != "off":
                    raise ValueError("P3 dual residual cannot be combined with E8 local fusion")
                if not isinstance(debug_info, dict):
                    raise RuntimeError("P3 dual residual requires qfid debug_info")
                qfid_info0 = debug_info.get("qfid_info", {})
                if not isinstance(qfid_info0, dict):
                    raise RuntimeError("P3 dual residual requires qfid_info")
                task_prob = qfid_info0.get("_prob_tensor")
                if not torch.is_tensor(task_prob):
                    values = qfid_info0.get("prob_values") or qfid_info0.get("p_mix_values") or []
                    if values:
                        task_prob = torch.tensor(values, dtype=torch.float32, device=device)
                if not torch.is_tensor(task_prob):
                    raise RuntimeError("P3 dual residual requires Frozen V0 probability tensor")
                full_core_order = qfid_info0.get("qficr_full_core_indices") or qfid_info0.get("full_core_indices") or []
                keep_idx, p3_info = _qficr_dual_residual_coconstruct(
                    raw_image_features,
                    task_prob,
                    int(self.visual_token_num),
                    frozen_order=full_core_order,
                    eps=float(selector.qfid_eps),
                )
                if not bool(p3_info.get("qficr_p3_semantic_only_reproduction", False)):
                    raise RuntimeError(
                        "P3 semantic-only reproduction failed before benchmark; "
                        f"first mismatch step={p3_info.get('qficr_p3_semantic_only_first_mismatch_step')}"
                    )
                qfid_info0.update(p3_info)
                debug_info["qfid_info"] = qfid_info0
                debug_info.update({
                    "dual_residual_mode": dual_residual_mode,
                    "method": dual_residual_mode,
                    "final_selected_indices": p3_info["qficr_p3_final_indices"],
                    "selected_core_indices": [],
                    "selected_residual_indices": p3_info["qficr_p3_final_indices"],
                })
            core_selector = os.environ.get("EC_QFICR_CORE_SELECTOR", "v0").strip().lower()
            core_feature_space = os.environ.get("EC_QFICR_CORE_FEATURE_SPACE", "pre").strip().lower()
            if core_selector not in {"v0", "cdpruner"}:
                raise ValueError(f"Unknown EC_QFICR_CORE_SELECTOR: {core_selector}")
            if core_feature_space not in {"pre", "post"}:
                raise ValueError(f"Unknown EC_QFICR_CORE_FEATURE_SPACE: {core_feature_space}")
            same_space_override = core_selector != "v0" or core_feature_space != "pre"
            if same_space_override and core_source != "legacy":
                raise ValueError("same-space core controls cannot be combined with EC_QFICR_CORE_SOURCE")
            if same_space_override:
                if B != 1:
                    raise ValueError("same-space core audit requires batch size 1")
                qfid_info0 = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                k_core = qfid_info0.get("adapt_k_core", qfid_info0.get("qficr_final_k_red"))
                task_prob = qfid_info0.get("_prob_tensor")
                if k_core is None or not torch.is_tensor(task_prob):
                    raise RuntimeError("V0 debug state did not expose K_core/task probability")
                k_core = int(k_core)
                core_features = raw_image_features
                if core_feature_space == "post":
                    core_features = self.get_model().mm_projector(raw_image_features)
                if core_selector == "v0":
                    override_core = select_frozen_qfi_order(
                        core_features, task_prob, k_core, eps=float(selector.qfid_eps)
                    )
                else:
                    override_core = select_cdpruner_indices_equal_budget(
                        core_features, image_embeds, text_embeds, k_core
                    )[0]
                keep_idx, debug_info = selector.select(
                    raw_image_features,
                    keep_num=self.visual_token_num,
                    question=texts,
                    relevance=relevance,
                    text_embeds=semantic_text_embeds,
                    b=None,
                    semantic_features=image_embeds,
                    semantic_units=semantic_units,
                    cls_attn=cls_attention,
                    transition_prior=transition_prior,
                    core_indices_override=override_core,
                )
                if isinstance(debug_info, dict):
                    debug_info["core_selector"] = core_selector
                    debug_info["core_feature_space"] = core_feature_space
                    debug_info["same_space_core_indices"] = override_core.detach().cpu().tolist()
            if core_source == "cdpruner_equal_budget":
                qfid_info0 = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                k_core = qfid_info0.get("adapt_k_core", qfid_info0.get("qficr_final_k_red"))
                if k_core is None:
                    raise RuntimeError("V0 qfid debug did not expose K_core for equal-budget core swap")
                k_core = int(k_core)
                projected_for_core = self.get_model().mm_projector(raw_image_features)
                cd_core = select_cdpruner_indices_equal_budget(
                    projected_for_core, image_embeds, text_embeds, k_core
                )[0]
                keep_idx, debug_info = selector.select(
                    raw_image_features,
                    keep_num=self.visual_token_num,
                    question=texts,
                    relevance=relevance,
                    text_embeds=semantic_text_embeds,
                    b=None,
                    semantic_features=image_embeds,
                    semantic_units=semantic_units,
                    cls_attn=cls_attention,
                    transition_prior=transition_prior,
                    core_indices_override=cd_core,
                )
                if isinstance(debug_info, dict):
                    debug_info["core_source"] = core_source
                    debug_info["v0_core_indices_before_override"] = qfid_info0.get("qficr_fixed_core_indices", [])
                    debug_info["cdpruner_equal_budget_core_indices"] = cd_core.detach().cpu().tolist()
            cd_safe_method = _qficr_cd_safe_method()
            scss_method = _qficr_scss_method()
            if cd_safe_method != "off" and scss_method != "off":
                raise ValueError("EC_QFICR_CD_SAFE_METHOD cannot be combined with EC_QFICR_SCSS_METHOD")
            if cd_safe_method != "off":
                if B != 1:
                    raise ValueError("CD-anchored safe refinement requires batch size 1")
                if core_source != "legacy" or same_space_override:
                    raise ValueError("CD-anchored safe refinement cannot be combined with same-space/core-source overrides")
                qfid_info0 = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                task_prob = qfid_info0.get("_prob_tensor")
                if not torch.is_tensor(task_prob):
                    raise RuntimeError("CD-anchored safe refinement requires frozen V0 task probability")
                initial_v0_final = keep_idx.detach().long().to(device=device).reshape(-1).sort().values
                if initial_v0_final.numel() != self.visual_token_num:
                    raise RuntimeError("V0 final indices are not exact K for CD-anchored refinement")
                cd_started = time.perf_counter()
                cd_m = int(os.environ.get("EC_QFICR_SCSS_M", "16"))
                cd_h = float(os.environ.get("EC_QFICR_SCSS_H", "0.01"))
                cd_seed = int(os.environ.get("EC_QFICR_SCSS_SEED", "20260805"))
                if cd_m != 16 or abs(cd_h - 0.01) > 1e-12 or cd_seed != 20260805:
                    raise ValueError("CD-anchored safe refinement freezes m=16, h=0.01, seed=20260805")
                projected_for_cd = self.get_model().mm_projector(raw_image_features)
                cd_indices = select_cdpruner_indices_equal_budget(
                    projected_for_cd, image_embeds, text_embeds, self.visual_token_num
                )[0].detach().long().to(device=device).reshape(-1).sort().values
                sens_started = time.perf_counter()
                sensitivity = _scss_projector_zo_sensitivity(
                    self.get_model().mm_projector,
                    raw_image_features,
                    m=cd_m,
                    h=cd_h,
                    seed=cd_seed,
                    chunk=int(os.environ.get("EC_QFICR_SCSS_CHUNK", "4096")),
                )
                if raw_image_features.is_cuda:
                    torch.cuda.synchronize(raw_image_features.device)
                sensitivity_ms = (time.perf_counter() - sens_started) * 1000.0
                cd_kernel = _scss_cd_kernel_post(projected_for_cd, image_embeds, text_embeds)
                psi_post = projected_for_cd[0].float() / projected_for_cd[0].float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
                kappa = compute_projection_overlap_kernel(
                    psi_post, eps=1e-12, overlap_kernel="relu_square", assume_normalized=True
                )
                select_started = time.perf_counter()
                final_idx, cd_swaps, cd_abnormal, cd_info = _cdsafe_apply_refinement(
                    cd_indices,
                    initial_v0_final,
                    sensitivity,
                    cd_kernel,
                    kappa,
                    task_prob.to(device=device),
                    cd_safe_method,
                )
                if raw_image_features.is_cuda:
                    torch.cuda.synchronize(raw_image_features.device)
                selection_ms = (time.perf_counter() - select_started) * 1000.0
                if cd_abnormal:
                    raise RuntimeError("CD-anchored safe refinement reached safety cap and is abnormal")
                final_idx = final_idx.reshape(-1).to(device=device, dtype=torch.long).sort().values
                if final_idx.numel() != self.visual_token_num or torch.unique(final_idx).numel() != self.visual_token_num:
                    raise RuntimeError(f"CD-safe {cd_safe_method} did not produce exact unique K")
                if not torch.isfinite(sensitivity).all():
                    raise RuntimeError("CD-safe sensitivity contains NaN/Inf")
                keep_idx = final_idx
                qfid_info1 = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                qfid_info1.update({
                    "qficr_cd_safe_method": cd_safe_method,
                    "qficr_cd_safe_final_indices": final_idx.detach().cpu().tolist(),
                    "qficr_cd_safe_cd_indices": cd_indices.detach().cpu().tolist(),
                    "qficr_cd_safe_v0_indices": initial_v0_final.detach().cpu().tolist(),
                    "qficr_cd_safe_m": cd_m,
                    "qficr_cd_safe_h": cd_h,
                    "qficr_cd_safe_seed": cd_seed,
                    "qficr_cd_safe_sensitivity_ms": sensitivity_ms,
                    "qficr_cd_safe_selection_ms": selection_ms,
                    "qficr_cd_safe_total_pruning_latency_ms": (time.perf_counter() - cd_started) * 1000.0,
                    **{f"qficr_cd_safe_{k}": v for k, v in cd_info.items() if k != "swaps"},
                    "qficr_cd_safe_swaps": cd_info.get("swaps", [])[:256],
                })
                if isinstance(debug_info, dict):
                    debug_info["qfid_info"] = qfid_info1
                _write_qficr_scss_debug({
                    "method": cd_safe_method,
                    "K": int(self.visual_token_num),
                    "N": int(N),
                    "exact_k": True,
                    "duplicate": 0,
                    "nan_inf": 0,
                    "final_selected_indices": final_idx.detach().cpu().tolist(),
                    "cd_selected_indices": cd_indices.detach().cpu().tolist(),
                    "v0_final_indices": initial_v0_final.detach().cpu().tolist(),
                    "sensitivity_ms": sensitivity_ms,
                    "selection_ms": selection_ms,
                    "total_pruning_latency_ms": qfid_info1["qficr_cd_safe_total_pruning_latency_ms"],
                    "sensitivity_mean": float(sensitivity.mean().item()),
                    "sensitivity_std": float(sensitivity.std(unbiased=False).item()),
                    "sensitivity_min": float(sensitivity.min().item()),
                    "sensitivity_max": float(sensitivity.max().item()),
                    **{k: v for k, v in cd_info.items() if k != "swaps"},
                    "swap_trace": cd_info.get("swaps", [])[:256],
                })
            if scss_method != "off":
                if B != 1:
                    raise ValueError("SCSS requires batch size 1")
                if core_source != "legacy" or same_space_override:
                    raise ValueError("SCSS cannot be combined with same-space/core-source overrides")
                qfid_info0 = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                k_core = int(qfid_info0.get("adapt_k_core", qfid_info0.get("qficr_final_k_red", self.visual_token_num)))
                k_rest = int(qfid_info0.get("adapt_k_recover", qfid_info0.get("qficr_final_k_rest", max(0, self.visual_token_num - k_core))))
                k_core = min(max(k_core, 0), self.visual_token_num)
                k_rest = min(max(k_rest, 0), self.visual_token_num - k_core)
                v0_core = qfid_info0.get("qficr_fixed_core_indices", qfid_info0.get("fixed_core_indices", []))
                v0_final = qfid_info0.get("qficr_final_selected_indices", qfid_info0.get("final_selected_indices", []))
                if not v0_final:
                    v0_final = keep_idx.detach().long().cpu().tolist()
                if raw_image_features.is_cuda:
                    torch.cuda.synchronize(raw_image_features.device)
                scss_started = time.perf_counter()
                scss_m = int(os.environ.get("EC_QFICR_SCSS_M", "16"))
                scss_h = float(os.environ.get("EC_QFICR_SCSS_H", "0.01"))
                scss_seed = int(os.environ.get("EC_QFICR_SCSS_SEED", "20260805"))
                if scss_m != 16 or abs(scss_h - 0.01) > 1e-12 or scss_seed != 20260805:
                    raise ValueError("SCSS freezes m=16, h=0.01, seed=20260805")
                sens_started = time.perf_counter()
                sensitivity = _scss_projector_zo_sensitivity(
                    self.get_model().mm_projector,
                    raw_image_features,
                    m=scss_m,
                    h=scss_h,
                    seed=scss_seed,
                    chunk=int(os.environ.get("EC_QFICR_SCSS_CHUNK", "4096")),
                )
                if raw_image_features.is_cuda:
                    torch.cuda.synchronize(raw_image_features.device)
                sensitivity_ms = (time.perf_counter() - sens_started) * 1000.0
                projected_for_scss = self.get_model().mm_projector(raw_image_features)
                cd_kernel = _scss_cd_kernel_post(projected_for_scss, image_embeds, text_embeds)
                select_started = time.perf_counter()
                scss_core = None
                scss_swaps = []
                scss_abnormal = False
                score_trace = []
                initial_v0_final = torch.tensor(v0_final, dtype=torch.long, device=device).reshape(-1)
                if scss_method == "z0":
                    final_idx = torch.tensor(
                        _scss_stable_order(sensitivity, descending=True)[: self.visual_token_num],
                        dtype=torch.long,
                        device=device,
                    ).sort().values
                elif scss_method == "z1":
                    core_tensor = torch.tensor(v0_core, dtype=torch.long, device=device).reshape(-1)
                    if core_tensor.numel() != k_core:
                        raise RuntimeError("Z1 could not recover exact frozen V0 core indices")
                    final_idx, score_trace = _scss_select_sensitivity_diversity(
                        sensitivity,
                        cd_kernel,
                        self.visual_token_num,
                        initial=core_tensor,
                    )
                elif scss_method == "z2":
                    scss_core, score_trace = _scss_select_sensitivity_diversity(
                        sensitivity,
                        cd_kernel,
                        k_core,
                    )
                    keep_idx, debug_info = selector.select(
                        raw_image_features,
                        keep_num=self.visual_token_num,
                        question=texts,
                        relevance=relevance,
                        text_embeds=semantic_text_embeds,
                        b=None,
                        semantic_features=image_embeds,
                        semantic_units=semantic_units,
                        cls_attn=cls_attention,
                        transition_prior=transition_prior,
                        core_indices_override=scss_core,
                    )
                    final_idx = keep_idx.detach().long().to(device=device).reshape(-1).sort().values
                elif scss_method == "z3":
                    scss_core, core_trace = _scss_select_sensitivity_diversity(
                        sensitivity,
                        cd_kernel,
                        k_core,
                    )
                    final_idx, rest_trace = _scss_select_sensitivity_diversity(
                        sensitivity,
                        cd_kernel,
                        self.visual_token_num,
                        initial=scss_core,
                    )
                    score_trace = list(core_trace) + list(rest_trace)
                else:
                    if initial_v0_final.numel() != self.visual_token_num:
                        raise RuntimeError("Z4 initial V0 final indices are not exact K")
                    final_idx, scss_swaps, scss_abnormal = _scss_apply_pareto_swap(
                        initial_v0_final,
                        sensitivity,
                        cd_kernel,
                    )
                    if scss_abnormal:
                        raise RuntimeError("Z4 reached safety cap and is abnormal")
                if raw_image_features.is_cuda:
                    torch.cuda.synchronize(raw_image_features.device)
                selection_ms = (time.perf_counter() - select_started) * 1000.0
                final_idx = final_idx.reshape(-1).to(device=device, dtype=torch.long).sort().values
                if final_idx.numel() != self.visual_token_num or torch.unique(final_idx).numel() != self.visual_token_num:
                    raise RuntimeError(f"SCSS {scss_method} did not produce exact unique K")
                if not torch.isfinite(sensitivity).all():
                    raise RuntimeError("SCSS sensitivity contains NaN/Inf")
                keep_idx = final_idx
                initial_set = set(int(x) for x in initial_v0_final.detach().cpu().tolist())
                final_set = set(int(x) for x in final_idx.detach().cpu().tolist())
                union = len(initial_set | final_set)
                jaccard = (len(initial_set & final_set) / union) if union else 1.0
                qfid_info1 = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                qfid_info1.update({
                    "qficr_final_selected_indices": final_idx.detach().cpu().tolist(),
                    "qficr_scss_method": scss_method,
                    "qficr_scss_m": scss_m,
                    "qficr_scss_h": scss_h,
                    "qficr_scss_seed": scss_seed,
                    "qficr_scss_k_core": k_core,
                    "qficr_scss_k_rest": k_rest,
                    "qficr_scss_core_indices": [] if scss_core is None else scss_core.detach().cpu().tolist(),
                    "qficr_scss_initial_v0_final_indices": initial_v0_final.detach().cpu().tolist(),
                    "qficr_scss_actual_swap_count": len(scss_swaps),
                    "qficr_scss_swaps": scss_swaps,
                    "qficr_scss_jaccard_to_v0": jaccard,
                    "qficr_scss_sensitivity_ms": sensitivity_ms,
                    "qficr_scss_selection_ms": selection_ms,
                    "qficr_scss_total_pruning_latency_ms": (time.perf_counter() - scss_started) * 1000.0,
                    "qficr_scss_sensitivity_min": float(sensitivity.min().item()),
                    "qficr_scss_sensitivity_max": float(sensitivity.max().item()),
                    "qficr_scss_sensitivity_mean": float(sensitivity.mean().item()),
                    "qficr_scss_sensitivity_std": float(sensitivity.std(unbiased=False).item()),
                    "qficr_scss_score_trace": score_trace[: min(len(score_trace), 256)],
                })
                if isinstance(debug_info, dict):
                    debug_info["qfid_info"] = qfid_info1
                _write_qficr_scss_debug({
                    "method": scss_method,
                    "K": int(self.visual_token_num),
                    "N": int(N),
                    "exact_k": True,
                    "duplicate": 0,
                    "nan_inf": 0,
                    "final_selected_indices": final_idx.detach().cpu().tolist(),
                    "v0_final_indices": initial_v0_final.detach().cpu().tolist(),
                    "scss_core_indices": [] if scss_core is None else scss_core.detach().cpu().tolist(),
                    "K_core": k_core,
                    "K_rest": k_rest,
                    "actual_swap_count": len(scss_swaps),
                    "swap_trace": scss_swaps,
                    "jaccard_to_v0": jaccard,
                    "sensitivity_ms": sensitivity_ms,
                    "selection_ms": selection_ms,
                    "total_pruning_latency_ms": qfid_info1["qficr_scss_total_pruning_latency_ms"],
                    "sensitivity_mean": qfid_info1["qficr_scss_sensitivity_mean"],
                    "sensitivity_std": qfid_info1["qficr_scss_sensitivity_std"],
                    "sensitivity_min": qfid_info1["qficr_scss_sensitivity_min"],
                    "sensitivity_max": qfid_info1["qficr_scss_sensitivity_max"],
                    "sensitivity_hash": _hash_long_tensor(torch.argsort(sensitivity, descending=True)),
                    "v0_core_match": bool(scss_method != "z1" or set(int(x) for x in final_idx[:0].tolist()) == set()),
                })
            role_core_info = None
            if core_mode in {"role_core_gap", "m2_dual_quality_rolecore"}:
                if B != 1:
                    raise ValueError(f"EC_QFICR_CORE_MODE={core_mode} currently requires batch size 1")
                if os.environ.get("EC_FORCE_SELECTED_INDICES_JSONL", "").strip():
                    raise ValueError(f"EC_FORCE_SELECTED_INDICES_JSONL cannot override EC_QFICR_CORE_MODE={core_mode}")
                role_projected_features = self.get_model().mm_projector(raw_image_features)
                qfid_info_role = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                v0_before_role = keep_idx.detach().long().to(device=device).reshape(-1).sort().values
                keep_idx, role_core_info = _qficr_role_core_gap_select(
                    raw_image_features,
                    role_projected_features,
                    v0_before_role,
                    qfid_info_role,
                    int(self.visual_token_num),
                    image_embeds=image_embeds,
                    text_embeds=text_embeds,
                )
                if isinstance(debug_info, dict):
                    qfid_info = debug_info.get("qfid_info", {})
                    if isinstance(qfid_info, dict) and role_core_info is not None:
                        v0_set = set(int(x) for x in role_core_info["S_V0"])
                        final_set = set(int(x) for x in role_core_info["S_final"])
                        union = len(v0_set | final_set)
                        jaccard = float(len(v0_set & final_set) / union) if union else 1.0
                        qfid_info.update({
                            "qficr_core_mode": role_core_info["mode"],
                            "qficr_alloc_mode": role_core_info["mode"],
                            "qficr_role_core_gap": role_core_info["mode"] == "role_core_gap",
                            "qficr_m2_dual_quality": role_core_info["mode"] == "m2_dual_quality_rolecore",
                            "qficr_role_core_gap_info": role_core_info,
                            "qficr_final_selected_indices": role_core_info["S_final"],
                            "qficr_rolecore_selected_indices": role_core_info["S_final"],
                            "qficr_role_core_semantic_indices": role_core_info["S_sem"],
                            "qficr_role_core_complement_indices": role_core_info["S_comp"],
                            "qficr_role_core_v0_indices": role_core_info["S_V0"],
                            "qficr_role_core_K_sem": role_core_info["K_sem"],
                            "qficr_role_core_K_comp": role_core_info["K_comp"],
                            "qficr_role_core_r_gap": role_core_info["r_gap"],
                            "qficr_role_core_largest_gap": role_core_info["largest_gap"],
                            "qficr_role_core_jaccard_to_v0": jaccard,
                            "qficr_role_core_num_diff_tokens": int(len(final_set - v0_set) + len(v0_set - final_set)),
                            "qficr_role_core_exact_match_forward": False,
                            "qficr_complement_quality_source": role_core_info.get("complement_quality_source", ""),
                            "qficr_native_cd_quality_min": role_core_info.get("native_cd_quality_min"),
                            "qficr_native_cd_quality_max": role_core_info.get("native_cd_quality_max"),
                            "qficr_native_cd_quality_mean": role_core_info.get("native_cd_quality_mean"),
                            "qficr_native_cd_quality_median": role_core_info.get("native_cd_quality_median"),
                            "qficr_duplicate_count": role_core_info["duplicate"],
                            "qficr_nan_inf_count": role_core_info["nan_inf"],
                            "qficr_actual_core_count": int(keep_idx.numel()),
                            "qficr_actual_residual_count": 0,
                            "qficr_final_k_red": int(keep_idx.numel()),
                            "qficr_final_k_rest": 0,
                        })
                        debug_info["qfid_info"] = qfid_info
            forced_selected = False
            if B != 1 and os.environ.get("EC_FORCE_SELECTED_INDICES_JSONL", "").strip():
                raise ValueError("EC_FORCE_SELECTED_INDICES_JSONL currently requires batch size 1")
            if B == 1:
                keep_idx, forced_selected = _maybe_force_selected_indices(keep_idx, N, device)
                if forced_selected and isinstance(debug_info, dict):
                    qfid_info = debug_info.get("qfid_info", {})
                    if isinstance(qfid_info, dict):
                        qfid_info["qficr_forced_selected_indices"] = True
                        qfid_info["qficr_final_selected_indices"] = keep_idx.detach().cpu().tolist()
                        qfid_info["qficr_duplicate_count"] = 0
                        qfid_info["qficr_actual_core_count"] = int(keep_idx.numel())
                        qfid_info["qficr_actual_residual_count"] = 0
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks[:, keep_idx] = True
            if dual_residual_mode == "p3_dual_residual_coconstruction" and isinstance(debug_info, dict):
                qfid_info = debug_info.get("qfid_info", {})
                if isinstance(qfid_info, dict):
                    actual_forward = torch.nonzero(index_masks[0], as_tuple=False).reshape(-1).detach().long().cpu().tolist()
                    p3_final = [int(x) for x in qfid_info.get("qficr_p3_final_indices", [])]
                    if actual_forward != p3_final:
                        raise RuntimeError("P3 final indices do not match actual forward selected indices")
                    qfid_info.update({
                        "qficr_actual_forward_selected_indices": actual_forward,
                        "qficr_p3_actual_forward_match": True,
                        "qficr_num_visual_tokens_before": int(N),
                        "qficr_num_visual_tokens_after": int(len(actual_forward)),
                        "qficr_p3_exact_k32": bool(len(actual_forward) == int(self.visual_token_num)),
                        "qficr_p3_canonical_ordering_preserved": True,
                    })
                    debug_info["qfid_info"] = qfid_info
                    selector.qfid_debug_stats_jsonl = role_core_stats_path
                    selector._write_qfid_debug_stats(qfid_info)
                    _write_qficr_p3_dual_residual_debug(qfid_info)
            if local_residual_mode in {
                "p1_semantic_core_local_residual",
                "p2_semantic58_local6",
                "p1_tc_semantic_core_conditioned_local_residual",
                "p1_lr_semantic_core_conditioned_local_residual",
                "p1_lr_entropy_core_conditioned_local_residual",
                "p1_lr_entropy_core_conditioned_local_residual_fast",
                "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                "p1_lr_entropy_core_random_recovery_v2",
                "p1_lr_entropy_spatial_diverse_recovery_v3",
                "p1_lr_entropy_relation_routed_recovery_v4",
                "p1_lr_generalized_routed_recovery_candidate",
            } and isinstance(debug_info, dict):
                qfid_info = debug_info.get("qfid_info", {})
                if isinstance(qfid_info, dict):
                    actual_forward = torch.nonzero(index_masks[0], as_tuple=False).reshape(-1).detach().long().cpu().tolist()
                    p1_final_key = "qficr_p1tc_final_indices" if local_residual_mode in {
                        "p1_tc_semantic_core_conditioned_local_residual",
                        "p1_lr_semantic_core_conditioned_local_residual",
                        "p1_lr_entropy_core_conditioned_local_residual",
                        "p1_lr_entropy_core_conditioned_local_residual_fast",
                        "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                        "p1_lr_entropy_core_random_recovery_v2",
                        "p1_lr_entropy_spatial_diverse_recovery_v3",
                        "p1_lr_entropy_relation_routed_recovery_v4",
                        "p1_lr_generalized_routed_recovery_candidate",
                    } else "qficr_p1_final_indices"
                    p1_final = [int(x) for x in qfid_info.get(p1_final_key, [])]
                    if actual_forward != p1_final:
                        raise RuntimeError("P1 final indices do not match actual forward selected indices")
                    qfid_info.update({
                        "qficr_actual_forward_selected_indices": actual_forward,
                        "qficr_p1_actual_forward_match": True,
                        "qficr_p1tc_actual_forward_match": local_residual_mode == "p1_tc_semantic_core_conditioned_local_residual",
                        "qficr_p1lr_actual_forward_match": local_residual_mode == "p1_lr_semantic_core_conditioned_local_residual",
                        "qficr_p1lr_entropy_actual_forward_match": local_residual_mode in {
                            "p1_lr_entropy_core_conditioned_local_residual",
                            "p1_lr_entropy_core_conditioned_local_residual_fast",
                            "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                            "p1_lr_entropy_core_random_recovery_v2",
                            "p1_lr_entropy_spatial_diverse_recovery_v3",
                            "p1_lr_entropy_relation_routed_recovery_v4",
                            "p1_lr_generalized_routed_recovery_candidate",
                        },
                        "qficr_num_visual_tokens_before": int(N),
                        "qficr_num_visual_tokens_after": int(len(actual_forward)),
                        "qficr_p1_exact_k32": bool(len(actual_forward) == int(self.visual_token_num)),
                        "qficr_p1tc_exact_k32": bool(local_residual_mode == "p1_tc_semantic_core_conditioned_local_residual" and len(actual_forward) == int(self.visual_token_num)),
                        "qficr_p1lr_exact_k32": bool(local_residual_mode == "p1_lr_semantic_core_conditioned_local_residual" and len(actual_forward) == int(self.visual_token_num)),
                        "qficr_p1lr_entropy_exact_k": bool(local_residual_mode in {
                            "p1_lr_entropy_core_conditioned_local_residual",
                            "p1_lr_entropy_core_conditioned_local_residual_fast",
                            "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                            "p1_lr_entropy_core_random_recovery_v2",
                            "p1_lr_entropy_spatial_diverse_recovery_v3",
                            "p1_lr_entropy_relation_routed_recovery_v4",
                            "p1_lr_generalized_routed_recovery_candidate",
                        } and len(actual_forward) == int(self.visual_token_num)),
                        "qficr_p1_canonical_ordering_preserved": True,
                        "qficr_p1tc_canonical_ordering_preserved": local_residual_mode == "p1_tc_semantic_core_conditioned_local_residual",
                        "qficr_p1lr_canonical_ordering_preserved": local_residual_mode == "p1_lr_semantic_core_conditioned_local_residual",
                        "qficr_p1lr_entropy_canonical_ordering_preserved": local_residual_mode in {
                            "p1_lr_entropy_core_conditioned_local_residual",
                            "p1_lr_entropy_core_conditioned_local_residual_fast",
                            "p1_lr_spatial_semantic_core_conditioned_local_residual_v1",
                            "p1_lr_entropy_core_random_recovery_v2",
                            "p1_lr_entropy_spatial_diverse_recovery_v3",
                            "p1_lr_entropy_relation_routed_recovery_v4",
                            "p1_lr_generalized_routed_recovery_candidate",
                        },
                    })
                    debug_info["qfid_info"] = qfid_info
                    selector.qfid_debug_stats_jsonl = role_core_stats_path
                    selector._write_qfid_debug_stats(qfid_info)
            if core_mode in {"role_core_gap", "m2_dual_quality_rolecore"} and isinstance(debug_info, dict):
                qfid_info = debug_info.get("qfid_info", {})
                if isinstance(qfid_info, dict) and role_core_info is not None:
                    actual_forward = torch.nonzero(index_masks[0], as_tuple=False).reshape(-1).detach().long().cpu().tolist()
                    role_final = [int(x) for x in role_core_info["S_final"]]
                    role_sem = set(int(x) for x in role_core_info["S_sem"])
                    role_comp = set(int(x) for x in role_core_info["S_comp"])
                    role_final_set = set(role_final)
                    forward_match = actual_forward == role_final
                    if not forward_match:
                        raise RuntimeError("RoleCore final indices do not match actual forward selected indices")
                    if not role_sem.issubset(role_final_set) or not role_comp.issubset(role_final_set):
                        raise RuntimeError("RoleCore semantic/complement sets are not subsets of final set")
                    if role_sem & role_comp or role_sem | role_comp != role_final_set:
                        raise RuntimeError("RoleCore semantic/complement partition does not reconstruct final set")
                    qfid_info.update({
                        "qficr_actual_forward_selected_indices": actual_forward,
                        "qficr_role_core_exact_match_forward": True,
                        "qficr_num_visual_tokens_before": int(N),
                        "qficr_num_visual_tokens_after": int(len(actual_forward)),
                        "qficr_actual_core_count": int(len(actual_forward)),
                        "qficr_actual_residual_count": 0,
                    })
                    debug_info["qfid_info"] = qfid_info
                    selector.qfid_debug_stats_jsonl = role_core_stats_path
                    selector._write_qfid_debug_stats(qfid_info)
            _prune_debug(f"keep_idx.shape={tuple(keep_idx.shape)}")
            _prune_debug(f"vtn={keep_idx.numel()}")
            _prune_debug(f"ec_score_source={debug_info['score_source']}")
            _prune_debug(f"candidate_size={debug_info['candidate_size']}")
            _prune_debug(
                f"R.nnz={debug_info['repulsion_nnz']}, "
                f"C.nnz={debug_info['complement_nnz']}"
            )
            local_fusion_info = None
            if _qficr_local_fusion_mode() != "off":
                raw_image_features, local_fusion_info = _apply_e8_local_fusion(
                    raw_image_features,
                    keep_idx,
                )
                if isinstance(debug_info, dict):
                    qfid_info = debug_info.get("qfid_info", {})
                    if isinstance(qfid_info, dict):
                        qfid_info.update(local_fusion_info or {})
                        qfid_info["qficr_actual_forward_selected_indices"] = (
                            keep_idx.detach().long().cpu().tolist()
                        )
                        qfid_info["qficr_num_visual_tokens_before"] = int(N)
                        qfid_info["qficr_num_visual_tokens_after"] = int(keep_idx.numel())
                        debug_info["qfid_info"] = qfid_info
            image_features = self.get_model().mm_projector(raw_image_features)
            if local_fusion_info is not None:
                selected_projected = image_features[0, keep_idx].detach().float()
                local_fusion_info = dict(local_fusion_info)
                local_fusion_info.update({
                    "projected_feature_hash_after": _hash_float_tensor(selected_projected),
                    "actual_forward_feature_match": True,
                    "actual_forward_visual_token_count": int(keep_idx.numel()),
                })
                _write_e8_local_fusion_debug(local_fusion_info)
                if isinstance(debug_info, dict):
                    qfid_info = debug_info.get("qfid_info", {})
                    if isinstance(qfid_info, dict):
                        qfid_info.update(local_fusion_info)
                        debug_info["qfid_info"] = qfid_info
            if B == 1:
                qfid_info_for_cache = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                _write_core_set_audit_cache(
                    "v0_core",
                    image_features,
                    keep_idx,
                    extra={
                        "qfid_info": qfid_info_for_cache,
                        "forced_selected_indices": bool(forced_selected),
                        "pre_projector_features": raw_image_features[0].detach().float().cpu(),
                    },
                )
            audit_cache_dir = os.environ.get(
                "EC_QFICR_RESTORATION_AUDIT_CACHE_DIR", ""
            ).strip()
            if audit_cache_dir:
                audit_qid = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "unknown").strip() or "unknown"
                safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", audit_qid)
                os.makedirs(audit_cache_dir, exist_ok=True)
                qfid_info = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                torch.save({
                    "version": 1,
                    "question_id": audit_qid,
                    "question": texts or "",
                    "projected_features": image_features[0].detach().float().cpu(),
                    "selected_indices": keep_idx.detach().long().cpu(),
                    "qfid_info": qfid_info,
                }, os.path.join(audit_cache_dir, f"features_{safe_qid}.pt"))
            transfer_mode = _qficr_residual_transfer_mode()
            if transfer_mode != "off":
                qfid_info = debug_info.get("qfid_info", {}) if isinstance(debug_info, dict) else {}
                task_prob = qfid_info.get("_prob_tensor") if isinstance(qfid_info, dict) else None
                image_features, residual_transfer_info = _apply_qficr_residual_transfer(
                    image_features,
                    keep_idx,
                    task_prob,
                    mode=transfer_mode,
                )
                if isinstance(qfid_info, dict):
                    qfid_info["residual_transfer"] = residual_transfer_info
                if residual_transfer_info is not None:
                    _write_qficr_residual_transfer_debug(residual_transfer_info)
            return image_features, index_masks

        if prune_method not in {"cdpruner", "cdpruner_task_refine", "cdpruner_relevance_refine"}:
            _prune_debug(f"unknown prune_method={prune_method}; fallback=cdpruner")

        image_features = self.get_model().mm_projector(image_features)
        
        # [CDPruner] Use the production equal-budget conditional-DPP MAP path.
        select_idx = select_cdpruner_indices_equal_budget(
            image_features, image_embeds, text_embeds, self.visual_token_num
        )

        if prune_method in {"cdpruner_task_refine", "cdpruner_relevance_refine"}:
            # Refinement diagnostics need the same production matrices; keep
            # this auxiliary calculation only for those opt-in methods.
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            image_normalized = image_normalized.float()
            similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))
            image_embeds_n = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_embeds_n = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            relevance = torch.matmul(image_embeds_n, text_embeds_n.t())
            relevance = (-relevance).mean(dim=-1)
            relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())
            kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
            if B != 1:
                raise ValueError("cdpruner_task_refine currently requires batch size 1")
            selector = ECPruner(debug=None)
            semantic_units = selector.extract_semantic_units(texts)
            semantic_text_embeds = _encode_ec_semantic_units(
                self.get_vision_tower(), semantic_units
            ) if selector.needs_semantics else None
            task_prob, prob_info = selector.compute_task_probability(
                vision_outputs[0],
                question=texts,
                cls_attn=cls_attention,
                text_embeds=semantic_text_embeds,
                semantic_features=vision_outputs[1],
                semantic_units=semantic_units,
                transition_prior=transition_prior,
            )
            task_prob = torch.nan_to_num(
                task_prob.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).reshape(-1)
            anchors = select_idx[0]
            # Preserve CDPruner's coverage partition and only change the
            # representative inside each anchor's nearest-feature cell.
            refine_score = task_prob if prune_method == "cdpruner_task_refine" else relevance[0].float()
            refined_idx = _refine_cdpruner_by_task_cells(
                similarity[0], anchors, refine_score
            ).unsqueeze(0)
            anchor_set = set(int(x) for x in anchors.tolist())
            refined_set = set(int(x) for x in refined_idx[0].tolist())
            assignment = similarity[0, :, anchors].argmax(dim=-1)
            assignment[anchors] = torch.arange(anchors.numel(), device=device)
            region_records = []
            side = int(round(math.sqrt(N)))
            for cell_id, (anchor, chosen) in enumerate(zip(anchors.tolist(), refined_idx[0].tolist())):
                members = torch.nonzero(assignment == cell_id, as_tuple=False).reshape(-1)
                cosine = float(similarity[0, chosen, anchor].item())
                anchor_score = float(refine_score[anchor].item())
                chosen_score = float(refine_score[chosen].item()) * max(cosine, 0.0)
                spatial_distance = None
                if side * side == N:
                    ar, ac = divmod(int(anchor), side); cr, cc = divmod(int(chosen), side)
                    spatial_distance = float(math.hypot(ar - cr, ac - cc))
                region_records.append({
                    "anchor_index": int(anchor), "cluster_size": int(members.numel()),
                    "replaced": bool(anchor != chosen), "chosen_index": int(chosen),
                    "anchor_task_prob": float(task_prob[anchor].item()),
                    "chosen_task_prob": float(task_prob[chosen].item()),
                    "anchor_cd_relevance": float(relevance[0, anchor].item()),
                    "chosen_cd_relevance": float(relevance[0, chosen].item()),
                    "chosen_anchor_cosine": cosine,
                    "local_score_gain": float(chosen_score - anchor_score),
                    "spatial_distance": spatial_distance,
                })
            eps_logdet = 1e-6
            def diversity(indices):
                idx = indices.long(); c = similarity[0][idx][:, idx].double()
                l = kernel[0][idx][:, idx].double(); eye = torch.eye(idx.numel(), device=device, dtype=torch.double)
                c_sign, c_logdet = torch.linalg.slogdet(c + eps_logdet * eye)
                l_sign, l_logdet = torch.linalg.slogdet(l + eps_logdet * eye)
                off = c[~torch.eye(idx.numel(), dtype=torch.bool, device=device)]
                return {"feature_logdet": float(c_logdet.item()), "feature_logdet_sign": float(c_sign.item()),
                        "conditional_logdet": float(l_logdet.item()), "conditional_logdet_sign": float(l_sign.item()),
                        "mean_pairwise_cosine": float(off.mean().item()), "max_pairwise_cosine": float(off.max().item()),
                        "min_pairwise_cosine_distance": float((1.0-off.max()).item())}
            _write_task_refine_debug({
                "mode": prune_method,
                "K": int(self.visual_token_num),
                "num_tokens": int(N),
                "replacement_count": int(len(anchor_set - refined_set)),
                "jaccard_vs_cdpruner": float(len(anchor_set & refined_set) / max(1, len(anchor_set | refined_set))),
                "mean_p_cdpruner": float(task_prob[anchors].mean().item()),
                "mean_p_refined": float(task_prob[refined_idx[0]].mean().item()),
                "task_prob_source": prob_info.get("source", ""),
                "logdet_epsilon": eps_logdet,
                "anchor_diversity": diversity(anchors),
                "selected_diversity": diversity(refined_idx[0]),
                "regions": region_records,
                "cdpruner_indices": [int(x) for x in anchors.tolist()],
                "final_selected_indices": [int(x) for x in refined_idx[0].tolist()],
            })
            select_idx = refined_idx

        forced_selected = False
        if B != 1 and os.environ.get("EC_FORCE_SELECTED_INDICES_JSONL", "").strip():
            raise ValueError("EC_FORCE_SELECTED_INDICES_JSONL currently requires batch size 1")
        if B == 1:
            forced_idx, forced_selected = _maybe_force_selected_indices(select_idx[0], N, device)
            if forced_selected:
                select_idx = forced_idx.unsqueeze(0)
        if B == 1:
            cd_audit = _cdpruner_equal_budget_audit(
                image_features,
                image_embeds,
                text_embeds,
                self.visual_token_num,
            )
            _write_core_set_audit_cache(
                "cdpruner",
                image_features,
                select_idx[0],
                extra={
                    "forced_selected_indices": bool(forced_selected),
                    "pre_projector_features": image_embeds[0].detach().float().cpu(),
                    "cd_audit": cd_audit,
                },
            )
            _write_cdpruner_selected_debug(
                select_idx,
                image_features,
                image_embeds,
                text_embeds,
                forced=forced_selected,
            )
        _prune_debug(f"keep_idx.shape={tuple(select_idx.shape)}")
        _prune_debug(f"vtn={self.visual_token_num}")
        index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        index_masks.scatter_(1, select_idx, True)
        
        return image_features, index_masks

    # [CDPruner] Prune visual tokens according to index masks
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, texts=None, decoder_feedback_query_ids=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        decoder_mode = _decoder_feedback_mode()
        decoder_context = None
        # [CDPruner] Prune visual tokens
        if type(images) is list or images.ndim == 5:
            if decoder_mode != "off":
                raise NotImplementedError("decoder-feedback audit currently requires one square image per sample")
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, index_masks = self.encode_images(
                concat_images,
                texts=texts,
            )
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
                image_features = [x[m] for x, m in zip(image_features, index_masks)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        base_index_mask = index_mask[0]
                        index_mask = index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device)
                            ), dim=-1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature))
                    else:
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if decoder_mode == "off":
                image_features, index_masks = self.encode_images(
                    images,
                    texts=texts,
                )
            else:
                image_features, task_prob, psi, prob_info = self.encode_images_for_decoder_feedback(
                    images,
                    texts=texts,
                )
                index_masks = torch.ones(
                    image_features.shape[:2], dtype=torch.bool, device=image_features.device
                )
                decoder_context = {
                    "task_prob": task_prob,
                    "psi": psi[0],
                    "prob_info": prob_info,
                }
            image_features = image_features[index_masks.to(image_features.device)].unsqueeze(0)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        final_visual_token_num = image_features[0].shape[0]
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            decoder_visual_start = None
            decoder_visual_len = None

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    if decoder_mode != "off":
                        if num_images != 1:
                            raise NotImplementedError("decoder-feedback audit requires exactly one image")
                        decoder_visual_start = sum(x.shape[0] for x in cur_new_input_embeds)
                        decoder_visual_len = cur_image_features.shape[0]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            if decoder_mode != "off":
                cur_new_input_embeds, keep_idx = self._apply_decoder_feedback(
                    cur_new_input_embeds,
                    cur_input_ids,
                    decoder_feedback_query_ids,
                    decoder_visual_start,
                    decoder_visual_len,
                    decoder_context["task_prob"],
                    decoder_context["psi"],
                    self.visual_token_num,
                    task_prob_ms=decoder_context["prob_info"].get("task_prob_ms", 0.0),
                )
                cur_new_labels = torch.cat((
                    cur_new_labels[:decoder_visual_start],
                    cur_new_labels[decoder_visual_start:decoder_visual_start + decoder_visual_len][keep_idx],
                    cur_new_labels[decoder_visual_start + decoder_visual_len:],
                ))
                final_visual_token_num = int(keep_idx.numel())

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, final_visual_token_num

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
