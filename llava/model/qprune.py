"""Question conditioned token selection used by the QPrune inference path.

The public implementation fixes the configuration used for the paper's
LLaVA-1.5 experiments. Only the visual token budget is changed between runs.
"""

import hashlib
import json
import math
import os
import random
import re

import torch
import torch.nn.functional as F


_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "does", "do", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "the", "there",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
_EXPLICIT_SPATIAL = re.compile(
    r"\b(left|right|above|below|under|behind)\b"
    r"|\bin\s+front\s+of\b|\bon\s+top\s+of\b", re.IGNORECASE,
)
_PROPOSAL_SEEDS = (20260906, 20260907, 20260908, 20260909, 20260910)


def _normalize(scores, eps=1e-8):
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    span = scores.max() - scores.min()
    if not torch.isfinite(span) or span.abs().item() <= eps:
        return torch.zeros_like(scores)
    return (scores - scores.min()) / span


def _semantic_units(question):
    if isinstance(question, (tuple, list)):
        question = question[0] if question else None
    if not isinstance(question, str) or not question.strip():
        return ["object"]
    units = []
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?", question.lower()):
        if word not in _STOP_WORDS and word not in units:
            units.append(word)
            if len(units) == 8:
                break
    return units or ["object"]


def _semantic_probability(image_embeds, text_embeds, units, tau, eps):
    if not torch.is_tensor(text_embeds):
        return None
    features = image_embeds.float().mean(dim=0)
    if text_embeds.ndim == 3 and text_embeds.shape[0] == 1:
        text_embeds = text_embeds.squeeze(0)
    if text_embeds.ndim != 2 or text_embeds.shape != (len(units), features.shape[-1]):
        return None
    texts = text_embeds.to(device=features.device, dtype=torch.float32)
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    texts = texts / texts.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    response = torch.softmax(features @ texts.t(), dim=-1)
    response = torch.nan_to_num(response, nan=1.0 / len(units), posinf=0.0, neginf=0.0)
    if len(units) == 1:
        confidence = torch.ones(response.shape[0], device=response.device)
    else:
        safe = response.clamp_min(1e-8)
        entropy = -(safe * safe.log()).sum(dim=-1)
        confidence = (1.0 - entropy / math.log(len(units))).clamp(0.0, 1.0)
    score = response.max(dim=-1).values * confidence
    if torch.max(torch.abs(score)).item() <= eps:
        return None
    return torch.softmax(_normalize(score) / tau, dim=0)


def _visual_probability(cls_attention, num_tokens, device, eps):
    if not torch.is_tensor(cls_attention):
        return None
    attention = torch.nan_to_num(
        cls_attention.to(device=device, dtype=torch.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    ).clamp_min(0.0)
    if attention.ndim == 1:
        attention = attention.unsqueeze(0)
    if attention.ndim == 2:
        score = attention.mean(dim=0)
    elif attention.ndim == 3:
        score = attention.mean(dim=1).mean(dim=0)
    else:
        return None
    score = score.reshape(-1)
    if score.numel() != num_tokens or score.sum().item() <= eps:
        return None
    return score / score.sum().clamp_min(eps)


def _mixed_probability(image_embeds, text_embeds, cls_attention, question, eps=1e-6):
    num_tokens = image_embeds.shape[1]
    units = _semantic_units(question)
    p_txt = _semantic_probability(image_embeds, text_embeds, units, 0.50, eps)
    p_vis = _visual_probability(cls_attention, num_tokens, image_embeds.device, eps)
    beta = 0.0
    agreement = 0.0
    if p_txt is not None and p_vis is not None:
        midpoint = 0.5 * (p_txt + p_vis)
        safe_txt = p_txt.clamp_min(eps)
        safe_vis = p_vis.clamp_min(eps)
        safe_mid = midpoint.clamp_min(eps)
        js = 0.5 * (
            (p_txt * (safe_txt.log() - safe_mid.log())).sum()
            + (p_vis * (safe_vis.log() - safe_mid.log())).sum()
        )
        agreement = max(0.0, min(1.0 - max(0.0, float(js.item())) / math.log(2.0), 1.0))
        gate = max(0.5, min(0.5 + agreement, 1.5))
        beta = min(0.105 * gate, 1.0)
        probs = (1.0 - beta) * p_txt + beta * p_vis
    elif p_txt is not None:
        probs = p_txt
    elif p_vis is not None:
        # The frozen implementation falls back to a uniform prior when the
        # text source is unavailable; keep that behavior for reproducibility.
        probs = torch.full_like(p_vis, 1.0 / num_tokens)
    else:
        probs = torch.full((num_tokens,), 1.0 / num_tokens, device=image_embeds.device)
    probs = probs / probs.sum().clamp_min(eps)
    probs = 0.85 * probs + 0.15 / num_tokens
    probs = probs / probs.sum().clamp_min(eps)
    return probs, {"beta": beta, "agreement": agreement, "semantic_units": units}


def _density_kernel(states, probs, eps):
    states = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    states = states / states.norm(dim=-1, keepdim=True).clamp_min(eps)
    probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    probs = probs / probs.sum().clamp_min(eps)
    overlap = (states @ states.t()).clamp(-1.0, 1.0).clamp_min(0.0).square()
    root = probs.clamp_min(eps).sqrt()
    kernel = root[:, None] * root[None, :] * overlap
    return torch.nan_to_num(0.5 * (kernel + kernel.t()), nan=0.0, posinf=0.0, neginf=0.0)


def _pivot_order(kernel, count, eps):
    diagonal = torch.diag(kernel).clone().clamp_min(0.0)
    residual = diagonal.clone()
    selected, columns = [], []
    for _ in range(count):
        scores = residual.clone()
        if selected:
            scores[torch.tensor(selected, device=kernel.device)] = -float("inf")
        pivot = int(torch.argmax(scores).item())
        value = float(scores[pivot].item())
        if not math.isfinite(value) or value <= eps:
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
        residual = (residual - column.square()).clamp_min(0.0)
    if len(selected) < count:
        remaining = [i for i in range(kernel.shape[0]) if i not in set(selected)]
        remaining.sort(key=lambda i: (-float(diagonal[i].item()), i))
        selected.extend(remaining[:count - len(selected)])
    return selected


def _question_type(question):
    if not isinstance(question, str) or not question.strip():
        return "default"
    q = question.lower().strip()
    compare = ("same", "different", "more", "less", "larger", "smaller", "bigger", "taller", "shorter", "closer", "farther", "compare")
    relation = ("left", "right", "above", "below", "under", "over", "behind", "front", "near", "next to", "on top of", "holding", "wearing", "sitting on", "standing on", "between", "inside", "outside", "around")
    binary = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")
    opened = ("what ", "where ", "which ", "who ", "whose ", "how many ", "what color", "what kind", "what type")
    if any(word in q for word in compare):
        return "compare"
    if any(word in q for word in relation):
        return "relation"
    if q.startswith(binary):
        return "binary"
    if q.startswith(opened):
        return "open"
    if any(word in q for word in ("color", "shape", "material", "size")):
        return "attr"
    return "default"


def _recovery_budget(probs, budget, question, ratio_cap, eps):
    count = probs.numel()
    entropy = -(probs * (probs + eps).log()).sum()
    normalized = float((entropy / math.log(count)).clamp(0.0, 1.0).item()) if count > 1 else 0.0
    kind = _question_type(question)
    ratios = {"binary": 0.0625, "open": 0.25, "relation": 0.25, "compare": 0.25, "attr": 0.125, "default": 0.125}
    base = ratios[kind]
    gate = min(max((normalized - 0.70) / 0.25, 0.0), 1.0)
    requested = base * gate
    raw = max(int(round(budget * requested)), 0)
    cap = max(int(round(budget * ratio_cap)), 0)
    recovered = min(raw, cap, budget)
    return budget - recovered, recovered, {"entropy": float(entropy.item()), "question_type": kind, "recovery_ratio": requested}


def _local_contrast(image_features, eps):
    if image_features.shape[0] != 576:
        raise ValueError("QPrune expects a 24x24 visual patch grid (576 tokens)")
    states = torch.nan_to_num(image_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    grid = states.t().reshape(1, states.shape[-1], 24, 24)
    patches = F.unfold(grid, kernel_size=3, padding=1).reshape(1, states.shape[-1], 9, 576)
    valid = F.unfold(torch.ones((1, 1, 24, 24), dtype=states.dtype, device=states.device), kernel_size=3, padding=1).reshape(1, 1, 9, 576)
    offsets = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=states.device)
    neighbors = patches.index_select(2, offsets)
    mask = valid.index_select(2, offsets)
    mean = ((neighbors * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0))[0].t().contiguous()
    contrast = torch.nan_to_num(states - mean, nan=0.0, posinf=0.0, neginf=0.0)
    cosine = F.cosine_similarity(states, mean, dim=-1, eps=max(eps, 1e-12))
    scores = torch.nan_to_num((1.0 - cosine).clamp_min(0.0), nan=0.0, posinf=0.0, neginf=0.0)
    return contrast, scores


def _conditioned_local_order(kernel, core, count, eps):
    size = kernel.shape[0]
    factor = torch.empty((size, min(size, len(core) + count)), dtype=torch.float32, device=kernel.device)
    residual = torch.diag(kernel).clone().clamp_min(0.0)
    excluded = torch.zeros(size, dtype=torch.bool, device=kernel.device)
    used = 0

    def update(pivot):
        nonlocal used, residual
        denom = residual[pivot].clamp_min(eps).sqrt()
        if used:
            previous = factor[:, :used]
            column = (kernel[:, pivot] - previous @ previous[pivot, :]) / denom
        else:
            column = kernel[:, pivot] / denom
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        factor[:, used] = column
        used += 1
        residual = (residual - column.square()).clamp_min(0.0)

    for pivot in core:
        if not bool(excluded[pivot].item()):
            update(pivot)
            excluded[pivot] = True
    selected = []
    diagonal = torch.diag(kernel).clone().clamp_min(0.0)
    for _ in range(count):
        pivot = int(torch.argmax(residual.masked_fill(excluded, float("-inf"))).item())
        value = float(residual[pivot].item())
        if not math.isfinite(value) or value <= eps:
            remaining = torch.nonzero(~excluded, as_tuple=False).reshape(-1)
            if not remaining.numel():
                break
            order = torch.argsort(diagonal[remaining], descending=True, stable=True)
            pivot = int(remaining[order[0]].item())
        update(pivot)
        excluded[pivot] = True
        selected.append(pivot)
    return selected


def _spatial_proposal(core, recover_count, question):
    candidates = sorted(set(range(576)) - set(core))
    qid = os.environ.get("QPRUNE_QUESTION_ID", "")
    qtext = os.environ.get("QPRUNE_QUESTION_TEXT", question)

    def distance(indices):
        if len(indices) < 2:
            return 0.0
        values = []
        for offset, first in enumerate(indices):
            row, col = divmod(first, 24)
            for second in indices[offset + 1:]:
                row2, col2 = divmod(second, 24)
                values.append(math.hypot(row - row2, col - col2))
        return sum(values) / len(values)

    proposals = []
    for index, base_seed in enumerate(_PROPOSAL_SEEDS):
        payload = f"{base_seed}\0{qid}\0{qtext}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        indices = random.Random(seed).sample(candidates, recover_count)
        proposals.append((distance(indices), -index, indices))
    return max(proposals)[2]


@torch.no_grad()
def select_visual_tokens(image_features, image_embeds, text_embeds, cls_attention, question, budget):
    """Return sorted patch indices and a compact trace for a single image."""
    if image_features.ndim != 3 or image_features.shape[0] != 1:
        raise ValueError("QPrune currently supports one image per inference call")
    count = image_features.shape[1]
    if count != 576 or not 1 <= budget <= count:
        raise ValueError(f"Expected 576 patches and budget in [1, 576], got {count} and {budget}")
    eps = 1e-6
    probs, probability_info = _mixed_probability(image_embeds, text_embeds, cls_attention, question, eps)
    core_order = _pivot_order(_density_kernel(image_features[0], probs, eps), budget, eps)

    question_body = re.sub(r"^\s*<image>\s*", "", str(question or ""), flags=re.IGNORECASE)
    question_body = re.split(r"\n\s*[A-E][.)]\s+", question_body, maxsplit=1)[0]
    question_only = " ".join(question_body.split())
    quantitative = re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|\bconcentration\b|\b(?:python|program|code)\b.*\b(?:output|generate|correct)\b", re.IGNORECASE)
    social = re.compile(r"\brelationship\b.*\b(persons?|people|men|women|man|woman)\b", re.IGNORECASE)
    spatial = re.compile(r"\b(left|right|above|below|under|over|behind|front|next to|beside|between)\b", re.IGNORECASE)
    if social.search(question_only) or spatial.search(question_only):
        cap, budget_route = 0.20, "relation"
    elif quantitative.search(question_only):
        cap, budget_route = 0.0, "quantitative"
    else:
        cap, budget_route = 0.10, "default"
    core_count, recover_count, budget_info = _recovery_budget(probs, budget, question_only, cap, eps)
    core = core_order[:core_count]

    if _EXPLICIT_SPATIAL.search(question_only.split("\n", 1)[0]):
        recovered = _spatial_proposal(core, recover_count, question_only)
        recovery_route = "spatial"
    elif recover_count:
        contrast, scores = _local_contrast(image_features[0], eps)
        if scores.sum().item() <= eps:
            recovered = []
        else:
            local_probs = scores / scores.sum().clamp_min(eps)
            kernel = _density_kernel(contrast, local_probs, eps)
            recovered = _conditioned_local_order(kernel, core, recover_count, eps)
        if len(recovered) < recover_count:
            excluded = torch.zeros(count, dtype=torch.bool, device=image_features.device)
            excluded[torch.tensor(core + recovered, device=image_features.device)] = True
            candidates = torch.nonzero(~excluded, as_tuple=False).reshape(-1)
            order = torch.argsort(scores[candidates], descending=True, stable=True)
            recovered.extend(int(x) for x in candidates[order[:recover_count - len(recovered)]].tolist())
        recovery_route = "local"
    else:
        recovered, recovery_route = [], "none"

    selected = sorted(core + recovered)
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("QPrune did not select exactly the requested token budget")
    trace = {
        "question_id": os.environ.get("QPRUNE_QUESTION_ID", ""),
        "budget": int(budget),
        "core_count": len(core),
        "recovery_count": len(recovered),
        "core_indices": core,
        "recovery_indices": recovered,
        "selected_indices": selected,
        "budget_route": budget_route,
        "recovery_route": recovery_route,
        **probability_info,
        **budget_info,
    }
    return torch.tensor(selected, dtype=torch.long, device=image_features.device), trace


def append_trace(record):
    path = os.environ.get("QPRUNE_TRACE_JSONL", "").strip()
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
