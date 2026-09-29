import os
import re
import json
import math
import time

import torch
import torch.nn.functional as F


_EC_DEBUG_COUNT = 0
_QFID_STATS_COUNT = 0
_QFICR_RUNTIME_PROFILE_COUNT = 0
_QFID_SPECTRAL_WARNED = False
_SPATIAL_WORDS = {"left", "right", "above", "below", "behind", "front", "near", "under", "over"}
_ACTION_WORDS = {
    "holding", "wearing", "riding", "carrying", "eating", "drinking",
    "playing", "standing", "sitting", "walking", "running",
}
_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "does", "do", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "the", "there",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def normalize(x, eps=1e-8):
    """Min-max normalize a score vector without producing NaN or Inf."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x

    x_min = x.min()
    x_max = x.max()
    scale = x_max - x_min
    if not torch.isfinite(scale) or scale.abs().item() <= eps:
        return torch.zeros_like(x)
    return (x - x_min) / scale


@torch.no_grad()
def select_frozen_qfi_order(states, probs, steps, eps=1e-12, return_trace=False):
    """Reproduce the frozen V0 density-kernel pivot order in either feature space.

    This helper is intentionally narrower than ``ECPruner.select``: it accepts
    the already-computed frozen task probability, changes no budget or prior,
    and returns indices only. It is used by SSCR-Audit to isolate feature space.
    """
    if not torch.is_tensor(states) or states.ndim not in {2, 3}:
        raise ValueError("states must be [N,D] or [1,N,D]")
    if states.ndim == 3:
        if states.shape[0] != 1:
            raise ValueError("same-space QFi order requires batch size 1")
        states = states[0]
    probs = torch.as_tensor(probs, device=states.device, dtype=torch.float32).reshape(-1)
    num_tokens = int(states.shape[0])
    if probs.numel() != num_tokens:
        raise ValueError("states/probs token mismatch")
    steps = min(max(int(steps), 0), num_tokens)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    probs = probs / probs.sum().clamp_min(eps)
    psi = states.float() / states.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    overlap = compute_projection_overlap_kernel(
        psi, eps=eps, overlap_kernel="relu_square", assume_normalized=True
    )
    sqrt_p = torch.sqrt(probs.clamp_min(eps))
    kernel = sqrt_p[:, None] * sqrt_p[None, :] * overlap
    kernel = torch.nan_to_num(0.5 * (kernel + kernel.t()), nan=0.0, posinf=0.0, neginf=0.0)
    diagonal = torch.diag(kernel).clone().clamp_min(0.0)
    residual = diagonal.clone()
    selected = []
    columns = []
    trace = []
    while len(selected) < steps:
        scores = residual.clone()
        if selected:
            scores[torch.tensor(selected, device=states.device, dtype=torch.long)] = -float("inf")
        trace.append(scores.detach().clone())
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
        residual = (residual - column.square()).clamp_min(0.0)
    if len(selected) < steps:
        remaining = [i for i in range(num_tokens) if i not in set(selected)]
        remaining.sort(key=lambda i: (-float(diagonal[i].item()), i))
        selected.extend(remaining[:steps - len(selected)])
    order = torch.tensor(selected, dtype=torch.long, device=states.device)
    if return_trace:
        return order, trace
    return order


def safe_normalize(x, eps=1e-8):
    return normalize(x, eps=eps)


def get_qfid_overlap_kernel(default="relu_square"):
    kernel = os.environ.get("EC_QFID_OVERLAP_KERNEL", default).strip().lower()
    if kernel not in {"relu_square", "abs_square", "cosine"}:
        kernel = default
    return kernel


def compute_projection_overlap_kernel(tokens, eps=1e-12, overlap_kernel=None, assume_normalized=False):
    """
    Build the token-to-token overlap matrix used by QFi-CR coverage and recovery.

    relu_square is the positive projection overlap kernel:
        kappa_ij = max(0, <psi_i, psi_j>)^2
    It is not ordinary abs cosine squared. Negative correlations are clamped
    before squaring so they are not incorrectly treated as high-overlap states.
    abs_square keeps the previous ablation behavior, and cosine uses max(0, cos).
    """
    if not torch.is_tensor(tokens) or tokens.ndim != 2:
        shape = tuple(tokens.shape) if torch.is_tensor(tokens) else type(tokens)
        raise ValueError(f"tokens must be [N, D], got {shape}")
    safe_eps = max(float(eps), 1e-12)
    kernel = overlap_kernel or get_qfid_overlap_kernel()
    x = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if not assume_normalized:
        x = F.normalize(x, dim=-1, eps=safe_eps)
    sim = torch.matmul(x, x.transpose(-1, -2)).clamp(-1.0, 1.0)
    if kernel == "abs_square":
        kappa = sim.abs().pow(2)
    elif kernel == "cosine":
        kappa = sim.clamp_min(0.0)
    else:
        kappa = sim.clamp_min(0.0).pow(2)
    return torch.nan_to_num(kappa.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def get_env_bool(name, default=False):
    value = os.environ.get(name, None)
    if value is None:
        return default
    return str(value).strip().lower() in ["1", "true", "yes", "y", "on"]


def budget_calibrate_prob(
    prob,
    keep_num,
    alpha=3.0,
    gamma_min=0.05,
    gamma_max=20.0,
    iters=30,
    eps=1e-12,
):
    """Power-rescale probabilities to match a budget-dependent effective support."""
    if not torch.is_tensor(prob) or prob.ndim not in {1, 2}:
        raise ValueError("prob must be a [N] or [B, N] tensor")
    if prob.shape[-1] == 0:
        raise ValueError("prob must contain at least one token")
    if not prob.is_floating_point():
        raise ValueError("prob must use a floating-point dtype")

    squeeze = prob.ndim == 1
    rows = prob.unsqueeze(0) if squeeze else prob
    original_dtype = rows.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    work = torch.nan_to_num(rows.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0)
    safe_eps = max(float(eps), float(torch.finfo(work_dtype).tiny))
    work = work.clamp_min(safe_eps)
    work = work / work.sum(dim=-1, keepdim=True).clamp_min(safe_eps)

    num_tokens = int(work.shape[-1])
    keep_num = max(1, min(int(keep_num), num_tokens))
    target_neff = min(float(num_tokens), max(float(keep_num), float(alpha) * keep_num))
    gamma_lo = max(float(gamma_min), safe_eps)
    gamma_hi = max(float(gamma_max), safe_eps)
    if gamma_lo > gamma_hi:
        gamma_lo, gamma_hi = gamma_hi, gamma_lo
    search_iters = max(0, int(iters))

    calibrated_rows = []
    neff_before_values = []
    neff_after_values = []
    gamma_values = []

    def rescale(row, gamma):
        logits = float(gamma) * row.log()
        return torch.softmax(logits, dim=-1)

    def effective_support(row):
        safe_row = row.clamp_min(safe_eps)
        entropy = -(safe_row * safe_row.log()).sum()
        return float(torch.exp(entropy).item())

    for row in work:
        neff_before = effective_support(row)
        low_prob = rescale(row, gamma_lo)
        high_prob = rescale(row, gamma_hi)
        low_neff = effective_support(low_prob)
        high_neff = effective_support(high_prob)

        if target_neff >= low_neff:
            best_gamma, best_prob, best_neff = gamma_lo, low_prob, low_neff
        elif target_neff <= high_neff:
            best_gamma, best_prob, best_neff = gamma_hi, high_prob, high_neff
        else:
            lo = gamma_lo
            hi = gamma_hi
            for _ in range(search_iters):
                mid = 0.5 * (lo + hi)
                mid_prob = rescale(row, mid)
                mid_neff = effective_support(mid_prob)
                if mid_neff > target_neff:
                    lo = mid
                else:
                    hi = mid

            candidates = []
            for gamma in (lo, 0.5 * (lo + hi), hi):
                candidate_prob = rescale(row, gamma)
                candidate_neff = effective_support(candidate_prob)
                candidates.append((abs(candidate_neff - target_neff), gamma, candidate_prob, candidate_neff))
            _, best_gamma, best_prob, best_neff = min(candidates, key=lambda item: item[0])

        calibrated_rows.append(best_prob)
        neff_before_values.append(neff_before)
        neff_after_values.append(best_neff)
        gamma_values.append(float(best_gamma))

    calibrated = torch.stack(calibrated_rows, dim=0).to(dtype=original_dtype, device=prob.device)
    calibrated = calibrated / calibrated.sum(dim=-1, keepdim=True).clamp_min(
        max(float(eps), float(torch.finfo(original_dtype).tiny))
    )

    def scalar_or_list(values):
        return float(values[0]) if squeeze else [float(value) for value in values]

    stats = {
        "budget_alpha": float(alpha),
        "budget_target_neff": float(target_neff) if squeeze else [float(target_neff)] * len(calibrated_rows),
        "budget_neff_before": scalar_or_list(neff_before_values),
        "budget_neff_after": scalar_or_list(neff_after_values),
        "budget_gamma": scalar_or_list(gamma_values),
    }
    return (calibrated.squeeze(0) if squeeze else calibrated), stats


def spectral_filter_kernel(
    kernel: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-12,
    trace_norm: bool = True,
):
    """
    Apply a PSD spectral transform K -> U diag(lambda^gamma) U^T.

    Returns:
        filtered_kernel: tensor with the same shape/device/dtype as kernel
        spectral_info: diagnostic dictionary for debug logging / sidecars
    """
    if not torch.is_tensor(kernel) or kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square [N, N] tensor")
    if not kernel.is_floating_point():
        raise ValueError("kernel must use a floating-point dtype")

    original_dtype = kernel.dtype
    original_device = kernel.device
    work = 0.5 * (kernel + kernel.t())
    work = torch.nan_to_num(work.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_eps = max(float(eps), float(torch.finfo(work.dtype).tiny))
    gamma_value = float(gamma)

    evals, evecs = torch.linalg.eigh(work)
    evals = torch.nan_to_num(evals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    evals_f = torch.where(evals > safe_eps, evals.pow(gamma_value), torch.zeros_like(evals))
    filtered = (evecs * evals_f.unsqueeze(0)) @ evecs.t()
    filtered = 0.5 * (filtered + filtered.t())
    filtered = torch.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

    trace_before = torch.trace(work).clamp_min(safe_eps)
    trace_after = torch.trace(filtered).clamp_min(safe_eps)
    if trace_norm:
        filtered = filtered * (trace_before / trace_after)
        filtered = 0.5 * (filtered + filtered.t())
        filtered = torch.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        trace_after = torch.trace(filtered).clamp_min(safe_eps)

    info = {
        "spectral_filter": True,
        "spectral_gamma": gamma_value,
        "spectral_trace_before": float(trace_before.item()),
        "spectral_trace_after": float(trace_after.item()),
        "spectral_rank_pos": int((evals > safe_eps).sum().item()),
        "spectral_min_eval": float(evals.min().item()) if evals.numel() > 0 else 0.0,
        "spectral_max_eval": float(evals.max().item()) if evals.numel() > 0 else 0.0,
        "spectral_trace_norm": bool(trace_norm),
    }
    return filtered.to(device=original_device, dtype=original_dtype), info


def select_by_evidence_recovery(states, probs, k, eps=1e-12):
    """
    Greedy selection by uncovered evidence recovery.

    states: visual token features, shape [N, D]
    probs: final calibrated observability / evidence probability, shape [N]
    k: number of selected tokens

    Return:
        selected_indices, shape [min(k, N)]
    """
    if not torch.is_tensor(states) or states.ndim != 2:
        shape = tuple(states.shape) if torch.is_tensor(states) else type(states)
        raise ValueError(f"states must be [N, D], got {shape}")
    if not torch.is_tensor(probs) or probs.ndim != 1:
        shape = tuple(probs.shape) if torch.is_tensor(probs) else type(probs)
        raise ValueError(f"probs must be [N], got {shape}")
    if states.shape[0] != probs.numel():
        raise ValueError(f"states/probs token mismatch: {states.shape[0]} vs {probs.numel()}")

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    if select_num == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    safe_eps = max(float(eps), 1e-12)
    psi = F.normalize(
        torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=safe_eps,
    )
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)

    evidence_kernel = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)

    selected = []
    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)

    for _ in range(select_num):
        marginal = (evidence_kernel - coverage[:, None]).clamp_min(0.0)
        gain = (p[:, None] * marginal).sum(dim=0)
        gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        gain = gain.masked_fill(~available, float("-inf"))
        idx = torch.argmax(gain)
        selected.append(idx)
        available[idx] = False
        coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    return torch.stack(selected).long()


def select_by_direct_coverage_audit(states, probs, k, audit_mode, eps=1e-12):
    """Run one controlled direct facility-location objective from an empty set."""
    valid_modes = {
        "direct_task_relu_square": ("relu_square", False),
        "direct_task_relu_cosine": ("cosine", False),
        "direct_uniform_relu_cosine": ("cosine", True),
    }
    if audit_mode not in valid_modes:
        raise ValueError(f"unsupported direct objective audit mode: {audit_mode}")
    if not torch.is_tensor(states) or states.ndim != 2:
        raise ValueError(f"states must be [N, D], got {getattr(states, 'shape', type(states))}")
    if not torch.is_tensor(probs) or probs.ndim != 1 or probs.numel() != states.shape[0]:
        raise ValueError("probs must be [N] and match states")

    kernel_name, use_uniform = valid_modes[audit_mode]
    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    safe_eps = max(float(eps), 1e-12)
    p_task = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p_task = p_task / p_task.sum().clamp_min(safe_eps)
    p_select = torch.full_like(p_task, 1.0 / max(num_tokens, 1)) if use_uniform else p_task

    affinity_start = time.perf_counter()
    psi = F.normalize(
        torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=safe_eps,
    )
    affinity = compute_projection_overlap_kernel(
        psi,
        eps=safe_eps,
        overlap_kernel=kernel_name,
        assume_normalized=True,
    )
    affinity_latency_ms = (time.perf_counter() - affinity_start) * 1000.0

    greedy_start = time.perf_counter()
    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    coverage = torch.zeros(num_tokens, dtype=affinity.dtype, device=device)
    coverage_trace = [float((p_select * coverage).sum().item())]
    selected = []
    for _ in range(select_num):
        gain = (p_select[:, None] * (affinity - coverage[:, None]).clamp_min(0.0)).sum(dim=0)
        gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        gain = gain.masked_fill(~available, float("-inf"))
        # torch.argmax returns the lowest index on ties, making the audit deterministic.
        idx = int(torch.argmax(gain).item())
        selected.append(idx)
        available[idx] = False
        coverage = torch.maximum(coverage, affinity[:, idx])
        coverage_trace.append(float((p_select * coverage).sum().item()))
    greedy_latency_ms = (time.perf_counter() - greedy_start) * 1000.0

    keep = torch.tensor(selected, dtype=torch.long, device=device)
    selected_p = p_task[keep] if keep.numel() else p_task[:0]
    info = {
        "objective_audit_mode": audit_mode,
        "objective_target_weight": "uniform" if use_uniform else "task_conditioned",
        "objective_overlap_kernel": kernel_name,
        "ratio_cap_active": False,
        "recover_budget_mode": "inactive",
        "recover_ratio": 0.0,
        "recover_ratio_requested": 0.0,
        "recover_ratio_min": 0.0,
        "recover_ratio_max": 0.0,
        "recover_ratio_cap": 0.0,
        "recover_abs_cap": 0,
        "recover_raw_k": 0,
        "recover_cap_applied": False,
        "recover_entropy_hat": 0.0,
        "recover_entropy_gamma": 1.0,
        "adapt_mode": "direct_greedy",
        "question_type": "not_used",
        "entropy": float((-(p_task * (p_task + safe_eps).log()).sum()).item()),
        "entropy_norm": float((-(p_task * (p_task + safe_eps).log()).sum() / math.log(max(num_tokens, 2))).item()),
        "k_core": select_num,
        "k_recover": 0,
        "recover_prior_anchor": False,
        "recover_prior_gamma": 0.0,
        "anchor_alpha": 0.0,
        "comp_score_mode": "direct_unanchored",
        "recover_cand_pool": False,
        "recover_cand_mult": 0.0,
        "recover_cand_size": num_tokens,
        "residual_alloc_mode": "direct",
        "compete_margin": 0.0,
        "K": select_num,
        "max_flexible_slots": 0,
        "fixed_core_count": select_num,
        "actual_residual_count": 0,
        "actual_core_count": select_num,
        "candidate_pool_enabled": False,
        "candidate_pool_size": num_tokens,
        "candidate_pool_multiplier": 0.0,
        "full_core_indices": selected,
        "fixed_core_indices": selected,
        "reserved_core_indices": [],
        "selected_core_indices": selected,
        "selected_residual_indices": [],
        "fallback_core_indices": [],
        "final_selected_indices": selected,
        "question_text_available": False,
        "restoration_mode": "inactive",
        "fixed_restoration_ratio": 0.0,
        "yn_gate_applied": False,
        "residual_utilization": 0.0,
        "fallback_to_core_rate": 0.0,
        "coverage_trace": coverage_trace,
        "coverage_monotonic": all(
            coverage_trace[i + 1] + 1e-8 >= coverage_trace[i]
            for i in range(len(coverage_trace) - 1)
        ),
        "task_weighted_coverage": float((p_task * coverage).sum().item()),
        "uniform_coverage": float(coverage.mean().item()) if coverage.numel() else 0.0,
        "mean_selected_p": float(selected_p.mean().item()) if selected_p.numel() else 0.0,
        "affinity_latency_ms": affinity_latency_ms,
        "greedy_latency_ms": greedy_latency_ms,
    }
    return keep, info


def _relation_residual_trace_from_gram(gram, indices, eps=1e-12):
    """Return the residual trace after projecting all relation rows onto indices."""
    safe_eps = max(float(eps), 1e-12)
    residual = torch.diag(gram).clone().clamp_min(0.0)
    columns = []
    trace = [float(residual.sum().item())]
    for pivot in indices:
        pivot = int(pivot)
        pivot_residual = residual[pivot]
        if float(pivot_residual.item()) <= safe_eps:
            trace.append(trace[-1])
            continue
        if columns:
            prev = torch.stack(columns, dim=1)
            correction = prev @ prev[pivot, :]
            column = (gram[:, pivot] - correction) / pivot_residual.sqrt()
        else:
            column = gram[:, pivot] / pivot_residual.sqrt()
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(column)
        residual = (residual - column.square()).clamp_min(0.0)
        next_trace = min(trace[-1], float(residual.sum().item()))
        trace.append(next_trace)
    return residual, trace


def select_by_task_relation_structure(states, probs, k, mode, initial_indices=None, eps=1e-12):
    """Greedily preserve signed task-conditioned relation-row structure."""
    if mode not in {"relation_direct", "relation_restore"}:
        raise ValueError(f"unsupported relation audit mode: {mode}")
    if not torch.is_tensor(states) or states.ndim != 2:
        raise ValueError("states must be [N, D]")
    if not torch.is_tensor(probs) or probs.ndim != 1 or probs.numel() != states.shape[0]:
        raise ValueError("probs must be [N] and match states")

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    safe_eps = max(float(eps), 1e-12)
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)

    def sync_timing():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync_timing()
    affinity_start = time.perf_counter()
    psi = F.normalize(
        torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=safe_eps,
    )
    root_p = p.sqrt()
    relation = (root_p[:, None] * (psi @ psi.t())) * root_p[None, :]
    relation = torch.nan_to_num(relation.float(), nan=0.0, posinf=0.0, neginf=0.0)
    gram = relation @ relation.t()
    gram = torch.nan_to_num(0.5 * (gram + gram.t()), nan=0.0, posinf=0.0, neginf=0.0)
    sync_timing()
    affinity_latency_ms = (time.perf_counter() - affinity_start) * 1000.0

    initial = []
    seen = set()
    for value in initial_indices or []:
        idx = int(value)
        if 0 <= idx < num_tokens and idx not in seen and len(initial) < select_num:
            initial.append(idx)
            seen.add(idx)

    sync_timing()
    greedy_start = time.perf_counter()
    residual, relation_trace = _relation_residual_trace_from_gram(gram, initial, eps=safe_eps)
    selected = list(initial)
    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if selected:
        available[torch.tensor(selected, dtype=torch.long, device=device)] = False
    columns = []
    # Reconstruct the incremental factor once so subsequent pivots update in O(NK).
    working = torch.diag(gram).clone().clamp_min(0.0)
    for pivot in initial:
        pivot_residual = working[pivot]
        if float(pivot_residual.item()) <= safe_eps:
            continue
        if columns:
            prev = torch.stack(columns, dim=1)
            column = (gram[:, pivot] - prev @ prev[pivot, :]) / pivot_residual.sqrt()
        else:
            column = gram[:, pivot] / pivot_residual.sqrt()
        column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(column)
        working = (working - column.square()).clamp_min(0.0)
    residual = working

    while len(selected) < select_num:
        score = residual.masked_fill(~available, float("-inf"))
        pivot = int(torch.argmax(score).item())
        selected.append(pivot)
        available[pivot] = False
        pivot_residual = residual[pivot]
        if float(pivot_residual.item()) > safe_eps:
            if columns:
                prev = torch.stack(columns, dim=1)
                column = (gram[:, pivot] - prev @ prev[pivot, :]) / pivot_residual.sqrt()
            else:
                column = gram[:, pivot] / pivot_residual.sqrt()
            column = torch.nan_to_num(column.float(), nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(column)
            residual = (residual - column.square()).clamp_min(0.0)
        relation_trace.append(min(relation_trace[-1], float(residual.sum().item())))
    sync_timing()
    greedy_latency_ms = (time.perf_counter() - greedy_start) * 1000.0

    keep = torch.tensor(selected, dtype=torch.long, device=device)
    initial_trace = relation_trace[0] if relation_trace else 0.0
    final_trace = relation_trace[-1] if relation_trace else 0.0
    if keep.numel():
        q, _ = torch.linalg.qr(relation[keep].t(), mode="reduced")
        eye = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
        orth_error = float((q.t() @ q - eye).abs().max().item()) if q.numel() else 0.0
    else:
        orth_error = 0.0

    relu_square = (psi @ psi.t()).clamp_min(0.0).square()
    coverage = relu_square[:, keep].max(dim=1).values if keep.numel() else torch.zeros_like(p)
    core_count = len(initial)
    residual_indices = selected[core_count:]
    info = {
        "relation_audit_mode": mode,
        "objective_audit_mode": "legacy_current",
        "objective_target_weight": "task_conditioned",
        "objective_overlap_kernel": "signed_relation",
        "adapt_mode": mode,
        "question_type": "not_used",
        "entropy": float((-(p * (p + safe_eps).log()).sum()).item()),
        "entropy_norm": float((-(p * (p + safe_eps).log()).sum() / math.log(max(num_tokens, 2))).item()),
        "recover_ratio": 0.0 if mode == "relation_direct" else float(len(residual_indices) / max(select_num, 1)),
        "recover_ratio_requested": 0.0 if mode == "relation_direct" else float(len(residual_indices) / max(select_num, 1)),
        "recover_ratio_min": 0.0,
        "recover_ratio_max": 0.0,
        "recover_ratio_cap": 0.0,
        "recover_abs_cap": 0,
        "recover_raw_k": 0 if mode == "relation_direct" else len(residual_indices),
        "recover_cap_applied": False,
        "recover_entropy_hat": 0.0,
        "recover_entropy_gamma": 1.0,
        "recover_budget_mode": "inactive" if mode == "relation_direct" else "legacy_core_budget",
        "k_core": select_num if mode == "relation_direct" else core_count,
        "k_recover": 0 if mode == "relation_direct" else len(residual_indices),
        "recover_prior_anchor": False,
        "recover_prior_gamma": 0.0,
        "anchor_alpha": 0.0,
        "comp_score_mode": "relation_residual",
        "recover_cand_pool": False,
        "recover_cand_mult": 0.0,
        "recover_cand_size": num_tokens - core_count,
        "ratio_cap_active": mode == "relation_restore",
        "residual_alloc_mode": mode,
        "compete_margin": 0.0,
        "K": select_num,
        "max_flexible_slots": 0 if mode == "relation_direct" else len(residual_indices),
        "fixed_core_count": core_count if mode == "relation_restore" else select_num,
        "actual_residual_count": len(residual_indices) if mode == "relation_restore" else 0,
        "actual_core_count": core_count if mode == "relation_restore" else select_num,
        "candidate_pool_enabled": False,
        "candidate_pool_size": num_tokens - core_count,
        "candidate_pool_multiplier": 0.0,
        "full_core_indices": initial,
        "fixed_core_indices": initial if mode == "relation_restore" else selected,
        "reserved_core_indices": [],
        "selected_core_indices": initial if mode == "relation_restore" else selected,
        "selected_residual_indices": residual_indices if mode == "relation_restore" else [],
        "fallback_core_indices": [],
        "final_selected_indices": selected,
        "question_text_available": False,
        "restoration_mode": "relation" if mode == "relation_restore" else "inactive",
        "fixed_restoration_ratio": 0.0,
        "yn_gate_applied": False,
        "residual_utilization": 1.0 if mode == "relation_restore" and residual_indices else 0.0,
        "fallback_to_core_rate": 0.0,
        "relation_residual_trace": relation_trace,
        "relation_residual_monotonic": all(relation_trace[i + 1] <= relation_trace[i] + 1e-7 for i in range(len(relation_trace) - 1)),
        "relation_reconstruction_residual": float(final_trace / max(initial_trace, safe_eps)),
        "relation_reconstruction_residual_raw": final_trace,
        "relation_initial_residual_raw": initial_trace,
        "relation_qr_orthogonality_error": orth_error,
        "relation_nan_inf_count": int((~torch.isfinite(relation)).sum().item()),
        "task_weighted_coverage": float((p * coverage).sum().item()),
        "uniform_coverage": float(coverage.mean().item()) if coverage.numel() else 0.0,
        "mean_selected_p": float(p[keep].mean().item()) if keep.numel() else 0.0,
        "affinity_latency_ms": affinity_latency_ms,
        "greedy_latency_ms": greedy_latency_ms,
    }
    return keep, info, gram


def select_by_core_then_recover(states, probs, k, qfi_core_indices, core_ratio=0.5, eps=1e-12):
    """
    Core-first evidence recovery.

    Stage 1 uses the original QFi residual pivot order for the core tokens.
    Stage 2 greedily recovers uncovered evidence from the remaining tokens.
    """
    if not torch.is_tensor(states) or states.ndim != 2:
        shape = tuple(states.shape) if torch.is_tensor(states) else type(states)
        raise ValueError(f"states must be [N, D], got {shape}")
    if not torch.is_tensor(probs) or probs.ndim != 1:
        shape = tuple(probs.shape) if torch.is_tensor(probs) else type(probs)
        raise ValueError(f"probs must be [N], got {shape}")
    if states.shape[0] != probs.numel():
        raise ValueError(f"states/probs token mismatch: {states.shape[0]} vs {probs.numel()}")
    if not torch.is_tensor(qfi_core_indices) or qfi_core_indices.ndim != 1:
        shape = tuple(qfi_core_indices.shape) if torch.is_tensor(qfi_core_indices) else type(qfi_core_indices)
        raise ValueError(f"qfi_core_indices must be [M], got {shape}")

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    if select_num == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    ratio = min(max(float(core_ratio), 0.0), 1.0)
    core_num = min(max(int(round(select_num * ratio)), 0), select_num)
    safe_eps = max(float(eps), 1e-12)

    qfi_order = qfi_core_indices.to(device=device, dtype=torch.long)
    qfi_order = qfi_order[(qfi_order >= 0) & (qfi_order < num_tokens)]
    if qfi_order.numel() > 0:
        seen = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        ordered_unique = []
        for idx in qfi_order.tolist():
            if not bool(seen[idx].item()):
                seen[idx] = True
                ordered_unique.append(int(idx))
        qfi_order = torch.tensor(ordered_unique, dtype=torch.long, device=device)

    if core_num > 0:
        core_indices = qfi_order[:core_num]
    else:
        core_indices = torch.empty(0, dtype=torch.long, device=device)

    psi = F.normalize(
        torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=safe_eps,
    )
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)

    evidence_kernel = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)

    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if core_indices.numel() > 0:
        available[core_indices] = False
        coverage = evidence_kernel[:, core_indices].max(dim=1).values
    else:
        coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)

    selected = [int(idx) for idx in core_indices.tolist()]
    while len(selected) < select_num:
        marginal = (evidence_kernel - coverage[:, None]).clamp_min(0.0)
        gain = (p[:, None] * marginal).sum(dim=0)
        gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        gain = gain.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(gain).item())
        if not bool(available[idx].item()):
            break
        selected.append(idx)
        available[idx] = False
        coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    if len(selected) < select_num:
        filler = torch.nonzero(available, as_tuple=False).reshape(-1)
        selected.extend(filler[: select_num - len(selected)].tolist())

    return torch.tensor(selected[:select_num], dtype=torch.long, device=device)


def infer_question_recovery_type(question):
    """Infer a coarse recovery type from raw question text without using labels."""
    if not isinstance(question, str) or not question.strip():
        return "default"

    q = question.lower().strip()
    binary_starts = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "has ", "have ", "had ",
    )
    relation_words = (
        "left", "right", "above", "below", "under", "over",
        "behind", "front", "near", "next to", "on top of",
        "holding", "wearing", "sitting on", "standing on",
        "between", "inside", "outside", "around",
    )
    compare_words = (
        "same", "different", "more", "less", "larger", "smaller",
        "bigger", "taller", "shorter", "closer", "farther", "compare",
    )
    open_starts = (
        "what ", "where ", "which ", "who ", "whose ", "how many ",
        "what color", "what kind", "what type",
    )
    attr_words = ("color", "shape", "material", "size")

    if any(word in q for word in compare_words):
        return "compare"
    if any(word in q for word in relation_words):
        return "relation"
    if q.startswith(binary_starts):
        return "binary"
    if q.startswith(open_starts):
        return "open"
    if any(word in q for word in attr_words):
        return "attr"
    return "default"


def infer_yes_no_question(question):
    """Conservative text-only yes/no detector used by optional budget gating."""
    if not isinstance(question, str) or not question.strip():
        return False
    q = question.lower().replace("<image>", " ").strip()
    first = q.split(None, 1)[0] if q else ""
    return first in {
        "is", "are", "do", "does", "did", "can", "could",
        "has", "have", "had", "was", "were", "will", "would", "should",
    }


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def compute_qficr_entropy_recovery_budget(
    probs,
    k,
    question_text=None,
    eps=1e-12,
    ratio_cap_override=None,
):
    """Replay the frozen QFi-CR entropy/question recovery-budget rule.

    This helper computes only a budget.  It never constructs recovery scores or
    ranks tokens, which lets selector variants reuse the validated Old QFi-CR
    allocation without reintroducing task probability into their local ranking.
    """
    safe_eps = max(float(eps), 1e-12)
    p = torch.as_tensor(probs, dtype=torch.float32).reshape(-1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)
    num_tokens = int(p.numel())
    select_num = min(max(0, int(k)), num_tokens)

    entropy = -(p * (p + safe_eps).log()).sum()
    entropy_norm = 0.0
    if num_tokens > 1:
        entropy_norm = float(
            (entropy / math.log(float(num_tokens))).clamp(0.0, 1.0).item()
        )

    mode_requested = os.environ.get("EC_QFID_ADAPT_MODE", "entropy_question").strip().lower()
    question_available = isinstance(question_text, str) and bool(question_text.strip())
    if mode_requested == "entropy_question" and question_available:
        adapt_mode = "entropy_question"
        question_type = infer_question_recovery_type(question_text)
    else:
        adapt_mode = "entropy_only"
        question_type = "default"

    ratio_min = min(max(_env_float("EC_QFID_ADAPT_RECOVER_MIN_RATIO", "0.00"), 0.0), 1.0)
    ratio_max = min(max(_env_float("EC_QFID_ADAPT_RECOVER_MAX_RATIO", "0.25"), 0.0), 1.0)
    if ratio_min > ratio_max:
        ratio_min, ratio_max = ratio_max, ratio_min
    ratio_by_type = {
        "binary": _env_float("EC_QFID_ADAPT_RECOVER_BINARY_RATIO", "0.0625"),
        "open": _env_float("EC_QFID_ADAPT_RECOVER_OPEN_RATIO", "0.25"),
        "relation": _env_float("EC_QFID_ADAPT_RECOVER_REL_RATIO", "0.25"),
        "compare": _env_float("EC_QFID_ADAPT_RECOVER_COMPARE_RATIO", "0.25"),
        "attr": _env_float("EC_QFID_ADAPT_RECOVER_ATTR_RATIO", "0.125"),
        "default": _env_float("EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO", "0.125"),
    }
    ratio_base = min(
        max(float(ratio_by_type.get(question_type, ratio_by_type["default"])), ratio_min),
        ratio_max,
    )
    entropy_gamma = max(_env_float("EC_QFICR_RECOVER_ENTROPY_GAMMA", "1.0"), safe_eps)
    entropy_gate = get_env_bool("EC_QFID_ADAPT_ENTROPY_GATE", default=True)
    if entropy_gate:
        entropy_low = _env_float("EC_QFID_ADAPT_ENTROPY_LOW", "0.70")
        entropy_high = _env_float("EC_QFID_ADAPT_ENTROPY_HIGH", "0.95")
        denom = max(entropy_high - entropy_low, safe_eps)
        entropy_hat = min(max((entropy_norm - entropy_low) / denom, 0.0), 1.0)
        recover_ratio = ratio_min + (ratio_base - ratio_min) * (entropy_hat ** entropy_gamma)
    else:
        entropy_low = None
        entropy_high = None
        entropy_hat = min(max(float(entropy_norm), 0.0), 1.0)
        recover_ratio = ratio_base
    recover_ratio = min(max(float(recover_ratio), ratio_min), ratio_max)

    raw_recover_num = max(int(round(select_num * recover_ratio)), 0)
    ratio_cap = (
        float(ratio_cap_override)
        if ratio_cap_override is not None
        else _env_float("EC_QFICR_RECOVER_RATIO_CAP", "0.25")
    )
    ratio_cap = min(max(ratio_cap, 0.0), 1.0)
    ratio_cap_num = max(int(round(select_num * ratio_cap)), 0)
    recover_num = min(raw_recover_num, ratio_cap_num, select_num)
    return {
        "adapt_mode": adapt_mode,
        "question_type": question_type,
        "question_text_available": bool(question_available),
        "entropy": float(entropy.item()),
        "entropy_norm": float(entropy_norm),
        "entropy_hat": float(entropy_hat),
        "entropy_gate": bool(entropy_gate),
        "entropy_low": entropy_low,
        "entropy_high": entropy_high,
        "entropy_gamma": float(entropy_gamma),
        "recover_ratio_min": float(ratio_min),
        "recover_ratio_max": float(ratio_max),
        "recover_ratio_base": float(ratio_base),
        "recover_ratio_requested": float(recover_ratio),
        "raw_recover_k": int(raw_recover_num),
        "ratio_cap": float(ratio_cap),
        "ratio_cap_k": int(ratio_cap_num),
        "ratio_cap_applied": bool(raw_recover_num > recover_num),
        "k_recover": int(recover_num),
        "k_core": int(select_num - recover_num),
    }


def _write_qficr_restoration_audit_probe(payload):
    """Write an opt-in, offline restoration probe cache.

    This is deliberately disabled unless the caller supplies an explicit
    directory.  It records selection-time tensors only; it does not alter
    ranking, recovery, or the returned indices.
    """
    root = os.environ.get("EC_QFICR_RESTORATION_AUDIT_CACHE_DIR", "").strip()
    if not root:
        return
    question_id = os.environ.get("EC_QFID_DEBUG_QUESTION_ID", "unknown").strip() or "unknown"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", question_id)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"selection_{safe_id}.pt")
    torch.save(payload, path)


def _qficr_grid_positions(num_tokens, device):
    side = int(round(math.sqrt(max(1, int(num_tokens)))))
    if side * side != int(num_tokens):
        return None
    rows = torch.arange(side, device=device, dtype=torch.float32).repeat_interleave(side)
    cols = torch.arange(side, device=device, dtype=torch.float32).repeat(side)
    return torch.stack([rows, cols], dim=-1)


def _qficr_spatial_local_prior(num_tokens, core_indices, probs, eps, prior_ablation="full"):
    """Build the optional spatial-local recovery prior b_j for QFi-CR ablations."""
    device = probs.device
    prior_ablation = (prior_ablation or "full").strip().lower()
    zero = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    if prior_ablation == "none" or num_tokens <= 0:
        return zero, zero, zero

    spatial_prior = zero.clone()
    positions = _qficr_grid_positions(num_tokens, device)
    if positions is not None and torch.is_tensor(core_indices) and core_indices.numel() > 0:
        core = core_indices.to(device=device, dtype=torch.long)
        core = core[(core >= 0) & (core < num_tokens)]
        if core.numel() > 0:
            dist = torch.cdist(positions, positions[core], p=2.0).min(dim=1).values
            spatial_prior = normalize(dist, eps=eps)

    local_prior = zero.clone()
    side = int(round(math.sqrt(max(1, int(num_tokens)))))
    if side * side == num_tokens:
        p_grid = probs.float().reshape(side, side)
        local_mean = torch.zeros_like(p_grid)
        local_count = torch.zeros_like(p_grid)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src_r0 = max(0, -dr)
            src_r1 = side - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = side - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = side - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = side - max(0, -dc)
            local_mean[dst_r0:dst_r1, dst_c0:dst_c1] += p_grid[src_r0:src_r1, src_c0:src_c1]
            local_count[dst_r0:dst_r1, dst_c0:dst_c1] += 1.0
        local_mean = local_mean / local_count.clamp_min(float(eps))
        local_prior = normalize((p_grid - local_mean).clamp_min(0.0).reshape(-1), eps=eps)

    if prior_ablation == "no_spatial":
        combined = local_prior
    elif prior_ablation == "no_local":
        combined = spatial_prior
    else:
        combined = 0.5 * (spatial_prior + local_prior)
    return normalize(combined, eps=eps), spatial_prior, local_prior


def _qficr_vector_stats(x):
    x = torch.nan_to_num(x.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return {k: 0.0 for k in ("mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99")}
    qs = torch.quantile(x, torch.tensor([0.25, 0.50, 0.75, 0.90, 0.95, 0.99], device=x.device))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "p25": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p75": float(qs[2].item()),
        "p90": float(qs[3].item()),
        "p95": float(qs[4].item()),
        "p99": float(qs[5].item()),
    }


def _qficr_robust_prominence(scores, mask, eps=1e-12):
    """Robust top-candidate prominence for an arbitrary score distribution."""
    if not torch.is_tensor(scores) or not torch.is_tensor(mask):
        return {
            "best_index": None, "best_score": None, "median": None, "q25": None,
            "q75": None, "mean": None, "std": None, "iqr": None, "z": 0.0,
            "pool_size": 0, "fallback": "empty",
        }
    device = scores.device
    safe_eps = max(float(eps), 1e-12)
    work = torch.nan_to_num(scores.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    idxs = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idxs.numel() == 0:
        return {
            "best_index": None, "best_score": None, "median": None, "q25": None,
            "q75": None, "mean": None, "std": None, "iqr": None, "z": 0.0,
            "pool_size": 0, "fallback": "empty",
        }
    idx_list = [int(x) for x in idxs.tolist()]
    idx_list.sort(key=lambda z: (-float(work[z].item()), z))
    best_index = idx_list[0]
    vals = work[idxs]
    q25 = torch.quantile(vals, 0.25)
    median = torch.quantile(vals, 0.50)
    q75 = torch.quantile(vals, 0.75)
    iqr = q75 - q25
    best_score = work[best_index]
    fallback = "iqr"
    if torch.isfinite(iqr) and float(iqr.abs().item()) >= 1e-12:
        z = (best_score - median) / (iqr + safe_eps)
    else:
        mean = vals.mean()
        std = vals.std(unbiased=False)
        fallback = "std"
        if torch.isfinite(std) and float(std.abs().item()) >= 1e-12:
            z = (best_score - mean) / (std + safe_eps)
        else:
            z = torch.zeros((), dtype=torch.float32, device=device)
            fallback = "zero"
    return {
        "best_index": int(best_index),
        "best_score": float(torch.nan_to_num(best_score.float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "median": float(torch.nan_to_num(median.float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "q25": float(torch.nan_to_num(q25.float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "q75": float(torch.nan_to_num(q75.float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "mean": float(torch.nan_to_num(vals.mean().float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "std": float(torch.nan_to_num(vals.std(unbiased=False).float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "iqr": float(torch.nan_to_num(iqr.float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "z": float(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
        "pool_size": int(idxs.numel()),
        "fallback": fallback,
    }


def _qficr_tie_aware_percentile(scores, mask):
    """Tie-aware percentile rank in [0,1], higher score means higher percentile.

    For candidate j: R(j) = (N_<j + 0.5 * N_=j) / N.
    Values outside mask are set to -inf and should not be selected.
    """
    if not torch.is_tensor(scores) or not torch.is_tensor(mask):
        raise ValueError("scores and mask must be tensors")
    device = scores.device
    work = torch.nan_to_num(scores.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    out = torch.full_like(work, float("-inf"), dtype=torch.float32, device=device)
    idxs = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idxs.numel() == 0:
        return out
    vals = work[idxs]
    n = float(vals.numel())
    # N is <= visual token count, so the explicit comparison is small and
    # deterministic; no token index is used in the percentile itself.
    less = (vals[None, :] < vals[:, None]).sum(dim=1).float()
    equal = (vals[:, None] == vals[None, :]).sum(dim=1).float()
    out[idxs] = ((less + 0.5 * equal) / max(n, 1.0)).clamp(0.0, 1.0)
    return out


def _qficr_corr(x, y, eps):
    x = torch.nan_to_num(x.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0 or x.numel() != y.numel():
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() <= eps:
        return 0.0
    return float((x @ y / denom.clamp_min(eps)).clamp(-1.0, 1.0).item())


def _qficr_mean_at(x, indices):
    if not torch.is_tensor(indices) or indices.numel() == 0:
        return 0.0
    return float(x[indices.to(device=x.device, dtype=torch.long)].float().mean().item())


def _select_qficr_competitive_allocation(
    p,
    evidence_kernel,
    qfi_core_order,
    select_num,
    max_flexible_slots,
    anchor_alpha,
    compete_margin,
    candidate_pool_enabled,
    candidate_pool_multiplier,
    eps=1e-12,
    save_step_debug=False,
):
    """Select flexible slots by comparing core and residual marginal gains."""
    device = p.device
    num_tokens = int(p.numel())
    select_num = min(max(int(select_num), 0), num_tokens)
    max_flexible_slots = min(max(int(max_flexible_slots), 0), select_num)
    safe_eps = max(float(eps), 1e-12)
    margin = max(float(compete_margin), 0.0)

    seen = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    full_core = []
    for idx in qfi_core_order.to(device=device, dtype=torch.long).tolist():
        idx = int(idx)
        if 0 <= idx < num_tokens and not bool(seen[idx].item()):
            full_core.append(idx)
            seen[idx] = True
            if len(full_core) >= select_num:
                break

    if len(full_core) != select_num:
        raise RuntimeError(
            f"competitive allocation could not build K core candidates: {len(full_core)} != {select_num}"
        )

    fixed_core_count = select_num - max_flexible_slots
    fixed_core = full_core[:fixed_core_count]
    reserved_core = full_core[fixed_core_count:]
    selected = list(fixed_core)
    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if selected:
        fixed_tensor = torch.tensor(selected, dtype=torch.long, device=device)
        selected_mask[fixed_tensor] = True
        coverage = evidence_kernel[:, fixed_tensor].max(dim=1).values
    else:
        coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)

    full_core_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    full_core_tensor = torch.tensor(full_core, dtype=torch.long, device=device)
    full_core_mask[full_core_tensor] = True
    residual_mask = ~full_core_mask
    residual_indices = torch.nonzero(residual_mask, as_tuple=False).reshape(-1)
    if candidate_pool_enabled:
        pool_size = min(
            max(int(round(float(select_num) * float(candidate_pool_multiplier))), 0),
            int(residual_indices.numel()),
        )
        candidate_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        if pool_size > 0:
            order = torch.argsort(p[residual_indices], descending=True)
            candidate_mask[residual_indices[order[:pool_size]]] = True
    else:
        candidate_mask = residual_mask.clone()
    candidate_pool_size = int(candidate_mask.sum().item())

    selected_residual = []
    fallback_core = []
    competition_steps = []
    selected_residual_raw_gains = []
    selected_residual_scores = []

    for step, core_candidate in enumerate(reserved_core, start=1):
        raw_gain = (
            p[:, None] * (evidence_kernel - coverage[:, None]).clamp_min(0.0)
        ).sum(dim=0)
        raw_gain = torch.nan_to_num(raw_gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        score = raw_gain * p.clamp_min(safe_eps).pow(float(anchor_alpha))
        score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)

        core_score = float(score[core_candidate].item())
        raw_core_gain = float(raw_gain[core_candidate].item())
        available_residual = candidate_mask & ~selected_mask
        residual_candidate = None
        residual_score = None
        raw_residual_gain = None
        choose_residual = False
        if bool(available_residual.any().item()):
            residual_scores = score.masked_fill(~available_residual, float("-inf"))
            candidate = int(torch.argmax(residual_scores).item())
            if bool(available_residual[candidate].item()):
                residual_candidate = candidate
                residual_score = float(score[candidate].item())
                raw_residual_gain = float(raw_gain[candidate].item())
                choose_residual = residual_score > core_score * (1.0 + margin)

        if choose_residual:
            chosen = int(residual_candidate)
            selected_residual.append(chosen)
            selected_residual_raw_gains.append(float(raw_residual_gain))
            selected_residual_scores.append(float(residual_score))
            decision = "residual"
        else:
            chosen = int(core_candidate)
            fallback_core.append(chosen)
            decision = "core"

        selected.append(chosen)
        selected_mask[chosen] = True
        candidate_mask[chosen] = False
        coverage = torch.maximum(coverage, evidence_kernel[:, chosen])

        if save_step_debug:
            score_ratio = None
            if residual_score is not None and core_score > safe_eps:
                score_ratio = float(residual_score / core_score)
            remaining_uncovered_mass = float(
                (p * (1.0 - coverage.clamp(0.0, 1.0))).sum().clamp_min(0.0).item()
            )
            competition_steps.append(
                {
                    "step": int(step),
                    "core_candidate_idx": int(core_candidate),
                    "core_score": core_score,
                    "residual_candidate_idx": residual_candidate,
                    "residual_score": residual_score,
                    "score_ratio": score_ratio,
                    "compete_margin": margin,
                    "decision": decision,
                    "raw_core_gain": raw_core_gain,
                    "raw_residual_gain": raw_residual_gain,
                    "remaining_uncovered_mass": remaining_uncovered_mass,
                }
            )

    if len(selected) != select_num or len(set(selected)) != select_num:
        raise RuntimeError(
            f"competitive allocation selected invalid token count: {len(selected)} for K={select_num}"
        )
    if set(selected_residual) & set(full_core):
        raise RuntimeError("competitive residual selection overlaps the full K-token core set")
    if len(selected_residual) > max_flexible_slots:
        raise RuntimeError("competitive residual selection exceeds the flexible-slot cap")

    info = {
        "residual_alloc_mode": "competitive",
        "K": int(select_num),
        "max_flexible_slots": int(max_flexible_slots),
        "fixed_core_count": int(fixed_core_count),
        "actual_residual_count": int(len(selected_residual)),
        "actual_core_count": int(len(fixed_core) + len(fallback_core)),
        "candidate_pool_enabled": bool(candidate_pool_enabled),
        "candidate_pool_size": int(candidate_pool_size),
        "candidate_pool_multiplier": float(candidate_pool_multiplier),
        "full_core_indices": [int(x) for x in full_core],
        "fixed_core_indices": [int(x) for x in fixed_core],
        "reserved_core_indices": [int(x) for x in reserved_core],
        "selected_residual_indices": [int(x) for x in selected_residual],
        "fallback_core_indices": [int(x) for x in fallback_core],
        "final_selected_indices": [int(x) for x in selected],
        "competition_steps": competition_steps,
        "selected_residual_raw_gains": selected_residual_raw_gains,
        "selected_residual_scores": selected_residual_scores,
    }
    return torch.tensor(selected, dtype=torch.long, device=device), info


def _select_qficr_balanced_competitive_allocation(
    p,
    evidence_kernel,
    qfi_core_order,
    select_num,
    max_flexible_slots,
    anchor_alpha,
    compete_margin,
    external_candidate_pool_enabled,
    external_candidate_pool_size,
    external_candidate_pool_multiplier,
    eps=1e-12,
    save_step_debug=False,
):
    """Compete equally sized QFi-boundary core and residual pools."""
    device = p.device
    num_tokens = int(p.numel())
    select_num = min(max(int(select_num), 0), num_tokens)
    max_flexible_slots = min(max(int(max_flexible_slots), 0), select_num)
    safe_eps = max(float(eps), 1e-12)
    margin = max(float(compete_margin), 0.0)

    ordered = []
    seen = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    for idx in qfi_core_order.to(device=device, dtype=torch.long).tolist():
        idx = int(idx)
        if 0 <= idx < num_tokens and not bool(seen[idx].item()):
            ordered.append(idx)
            seen[idx] = True

    if len(ordered) < select_num:
        raise RuntimeError(
            f"balanced competitive allocation needs at least K QFi candidates: {len(ordered)} < {select_num}"
        )

    full_core = ordered[:select_num]
    fixed_core_count = select_num - max_flexible_slots
    fixed_core = full_core[:fixed_core_count]
    initial_core_pool = full_core[fixed_core_count:]

    # The original competitive mode compares one ordered core token against the
    # maximum over a large residual set. Balanced mode instead takes both pools
    # from adjacent QFi-order boundary blocks, reducing that best-of-many bias.
    full_core_set = set(full_core)
    initial_residual_pool = []
    for idx in ordered[select_num:]:
        if idx not in full_core_set:
            initial_residual_pool.append(int(idx))
        if len(initial_residual_pool) >= max_flexible_slots:
            break

    selected = list(fixed_core)
    if selected:
        fixed_tensor = torch.tensor(selected, dtype=torch.long, device=device)
        coverage = evidence_kernel[:, fixed_tensor].max(dim=1).values
    else:
        coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)

    remaining_core_pool = list(initial_core_pool)
    remaining_residual_pool = list(initial_residual_pool)
    selected_flexible_core = []
    selected_residual = []
    selected_residual_raw_gains = []
    selected_residual_scores = []
    core_pool_sizes_by_step = []
    residual_pool_sizes_by_step = []
    competition_steps = []

    for step in range(1, max_flexible_slots + 1):
        raw_gain = (
            p[:, None] * (evidence_kernel - coverage[:, None]).clamp_min(0.0)
        ).sum(dim=0)
        raw_gain = torch.nan_to_num(raw_gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        score = raw_gain * p.clamp_min(safe_eps).pow(float(anchor_alpha))
        score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)

        best_core = None
        best_core_score = None
        raw_core_gain = None
        if remaining_core_pool:
            core_tensor = torch.tensor(remaining_core_pool, dtype=torch.long, device=device)
            core_position = int(torch.argmax(score[core_tensor]).item())
            best_core = int(remaining_core_pool[core_position])
            best_core_score = float(score[best_core].item())
            raw_core_gain = float(raw_gain[best_core].item())

        best_residual = None
        best_residual_score = None
        raw_residual_gain = None
        if remaining_residual_pool:
            residual_tensor = torch.tensor(remaining_residual_pool, dtype=torch.long, device=device)
            residual_position = int(torch.argmax(score[residual_tensor]).item())
            best_residual = int(remaining_residual_pool[residual_position])
            best_residual_score = float(score[best_residual].item())
            raw_residual_gain = float(raw_gain[best_residual].item())

        if best_core is None and best_residual is None:
            break
        choose_residual = best_core is None
        if best_core is not None and best_residual is not None:
            choose_residual = best_residual_score > best_core_score * (1.0 + margin)

        if choose_residual:
            chosen = int(best_residual)
            remaining_residual_pool.remove(chosen)
            selected_residual.append(chosen)
            selected_residual_raw_gains.append(float(raw_residual_gain))
            selected_residual_scores.append(float(best_residual_score))
            decision = "residual"
        else:
            chosen = int(best_core)
            remaining_core_pool.remove(chosen)
            selected_flexible_core.append(chosen)
            decision = "core"

        selected.append(chosen)
        coverage = torch.maximum(coverage, evidence_kernel[:, chosen])
        core_pool_sizes_by_step.append(len(remaining_core_pool))
        residual_pool_sizes_by_step.append(len(remaining_residual_pool))

        if save_step_debug:
            score_ratio = None
            if best_residual_score is not None and best_core_score is not None and best_core_score > safe_eps:
                score_ratio = float(best_residual_score / best_core_score)
            uncovered_mass = float(
                (p * (1.0 - coverage.clamp(0.0, 1.0))).sum().clamp_min(0.0).item()
            )
            competition_steps.append(
                {
                    "step": int(step),
                    "best_core_idx": best_core,
                    "best_core_score": best_core_score,
                    "best_residual_idx": best_residual,
                    "best_residual_score": best_residual_score,
                    "score_ratio": score_ratio,
                    "compete_margin": margin,
                    "decision": decision,
                    "raw_core_gain": raw_core_gain,
                    "raw_residual_gain": raw_residual_gain,
                    "uncovered_mass": uncovered_mass,
                    "remaining_core_pool_size": len(remaining_core_pool),
                    "remaining_residual_pool_size": len(remaining_residual_pool),
                    # Compatibility aliases for existing competitive collectors.
                    "core_candidate_idx": best_core,
                    "core_score": best_core_score,
                    "residual_candidate_idx": best_residual,
                    "residual_score": best_residual_score,
                    "remaining_uncovered_mass": uncovered_mass,
                }
            )

    if len(selected) != select_num or len(set(selected)) != select_num:
        raise RuntimeError(
            f"balanced competitive allocation selected invalid token count: {len(selected)} for K={select_num}"
        )
    if set(selected_residual) & full_core_set:
        raise RuntimeError("balanced competitive residual selection overlaps the full K-token core set")
    if len(selected_residual) > max_flexible_slots:
        raise RuntimeError("balanced competitive residual selection exceeds the flexible-slot cap")

    actual_residual_count = len(selected_residual)
    actual_core_count = len(fixed_core) + len(selected_flexible_core)
    utilization = (
        float(actual_residual_count) / float(max_flexible_slots)
        if max_flexible_slots > 0
        else 0.0
    )
    info = {
        "residual_alloc_mode": "competitive_balanced",
        "K": int(select_num),
        "max_flexible_slots": int(max_flexible_slots),
        "fixed_core_count": int(fixed_core_count),
        "actual_residual_count": int(actual_residual_count),
        "actual_core_count": int(actual_core_count),
        "residual_utilization": utilization,
        "fallback_to_core_rate": 1.0 - utilization if max_flexible_slots > 0 else 0.0,
        "candidate_pool_enabled": bool(external_candidate_pool_enabled),
        "candidate_pool_size": int(external_candidate_pool_size),
        "candidate_pool_multiplier": float(external_candidate_pool_multiplier),
        "external_candidate_pool_enabled": bool(external_candidate_pool_enabled),
        "external_candidate_pool_size": int(external_candidate_pool_size),
        "external_candidate_pool_multiplier": float(external_candidate_pool_multiplier),
        "balanced_pool_overrides_external_pool": True,
        "balanced_pool_size_requested": int(max_flexible_slots),
        "initial_core_pool_size": int(len(initial_core_pool)),
        "initial_residual_pool_size": int(len(initial_residual_pool)),
        "remaining_core_pool_size_by_step": [int(x) for x in core_pool_sizes_by_step],
        "remaining_residual_pool_size_by_step": [int(x) for x in residual_pool_sizes_by_step],
        "balanced_core_pool_indices": [int(x) for x in initial_core_pool],
        "balanced_residual_pool_indices": [int(x) for x in initial_residual_pool],
        "full_core_indices": [int(x) for x in full_core],
        "fixed_core_indices": [int(x) for x in fixed_core],
        "reserved_core_indices": [int(x) for x in initial_core_pool],
        "selected_core_indices": [int(x) for x in fixed_core + selected_flexible_core],
        "selected_residual_indices": [int(x) for x in selected_residual],
        "fallback_core_indices": [int(x) for x in selected_flexible_core],
        "final_selected_indices": [int(x) for x in selected],
        "competition_steps": competition_steps,
        "selected_residual_raw_gains": selected_residual_raw_gains,
        "selected_residual_scores": selected_residual_scores,
    }
    return torch.tensor(selected, dtype=torch.long, device=device), info


def select_by_adaptive_core_recover(
    states,
    probs,
    k,
    qfi_core_order,
    question_text=None,
    cls_prior=None,
    eps=1e-12,
    core_indices_override=None,
):
    """
    Adaptive Core-Recover selection.

    The recovery budget is determined by question text when available and by
    normalized QFi probability entropy. Recovery gain can optionally be anchored
    by each candidate's own QFi probability and restricted to a top-prob pool.
    """
    if not torch.is_tensor(states) or states.ndim != 2:
        shape = tuple(states.shape) if torch.is_tensor(states) else type(states)
        raise ValueError(f"states must be [N, D], got {shape}")
    if not torch.is_tensor(probs) or probs.ndim != 1:
        shape = tuple(probs.shape) if torch.is_tensor(probs) else type(probs)
        raise ValueError(f"probs must be [N], got {shape}")
    if states.shape[0] != probs.numel():
        raise ValueError(f"states/probs token mismatch: {states.shape[0]} vs {probs.numel()}")
    if not torch.is_tensor(qfi_core_order) or qfi_core_order.ndim != 1:
        shape = tuple(qfi_core_order.shape) if torch.is_tensor(qfi_core_order) else type(qfi_core_order)
        raise ValueError(f"qfi_core_order must be [M], got {shape}")

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    alloc_mode = os.environ.get("EC_QFICR_ALLOC_MODE", "frozen").strip().lower()
    if alloc_mode not in {"frozen", "adaptive_competition"}:
        raise ValueError(f"Unknown EC_QFICR_ALLOC_MODE: {alloc_mode}")
    residual_alloc_mode = os.environ.get("EC_QFICR_RESIDUAL_ALLOC_MODE", "legacy").strip().lower()
    if residual_alloc_mode not in {"legacy", "competitive", "competitive_balanced"}:
        residual_alloc_mode = "legacy"
    trainfree_restoration_mode = os.environ.get(
        "EC_QFICR_TRAINFREE_RESTORATION_MODE", "legacy"
    ).strip().lower()
    if trainfree_restoration_mode not in {"legacy", "single_step", "pareto_competitive"}:
        trainfree_restoration_mode = "legacy"
    compete_margin = max(_env_float("EC_QFICR_COMPETE_MARGIN", "0.0"), 0.0)
    if select_num == 0:
        info = {
            "adapt_mode": "entropy_only",
            "alloc_mode": alloc_mode,
            "question_type": "default",
            "entropy_norm": 0.0,
            "recover_ratio": 0.0,
            "k_core": 0,
            "k_recover": 0,
            "recover_prior_anchor": False,
            "recover_prior_gamma": 0.0,
            "anchor_alpha": 0.0,
            "recover_cand_pool": False,
            "recover_cand_mult": 0.0,
            "recover_cand_size": 0,
            "question_text_available": False,
            "restoration_mode": "full",
            "fixed_restoration_ratio": 0.2,
            "use_spatial_local_prior": False,
            "prior_ablation": "none",
            "prior_lambda": 0.0,
            "random_seed": 42,
            "spatial_prior_mean": 0.0,
            "local_prior_mean": 0.0,
            "spatial_local_prior_mean": 0.0,
            "entropy_anchor_ratio": 0.0,
            "entropy_anchor_count": 0,
            "residual_budget_gamma": -1.0,
            "residual_budget_applied": False,
            "residual_mass": 0.0,
            "k_core_original": 0,
            "k_recover_original": 0,
            "restore_extra_prior": "none",
            "restore_extra_lambda": 0.0,
            "subspace_residual_mode": "fast",
            "restore_extra_enabled": False,
            "restore_extra_mean": 0.0,
            "restore_extra_std": 0.0,
            "restore_extra_min": 0.0,
            "restore_extra_max": 0.0,
            "restore_extra_norm_mean": 0.0,
            "restore_extra_base_corr": 0.0,
            "restore_extra_mean_g_res": 0.0,
            "comp_score_mode": "current_anchor",
            "residual_alloc_mode": residual_alloc_mode,
            "compete_margin": float(compete_margin),
            "K": 0,
            "max_flexible_slots": 0,
            "fixed_core_count": 0,
            "actual_residual_count": 0,
            "actual_core_count": 0,
            "candidate_pool_enabled": False,
            "candidate_pool_size": 0,
            "candidate_pool_multiplier": 0.0,
            "full_core_indices": [],
            "fixed_core_indices": [],
            "reserved_core_indices": [],
            "selected_residual_indices": [],
            "fallback_core_indices": [],
            "final_selected_indices": [],
            "competition_steps": [],
            "external_candidate_pool_enabled": False,
            "external_candidate_pool_size": 0,
            "external_candidate_pool_multiplier": 0.0,
            "balanced_pool_overrides_external_pool": False,
            "balanced_pool_size_requested": 0,
            "initial_core_pool_size": 0,
            "initial_residual_pool_size": 0,
            "remaining_core_pool_size_by_step": [],
            "remaining_residual_pool_size_by_step": [],
            "balanced_core_pool_indices": [],
            "balanced_residual_pool_indices": [],
            "selected_core_indices": [],
            "residual_utilization": 0.0,
            "fallback_to_core_rate": 0.0,
        }
        return torch.empty(0, dtype=torch.long, device=device), info

    safe_eps = max(float(eps), 1e-12)
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)
    obs_temperature = max(_env_float("EC_QFICR_OBS_TEMPERATURE", "1.0"), safe_eps)
    if abs(obs_temperature - 1.0) > 1e-12:
        # QFi-CR micro tuning: apply temperature after the existing full
        # observation fusion, then renormalize to keep p a probability vector.
        p = p.clamp_min(safe_eps).pow(obs_temperature)
        p = p / p.sum().clamp_min(safe_eps)

    entropy = -(p * (p + safe_eps).log()).sum()
    entropy_norm = 0.0
    if num_tokens > 1:
        entropy_norm = float((entropy / math.log(float(num_tokens))).clamp(0.0, 1.0).item())

    mode_requested = os.environ.get("EC_QFID_ADAPT_MODE", "entropy_question").strip().lower()
    question_available = isinstance(question_text, str) and bool(question_text.strip())
    if mode_requested == "entropy_question" and question_available:
        adapt_mode = "entropy_question"
        question_type = infer_question_recovery_type(question_text)
    else:
        adapt_mode = "entropy_only"
        question_type = "default"

    ratio_min = min(max(_env_float("EC_QFID_ADAPT_RECOVER_MIN_RATIO", "0.00"), 0.0), 1.0)
    ratio_max = min(max(_env_float("EC_QFID_ADAPT_RECOVER_MAX_RATIO", "0.25"), 0.0), 1.0)
    if ratio_min > ratio_max:
        ratio_min, ratio_max = ratio_max, ratio_min

    ratio_by_type = {
        "binary": _env_float("EC_QFID_ADAPT_RECOVER_BINARY_RATIO", "0.0625"),
        "open": _env_float("EC_QFID_ADAPT_RECOVER_OPEN_RATIO", "0.25"),
        "relation": _env_float("EC_QFID_ADAPT_RECOVER_REL_RATIO", "0.25"),
        "compare": _env_float("EC_QFID_ADAPT_RECOVER_COMPARE_RATIO", "0.25"),
        "attr": _env_float("EC_QFID_ADAPT_RECOVER_ATTR_RATIO", "0.125"),
        "default": _env_float("EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO", "0.125"),
    }
    ratio_base = min(max(float(ratio_by_type.get(question_type, ratio_by_type["default"])), ratio_min), ratio_max)

    entropy_gamma = max(_env_float("EC_QFICR_RECOVER_ENTROPY_GAMMA", "1.0"), safe_eps)
    entropy_gate = get_env_bool("EC_QFID_ADAPT_ENTROPY_GATE", default=True)
    if entropy_gate:
        low = _env_float("EC_QFID_ADAPT_ENTROPY_LOW", "0.70")
        high = _env_float("EC_QFID_ADAPT_ENTROPY_HIGH", "0.95")
        denom = max(high - low, safe_eps)
        u = min(max((entropy_norm - low) / denom, 0.0), 1.0)
        entropy_hat = min(max(float(u), 0.0), 1.0)
        recover_ratio = ratio_min + (ratio_base - ratio_min) * (entropy_hat ** entropy_gamma)
    else:
        entropy_hat = min(max(float(entropy_norm), 0.0), 1.0)
        recover_ratio = ratio_base
    recover_ratio = min(max(float(recover_ratio), ratio_min), ratio_max)
    recover_ratio_requested = float(recover_ratio)

    budget_mode = os.environ.get("EC_QFICR_RECOVER_BUDGET_MODE", "abs_cap").strip().lower()
    if budget_mode not in {"abs_cap", "ratio_only", "ratio_cap"}:
        budget_mode = "abs_cap"
    abs_cap_default = os.environ.get("EC_QFID_ADAPT_RECOVER_CAP", "16")
    abs_cap = max(0, _env_int("EC_QFICR_RECOVER_ABS_CAP", abs_cap_default))
    ratio_cap = min(max(_env_float("EC_QFICR_RECOVER_RATIO_CAP", "0.25"), 0.0), 1.0)
    raw_recover_num = max(int(round(select_num * recover_ratio)), 0)
    cap_applied = False
    if budget_mode == "abs_cap":
        recover_num = min(raw_recover_num, abs_cap)
        cap_applied = raw_recover_num > recover_num
    elif budget_mode == "ratio_cap":
        ratio_cap_num = max(int(round(select_num * ratio_cap)), 0)
        recover_num = min(raw_recover_num, ratio_cap_num)
        cap_applied = raw_recover_num > recover_num
    else:
        recover_num = raw_recover_num
    recover_num = min(max(recover_num, 0), select_num)
    core_num = select_num - recover_num

    restoration_mode = os.environ.get("EC_QFICR_RESTORATION_MODE", "full").strip().lower()
    if restoration_mode not in {"full", "none", "random", "fixed_ratio"}:
        restoration_mode = "full"
    fixed_restoration_ratio = min(
        max(_env_float("EC_QFICR_FIXED_RESTORATION_RATIO", "0.2"), 0.0),
        1.0,
    )
    if restoration_mode == "none":
        recover_ratio = 0.0
        recover_num = 0
        core_num = select_num
    elif restoration_mode == "fixed_ratio":
        recover_ratio = fixed_restoration_ratio
        recover_num = min(max(int(round(select_num * fixed_restoration_ratio)), 0), select_num)
        core_num = select_num - recover_num
    yn_gate_mode = os.environ.get("EC_QFICR_YN_GATE_MODE", "frozen").strip().lower()
    if yn_gate_mode not in {"frozen", "ungated"}:
        raise ValueError(f"Unknown EC_QFICR_YN_GATE_MODE: {yn_gate_mode}")
    evidence_recovery_mode = os.environ.get("EC_QFICR_EVIDENCE_RECOVERY_MODE", "frozen").strip().lower()
    if evidence_recovery_mode not in {"frozen", "detail", "dual_max"}:
        raise ValueError(f"Unknown EC_QFICR_EVIDENCE_RECOVERY_MODE: {evidence_recovery_mode}")
    recovery_score_mode = os.environ.get("EC_QFICR_RECOVERY_SCORE_MODE", "frozen").strip().lower()
    if recovery_score_mode not in {"frozen", "rc_rank_geom"}:
        raise ValueError(f"Unknown EC_QFICR_RECOVERY_SCORE_MODE: {recovery_score_mode}")
    # EC_QFICR_YN_BUDGET_GATE scales yes/no recovery only. Setting it to 0
    # means yes/no questions use no recovery and keep the full core budget.
    yn_budget_gate = min(max(_env_float("EC_QFICR_YN_BUDGET_GATE", "1.0"), 0.0), 1.0)
    yn_gate_applied = False
    if (
        yn_gate_mode == "frozen"
        and restoration_mode != "none"
        and yn_budget_gate < 1.0
        and recover_num > 0
        and infer_yes_no_question(question_text)
    ):
        recover_num = min(max(int(round(float(recover_num) * yn_budget_gate)), 0), select_num)
        core_num = select_num - recover_num
        recover_ratio = float(recover_num) / max(float(select_num), 1.0)
        yn_gate_applied = True
        cap_applied = cap_applied or raw_recover_num > recover_num

    adaptive_base_core_count = None
    adaptive_max_recover_slots = None
    if alloc_mode == "adaptive_competition":
        if budget_mode != "ratio_cap":
            raise ValueError("adaptive_competition requires EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap")
        if restoration_mode != "full":
            raise ValueError("adaptive_competition requires EC_QFICR_RESTORATION_MODE=full")
        adaptive_max_recover_slots = min(
            max(int(round(float(select_num) * float(ratio_cap))), 0),
            select_num,
        )
        adaptive_base_core_count = select_num - adaptive_max_recover_slots
        recover_num = int(adaptive_max_recover_slots)
        core_num = int(adaptive_base_core_count)
        recover_ratio = float(recover_num) / max(float(select_num), 1.0)
        cap_applied = True
        # The adaptive allocator intentionally replaces answer-format budget
        # suppression with sample-wise Core-vs-Recovery competition.  This flag
        # records whether the Frozen gate would have fired before replacement.

    psi = F.normalize(
        torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=safe_eps,
    )
    evidence_kernel = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)

    qfi_order = qfi_core_order.to(device=device, dtype=torch.long)
    qfi_order = qfi_order[(qfi_order >= 0) & (qfi_order < num_tokens)]
    if qfi_order.numel() > 0:
        seen = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        ordered_unique = []
        for idx in qfi_order.tolist():
            if not bool(seen[idx].item()):
                seen[idx] = True
                ordered_unique.append(int(idx))
        qfi_order = torch.tensor(ordered_unique, dtype=torch.long, device=device)

    entropy_anchor_ratio = min(max(_env_float("EC_QFICR_ENTROPY_ANCHOR_RATIO", "0.0"), 0.0), 1.0)
    residual_budget_gamma = _env_float("EC_QFICR_RESIDUAL_BUDGET_GAMMA", "-1.0")
    residual_budget_enabled = residual_budget_gamma > 0.0 and restoration_mode == "full"
    k_core_original = int(core_num)
    k_recover_original = int(recover_num)
    entropy_anchor_count = 0
    residual_mass = 0.0
    residual_budget_applied = False

    def build_core_indices(target_core_num):
        target_core_num = min(max(int(target_core_num), 0), select_num, num_tokens)
        if target_core_num <= 0:
            return torch.empty(0, dtype=torch.long, device=device), 0

        if core_indices_override is not None:
            override = torch.as_tensor(core_indices_override, dtype=torch.long, device=device).reshape(-1)
            override = override[(override >= 0) & (override < num_tokens)]
            if override.numel() != target_core_num or torch.unique(override).numel() != target_core_num:
                raise ValueError(
                    f"core_indices_override must contain exactly target K_core={target_core_num} unique valid indices"
                )
            return override, 0

        # Fast path preserves the current stable QFi-CR behavior exactly unless
        # one of the new generalization ablations is explicitly enabled.
        if entropy_anchor_ratio <= 0.0 and not residual_budget_enabled:
            return qfi_order[:target_core_num], 0

        anchor_num = min(
            max(int(round(entropy_anchor_ratio * float(select_num) * (1.0 - entropy_norm))), 0),
            target_core_num,
        )
        selected_core = []
        core_available = torch.ones(num_tokens, dtype=torch.bool, device=device)
        core_coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)
        if anchor_num > 0:
            anchor_idx = torch.topk(p, k=anchor_num).indices
            selected_core.extend([int(x) for x in anchor_idx.tolist()])
            core_available[anchor_idx] = False
            core_coverage = evidence_kernel[:, anchor_idx].max(dim=1).values

        # Entropy-aware core anchor protection: anchors initialize coverage, then
        # the remaining reduction budget follows the original projection coverage
        # objective over the uncovered residual mass.
        while len(selected_core) < target_core_num and bool(core_available.any().item()):
            core_gain = (p[:, None] * (evidence_kernel - core_coverage[:, None]).clamp_min(0.0)).sum(dim=0)
            core_gain = torch.nan_to_num(core_gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
            core_gain = core_gain.masked_fill(~core_available, float("-inf"))
            idx = int(torch.argmax(core_gain).item())
            if not bool(core_available[idx].item()):
                break
            selected_core.append(idx)
            core_available[idx] = False
            core_coverage = torch.maximum(core_coverage, evidence_kernel[:, idx])

        if len(selected_core) < target_core_num:
            filler = torch.nonzero(core_available, as_tuple=False).reshape(-1)
            selected_core.extend([int(x) for x in filler[: target_core_num - len(selected_core)].tolist()])

        return torch.tensor(selected_core[:target_core_num], dtype=torch.long, device=device), anchor_num

    core_indices, entropy_anchor_count = build_core_indices(core_num)

    if residual_budget_enabled and recover_num > 0:
        if core_indices.numel() > 0:
            tmp_coverage = evidence_kernel[:, core_indices].max(dim=1).values
        else:
            tmp_coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)
        # Residual-mass restoration budget gating: only reduce recovery budget
        # when the already reduced core explains most task-conditioned mass.
        residual_mass = float((p * (1.0 - tmp_coverage.clamp(0.0, 1.0))).sum().clamp(0.0, 1.0).item())
        recover_num_new = min(
            max(int(round(float(residual_budget_gamma) * float(select_num) * residual_mass)), 0),
            recover_num,
            select_num,
        )
        if recover_num_new != recover_num:
            residual_budget_applied = True
            recover_num = recover_num_new
            core_num = select_num - recover_num
            recover_ratio = float(recover_num) / max(float(select_num), 1.0)
            core_indices, entropy_anchor_count = build_core_indices(core_num)

    selected = [int(idx) for idx in core_indices.tolist()]

    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if core_indices.numel() > 0:
        available[core_indices] = False
        coverage = evidence_kernel[:, core_indices].max(dim=1).values
    else:
        coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)

    cand_pool_enabled = get_env_bool("EC_QFID_RECOVER_CAND_POOL", default=True)
    cand_mult = max(_env_float("EC_QFID_RECOVER_CAND_MULT", "3.0"), 0.0)
    candidate_mask = available.clone()
    candidate_size = int(candidate_mask.sum().item())
    if cand_pool_enabled and recover_num > 0:
        remaining_idx = torch.nonzero(available, as_tuple=False).reshape(-1)
        pool_num = min(int(round(select_num * cand_mult)), int(remaining_idx.numel()))
        if pool_num >= recover_num and pool_num > 0:
            top_order = torch.argsort(p[remaining_idx], descending=True)
            top_idx = remaining_idx[top_order[:pool_num]]
            candidate_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            candidate_mask[top_idx] = True
            candidate_size = int(top_idx.numel())
        else:
            candidate_size = int(remaining_idx.numel())

    prior_anchor = get_env_bool("EC_QFID_RECOVER_PRIOR_ANCHOR", default=True)
    prior_gamma = max(_env_float("EC_QFID_RECOVER_PRIOR_GAMMA", "0.5"), 0.0)
    anchor_alpha = max(_env_float("EC_QFICR_ANCHOR_ALPHA", str(prior_gamma)), 0.0)
    comp_score_mode = os.environ.get("EC_QFICR_COMP_SCORE_MODE", "current_anchor").strip().lower()
    if comp_score_mode not in {
        "current_anchor",
        "weak_anchor",
        "no_anchor",
        "marginal_gain_weak_gate",
        "near_boundary_fill",
    }:
        comp_score_mode = "current_anchor"

    if residual_alloc_mode in {"competitive", "competitive_balanced"}:
        if budget_mode != "ratio_cap":
            raise ValueError(
                "competitive residual allocation requires EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap"
            )
        if comp_score_mode != "current_anchor":
            raise ValueError(
                "competitive residual allocation requires EC_QFICR_COMP_SCORE_MODE=current_anchor"
            )
        if restoration_mode not in {"full", "none"}:
            raise ValueError(
                "competitive residual allocation supports restoration_mode=full or none"
            )

        save_step_debug = bool(os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip())
        if residual_alloc_mode == "competitive_balanced":
            competitive_selected, competitive_info = _select_qficr_balanced_competitive_allocation(
                p=p,
                evidence_kernel=evidence_kernel,
                qfi_core_order=qfi_order,
                select_num=select_num,
                max_flexible_slots=recover_num,
                anchor_alpha=anchor_alpha,
                compete_margin=compete_margin,
                external_candidate_pool_enabled=cand_pool_enabled,
                external_candidate_pool_size=candidate_size,
                external_candidate_pool_multiplier=cand_mult,
                eps=safe_eps,
                save_step_debug=save_step_debug,
            )
        else:
            competitive_selected, competitive_info = _select_qficr_competitive_allocation(
                p=p,
                evidence_kernel=evidence_kernel,
                qfi_core_order=qfi_order,
                select_num=select_num,
                max_flexible_slots=recover_num,
                anchor_alpha=anchor_alpha,
                compete_margin=compete_margin,
                candidate_pool_enabled=cand_pool_enabled,
                candidate_pool_multiplier=cand_mult,
                eps=safe_eps,
                save_step_debug=save_step_debug,
            )

        residual_indices = competitive_info["selected_residual_indices"]
        core_selected = competitive_info["fixed_core_indices"] + competitive_info["fallback_core_indices"]

        def _competitive_mean(values):
            return float(sum(values) / len(values)) if values else None

        def _competitive_mean_p(indices):
            if not indices:
                return None
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            return float(p[idx_tensor].mean().item())

        residual_p_values = [float(p[idx].item()) for idx in residual_indices]
        actual_residual_count = int(competitive_info["actual_residual_count"])
        actual_core_count = int(competitive_info["actual_core_count"])
        info = {
            "adapt_mode": adapt_mode,
            "question_type": question_type,
            "entropy": float(entropy.item()),
            "entropy_norm": entropy_norm,
            "recover_ratio_min": float(ratio_min),
            "recover_ratio_max": float(ratio_max),
            "recover_ratio_requested": float(recover_ratio_requested),
            "recover_ratio": float(actual_residual_count) / max(float(select_num), 1.0),
            "recover_budget_mode": budget_mode,
            "recover_abs_cap": int(abs_cap),
            "recover_ratio_cap": float(ratio_cap),
            "recover_entropy_gamma": float(entropy_gamma),
            "recover_entropy_hat": float(entropy_hat),
            "recover_raw_k": int(raw_recover_num),
            "recover_cap_applied": bool(cap_applied),
            "k_core": actual_core_count,
            "k_recover": actual_residual_count,
            "recover_prior_anchor": True,
            "recover_prior_gamma": float(prior_gamma),
            "anchor_alpha": float(anchor_alpha),
            "comp_score_mode": comp_score_mode,
            "recover_cand_pool": bool(competitive_info["candidate_pool_enabled"]),
            "recover_cand_mult": float(competitive_info["candidate_pool_multiplier"]),
            "recover_cand_size": int(competitive_info["candidate_pool_size"]),
            "question_text_available": bool(question_available),
            "restoration_mode": restoration_mode,
            "fixed_restoration_ratio": float(fixed_restoration_ratio),
            "yn_budget_gate": float(yn_budget_gate),
            "yn_gate_mode": yn_gate_mode,
            "yn_gate_applied": bool(yn_gate_applied),
            "entropy_anchor_ratio": float(entropy_anchor_ratio),
            "entropy_anchor_count": 0,
            "residual_budget_gamma": float(residual_budget_gamma),
            "residual_budget_applied": bool(residual_budget_applied),
            "residual_mass": float(residual_mass),
            "k_core_original": int(k_core_original),
            "k_recover_original": int(k_recover_original),
            "use_spatial_local_prior": False,
            "prior_ablation": "none",
            "prior_lambda": 0.0,
            "rest_beta": 1.0,
            "support_beta": 0.0,
            "self_discount": 0.0,
            "obs_temperature": float(obs_temperature),
            "restore_div_lambda": 0.0,
            "restore_candidate_factor": 0.0,
            "restore_extra_prior": "none",
            "restore_extra_lambda": 0.0,
            "subspace_residual_mode": "fast",
            "restore_extra_enabled": False,
            "restore_extra_mean": 0.0,
            "restore_extra_std": 0.0,
            "restore_extra_min": 0.0,
            "restore_extra_max": 0.0,
            "restore_extra_norm_mean": 0.0,
            "restore_extra_base_corr": 0.0,
            "restore_extra_mean_g_res": 0.0,
            "random_seed": int(_env_int("EC_QFICR_RANDOM_SEED", "42")),
            "spatial_prior_mean": 0.0,
            "local_prior_mean": 0.0,
            "spatial_local_prior_mean": 0.0,
            "mean_p_core": _competitive_mean_p(core_selected),
            "mean_p_rest": _competitive_mean_p(residual_indices),
            "median_p_rest": None if not residual_p_values else float(torch.tensor(residual_p_values).median().item()),
            "mean_residual_gain_rest": _competitive_mean(competitive_info["selected_residual_raw_gains"]),
            "median_residual_gain_rest": None,
            "mean_max_overlap_to_core_rest": None,
            "median_max_overlap_to_core_rest": None,
            "mean_qfi_rank_rest": None,
            "median_qfi_rank_rest": None,
            "near_boundary_rank_threshold": None,
            "near_boundary_rank_hit_rate": None,
            "mean_p_discarded": None,
            "median_p_discarded": None,
            "mean_residual_gain_discarded": None,
            "median_residual_gain_discarded": None,
            "mean_max_overlap_to_core_discarded": None,
            "median_max_overlap_to_core_discarded": None,
            "mean_qfi_rank_discarded": None,
            "median_qfi_rank_discarded": None,
            "mean_support_rest": None,
            "mean_self_gain_rest": None,
            "mean_restoration_score_rest": _competitive_mean(competitive_info["selected_residual_scores"]),
            "compete_margin": float(compete_margin),
            **competitive_info,
        }
        return competitive_selected, info

    use_spatial_local_prior = get_env_bool("EC_QFICR_USE_SPATIAL_LOCAL_PRIOR", default=False)
    prior_ablation = os.environ.get("EC_QFICR_PRIOR_ABLATION", "full").strip().lower()
    if prior_ablation not in {"full", "no_spatial", "no_local", "none"}:
        prior_ablation = "full"
    if not use_spatial_local_prior:
        prior_ablation = "none"
    prior_lambda = max(_env_float("EC_QFICR_PRIOR_LAMBDA", "0.0"), 0.0)
    if not use_spatial_local_prior:
        prior_lambda = 0.0
    rest_beta = max(_env_float("EC_QFICR_REST_BETA", "1.0"), safe_eps)
    restore_div_lambda = max(_env_float("EC_QFICR_RESTORE_DIV_LAMBDA", "0.0"), 0.0)
    restore_candidate_factor = max(_env_float("EC_QFICR_RESTORE_CANDIDATE_FACTOR", "2.0"), 1.0)
    restore_extra_prior = os.environ.get("EC_QFICR_RESTORE_EXTRA_PRIOR", "none").strip().lower()
    if restore_extra_prior not in {"none", "subspace_residual", "uncovered_cls", "context_anti"}:
        restore_extra_prior = "none"
    restore_extra_lambda = max(_env_float("EC_QFICR_RESTORE_EXTRA_LAMBDA", "0.0"), 0.0)
    support_beta = max(_env_float("EC_QFICR_SUPPORT_BETA", "0.0"), 0.0)
    self_discount = max(_env_float("EC_QFICR_SELF_DISCOUNT", "0.0"), 0.0)
    subspace_residual_mode = os.environ.get("EC_QFICR_SUBSPACE_RESIDUAL_MODE", "fast").strip().lower()
    if subspace_residual_mode not in {"fast", "full"}:
        subspace_residual_mode = "fast"
    restore_extra_enabled = (
        restoration_mode == "full"
        and restore_extra_prior != "none"
        and restore_extra_lambda > 0.0
    )
    p_cls_vec = None
    if torch.is_tensor(cls_prior) and cls_prior.numel() == num_tokens:
        p_cls_vec = torch.nan_to_num(cls_prior.to(device=device).float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        p_cls_vec = p_cls_vec / p_cls_vec.sum().clamp_min(safe_eps)
    spatial_local_prior, spatial_prior, local_prior = _qficr_spatial_local_prior(
        num_tokens,
        core_indices,
        p,
        safe_eps,
        prior_ablation=prior_ablation,
    )
    random_seed = _env_int("EC_QFICR_RANDOM_SEED", "42")
    qfi_rank = torch.full((num_tokens,), float("nan"), dtype=torch.float32, device=device)
    if qfi_order.numel() > 0:
        rank_vals = torch.arange(1, qfi_order.numel() + 1, dtype=torch.float32, device=device)
        qfi_rank[qfi_order] = rank_vals

    extra_stats = {
        "restore_extra_mean": 0.0,
        "restore_extra_std": 0.0,
        "restore_extra_min": 0.0,
        "restore_extra_max": 0.0,
        "restore_extra_norm_mean": 0.0,
        "restore_extra_base_corr": 0.0,
        "restore_extra_mean_g_res": 0.0,
    }
    support = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    if support_beta > 0.0:
        # Feature/evidence support excludes self contribution and does not use
        # image coordinates or spatial-local priors.
        support = (p[:, None] * evidence_kernel).sum(dim=0) - p
        support = torch.nan_to_num(support.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    core_coverage_for_marginal = coverage.clone()

    def restoration_base_gain(current_coverage):
        coverage_for_gain = core_coverage_for_marginal if comp_score_mode == "marginal_gain_weak_gate" else current_coverage
        g_res_local = (p[:, None] * (evidence_kernel - coverage_for_gain[:, None]).clamp_min(0.0)).sum(dim=0)
        if self_discount > 0.0:
            self_gain_local = p * (1.0 - coverage_for_gain.clamp(0.0, 1.0)).clamp_min(0.0)
            gain_local = (g_res_local - self_discount * self_gain_local).clamp_min(0.0)
        else:
            gain_local = g_res_local
        if abs(rest_beta - 1.0) > 1e-12:
            gain_local = gain_local.clamp_min(safe_eps).pow(rest_beta)
        if comp_score_mode == "weak_anchor":
            gain_local = gain_local * p.clamp_min(safe_eps).pow(0.25)
        elif comp_score_mode == "no_anchor":
            pass
        elif comp_score_mode == "marginal_gain_weak_gate":
            gain_local = gain_local * p.clamp_min(safe_eps).pow(0.1)
        elif prior_anchor:
            gain_local = gain_local * p.clamp_min(safe_eps).pow(anchor_alpha)
        if support_beta > 0.0:
            gain_local = gain_local * support.clamp_min(safe_eps).pow(support_beta)
        if use_spatial_local_prior and prior_ablation != "none" and prior_lambda > 0.0:
            # Legacy spatial/local prior remains disabled for the stable method
            # and for replacement-prior experiments unless explicitly enabled.
            gain_local = gain_local * (1.0 + prior_lambda * entropy_norm * spatial_local_prior)
        return torch.nan_to_num(gain_local.float(), nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(
            g_res_local.float(), nan=0.0, posinf=0.0, neginf=0.0
        )

    # Training-free competitive restoration.  This branch is opt-in and the
    # default legacy path below is intentionally untouched.  Both variants use
    # only projector-space features, p, overlap coverage, and qfi_order; they
    # never inspect answers, GT, decoder states, or benchmark identifiers.
    tfcr_selected_rest = []
    tfcr_selected_core_cont = []
    tfcr_competition_steps = []
    tfcr_active = trainfree_restoration_mode != "legacy" or alloc_mode == "adaptive_competition"
    if tfcr_active:
        # The first core_num entries are already the frozen V0 core.  The
        # continuation pointer starts immediately after that prefix.
        selected = [int(x) for x in core_indices.tolist()]
        available_tfcr = available.clone()
        continuation = [int(x) for x in qfi_order[core_num:select_num].tolist()]

        def _tfcr_next_core():
            for idx_local in continuation:
                if 0 <= idx_local < num_tokens and bool(available_tfcr[idx_local].item()):
                    return idx_local
            # qfi_order may be shorter than K in some selector fallbacks.  Use
            # deterministic original-index completion only as a safety filler.
            for idx_local in range(num_tokens):
                if bool(available_tfcr[idx_local].item()):
                    return idx_local
            return None

        def _tfcr_restoration_proposal():
            gain_local, weighted_local = restoration_base_gain(coverage)
            residual_mass_local = (evidence_kernel - coverage[:, None]).clamp_min(0.0).sum(dim=0)
            uniform_local = residual_mass_local / max(float(num_tokens), 1.0)
            mask_local = available_tfcr & candidate_mask
            if not bool(mask_local.any().item()):
                return None, gain_local, weighted_local, uniform_local
            score_local = gain_local.masked_fill(~mask_local, float("-inf"))
            idx_local = int(torch.argmax(score_local).item())
            if not bool(mask_local[idx_local].item()):
                return None, gain_local, weighted_local, uniform_local
            return idx_local, gain_local, weighted_local, uniform_local

        if alloc_mode == "adaptive_competition":
            if comp_score_mode != "current_anchor":
                raise ValueError("adaptive_competition requires EC_QFICR_COMP_SCORE_MODE=current_anchor")
            if evidence_recovery_mode != "frozen":
                raise ValueError("adaptive_competition requires EC_QFICR_EVIDENCE_RECOVERY_MODE=frozen")
            core_trace = None
            try:
                _, core_trace = select_frozen_qfi_order(
                    states,
                    p,
                    select_num,
                    eps=safe_eps,
                    return_trace=True,
                )
            except Exception:
                core_trace = None
            selected_core_adaptive = [int(x) for x in core_indices.tolist()]
            slot_id = 0
            while len(selected) < select_num:
                slot_id += 1
                active_core_mask = available_tfcr.clone()
                core_step_idx = len(selected_core_adaptive)
                if core_trace is not None and core_step_idx < len(core_trace):
                    core_scores = core_trace[core_step_idx].to(device=device).float()
                else:
                    # Deterministic safety fallback using the existing qfi_order
                    # only if the frozen trace is unavailable.
                    core_scores = torch.full((num_tokens,), float("-inf"), dtype=torch.float32, device=device)
                    for rank_local, idx_local in enumerate(qfi_order.tolist()):
                        if 0 <= int(idx_local) < num_tokens:
                            core_scores[int(idx_local)] = -float(rank_local)
                core_stats = _qficr_robust_prominence(core_scores, active_core_mask, eps=safe_eps)

                gain_r, weighted_vec = restoration_base_gain(coverage)
                active_recovery_mask = available_tfcr & candidate_mask
                frozen_recovery_scores = gain_r.masked_fill(~active_recovery_mask, float("-inf"))
                frozen_recovery_stats = _qficr_robust_prominence(
                    frozen_recovery_scores, active_recovery_mask, eps=safe_eps
                )
                rank_g = _qficr_tie_aware_percentile(weighted_vec, active_recovery_mask)
                rank_p = _qficr_tie_aware_percentile(p, active_recovery_mask)
                rc_scores = torch.sqrt((rank_g.clamp_min(0.0) * rank_p.clamp_min(0.0)).clamp_min(0.0))
                rc_scores = torch.nan_to_num(rc_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
                rc_scores = rc_scores.masked_fill(~active_recovery_mask, float("-inf"))
                if recovery_score_mode == "rc_rank_geom":
                    recovery_scores = rc_scores
                else:
                    recovery_scores = frozen_recovery_scores
                recovery_stats = _qficr_robust_prominence(recovery_scores, active_recovery_mask, eps=safe_eps)

                z_core = float(core_stats["z"])
                z_recovery = float(recovery_stats["z"])
                choose_recovery = (
                    recovery_stats["best_index"] is not None
                    and (core_stats["best_index"] is None or z_recovery > z_core)
                )
                if choose_recovery:
                    chosen = int(recovery_stats["best_index"])
                    chosen_type = "recovery"
                    tfcr_selected_rest.append(chosen)
                else:
                    if core_stats["best_index"] is None:
                        break
                    chosen = int(core_stats["best_index"])
                    chosen_type = "core"
                    selected_core_adaptive.append(chosen)
                    tfcr_selected_core_cont.append(chosen)

                selected.append(chosen)
                available_tfcr[chosen] = False
                candidate_mask[chosen] = False
                coverage = torch.maximum(coverage, evidence_kernel[:, chosen])

                tfcr_competition_steps.append({
                    "slot_id": int(slot_id),
                    "step": int(slot_id),
                    "best_core_index": core_stats["best_index"],
                    "best_core_raw_score": core_stats["best_score"],
                    "core_score_median": core_stats["median"],
                    "core_score_q25": core_stats["q25"],
                    "core_score_q75": core_stats["q75"],
                    "core_score_mean": core_stats["mean"],
                    "core_score_std": core_stats["std"],
                    "core_score_iqr": core_stats["iqr"],
                    "core_score_pool_size": core_stats["pool_size"],
                    "core_prominence_fallback": core_stats["fallback"],
                    "Z_core": z_core,
                    "recovery_score_mode": recovery_score_mode,
                    "frozen_best_recovery_index": frozen_recovery_stats["best_index"],
                    "frozen_best_recovery_raw_score": frozen_recovery_stats["best_score"],
                    "frozen_best_recovery_p": None if frozen_recovery_stats["best_index"] is None else float(p[int(frozen_recovery_stats["best_index"])].item()),
                    "frozen_best_recovery_raw_g": None if frozen_recovery_stats["best_index"] is None else float(weighted_vec[int(frozen_recovery_stats["best_index"])].item()),
                    "frozen_best_recovery_rank_p": None if frozen_recovery_stats["best_index"] is None else float(rank_p[int(frozen_recovery_stats["best_index"])].item()),
                    "frozen_best_recovery_rank_g": None if frozen_recovery_stats["best_index"] is None else float(rank_g[int(frozen_recovery_stats["best_index"])].item()),
                    "best_recovery_index": recovery_stats["best_index"],
                    "best_recovery_raw_score": recovery_stats["best_score"],
                    "best_recovery_p": None if recovery_stats["best_index"] is None else float(p[int(recovery_stats["best_index"])].item()),
                    "best_recovery_raw_g": None if recovery_stats["best_index"] is None else float(weighted_vec[int(recovery_stats["best_index"])].item()),
                    "best_recovery_rank_p": None if recovery_stats["best_index"] is None else float(rank_p[int(recovery_stats["best_index"])].item()),
                    "best_recovery_rank_g": None if recovery_stats["best_index"] is None else float(rank_g[int(recovery_stats["best_index"])].item()),
                    "best_recovery_rc_score": None if recovery_stats["best_index"] is None else float(rc_scores[int(recovery_stats["best_index"])].item()),
                    "recovery_score_median": recovery_stats["median"],
                    "recovery_score_q25": recovery_stats["q25"],
                    "recovery_score_q75": recovery_stats["q75"],
                    "recovery_score_mean": recovery_stats["mean"],
                    "recovery_score_std": recovery_stats["std"],
                    "recovery_score_iqr": recovery_stats["iqr"],
                    "recovery_score_pool_size": recovery_stats["pool_size"],
                    "recovery_prominence_fallback": recovery_stats["fallback"],
                    "Z_recovery": z_recovery,
                    "decision": chosen_type,
                    "selected_candidate_type": chosen_type,
                    "selected_candidate_index": int(chosen),
                })

        elif trainfree_restoration_mode == "single_step" and recover_num > 0:
            r_idx, gain_r, weighted_rv, uniform_rv = _tfcr_restoration_proposal()
            if r_idx is not None:
                selected.append(r_idx); tfcr_selected_rest.append(r_idx)
                available_tfcr[r_idx] = False; candidate_mask[r_idx] = False
                coverage = torch.maximum(coverage, evidence_kernel[:, r_idx])
            while len(selected) < select_num:
                c_idx = _tfcr_next_core()
                if c_idx is None: break
                selected.append(c_idx); tfcr_selected_core_cont.append(c_idx)
                available_tfcr[c_idx] = False; candidate_mask[c_idx] = False
                coverage = torch.maximum(coverage, evidence_kernel[:, c_idx])

        elif trainfree_restoration_mode == "pareto_competitive":
            while len(selected) < select_num:
                r_idx, gain_r, weighted_vec, uniform_vec = _tfcr_restoration_proposal()
                c_idx = _tfcr_next_core()
                weighted_r = float(weighted_vec[r_idx].item()) if r_idx is not None else float("-inf")
                uniform_r = float(uniform_vec[r_idx].item()) if r_idx is not None else float("-inf")
                weighted_c = float(weighted_vec[c_idx].item()) if c_idx is not None else float("-inf")
                uniform_c = float(uniform_vec[c_idx].item()) if c_idx is not None else float("-inf")
                pareto_accept = bool(
                    r_idx is not None and c_idx is not None and
                    weighted_r > weighted_c and uniform_r > uniform_c
                )
                if r_idx is not None and c_idx is not None and r_idx == c_idx:
                    pareto_accept = True
                chosen = r_idx if pareto_accept else c_idx
                chosen_type = "restoration" if pareto_accept else "core_continuation"
                if chosen is None:
                    break
                selected.append(int(chosen)); available_tfcr[int(chosen)] = False
                candidate_mask[int(chosen)] = False
                coverage = torch.maximum(coverage, evidence_kernel[:, int(chosen)])
                if chosen_type == "restoration": tfcr_selected_rest.append(int(chosen))
                else: tfcr_selected_core_cont.append(int(chosen))
                tfcr_competition_steps.append({
                    "step": int(len(selected) - core_num),
                    "restoration_candidate_index": None if r_idx is None else int(r_idx),
                    "core_candidate_index": None if c_idx is None else int(c_idx),
                    "weighted_restoration": weighted_r,
                    "weighted_core": weighted_c,
                    "uniform_restoration": uniform_r,
                    "uniform_core": uniform_c,
                    "pareto_accept": bool(pareto_accept),
                    "selected_candidate_type": chosen_type,
                    "selected_candidate_index": int(chosen),
                })

    # Offline candidate-utility audit hook.  It snapshots the exact first
    # restoration state before any restoration candidate is selected.  The
    # hook is opt-in and intentionally has no effect on the normal path.
    if os.environ.get("EC_QFICR_RESTORATION_AUDIT_CACHE_DIR", "").strip():
        audit_current_score, audit_weighted_residual = restoration_base_gain(coverage)
        audit_residual_mass = (evidence_kernel - coverage[:, None]).clamp_min(0.0).sum(dim=0)
        audit_uniform_residual = audit_residual_mass / max(float(num_tokens), 1.0)
        audit_anchor_factor = p.clamp_min(safe_eps).pow(anchor_alpha if prior_anchor else 1.0)
        audit_candidate_mask = available & candidate_mask
        _write_qficr_restoration_audit_probe({
            "version": 1,
            "num_tokens": int(num_tokens),
            "select_num": int(select_num),
            "states": states.detach().float().cpu(),
            "probs": p.detach().float().cpu(),
            "evidence_kernel": evidence_kernel.detach().float().cpu(),
            "coverage": coverage.detach().float().cpu(),
            "core_indices": core_indices.detach().long().cpu(),
            "qfi_order": qfi_order.detach().long().cpu(),
            "candidate_mask": audit_candidate_mask.detach().bool().cpu(),
            "current_score": audit_current_score.detach().float().cpu(),
            "weighted_residual": audit_weighted_residual.detach().float().cpu(),
            "uniform_residual": audit_uniform_residual.detach().float().cpu(),
            "candidate_probability": p.detach().float().cpu(),
            "anchor_factor": audit_anchor_factor.detach().float().cpu(),
            "residual_mass": audit_residual_mass.detach().float().cpu(),
            "question": question_text or "",
            "config": {
                "recover_budget_mode": budget_mode,
                "ratio_cap": float(ratio_cap),
                "comp_score_mode": comp_score_mode,
                "anchor_alpha": float(anchor_alpha),
                "rest_beta": float(rest_beta),
                "prior_anchor": bool(prior_anchor),
                "overlap_kernel": get_qfid_overlap_kernel(),
                "use_spatial_local_prior": bool(use_spatial_local_prior),
                "prior_ablation": prior_ablation,
                "prior_lambda": float(prior_lambda),
                "candidate_pool_enabled": bool(cand_pool_enabled),
            },
        })

    def normalize_extra(extra, mask):
        out = torch.zeros_like(extra, dtype=torch.float32, device=device)
        if not bool(mask.any().item()):
            return out
        vals = torch.nan_to_num(extra[mask].float(), nan=0.0, posinf=0.0, neginf=0.0)
        v_min = vals.min()
        v_max = vals.max()
        denom = v_max - v_min
        if not torch.isfinite(denom) or float(denom.abs().item()) <= safe_eps:
            return out
        out[mask] = ((vals - v_min) / denom).clamp(0.0, 1.0)
        return out

    def compute_restore_extra(g_res_local, base_gain_local, mask):
        if not restore_extra_enabled:
            return torch.zeros(num_tokens, dtype=torch.float32, device=device)
        core_coverage = coverage.clamp(0.0, 1.0)
        if restore_extra_prior == "subspace_residual":
            # Replacement prior: fast residual state signal, 1 - c_j(S_red).
            # It is aligned with QFi-CR's projection-overlap coverage and avoids
            # reintroducing the previously harmful spatial/local prior.
            if subspace_residual_mode == "full" and core_indices.numel() > 0:
                basis = []
                for idx_local in core_indices.tolist():
                    v = psi[int(idx_local)]
                    for b in basis:
                        v = v - torch.dot(v, b) * b
                    n = torch.linalg.vector_norm(v)
                    if float(n.item()) > safe_eps:
                        basis.append(v / n.clamp_min(safe_eps))
                if basis:
                    B = torch.stack(basis, dim=0)
                    residual = psi - (psi @ B.t()) @ B
                    extra = residual.pow(2).sum(dim=-1).clamp(0.0, 1.0)
                else:
                    extra = 1.0 - core_coverage
            else:
                extra = 1.0 - core_coverage
        elif restore_extra_prior == "uncovered_cls":
            cls_signal = p_cls_vec if p_cls_vec is not None else p
            extra = cls_signal * (1.0 - core_coverage)
        elif restore_extra_prior == "context_anti":
            extra = (1.0 - p.clamp(0.0, 1.0)) * g_res_local.clamp_min(0.0)
        else:
            extra = torch.zeros(num_tokens, dtype=torch.float32, device=device)
        extra = torch.nan_to_num(extra.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        norm_extra = normalize_extra(extra, mask)
        if bool(mask.any().item()):
            extra_vals = extra[mask]
            norm_vals = norm_extra[mask]
            base_vals = torch.nan_to_num(base_gain_local[mask].float(), nan=0.0, posinf=0.0, neginf=0.0)
            extra_stats["restore_extra_mean"] = float(extra_vals.mean().item()) if extra_vals.numel() else 0.0
            extra_stats["restore_extra_std"] = float(extra_vals.std(unbiased=False).item()) if extra_vals.numel() else 0.0
            extra_stats["restore_extra_min"] = float(extra_vals.min().item()) if extra_vals.numel() else 0.0
            extra_stats["restore_extra_max"] = float(extra_vals.max().item()) if extra_vals.numel() else 0.0
            extra_stats["restore_extra_norm_mean"] = float(norm_vals.mean().item()) if norm_vals.numel() else 0.0
            extra_stats["restore_extra_base_corr"] = _qficr_corr(norm_vals, base_vals, safe_eps) if norm_vals.numel() else 0.0
            extra_stats["restore_extra_mean_g_res"] = float(g_res_local[mask].mean().item()) if g_res_local[mask].numel() else 0.0
        return norm_extra

    rest_selected_gain_values = []
    rest_selected_res_values = []
    rest_selected_support_values = []
    rest_selected_self_values = []
    rest_selected_p_values = []
    rest_selected_core_overlap_values = []
    rest_selected_qfi_rank_values = []
    evidence_detail_selected = []
    evidence_complement_selected = []
    evidence_selected_branch = []

    def append_restoration_debug(idx_local, gain_local, g_res_local, current_coverage):
        if idx_local < 0 or idx_local >= num_tokens:
            return
        rest_selected_gain_values.append(float(torch.nan_to_num(gain_local[idx_local].float(), nan=0.0).item()))
        rest_selected_res_values.append(float(torch.nan_to_num(g_res_local[idx_local].float(), nan=0.0).item()))
        rest_selected_p_values.append(float(torch.nan_to_num(p[idx_local].float(), nan=0.0).item()))
        if core_indices.numel() > 0:
            overlap_val = evidence_kernel[idx_local, core_indices].max()
        else:
            overlap_val = torch.tensor(0.0, dtype=evidence_kernel.dtype, device=device)
        rest_selected_core_overlap_values.append(float(torch.nan_to_num(overlap_val.float(), nan=0.0).item()))
        rank_val = qfi_rank[idx_local]
        if torch.isfinite(rank_val):
            rest_selected_qfi_rank_values.append(float(rank_val.item()))
        if support_beta > 0.0:
            rest_selected_support_values.append(float(torch.nan_to_num(support[idx_local].float(), nan=0.0).item()))
        self_gain_local = p[idx_local] * (1.0 - current_coverage[idx_local].clamp(0.0, 1.0)).clamp_min(0.0)
        rest_selected_self_values.append(float(torch.nan_to_num(self_gain_local.float(), nan=0.0).item()))

    def rank_percentile_desc_index_asc(raw_local, mask_local):
        rank_score = torch.full((num_tokens,), float("-inf"), dtype=torch.float32, device=device)
        idxs = torch.nonzero(mask_local, as_tuple=False).reshape(-1)
        if idxs.numel() == 0:
            return rank_score
        idx_list = [int(x) for x in idxs.tolist()]
        idx_list.sort(key=lambda z: (-float(torch.nan_to_num(raw_local[z].float(), nan=0.0, posinf=0.0, neginf=0.0).item()), z))
        denom = max(len(idx_list) - 1, 1)
        for rank_local, idx_local in enumerate(idx_list):
            pct = 1.0 if len(idx_list) == 1 else 1.0 - (float(rank_local) / float(denom))
            rank_score[idx_local] = pct
        return rank_score

    def choose_by_score_desc_p_desc_index_asc(score_local, mask_local):
        idxs = torch.nonzero(mask_local, as_tuple=False).reshape(-1)
        if idxs.numel() == 0:
            return None
        idx_list = [int(x) for x in idxs.tolist()]
        idx_list.sort(key=lambda z: (
            -float(torch.nan_to_num(score_local[z].float(), nan=0.0, posinf=0.0, neginf=0.0).item()),
            -float(torch.nan_to_num(p[z].float().clamp_min(safe_eps).sqrt(), nan=0.0, posinf=0.0, neginf=0.0).item()),
            z,
        ))
        return idx_list[0]

    if not tfcr_active and comp_score_mode == "near_boundary_fill" and recover_num > 0:
        for idx_tensor in qfi_order.tolist():
            idx = int(idx_tensor)
            if len(selected) >= select_num:
                break
            if idx < 0 or idx >= num_tokens or not bool(available[idx].item()):
                continue
            gain, g_res = restoration_base_gain(coverage)
            append_restoration_debug(idx, gain, g_res, coverage)
            selected.append(idx)
            available[idx] = False
            candidate_mask[idx] = False
            coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    if not tfcr_active and restoration_mode == "random" and recover_num > 0:
        remaining_idx = torch.nonzero(available, as_tuple=False).reshape(-1)
        if remaining_idx.numel() > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(random_seed))
            order = torch.randperm(remaining_idx.numel(), device=device, generator=generator)
            random_idx = remaining_idx[order[:recover_num]]
            selected.extend([int(x) for x in random_idx.tolist()])
            available[random_idx] = False

    if not tfcr_active and comp_score_mode != "near_boundary_fill" and restoration_mode == "full" and restore_div_lambda > 0.0 and recover_num > 0:
        # Micro-tuning ablation: keep S_red fixed, build a small restoration
        # candidate pool by the normal residual gain, then greedily rerank with
        # a light kappa-based redundancy penalty. lambda=0 keeps the old path.
        base_gain, g_res = restoration_base_gain(coverage)
        if restore_extra_enabled:
            extra_norm = compute_restore_extra(g_res, base_gain, available & candidate_mask)
            base_gain = base_gain * (1.0 + restore_extra_lambda * extra_norm)
        base_gain = base_gain.masked_fill(~available, float("-inf"))
        base_gain = base_gain.masked_fill(~candidate_mask, float("-inf"))
        pool_size = min(
            max(int(math.ceil(float(restore_candidate_factor) * float(recover_num))), recover_num),
            int((available & candidate_mask).sum().item()),
        )
        if pool_size > 0:
            pool_idx = torch.topk(base_gain, k=pool_size).indices
            pool_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            pool_mask[pool_idx] = True
            while len(selected) < select_num and bool((available & pool_mask).any().item()):
                if selected:
                    selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
                    redundancy = evidence_kernel[:, selected_tensor].max(dim=1).values
                else:
                    redundancy = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)
                div_factor = (1.0 - redundancy.clamp(0.0, 1.0)).clamp_min(0.0).pow(restore_div_lambda)
                score = base_gain * div_factor
                score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
                score = score.masked_fill(~available, float("-inf"))
                score = score.masked_fill(~pool_mask, float("-inf"))
                idx = int(torch.argmax(score).item())
                if not bool(available[idx].item()) or not bool(pool_mask[idx].item()):
                    break
                append_restoration_debug(idx, score, g_res, coverage)
                selected.append(idx)
                available[idx] = False
                pool_mask[idx] = False
                candidate_mask[idx] = False
                coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    if (
        not tfcr_active
        and evidence_recovery_mode in {"detail", "dual_max"}
        and comp_score_mode != "near_boundary_fill"
        and restoration_mode == "full"
        and restore_div_lambda <= 0.0
        and recover_num > 0
        and len(selected) < select_num
    ):
        if core_indices.numel() > 0:
            core_kernel = evidence_kernel[:, core_indices]
            assigned_core_pos = torch.argmax(core_kernel, dim=1)
            assigned_core_index = core_indices[assigned_core_pos]
            detail_affinity = torch.gather(core_kernel, 1, assigned_core_pos.view(-1, 1)).reshape(-1)
        else:
            assigned_core_index = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
            detail_affinity = torch.zeros(num_tokens, dtype=torch.float32, device=device)
        sqrt_p = p.clamp_min(safe_eps).sqrt()
        while len(selected) < select_num:
            active_mask = available & candidate_mask
            if not bool(active_mask.any().item()):
                candidate_mask = available.clone()
                active_mask = available
                if not bool(active_mask.any().item()):
                    break

            complement_gain, complement_weighted = restoration_base_gain(coverage)
            complement_raw = complement_weighted.clamp_min(0.0) * sqrt_p
            if evidence_detail_selected:
                detail_red = torch.zeros(num_tokens, dtype=torch.float32, device=device)
                for ridx_local in evidence_detail_selected:
                    same_region = assigned_core_index == assigned_core_index[int(ridx_local)]
                    detail_red = torch.where(
                        same_region,
                        torch.maximum(detail_red, evidence_kernel[:, int(ridx_local)].float()),
                        detail_red,
                    )
                detail_red = detail_red.clamp(0.0, 1.0)
            else:
                detail_red = torch.zeros(num_tokens, dtype=torch.float32, device=device)
            detail_raw = sqrt_p * detail_affinity.float().clamp_min(0.0) * (1.0 - detail_red).clamp_min(0.0)
            detail_rank = rank_percentile_desc_index_asc(detail_raw, active_mask)
            complement_rank = rank_percentile_desc_index_asc(complement_raw, active_mask)
            if evidence_recovery_mode == "detail":
                evidence_score = detail_rank
                idx = choose_by_score_desc_p_desc_index_asc(evidence_score, active_mask)
                selected_branch = "detail"
            else:
                evidence_score = torch.maximum(detail_rank, complement_rank)
                idx = choose_by_score_desc_p_desc_index_asc(evidence_score, active_mask)
                if idx is None:
                    selected_branch = "none"
                else:
                    d_val = float(detail_rank[idx].item())
                    c_val = float(complement_rank[idx].item())
                    selected_branch = "detail" if d_val >= c_val else "complement"
            if idx is None or not bool(active_mask[idx].item()):
                break
            append_restoration_debug(idx, evidence_score, complement_weighted, coverage)
            selected.append(idx)
            available[idx] = False
            candidate_mask[idx] = False
            if selected_branch == "detail":
                evidence_detail_selected.append(idx)
            elif selected_branch == "complement":
                evidence_complement_selected.append(idx)
            evidence_selected_branch.append(selected_branch)
            coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    while not tfcr_active and evidence_recovery_mode == "frozen" and comp_score_mode != "near_boundary_fill" and len(selected) < select_num:
        gain, g_res = restoration_base_gain(coverage)
        if restore_extra_enabled:
            extra_norm = compute_restore_extra(g_res, gain, available & candidate_mask)
            gain = gain * (1.0 + restore_extra_lambda * extra_norm)
            gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        gain = gain.masked_fill(~available, float("-inf"))
        gain = gain.masked_fill(~candidate_mask, float("-inf"))
        idx = int(torch.argmax(gain).item())
        if not bool(available[idx].item()) or not bool(candidate_mask[idx].item()):
            candidate_mask = available.clone()
            gain, g_res = restoration_base_gain(coverage)
            if restore_extra_enabled:
                extra_norm = compute_restore_extra(g_res, gain, available)
                gain = gain * (1.0 + restore_extra_lambda * extra_norm)
                gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
            gain = gain.masked_fill(~available, float("-inf"))
            idx = int(torch.argmax(gain).item())
            if not bool(available[idx].item()):
                break
        append_restoration_debug(idx, gain, g_res, coverage)
        selected.append(idx)
        available[idx] = False
        candidate_mask[idx] = False
        coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    if len(selected) < select_num:
        filler = torch.nonzero(available, as_tuple=False).reshape(-1)
        selected.extend(filler[: select_num - len(selected)].tolist())

    selected_limited = selected[:select_num]
    core_count = min(core_num, len(selected_limited))
    core_selected = selected_limited[:core_count]
    rest_selected = selected_limited[core_count:]
    tfcr_actual_rest_count = len(tfcr_selected_rest) if tfcr_active else len(rest_selected)
    tfcr_actual_core_cont_count = len(tfcr_selected_core_cont) if tfcr_active else 0

    def _mean_indices(vec, indices):
        if not indices:
            return None
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        vals = torch.nan_to_num(vec[idx_tensor].float(), nan=0.0, posinf=0.0, neginf=0.0)
        return float(vals.mean().item()) if vals.numel() else None

    def _mean_list(values):
        return float(sum(values) / len(values)) if values else None

    def _median_list(values):
        if not values:
            return None
        vals = sorted(float(v) for v in values)
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return 0.5 * (vals[mid - 1] + vals[mid])

    def _values_at(vec, indices):
        if not indices:
            return []
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        vals = torch.nan_to_num(vec[idx_tensor].float(), nan=0.0, posinf=0.0, neginf=0.0)
        return [float(v) for v in vals.tolist()]

    def _rank_values(indices):
        vals = []
        for idx_local in indices:
            rank_val = qfi_rank[int(idx_local)]
            if torch.isfinite(rank_val):
                vals.append(float(rank_val.item()))
        return vals

    def _max_core_overlap_values(indices):
        if not indices:
            return []
        if core_indices.numel() <= 0:
            return [0.0 for _ in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        vals = evidence_kernel[idx_tensor][:, core_indices].max(dim=1).values
        vals = torch.nan_to_num(vals.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return [float(v) for v in vals.tolist()]

    if core_indices.numel() > 0:
        diagnostic_core_coverage = evidence_kernel[:, core_indices].max(dim=1).values
    else:
        diagnostic_core_coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)
    diagnostic_g_res = (p[:, None] * (evidence_kernel - diagnostic_core_coverage[:, None]).clamp_min(0.0)).sum(dim=0)
    rest_set = set(int(x) for x in rest_selected)
    core_set = set(int(x) for x in core_selected)
    discarded = [int(i) for i in range(num_tokens) if int(i) not in rest_set and int(i) not in core_set]
    discarded_p_values = _values_at(p, discarded)
    discarded_res_values = _values_at(diagnostic_g_res, discarded)
    discarded_overlap_values = _max_core_overlap_values(discarded)
    discarded_rank_values = _rank_values(discarded)
    near_boundary_threshold = float(select_num + max(1, recover_num) * 2)
    near_boundary_hits = [1.0 for v in rest_selected_qfi_rank_values if v <= near_boundary_threshold]

    info = {
        "adapt_mode": adapt_mode,
        "alloc_mode": alloc_mode,
        "question_type": question_type,
        "entropy": float(entropy.item()),
        "entropy_norm": entropy_norm,
        "recover_ratio_min": float(ratio_min),
        "recover_ratio_max": float(ratio_max),
        "recover_ratio_requested": float(recover_ratio_requested),
        "recover_ratio": recover_ratio,
        "recover_budget_mode": budget_mode,
        "recover_abs_cap": int(abs_cap),
        "recover_ratio_cap": float(ratio_cap),
        "recover_entropy_gamma": float(entropy_gamma),
        "recover_entropy_hat": float(entropy_hat),
        "recover_raw_k": int(raw_recover_num),
        "recover_cap_applied": bool(cap_applied),
        "k_core": min(core_num, len(selected)),
        "k_recover": tfcr_actual_rest_count if tfcr_active else max(0, min(select_num, len(selected)) - min(core_num, len(selected))),
        "recover_prior_anchor": bool(prior_anchor),
        "recover_prior_gamma": float(prior_gamma),
        "anchor_alpha": float(anchor_alpha),
        "comp_score_mode": comp_score_mode,
        "recover_cand_pool": bool(cand_pool_enabled),
        "recover_cand_mult": float(cand_mult),
        "recover_cand_size": int(candidate_size),
        "residual_alloc_mode": "legacy",
        "compete_margin": float(compete_margin),
        "K": int(select_num),
        "K_base": int(adaptive_base_core_count) if adaptive_base_core_count is not None else int(core_num),
        "K_rest_max": int(adaptive_max_recover_slots) if adaptive_max_recover_slots is not None else int(recover_num),
        "max_flexible_slots": int(recover_num),
        "fixed_core_count": int(core_num),
        "actual_residual_count": int(tfcr_actual_rest_count),
        "actual_core_count": int(core_count + tfcr_actual_core_cont_count) if tfcr_active else min(core_num, len(selected)),
        "candidate_pool_enabled": bool(cand_pool_enabled),
        "candidate_pool_size": int(candidate_size),
        "candidate_pool_multiplier": float(cand_mult),
        "full_core_indices": [int(x) for x in qfi_order[:select_num].tolist()],
        "fixed_core_indices": [int(x) for x in core_selected],
        "base_core_indices": [int(x) for x in core_indices.tolist()],
        "reserved_core_indices": [int(x) for x in qfi_order[core_num:select_num].tolist()],
        "selected_core_indices": [int(x) for x in ((core_indices.tolist() if core_indices.numel() else []) + tfcr_selected_core_cont)] if tfcr_active else [int(x) for x in core_selected],
        "selected_residual_indices": [int(x) for x in (tfcr_selected_rest if tfcr_active else rest_selected)],
        "recovery_indices": [int(x) for x in (tfcr_selected_rest if tfcr_active else rest_selected)],
        "fallback_core_indices": [int(x) for x in tfcr_selected_core_cont] if tfcr_active else [],
        "final_selected_indices": [int(x) for x in selected_limited],
        "competition_steps": [],
        "question_text_available": bool(question_available),
        "restoration_mode": restoration_mode,
        "fixed_restoration_ratio": float(fixed_restoration_ratio),
        "use_spatial_local_prior": bool(use_spatial_local_prior),
        "prior_ablation": prior_ablation,
        "prior_lambda": float(prior_lambda),
        "rest_beta": float(rest_beta),
        "support_beta": float(support_beta),
        "self_discount": float(self_discount),
        "obs_temperature": float(obs_temperature),
        "restore_div_lambda": float(restore_div_lambda),
        "restore_candidate_factor": float(restore_candidate_factor),
        "yn_budget_gate": float(yn_budget_gate),
        "yn_gate_mode": yn_gate_mode,
        "yn_gate_applied": bool(yn_gate_applied),
        "evidence_recovery_mode": evidence_recovery_mode,
        "recovery_score_mode": recovery_score_mode,
        "evidence_recovery_rank_convention": "descending_raw_index_asc_best1_worst0",
        "evidence_detail_indices": [int(x) for x in evidence_detail_selected],
        "evidence_complement_indices": [int(x) for x in evidence_complement_selected],
        "evidence_selected_branch": [str(x) for x in evidence_selected_branch],
        "entropy_anchor_ratio": float(entropy_anchor_ratio),
        "entropy_anchor_count": int(entropy_anchor_count),
        "residual_budget_gamma": float(residual_budget_gamma),
        "residual_budget_applied": bool(residual_budget_applied),
        "residual_mass": float(residual_mass),
        "k_core_original": int(k_core_original),
        "k_recover_original": int(k_recover_original),
        "trainfree_restoration_mode": trainfree_restoration_mode,
        "K_rest_cap": int(recover_num),
        "actual_restoration_count": int(tfcr_actual_rest_count),
        "core_continuation_count": int(tfcr_actual_core_cont_count),
        "restoration_step": int(len(tfcr_competition_steps)),
        "restoration_candidate_index": None if not tfcr_selected_rest else int(tfcr_selected_rest[0]),
        "core_candidate_index": None if not tfcr_selected_core_cont else int(tfcr_selected_core_cont[0]),
        "competition_steps": tfcr_competition_steps if tfcr_active else [],
        "duplicate_count": int(len(selected_limited) - len(set(selected_limited))),
        "nan_inf_count": int((~torch.isfinite(torch.tensor(selected_limited, device=device, dtype=torch.float32))).sum().item()) if selected_limited else 0,
        "restore_extra_prior": restore_extra_prior,
        "restore_extra_lambda": float(restore_extra_lambda),
        "subspace_residual_mode": subspace_residual_mode,
        "restore_extra_enabled": bool(restore_extra_enabled),
        **extra_stats,
        "random_seed": int(random_seed),
        "spatial_prior_mean": float(spatial_prior.mean().item()) if spatial_prior.numel() > 0 else 0.0,
        "local_prior_mean": float(local_prior.mean().item()) if local_prior.numel() > 0 else 0.0,
        "spatial_local_prior_mean": float(spatial_local_prior.mean().item()) if spatial_local_prior.numel() > 0 else 0.0,
        "mean_p_core": _mean_indices(p, core_selected),
        "mean_p_rest": _mean_indices(p, rest_selected),
        "median_p_rest": _median_list(rest_selected_p_values),
        "mean_residual_gain_rest": _mean_list(rest_selected_res_values),
        "median_residual_gain_rest": _median_list(rest_selected_res_values),
        "mean_max_overlap_to_core_rest": _mean_list(rest_selected_core_overlap_values),
        "median_max_overlap_to_core_rest": _median_list(rest_selected_core_overlap_values),
        "mean_qfi_rank_rest": _mean_list(rest_selected_qfi_rank_values),
        "median_qfi_rank_rest": _median_list(rest_selected_qfi_rank_values),
        "near_boundary_rank_threshold": near_boundary_threshold,
        "near_boundary_rank_hit_rate": None if not rest_selected_qfi_rank_values else float(len(near_boundary_hits) / len(rest_selected_qfi_rank_values)),
        "mean_p_discarded": _mean_list(discarded_p_values),
        "median_p_discarded": _median_list(discarded_p_values),
        "mean_residual_gain_discarded": _mean_list(discarded_res_values),
        "median_residual_gain_discarded": _median_list(discarded_res_values),
        "mean_max_overlap_to_core_discarded": _mean_list(discarded_overlap_values),
        "median_max_overlap_to_core_discarded": _median_list(discarded_overlap_values),
        "mean_qfi_rank_discarded": _mean_list(discarded_rank_values),
        "median_qfi_rank_discarded": _median_list(discarded_rank_values),
        "mean_support_rest": _mean_list(rest_selected_support_values),
        "mean_self_gain_rest": _mean_list(rest_selected_self_values),
        "mean_restoration_score_rest": _mean_list(rest_selected_gain_values),
    }
    debug_prior_path = os.environ.get("EC_QFICR_DEBUG_PRIOR_STATS_JSONL", "").strip()
    if debug_prior_path:
        try:
            if core_indices.numel() > 0:
                core_coverage = evidence_kernel[:, core_indices].max(dim=1).values
            else:
                core_coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)
            g_res = (p[:, None] * (evidence_kernel - core_coverage[:, None]).clamp_min(0.0)).sum(dim=0)
            g_rest_without_prior = g_res * p.clamp_min(safe_eps).pow(anchor_alpha if prior_anchor else 1.0)
            modulation = 1.0 + prior_lambda * entropy_norm * spatial_local_prior

            def greedy_rest(use_prior):
                rest_selected = []
                rest_available = torch.ones(num_tokens, dtype=torch.bool, device=device)
                if core_indices.numel() > 0:
                    rest_available[core_indices] = False
                rest_coverage = core_coverage.clone()
                for _ in range(max(0, recover_num)):
                    rest_gain = (p[:, None] * (evidence_kernel - rest_coverage[:, None]).clamp_min(0.0)).sum(dim=0)
                    if prior_anchor:
                        rest_gain = rest_gain * p.clamp_min(safe_eps).pow(anchor_alpha)
                    if use_prior:
                        rest_gain = rest_gain * (1.0 + entropy_norm * spatial_local_prior)
                    rest_gain = torch.nan_to_num(rest_gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    rest_gain = rest_gain.masked_fill(~rest_available, float("-inf"))
                    idx_local = int(torch.argmax(rest_gain).item())
                    if not bool(rest_available[idx_local].item()):
                        break
                    rest_selected.append(idx_local)
                    rest_available[idx_local] = False
                    rest_coverage = torch.maximum(rest_coverage, evidence_kernel[:, idx_local])
                return torch.tensor(rest_selected, dtype=torch.long, device=device)

            rest_no_prior = greedy_rest(False)
            rest_full_prior = greedy_rest(True)
            no_set = set(int(x) for x in rest_no_prior.tolist())
            full_set = set(int(x) for x in rest_full_prior.tolist())
            overlap = len(no_set & full_set) / max(1, min(len(no_set), len(full_set)))
            full_extra = torch.tensor(sorted(full_set - no_set), dtype=torch.long, device=device)
            no_extra = torch.tensor(sorted(no_set - full_set), dtype=torch.long, device=device)
            record = {
                "num_tokens": int(num_tokens),
                "k": int(select_num),
                "k_core": int(core_num),
                "k_recover": int(recover_num),
                "entropy_norm": float(entropy_norm),
                "anchor_alpha": float(anchor_alpha),
                "prior_lambda": float(prior_lambda),
                "prior_ablation": prior_ablation,
                "a_j": _qficr_vector_stats(spatial_prior),
                "l_j": _qficr_vector_stats(local_prior),
                "b_j": _qficr_vector_stats(spatial_local_prior),
                "m_j": _qficr_vector_stats(modulation),
                "corr": {
                    "a_p": _qficr_corr(spatial_prior, p, safe_eps),
                    "l_p": _qficr_corr(local_prior, p, safe_eps),
                    "b_p": _qficr_corr(spatial_local_prior, p, safe_eps),
                    "a_g_res": _qficr_corr(spatial_prior, g_res, safe_eps),
                    "l_g_res": _qficr_corr(local_prior, g_res, safe_eps),
                    "b_g_res": _qficr_corr(spatial_local_prior, g_res, safe_eps),
                    "b_g_rest_without_prior": _qficr_corr(spatial_local_prior, g_rest_without_prior, safe_eps),
                },
                "rest_selection": {
                    "overlap_ratio": float(overlap),
                    "full_prior_extra_mean_p": _qficr_mean_at(p, full_extra),
                    "full_prior_extra_mean_g_res": _qficr_mean_at(g_res, full_extra),
                    "no_prior_extra_mean_p": _qficr_mean_at(p, no_extra),
                    "no_prior_extra_mean_g_res": _qficr_mean_at(g_res, no_extra),
                    "full_prior_extra_count": int(full_extra.numel()),
                    "no_prior_extra_count": int(no_extra.numel()),
                },
            }
            os.makedirs(os.path.dirname(debug_prior_path) or ".", exist_ok=True)
            with open(debug_prior_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass
    return torch.tensor(selected[:select_num], dtype=torch.long, device=device), info


def infer_qmo_recovery_type(question):
    """QMO-CR recovery type with a few OCR/counting/text-specific aliases."""
    if not isinstance(question, str) or not question.strip():
        return "default"

    q = question.lower().strip()
    if any(word in q for word in ("ocr", "text", "letter", "word", "sign", "read", "written", "say")):
        return "ocr"
    if any(word in q for word in ("color", "colour", "shape", "material", "size")):
        return "attr"
    if "how many" in q or any(word in q for word in ("count", "number of", "amount of")):
        return "compare"

    return infer_question_recovery_type(question)


def infer_qmo_gate_type(question):
    """Conservative text-only gate for selective QMO use."""
    if not isinstance(question, str) or not question.strip():
        return "default"

    q = re.sub(r"\s+", " ", question.lower().strip())
    compare_terms = (
        "more", "less", "fewer", "larger", "smaller", "bigger", "taller",
        "shorter", "higher", "lower", "same", "different", "than", "equal",
        "closest", "farthest", "left of", "right of", "compare",
    )
    category_phrases = (
        "what kind", "what type", "what category", "what sort",
        "which kind", "which type",
    )
    object_phrases = (
        "what object", "what item", "what thing", "what is on", "what is in",
        "what is near", "what is next to", "what is beside",
    )

    if any(term in q for term in compare_terms):
        return "compare"
    if any(phrase in q for phrase in category_phrases):
        return "category"
    if any(phrase in q for phrase in object_phrases):
        return "object"
    return "default"


def _stable_minmax_probability(raw, eps=1e-12):
    safe_eps = max(float(eps), 1e-12)
    x = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if x.numel() == 0:
        return x, x
    x_min = x.min()
    x_max = x.max()
    scale = x_max - x_min
    if torch.isfinite(scale) and scale.abs().item() > safe_eps:
        h = (x - x_min) / scale.clamp_min(safe_eps)
    else:
        total = x.clamp_min(0.0).sum()
        if torch.isfinite(total) and total.item() > safe_eps:
            p_raw = x.clamp_min(0.0) / total.clamp_min(safe_eps)
            h = p_raw - p_raw.min()
            h = h / h.max().clamp_min(safe_eps)
        else:
            h = torch.ones_like(x)
    p = h.clamp_min(safe_eps)
    p = p / p.sum().clamp_min(safe_eps)
    return h.clamp(0.0, 1.0), p


def _qmo_norm_gain(J, covered, p, available):
    gain = ((J - covered[:, None]).clamp_min(0.0) * p[:, None]).sum(dim=0)
    gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
    finite_gain = gain[available]
    if finite_gain.numel() > 0:
        g_min = finite_gain.min()
        g_max = finite_gain.max()
        if torch.isfinite(g_max - g_min) and (g_max - g_min).abs().item() > 1e-12:
            gain = (gain - g_min) / (g_max - g_min).clamp_min(1e-12)
        else:
            gain = torch.zeros_like(gain)
    else:
        gain = torch.zeros_like(gain)
    return gain.clamp(0.0, 1.0)


def select_by_qmo_cr(
    states,
    probs,
    k,
    question_text=None,
    eps=1e-12,
):
    """
    QMO-CR: multi-observable field + state-fidelity coupled recovery.

    The selector reuses the already-computed QFi probability as the task prior,
    builds J_ij = sqrt(p_i p_j) kappa_ij, where kappa is the configured
    projection overlap kernel, then greedily selects a coupled core followed by
    prior-guided recovery tokens.
    """
    if not torch.is_tensor(states) or states.ndim != 2:
        shape = tuple(states.shape) if torch.is_tensor(states) else type(states)
        raise ValueError(f"states must be [N, D], got {shape}")
    if not torch.is_tensor(probs) or probs.ndim != 1:
        shape = tuple(probs.shape) if torch.is_tensor(probs) else type(probs)
        raise ValueError(f"probs must be [N], got {shape}")
    if states.shape[0] != probs.numel():
        raise ValueError(f"states/probs token mismatch: {states.shape[0]} vs {probs.numel()}")

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    safe_eps = max(float(eps), 1e-12)
    if select_num == 0:
        info = {
            "qmo_cr_enabled": True,
            "question_type": "default",
            "entropy": 0.0,
            "purity": 0.0,
            "D_q": 0.0,
            "recover_ratio": 0.0,
            "k_core": 0,
            "k_recover": 0,
            "core_indices": [],
            "recovery_indices": [],
            "selected_indices": [],
        }
        return torch.empty(0, dtype=torch.long, device=device), info

    h, p = _stable_minmax_probability(probs.to(device=device), eps=safe_eps)

    psi = F.normalize(
        torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    F_state = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)
    J = torch.sqrt(p[:, None] * p[None, :] + safe_eps) * F_state
    J = torch.nan_to_num(J.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    entropy = torch.zeros((), device=device, dtype=torch.float32)
    if num_tokens > 1:
        entropy = (-(p * torch.log(p + safe_eps)).sum() / math.log(float(num_tokens))).clamp(0.0, 1.0)
    purity = ((p[:, None] * p[None, :]) * F_state).sum().clamp(0.0, 1.0)

    use_purity = get_env_bool("EC_QMO_USE_PURITY", default=True)
    lambda_h = max(0.0, _env_float("EC_QMO_LAMBDA_H", "0.5"))
    lambda_p = max(0.0, _env_float("EC_QMO_LAMBDA_P", "0.5"))
    if use_purity:
        D_q = (lambda_h * entropy + lambda_p * (1.0 - purity)).clamp(0.0, 1.0)
    else:
        D_q = entropy

    qtype = infer_qmo_recovery_type(question_text)
    ratio_min = min(max(_env_float("EC_QMO_RECOVER_MIN_RATIO", "0.00"), 0.0), 1.0)
    ratio_max = min(max(_env_float("EC_QMO_RECOVER_MAX_RATIO", "0.25"), 0.0), 1.0)
    if ratio_min > ratio_max:
        ratio_min, ratio_max = ratio_max, ratio_min
    ratio_by_type = {
        "binary": _env_float("EC_QMO_RECOVER_BINARY_RATIO", "0.0625"),
        "attr": _env_float("EC_QMO_RECOVER_ATTR_RATIO", "0.125"),
        "relation": _env_float("EC_QMO_RECOVER_REL_RATIO", "0.25"),
        "compare": _env_float("EC_QMO_RECOVER_COMPARE_RATIO", "0.25"),
        "open": _env_float("EC_QMO_RECOVER_OPEN_RATIO", "0.25"),
        "ocr": _env_float("EC_QMO_RECOVER_OCR_RATIO", "0.25"),
        "default": _env_float("EC_QMO_RECOVER_DEFAULT_RATIO", "0.125"),
    }
    base_ratio = min(max(float(ratio_by_type.get(qtype, ratio_by_type["default"])), ratio_min), ratio_max)
    tau_d = _env_float("EC_QMO_D_TAU", "0.75")
    eta = _env_float("EC_QMO_D_ETA", "0.5")
    recover_ratio = base_ratio * (1.0 + eta * (float(D_q.item()) - tau_d))
    recover_ratio = min(max(float(recover_ratio), ratio_min), ratio_max)
    cap = max(0, _env_int("EC_QMO_RECOVER_CAP", "16"))
    recover_num = min(max(int(round(recover_ratio * select_num)), 0), select_num - 1 if select_num > 1 else 0, cap)
    core_num = max(1, select_num - recover_num) if select_num > 0 else 0
    recover_num = select_num - core_num

    w_cov = _env_float("EC_QMO_W_COV", "1.0")
    w_h = _env_float("EC_QMO_W_H", "0.5")
    lambda_j = max(0.0, _env_float("EC_QMO_LAMBDA_J", "0.2"))
    reduce_mode = os.environ.get("EC_QMO_COUPLING_REDUCE", "mean").strip().lower()
    if reduce_mode not in {"mean", "sum"}:
        reduce_mode = "mean"
    alpha_prior = _env_float("EC_QMO_RECOVER_ALPHA_PRIOR", "0.5")
    prior_gamma = max(0.0, _env_float("EC_QMO_RECOVER_GAMMA", "0.5"))

    selected = []
    core_indices = []
    recovery_indices = []
    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    covered = torch.zeros(num_tokens, dtype=J.dtype, device=device)

    def select_one(use_prior):
        cov_gain = _qmo_norm_gain(J, covered, p, available)
        if selected:
            selected_idx = torch.tensor(selected, dtype=torch.long, device=device)
            red = J[:, selected_idx].sum(dim=1) if reduce_mode == "sum" else J[:, selected_idx].mean(dim=1)
        else:
            red = torch.zeros(num_tokens, dtype=J.dtype, device=device)
        score = w_cov * cov_gain + w_h * h - lambda_j * red
        if use_prior:
            score = score + alpha_prior * p.clamp_min(safe_eps).pow(prior_gamma)
        score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        score = score.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(score).item())
        if not bool(available[idx].item()):
            return None
        return idx

    while len(core_indices) < core_num:
        idx = select_one(use_prior=False)
        if idx is None:
            break
        selected.append(idx)
        core_indices.append(idx)
        available[idx] = False
        covered = torch.maximum(covered, J[:, idx])

    while len(recovery_indices) < recover_num:
        idx = select_one(use_prior=True)
        if idx is None:
            break
        selected.append(idx)
        recovery_indices.append(idx)
        available[idx] = False
        covered = torch.maximum(covered, J[:, idx])

    if len(selected) < select_num:
        filler = torch.nonzero(available, as_tuple=False).reshape(-1)
        for idx in filler[: select_num - len(selected)].tolist():
            selected.append(int(idx))
            recovery_indices.append(int(idx))

    selected = selected[:select_num]
    keep_idx = torch.tensor(selected, dtype=torch.long, device=device)
    keep_idx = torch.unique(keep_idx, sorted=True)
    if keep_idx.numel() < select_num:
        mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        mask[keep_idx] = False
        filler = torch.nonzero(mask, as_tuple=False).reshape(-1)
        keep_idx = torch.cat([keep_idx, filler[: select_num - keep_idx.numel()]]).sort().values
    if keep_idx.numel() > select_num:
        keep_idx = keep_idx[:select_num].sort().values

    info = {
        "qmo_cr_enabled": True,
        "question_type": qtype,
        "entropy": float(entropy.item()),
        "purity": float(purity.item()),
        "D_q": float(D_q.item()),
        "use_purity": bool(use_purity),
        "lambda_h": float(lambda_h),
        "lambda_p": float(lambda_p),
        "d_tau": float(tau_d),
        "d_eta": float(eta),
        "recover_ratio": float(recover_ratio),
        "k_core": int(min(core_num, len(core_indices))),
        "k_recover": int(max(0, select_num - min(core_num, len(core_indices)))),
        "w_cov": float(w_cov),
        "w_h": float(w_h),
        "lambda_j": float(lambda_j),
        "coupling_reduce": reduce_mode,
        "recover_alpha_prior": float(alpha_prior),
        "recover_gamma": float(prior_gamma),
        "h_min": float(h.min().item()) if h.numel() else 0.0,
        "h_max": float(h.max().item()) if h.numel() else 0.0,
        "h_mean": float(h.mean().item()) if h.numel() else 0.0,
        "J_min": float(J.min().item()) if J.numel() else 0.0,
        "J_max": float(J.max().item()) if J.numel() else 0.0,
        "J_mean": float(J.mean().item()) if J.numel() else 0.0,
        "core_indices": [int(x) for x in core_indices],
        "recovery_indices": [int(x) for x in recovery_indices],
        "selected_indices": [int(x) for x in keep_idx.tolist()],
    }
    return keep_idx.long(), info


def _overlap_ratio(indices_a, indices_b):
    a = {int(x) for x in indices_a}
    b = {int(x) for x in indices_b}
    denom = max(1, len(a))
    return float(len(a & b) / denom)


def select_by_qmo_gated(
    states,
    probs,
    k,
    qfi_core_order,
    question_text=None,
    qmo_states=None,
    eps=1e-12,
):
    """Text-gated QMO selector with baseline fallback."""
    baseline_keep, baseline_info = select_by_adaptive_core_recover(
        states=states,
        probs=probs,
        k=k,
        qfi_core_order=qfi_core_order,
        question_text=question_text,
        eps=eps,
    )

    qmo_states = states if qmo_states is None else qmo_states
    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    safe_eps = max(float(eps), 1e-12)
    gate_type = infer_qmo_gate_type(question_text)
    gate_types = {
        item.strip().lower()
        for item in os.environ.get("EC_QMO_GATE_TYPES", "compare,category,object").split(",")
        if item.strip()
    }
    gate_enabled = gate_type in gate_types
    gate_mode = os.environ.get("EC_QMO_GATE_MODE", "recovery_only").strip().lower()
    if gate_mode not in {"full", "recovery_only"}:
        gate_mode = "recovery_only"

    qmo_keep = baseline_keep
    qmo_cr_info = {
        "qmo_cr_enabled": False,
        "question_type": "none",
        "entropy": 0.0,
        "purity": 0.0,
        "D_q": 0.0,
        "use_purity": False,
        "lambda_h": 0.0,
        "lambda_p": 0.0,
        "d_tau": 0.0,
        "d_eta": 0.0,
        "recover_ratio": 0.0,
        "k_core": 0,
        "k_recover": 0,
        "w_cov": 0.0,
        "w_h": 0.0,
        "lambda_j": 0.0,
        "coupling_reduce": "mean",
        "recover_alpha_prior": 0.0,
        "recover_gamma": 0.0,
        "h_min": 0.0,
        "h_max": 0.0,
        "h_mean": 0.0,
        "J_min": 0.0,
        "J_max": 0.0,
        "J_mean": 0.0,
        "core_indices": [],
        "recovery_indices": [],
        "selected_indices": [int(x) for x in baseline_keep.tolist()],
    }
    if gate_enabled:
        qmo_keep, qmo_cr_info = select_by_qmo_cr(
            states=qmo_states,
            probs=probs,
            k=k,
            question_text=question_text,
            eps=eps,
        )

    if not gate_enabled or gate_mode == "full" or select_num == 0:
        final_keep = qmo_keep if gate_enabled and gate_mode == "full" else baseline_keep
        final_keep = torch.unique(final_keep.long(), sorted=True)
        if final_keep.numel() < select_num:
            mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            mask[final_keep] = False
            filler = torch.nonzero(mask, as_tuple=False).reshape(-1)
            final_keep = torch.cat([final_keep, filler[: select_num - final_keep.numel()]]).sort().values
        if final_keep.numel() > select_num:
            final_keep = final_keep[:select_num].sort().values
        info = {
            "gate_enabled": bool(gate_enabled),
            "gate_type": gate_type,
            "gate_mode": gate_mode,
            "gate_types": sorted(gate_types),
            "gate_beta": 0.0,
            "baseline_info": baseline_info,
            "qmo_cr_info": qmo_cr_info,
            "baseline_indices": [int(x) for x in baseline_keep.tolist()],
            "qmo_cr_indices": [int(x) for x in qmo_keep.tolist()],
            "selected_indices": [int(x) for x in final_keep.tolist()],
            "baseline_overlap": _overlap_ratio(final_keep.tolist(), baseline_keep.tolist()),
            "qmo_cr_overlap": _overlap_ratio(final_keep.tolist(), qmo_keep.tolist()),
        }
        return final_keep.long(), info

    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)
    qfi_order = qfi_core_order.to(device=device, dtype=torch.long)
    qfi_order = qfi_order[(qfi_order >= 0) & (qfi_order < num_tokens)]
    if qfi_order.numel() > 0:
        seen = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        ordered_unique = []
        for idx in qfi_order.tolist():
            if not bool(seen[idx].item()):
                seen[idx] = True
                ordered_unique.append(int(idx))
        qfi_order = torch.tensor(ordered_unique, dtype=torch.long, device=device)

    recover_num = int(baseline_info["k_recover"])
    core_num = max(0, min(select_num, select_num - recover_num))
    core_indices = qfi_order[:core_num] if core_num > 0 else torch.empty(0, dtype=torch.long, device=device)
    selected = [int(idx) for idx in core_indices.tolist()]

    psi = F.normalize(
        torch.nan_to_num(qmo_states.float(), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=safe_eps,
    )
    F_state = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)
    evidence_kernel = F_state
    h, _ = _stable_minmax_probability(probs.to(device=device), eps=safe_eps)

    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if core_indices.numel() > 0:
        available[core_indices] = False
        coverage = evidence_kernel[:, core_indices].max(dim=1).values
        redundancy = evidence_kernel[:, core_indices].mean(dim=1)
    else:
        coverage = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)
        redundancy = torch.zeros(num_tokens, dtype=evidence_kernel.dtype, device=device)

    prior_anchor = get_env_bool("EC_QFID_RECOVER_PRIOR_ANCHOR", default=True)
    prior_gamma = max(_env_float("EC_QFID_RECOVER_PRIOR_GAMMA", "0.5"), 0.0)
    cand_pool_enabled = get_env_bool("EC_QFID_RECOVER_CAND_POOL", default=True)
    cand_mult = max(_env_float("EC_QFID_RECOVER_CAND_MULT", "3.0"), 0.0)
    candidate_mask = available.clone()
    candidate_size = int(candidate_mask.sum().item())
    if cand_pool_enabled and recover_num > 0:
        remaining_idx = torch.nonzero(available, as_tuple=False).reshape(-1)
        pool_num = min(int(round(select_num * cand_mult)), int(remaining_idx.numel()))
        if pool_num >= recover_num and pool_num > 0:
            top_order = torch.argsort(p[remaining_idx], descending=True)
            top_idx = remaining_idx[top_order[:pool_num]]
            candidate_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            candidate_mask[top_idx] = True
            candidate_size = int(top_idx.numel())

    beta_qmo = max(0.0, _env_float("EC_QMO_GATE_BETA", "0.1"))
    w_h = _env_float("EC_QMO_W_H", "0.1")
    lambda_j = max(0.0, _env_float("EC_QMO_LAMBDA_J", "0.01"))

    while len(selected) < select_num:
        marginal = (evidence_kernel - coverage[:, None]).clamp_min(0.0)
        baseline_gain = (p[:, None] * marginal).sum(dim=0)
        if prior_anchor:
            baseline_gain = baseline_gain * p.clamp_min(safe_eps).pow(prior_gamma)
        qmo_bonus = w_h * h - lambda_j * redundancy
        qmo_bonus = normalize(qmo_bonus, eps=safe_eps)
        gain = baseline_gain + beta_qmo * qmo_bonus
        gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
        gain = gain.masked_fill(~available, float("-inf"))
        gain = gain.masked_fill(~candidate_mask, float("-inf"))
        idx = int(torch.argmax(gain).item())
        if not bool(available[idx].item()) or not bool(candidate_mask[idx].item()):
            candidate_mask = available.clone()
            gain = baseline_gain + beta_qmo * qmo_bonus
            gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
            gain = gain.masked_fill(~available, float("-inf"))
            idx = int(torch.argmax(gain).item())
            if not bool(available[idx].item()):
                break
        selected.append(idx)
        available[idx] = False
        candidate_mask[idx] = False
        coverage = torch.maximum(coverage, evidence_kernel[:, idx])

    if len(selected) < select_num:
        filler = torch.nonzero(available, as_tuple=False).reshape(-1)
        selected.extend(filler[: select_num - len(selected)].tolist())

    final_keep = torch.tensor(selected[:select_num], dtype=torch.long, device=device)
    final_keep = torch.unique(final_keep, sorted=True)
    if final_keep.numel() < select_num:
        mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        mask[final_keep] = False
        filler = torch.nonzero(mask, as_tuple=False).reshape(-1)
        final_keep = torch.cat([final_keep, filler[: select_num - final_keep.numel()]]).sort().values
    if final_keep.numel() > select_num:
        final_keep = final_keep[:select_num].sort().values

    baseline_info = dict(baseline_info)
    baseline_info.update({
        "adapt_mode": "qmo_gated_recovery_only",
        "question_type": gate_type,
        "k_core": int(min(core_num, select_num)),
        "k_recover": int(max(0, select_num - min(core_num, select_num))),
        "recover_cand_pool": bool(cand_pool_enabled),
        "recover_cand_mult": float(cand_mult),
        "recover_cand_size": int(candidate_size),
    })
    info = {
        "gate_enabled": True,
        "gate_type": gate_type,
        "gate_mode": gate_mode,
        "gate_types": sorted(gate_types),
        "gate_beta": float(beta_qmo),
        "baseline_info": baseline_info,
        "qmo_cr_info": qmo_cr_info,
        "baseline_indices": [int(x) for x in baseline_keep.tolist()],
        "qmo_cr_indices": [int(x) for x in qmo_keep.tolist()],
        "selected_indices": [int(x) for x in final_keep.tolist()],
        "baseline_overlap": _overlap_ratio(final_keep.tolist(), baseline_keep.tolist()),
        "qmo_cr_overlap": _overlap_ratio(final_keep.tolist(), qmo_keep.tolist()),
        "core_indices": [int(x) for x in core_indices.tolist()],
        "recovery_indices": [int(x) for x in final_keep.tolist() if int(x) not in {int(v) for v in core_indices.tolist()}],
    }
    return final_keep.long(), info


def _qsp_unique_order(order, num_tokens, device):
    order = order.to(device=device, dtype=torch.long)
    order = order[(order >= 0) & (order < num_tokens)]
    if order.numel() == 0:
        return order
    seen = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    unique = []
    for idx in order.tolist():
        if not bool(seen[idx].item()):
            seen[idx] = True
            unique.append(int(idx))
    return torch.tensor(unique, dtype=torch.long, device=device)


def _qsp_finalize_indices(selected, select_num, num_tokens, device):
    keep_idx = torch.tensor(selected[:select_num], dtype=torch.long, device=device)
    keep_idx = torch.unique(keep_idx, sorted=True)
    if keep_idx.numel() < select_num:
        mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        if keep_idx.numel() > 0:
            mask[keep_idx] = False
        filler = torch.nonzero(mask, as_tuple=False).reshape(-1)
        keep_idx = torch.cat([keep_idx, filler[: select_num - keep_idx.numel()]]).sort().values
    if keep_idx.numel() > select_num:
        keep_idx = keep_idx[:select_num].sort().values
    return keep_idx.long()


def _qfi_cards_base_record(profile, keep_idx, adaptive_info, probs, num_tokens, fallback_used=False, fallback_reason=""):
    final_indices = [int(x) for x in keep_idx.tolist()]
    k_core = int(adaptive_info.get("k_core", 0))
    core_indices = final_indices[:k_core]
    rec_indices = final_indices[k_core:]
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(1e-12)

    def mean_at(indices):
        if not indices:
            return 0.0
        idx = torch.tensor(indices, dtype=torch.long, device=probs.device)
        idx = idx[(idx >= 0) & (idx < int(num_tokens))]
        return float(p[idx].mean().item()) if idx.numel() > 0 else 0.0

    return {
        "profile": profile,
        "k": int(len(final_indices)),
        "num_tokens": int(num_tokens),
        "k_core": k_core,
        "k_rec": max(0, int(len(final_indices)) - k_core),
        "final_indices": final_indices,
        "core_indices": core_indices,
        "rec_indices": rec_indices,
        "mean_p_final": mean_at(final_indices),
        "mean_p_core": mean_at(core_indices),
        "mean_p_rec": mean_at(rec_indices),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason)[:240],
    }


def compute_zeno_persistence_score(attn_obs, num_tokens=None, eps=1e-12):
    """
    Quantum Zeno-inspired persistent observation score.

    The preferred input is attention observations shaped [L, H, N] or [H, N].
    If only a single token observation vector is available, this function uses
    it as the persistence proxy and normalizes it to [0, 1].
    """
    if attn_obs is None or not torch.is_tensor(attn_obs):
        if num_tokens is None:
            return None
        return torch.zeros(int(num_tokens), dtype=torch.float32)
    obs = torch.nan_to_num(attn_obs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if obs.ndim == 1:
        score = obs
    elif obs.ndim == 2:
        top_ratio = max(0.0, min(_env_float("EC_QFI_ZENO_TOP_RATIO", "0.10"), 1.0))
        if top_ratio > 0.0 and obs.shape[-1] > 0:
            top_r = max(1, int(round(float(obs.shape[-1]) * top_ratio)))
            top_idx = torch.topk(obs, k=min(top_r, obs.shape[-1]), dim=-1).indices
            mask = torch.zeros_like(obs, dtype=torch.float32)
            mask.scatter_(-1, top_idx, 1.0)
            score = mask.mean(dim=0)
        else:
            score = obs.mean(dim=0)
    else:
        flat = obs.reshape(-1, obs.shape[-1])
        top_ratio = max(0.0, min(_env_float("EC_QFI_ZENO_TOP_RATIO", "0.10"), 1.0))
        if top_ratio > 0.0 and flat.shape[-1] > 0:
            top_r = max(1, int(round(float(flat.shape[-1]) * top_ratio)))
            top_idx = torch.topk(flat, k=min(top_r, flat.shape[-1]), dim=-1).indices
            mask = torch.zeros_like(flat, dtype=torch.float32)
            mask.scatter_(-1, top_idx, 1.0)
            score = mask.mean(dim=0)
        else:
            score = flat.mean(dim=0)
    if num_tokens is not None and score.numel() != int(num_tokens):
        score = score.reshape(-1)[: int(num_tokens)]
        if score.numel() < int(num_tokens):
            score = torch.cat([score, torch.zeros(int(num_tokens) - score.numel(), device=score.device)])
    return normalize(score.reshape(-1), eps=max(float(eps), 1e-12)).clamp(0.0, 1.0)


def apply_zeno_bonus(qfi_score, zeno_score, beta):
    """score = qfi_score * (1 + beta * zeno_score)."""
    beta = max(0.0, float(beta))
    return torch.nan_to_num(qfi_score.float() * (1.0 + beta * zeno_score.float()), nan=0.0, posinf=0.0, neginf=0.0)


def build_spatial_buffer_candidates(core_indices, grid_size, mode="8neighbor"):
    """Build 4-neighbor or 8-neighbor spatial buffer candidates around anchors."""
    if isinstance(grid_size, int):
        h = w = int(grid_size)
    else:
        h, w = int(grid_size[0]), int(grid_size[1])
    if h <= 0 or w <= 0:
        return []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if str(mode).lower() == "8neighbor":
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    core_set = {int(x) for x in core_indices}
    candidates = []
    seen = set()
    for idx in core_set:
        r, c = divmod(int(idx), w)
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                cand = rr * w + cc
                if cand not in core_set and cand not in seen:
                    seen.add(cand)
                    candidates.append(cand)
    return candidates


def select_spatial_buffer_tokens(core_indices, probs, recover_score, k_buffer, mode="8neighbor"):
    """Select spatial buffer tokens from neighbors of the QFI core anchors."""
    num_tokens = int(probs.numel())
    side = int(round(math.sqrt(float(num_tokens))))
    if side * side != num_tokens or k_buffer <= 0:
        return [], {"num_buffer_tokens": 0, "buffer_mode": mode, "mean_buffer_score": 0.0}
    candidates = build_spatial_buffer_candidates(core_indices, side, mode=mode)
    if not candidates:
        return [], {"num_buffer_tokens": 0, "buffer_mode": mode, "mean_buffer_score": 0.0}
    device = probs.device
    cand_idx = torch.tensor(candidates, dtype=torch.long, device=device)
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(1e-12)
    score = torch.nan_to_num(recover_score.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0) * p.clamp_min(1e-12)
    order = torch.argsort(score[cand_idx], descending=True)
    selected = cand_idx[order[: min(int(k_buffer), int(cand_idx.numel()))]].tolist()
    mean_score = float(score[torch.tensor(selected, dtype=torch.long, device=device)].mean().item()) if selected else 0.0
    return [int(x) for x in selected], {
        "num_buffer_tokens": int(len(selected)),
        "buffer_mode": mode,
        "mean_buffer_score": mean_score,
    }


def _qfi_rank_score_from_order(qfi_core_order, num_tokens, device):
    order = _qsp_unique_order(qfi_core_order, num_tokens, device)
    score = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    if order.numel() > 0:
        rank_score = torch.linspace(1.0, 0.0, steps=int(order.numel()), device=device)
        score[order] = rank_score
    return score, order


def _qfi_recover_score(states, probs, core_indices, eps=1e-12):
    device = states.device
    num_tokens = int(states.shape[0])
    safe_eps = max(float(eps), 1e-12)
    p = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)
    psi = F.normalize(torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=safe_eps)
    evidence_kernel = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)
    if core_indices:
        core_t = torch.tensor(core_indices, dtype=torch.long, device=device)
        coverage = evidence_kernel[:, core_t].max(dim=1).values
    else:
        coverage = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    gain = (p[:, None] * (evidence_kernel - coverage[:, None]).clamp_min(0.0)).sum(dim=0)
    return torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)


def select_by_qfi_zeno_stabilized(states, probs, k, qfi_core_order, question_text=None, attn_obs=None, eps=1e-12):
    """QFI baseline + small Quantum Zeno-inspired persistence bonus."""
    device = states.device
    num_tokens = int(states.shape[0])
    beta = max(0.0, _env_float("EC_QFI_ZENO_BETA", "0.10"))
    zeno = compute_zeno_persistence_score(attn_obs if attn_obs is not None else probs, num_tokens, eps).to(device=device)
    rank_score, _ = _qfi_rank_score_from_order(qfi_core_order, num_tokens, device)
    zeno_order = torch.argsort(apply_zeno_bonus(rank_score, zeno, beta), descending=True)
    keep_idx, info = select_by_adaptive_core_recover(states, probs, k, zeno_order, question_text=question_text, eps=eps)
    keep_idx = _qsp_finalize_indices([int(x) for x in keep_idx.tolist()], min(max(0, int(k)), num_tokens), num_tokens, device)
    record = _qfi_cards_base_record("qfi_zeno_stabilized", keep_idx, info, probs, num_tokens)
    core = record["core_indices"]
    rec = record["rec_indices"]
    record.update({
        "mean_zeno_final": float(zeno[keep_idx].mean().item()) if keep_idx.numel() else 0.0,
        "mean_zeno_core": float(zeno[torch.tensor(core, dtype=torch.long, device=device)].mean().item()) if core else 0.0,
        "mean_zeno_rec": float(zeno[torch.tensor(rec, dtype=torch.long, device=device)].mean().item()) if rec else 0.0,
        "zeno_beta": float(beta),
    })
    return keep_idx, info, record


def select_by_qfi_spatial_buffer_recovery(states, probs, k, qfi_core_order, question_text=None, eps=1e-12):
    """QFI baseline + spatial buffer tokens around core anchors."""
    device = states.device
    num_tokens = int(states.shape[0])
    baseline_keep, info = select_by_adaptive_core_recover(states, probs, k, qfi_core_order, question_text=question_text, eps=eps)
    select_num = min(max(0, int(k)), num_tokens)
    core_num = int(info.get("k_core", 0))
    qfi_order = _qsp_unique_order(qfi_core_order, num_tokens, device)
    core_indices = [int(x) for x in qfi_order[:core_num].tolist()]
    ratio = max(0.0, min(_env_float("EC_QFI_BUFFER_RATIO", "0.15"), 1.0))
    max_buffer = max(0, _env_int("EC_QFI_BUFFER_MAX", "16"))
    k_buffer = min(max_buffer, int(round(float(select_num) * ratio)), max(0, select_num - len(core_indices)))
    recover_score = _qfi_recover_score(states, probs, core_indices, eps=eps)
    buffer_mode = os.environ.get("EC_QFI_BUFFER_MODE", "8neighbor").strip() or "8neighbor"
    buffer_indices, buffer_info = select_spatial_buffer_tokens(core_indices, probs, recover_score, k_buffer, mode=buffer_mode)
    selected = []
    for idx in core_indices + buffer_indices + [int(x) for x in baseline_keep.tolist()] + [int(x) for x in qfi_order.tolist()]:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= select_num:
            break
    keep_idx = _qsp_finalize_indices(selected, select_num, num_tokens, device)
    info = dict(info)
    info["k_core"] = min(core_num, int(keep_idx.numel()))
    info["k_recover"] = max(0, int(keep_idx.numel()) - info["k_core"])
    record = _qfi_cards_base_record("qfi_spatial_buffer_recovery", keep_idx, info, probs, num_tokens)
    record.update(buffer_info)
    return keep_idx, info, record


def _select_adaptive_in_candidate_pool(states, probs, k, qfi_core_order, candidates, question_text=None, eps=1e-12):
    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    cand = _qsp_unique_order(torch.tensor(candidates, dtype=torch.long, device=device), num_tokens, device)
    if cand.numel() < select_num:
        full = _qsp_unique_order(qfi_core_order, num_tokens, device)
        seen = set(int(x) for x in cand.tolist())
        extra = [int(x) for x in full.tolist() if int(x) not in seen]
        cand = torch.tensor([int(x) for x in cand.tolist()] + extra, dtype=torch.long, device=device)
    cand = cand[: max(select_num, int(cand.numel()))]
    sub_states = states.index_select(0, cand)
    sub_probs = probs.index_select(0, cand)
    orig_to_sub = {int(orig): pos for pos, orig in enumerate(cand.tolist())}
    sub_order = [orig_to_sub[int(x)] for x in _qsp_unique_order(qfi_core_order, num_tokens, device).tolist() if int(x) in orig_to_sub]
    if not sub_order:
        sub_order = list(range(int(cand.numel())))
    sub_order_t = torch.tensor(sub_order, dtype=torch.long, device=device)
    sub_keep, info = select_by_adaptive_core_recover(sub_states, sub_probs, select_num, sub_order_t, question_text=question_text, eps=eps)
    keep_idx = cand.index_select(0, sub_keep)
    keep_idx = _qsp_finalize_indices([int(x) for x in keep_idx.tolist()], select_num, num_tokens, device)
    return keep_idx, info, int(cand.numel())


def select_by_qfi_progressive(states, probs, k, qfi_core_order, question_text=None, eps=1e-12):
    """Progressive pruning fallback: first keep K_mid candidates, then refine to K_final."""
    device = states.device
    num_tokens = int(states.shape[0])
    final_k = min(max(0, _env_int("EC_QFI_PROGRESSIVE_FINAL_K", str(k))), min(max(0, int(k)), num_tokens))
    mid_k = min(max(final_k, _env_int("EC_QFI_PROGRESSIVE_MID_K", "128")), num_tokens)
    mid_keep, _ = select_by_adaptive_core_recover(states, probs, mid_k, qfi_core_order, question_text=question_text, eps=eps)
    keep_idx, info, cand_size = _select_adaptive_in_candidate_pool(
        states, probs, final_k, qfi_core_order, mid_keep.tolist(), question_text=question_text, eps=eps
    )
    record = _qfi_cards_base_record("qfi_progressive_128_to_64", keep_idx, info, probs, num_tokens)
    record.update({
        "mid_k": int(mid_k),
        "final_k": int(final_k),
        "stage_a_layer": _env_int("EC_QFI_PROGRESSIVE_STAGE_A_LAYER", "2"),
        "stage_b_layer": _env_int("EC_QFI_PROGRESSIVE_STAGE_B_LAYER", "8"),
        "progressive_mode": "candidate_fallback",
        "candidate_size": int(cand_size),
    })
    return keep_idx, info, record


def select_by_qfi_coarse_to_fine(states, probs, k, qfi_core_order, question_text=None, coarse_score=None, eps=1e-12):
    """Coarse visual filtering followed by QFI adaptive fine selection."""
    device = states.device
    num_tokens = int(states.shape[0])
    fine_k = min(max(0, _env_int("EC_QFI_FINE_K", str(k))), min(max(0, int(k)), num_tokens))
    coarse_k = min(max(fine_k, _env_int("EC_QFI_COARSE_K", "192")), num_tokens)
    source = "p_cls" if torch.is_tensor(coarse_score) and coarse_score.numel() == num_tokens else "p"
    score = coarse_score.to(device=device).float() if source == "p_cls" else probs.float()
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    candidates = torch.topk(score, k=coarse_k).indices
    keep_idx, info, cand_size = _select_adaptive_in_candidate_pool(
        states, probs, fine_k, qfi_core_order, candidates.tolist(), question_text=question_text, eps=eps
    )
    record = _qfi_cards_base_record("qfi_coarse_to_fine_pruning", keep_idx, info, probs, num_tokens)
    record.update({
        "coarse_k": int(coarse_k),
        "fine_k": int(fine_k),
        "coarse_source": source,
        "candidate_size": int(cand_size),
    })
    return keep_idx, info, record


def _qsp_append_basis_vector(basis_vectors, psi_j, eps):
    if basis_vectors:
        basis = torch.stack(basis_vectors, dim=0)
        residual = psi_j - torch.matmul(torch.matmul(psi_j, basis.t()), basis)
    else:
        residual = psi_j
    residual_norm = residual.norm().clamp_min(eps)
    if float(residual_norm.item()) <= eps:
        return False, None, float(residual_norm.item())
    basis_vectors.append(torch.nan_to_num((residual / residual_norm).float(), nan=0.0, posinf=0.0, neginf=0.0))
    return True, basis_vectors[-1], float(residual_norm.item())


def _qsp_build_basis(psi, core_indices, eps):
    basis_vectors = []
    residual_norms = []
    for idx in core_indices:
        added, _, residual_norm = _qsp_append_basis_vector(basis_vectors, psi[int(idx)], eps)
        if added:
            residual_norms.append(float(residual_norm))
    return basis_vectors, residual_norms


def _qsp_entropy_from_prob(prob, eps):
    if prob.numel() <= 1:
        return 0.0
    safe_prob = torch.nan_to_num(prob.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
    safe_prob = safe_prob / safe_prob.sum().clamp_min(eps)
    entropy = -(safe_prob * safe_prob.log()).sum() / math.log(float(safe_prob.numel()))
    return float(entropy.clamp(0.0, 1.0).item())


def _qsp_complement_recovery_scores(
    psi,
    p,
    core_indices,
    eps,
    use_spatial_local_prior=False,
    lambda_b=0.1,
    prior_scores=None,
    complement_bonus_eta=0.15,
    mode="complement_projected",
):
    basis_vectors, residual_norms = _qsp_build_basis(psi, core_indices, eps)
    if not basis_vectors:
        zeros = torch.zeros_like(p)
        return zeros, {
            "trace_residual": 0.0,
            "entropy_g_perp": 0.0,
            "basis_residual_norms": residual_norms,
            "mean_residual_norm_core": 0.0,
        }

    basis = torch.stack(basis_vectors, dim=0)
    residual = psi - torch.matmul(torch.matmul(psi, basis.t()), basis)
    sim_res = torch.matmul(residual, residual.t())
    g_perp_raw = ((sim_res.pow(2)) * p[None, :]).sum(dim=1)
    trace_res = (p * residual.pow(2).sum(dim=1)).sum()
    g_perp = g_perp_raw / trace_res.clamp_min(eps)
    g_perp = torch.nan_to_num(g_perp.float(), nan=0.0, posinf=0.0, neginf=0.0)
    entropy_g_perp = _qsp_entropy_from_prob(g_perp.clamp_min(0.0), eps)

    if mode == "prior_recovery":
        rec_score = p.clone() if prior_scores is None else prior_scores.clone()
    elif mode == "complement_bonus":
        g_prior = p.clone() if prior_scores is None else prior_scores.clone()
        rec_score = g_prior * (1.0 + max(0.0, float(complement_bonus_eta)) * normalize(g_perp, eps=eps))
    else:
        rec_score = g_perp

    if use_spatial_local_prior:
        b_j = normalize(p, eps=eps).to(device=psi.device)
        rec_score = rec_score * (1.0 + max(0.0, float(lambda_b)) * b_j)
    rec_score = torch.nan_to_num(rec_score.float(), nan=0.0, posinf=0.0, neginf=0.0)

    diag = {
        "trace_residual": float(trace_res.item()),
        "entropy_g_perp": float(entropy_g_perp),
        "basis_residual_norms": residual_norms,
        "mean_residual_norm_core": float(sum(residual_norms) / len(residual_norms)) if residual_norms else 0.0,
        "g_perp": g_perp,
        "residual": residual,
    }
    return rec_score, diag


def select_by_qfi_core_complement_recovery(
    states,
    probs,
    k,
    qfi_core_order,
    question_text=None,
    eps=1e-12,
):
    """Reuse adaptive QFI core sizing, but swap recovery for complement projection."""
    baseline_keep, baseline_info = select_by_adaptive_core_recover(
        states=states,
        probs=probs,
        k=k,
        qfi_core_order=qfi_core_order,
        question_text=question_text,
        eps=eps,
    )

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    safe_eps = max(float(eps), 1e-12)
    if select_num == 0:
        return baseline_keep.long(), {
            "qsp_enabled": True,
            "qsp_fallback": False,
            "qsp_fallback_reason": "",
            "qsp_core_selection": "qfi_core",
            "qsp_recovery": "complement_projected",
            "qsp_use_spatial_local_prior": False,
            "qsp_lambda_b": 0.0,
            "qsp_candidate_topm": 0,
            "qsp_candidate_size": 0,
            "qsp_k_core": 0,
            "qsp_k_recover": 0,
            "qsp_trace_res": 0.0,
            "qsp_delta_max": 0.0,
            "qsp_delta_mean": 0.0,
            "qsp_selector_label": "qfi_core_complement_recovery",
            "core_indices": [],
            "recovery_indices": [],
            "selected_indices": [],
        }

    p = torch.nan_to_num(probs.to(device=device).float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)
    psi = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(safe_eps)

    qfi_order = _qsp_unique_order(qfi_core_order, num_tokens, device)
    core_num = int(min(max(0, baseline_info["k_core"]), select_num))
    core_indices = [int(x) for x in qfi_order[:core_num].tolist()]

    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if core_indices:
        available[torch.tensor(core_indices, dtype=torch.long, device=device)] = False

    use_spatial_local_prior = get_env_bool("EC_QSP_USE_SPATIAL_LOCAL_PRIOR", default=False)
    lambda_b = max(0.0, _env_float("EC_QSP_LAMBDA_B", "0.1"))
    rec_score, comp_diag = _qsp_complement_recovery_scores(
        psi,
        p,
        core_indices,
        safe_eps,
        use_spatial_local_prior=use_spatial_local_prior,
        lambda_b=lambda_b,
        mode="complement_projected",
    )
    rec_score = rec_score.masked_fill(~available, float("-inf"))
    need = max(0, select_num - len(core_indices))
    recovery_indices = []
    if need > 0:
        order = torch.argsort(rec_score, descending=True)
        for idx in order.tolist():
            idx = int(idx)
            if len(recovery_indices) >= need:
                break
            if bool(available[idx].item()):
                recovery_indices.append(idx)
                available[idx] = False

    keep_idx = _qsp_finalize_indices(core_indices + recovery_indices, select_num, num_tokens, device)
    selected_set = {int(x) for x in keep_idx.tolist()}
    recovery_indices = [int(x) for x in keep_idx.tolist() if int(x) not in set(core_indices)]
    info = {
        "qsp_enabled": True,
        "qsp_fallback": False,
        "qsp_fallback_reason": "",
        "qsp_core_selection": "qfi_core",
        "qsp_recovery": "complement_projected",
        "qsp_use_spatial_local_prior": bool(use_spatial_local_prior),
        "qsp_lambda_b": float(lambda_b),
        "qsp_candidate_topm": 0,
        "qsp_candidate_size": int(num_tokens - len(core_indices)),
        "qsp_k_core": int(len([x for x in core_indices if x in selected_set])),
        "qsp_k_recover": int(len(recovery_indices)),
        "qsp_trace_res": float(comp_diag["trace_residual"]),
        "qsp_delta_max": 0.0,
        "qsp_delta_mean": 0.0,
        "qsp_entropy_g_perp": float(comp_diag["entropy_g_perp"]),
        "qsp_mean_residual_norm_core": float(comp_diag["mean_residual_norm_core"]),
        "qsp_task_anchor_alpha": 0.0,
        "qsp_complement_bonus_eta": 0.0,
        "qsp_selector_label": "qfi_core_complement_recovery",
        "mean_p_core": float(p[core_indices].mean().item()) if core_indices else 0.0,
        "mean_p_recovery": float(p[recovery_indices].mean().item()) if recovery_indices else 0.0,
        "mean_p_final": float(p[keep_idx].mean().item()) if keep_idx.numel() else 0.0,
        "overlap_core_top_p": 0.0,
        "overlap_final_top_p": 0.0,
        "num_low_p_high_delta_selected": 0,
        "core_indices": [int(x) for x in core_indices if int(x) in selected_set],
        "recovery_indices": recovery_indices,
        "selected_indices": [int(x) for x in keep_idx.tolist()],
    }
    return keep_idx.long(), info


def select_by_qsp_density_projective(
    states,
    probs,
    k,
    qfi_core_order=None,
    eps=1e-12,
):
    """
    QSP-CR Density Projective selector.

    Task-Conditioned Mixed-State Modeling:
    visual tokens are normalized into psi_i and the existing QFi task prior is
    reused as p_i. The density matrix rho = sum_i p_i |psi_i><psi_i| is never
    materialized; all gains are computed through equivalent projections.

    Density-State Projective Core Selection:
    core pivots maximize Tr[(P_{S union {j}} - P_S) rho] by maintaining an
    orthonormal basis for the currently selected token subspace.

    Complement-Projected State Recovery:
    recovery ranks non-core tokens by their response inside the orthogonal
    complement of the core subspace.
    """
    if not torch.is_tensor(states) or states.ndim != 2:
        shape = tuple(states.shape) if torch.is_tensor(states) else type(states)
        raise ValueError(f"states must be [N, D], got {shape}")
    if not torch.is_tensor(probs) or probs.ndim != 1:
        shape = tuple(probs.shape) if torch.is_tensor(probs) else type(probs)
        raise ValueError(f"probs must be [N], got {shape}")
    if states.shape[0] != probs.numel():
        raise ValueError(f"states/probs token mismatch: {states.shape[0]} vs {probs.numel()}")

    device = states.device
    num_tokens = int(states.shape[0])
    select_num = min(max(0, int(k)), num_tokens)
    safe_eps = max(float(eps), 1e-12)
    if select_num == 0:
        return torch.empty(0, dtype=torch.long, device=device), {
            "qsp_enabled": True,
            "qsp_fallback": False,
            "qsp_fallback_reason": "",
            "qsp_core_selection": "density_projection",
            "qsp_recovery": "complement_projected",
            "qsp_use_spatial_local_prior": False,
            "qsp_lambda_b": 0.0,
            "qsp_candidate_topm": 0,
            "qsp_candidate_size": 0,
            "qsp_k_core": 0,
            "qsp_k_recover": 0,
            "qsp_trace_res": 0.0,
            "qsp_delta_max": 0.0,
            "qsp_delta_mean": 0.0,
            "qsp_entropy_g_perp": 0.0,
            "qsp_mean_residual_norm_core": 0.0,
            "qsp_task_anchor_alpha": 0.0,
            "qsp_complement_bonus_eta": 0.0,
            "qsp_selector_label": "qsp_cr_density_projective",
            "mean_p_core": 0.0,
            "mean_p_recovery": 0.0,
            "mean_p_final": 0.0,
            "overlap_core_top_p": 0.0,
            "overlap_final_top_p": 0.0,
            "num_low_p_high_delta_selected": 0,
            "core_indices": [],
            "recovery_indices": [],
            "selected_indices": [],
        }

    # Task-Conditioned Mixed-State Modeling.
    psi = torch.nan_to_num(states.float(), nan=0.0, posinf=0.0, neginf=0.0)
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(safe_eps)
    p = torch.nan_to_num(probs.to(device=device).float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    p = p / p.sum().clamp_min(safe_eps)

    core_selection = os.environ.get("EC_QSP_CORE_SELECTION", "density_projection").strip().lower()
    if core_selection not in {"top_p", "pairwise_fidelity", "density_projection"}:
        core_selection = "density_projection"
    recovery_mode = os.environ.get("EC_QSP_RECOVERY", "complement_projected").strip().lower()
    if recovery_mode not in {"none", "prior_recovery", "complement_projected", "complement_bonus"}:
        recovery_mode = "complement_projected"
    use_spatial_local_prior = get_env_bool("EC_QSP_USE_SPATIAL_LOCAL_PRIOR", default=False)
    lambda_b = max(0.0, _env_float("EC_QSP_LAMBDA_B", "0.1"))
    candidate_topm = max(0, _env_int("EC_QSP_CANDIDATE_TOPM", "0"))
    core_ratio = min(max(_env_float("EC_QSP_CORE_RATIO", "0.75"), 0.0), 1.0)
    recover_cap = max(0, _env_int("EC_QSP_RECOVER_CAP", "16"))
    task_anchor_alpha = max(0.0, _env_float("EC_QSP_TASK_ANCHOR_ALPHA", "0.0"))
    complement_bonus_eta = max(0.0, _env_float("EC_QSP_COMPLEMENT_BONUS_ETA", "0.15"))
    selector_label = os.environ.get("EC_QSP_SELECTOR_LABEL", "qsp_cr_density_projective").strip() or "qsp_cr_density_projective"

    if recovery_mode == "none":
        recover_num = 0
    else:
        recover_num = min(max(select_num - int(round(select_num * core_ratio)), 0), select_num, recover_cap)
    core_num = select_num - recover_num

    available = torch.ones(num_tokens, dtype=torch.bool, device=device)
    candidate_mask = available.clone()
    if candidate_topm > 0:
        pool_num = min(candidate_topm, num_tokens)
        top_idx = torch.argsort(p, descending=True)[:pool_num]
        candidate_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        candidate_mask[top_idx] = True
    candidate_size = int(candidate_mask.sum().item())

    core_indices = []
    basis_vectors = []
    delta_history = []

    if core_selection == "top_p":
        order = torch.argsort(p, descending=True)
        if candidate_topm > 0:
            order = order[candidate_mask[order]]
        for idx in order[:core_num].tolist():
            idx = int(idx)
            core_indices.append(idx)
            available[idx] = False
            _qsp_append_basis_vector(basis_vectors, psi[idx], safe_eps)
    elif core_selection == "pairwise_fidelity":
        gram = compute_projection_overlap_kernel(psi, eps=safe_eps, assume_normalized=True)
        coverage = torch.zeros(num_tokens, dtype=gram.dtype, device=device)
        while len(core_indices) < core_num:
            marginal = (gram - coverage[:, None]).clamp_min(0.0)
            gain = (p[:, None] * marginal).sum(dim=0)
            gain = torch.nan_to_num(gain.float(), nan=0.0, posinf=0.0, neginf=0.0)
            gain = gain.masked_fill(~available, float("-inf"))
            gain = gain.masked_fill(~candidate_mask, float("-inf"))
            idx = int(torch.argmax(gain).item())
            if not bool(available[idx].item()) or not bool(candidate_mask[idx].item()):
                break
            core_indices.append(idx)
            available[idx] = False
            coverage = torch.maximum(coverage, gram[:, idx])
            delta_history.append(float(gain[idx].item()))
            _qsp_append_basis_vector(basis_vectors, psi[idx], safe_eps)
    else:
        # Density-State Projective Core Selection.
        while len(core_indices) < core_num:
            if basis_vectors:
                basis = torch.stack(basis_vectors, dim=0)
                residual = psi - torch.matmul(torch.matmul(psi, basis.t()), basis)
            else:
                residual = psi
            residual_norm = residual.norm(dim=-1)
            q = residual / residual_norm.unsqueeze(-1).clamp_min(safe_eps)
            sim_proj = torch.matmul(psi, q.t()).clamp_min(0.0).pow(2).clamp(0.0, 1.0)
            delta = (p[:, None] * sim_proj).sum(dim=0)
            if task_anchor_alpha > 0.0:
                delta = delta * p.clamp_min(safe_eps).pow(task_anchor_alpha)
            delta = torch.where(residual_norm > safe_eps, delta, torch.zeros_like(delta))
            delta = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            delta = delta.masked_fill(~available, float("-inf"))
            delta = delta.masked_fill(~candidate_mask, float("-inf"))
            idx = int(torch.argmax(delta).item())
            if not bool(available[idx].item()) or not bool(candidate_mask[idx].item()):
                candidate_mask = available.clone()
                delta = delta.masked_fill(~available, float("-inf"))
                idx = int(torch.argmax(delta).item())
                if not bool(available[idx].item()):
                    break
            added, _, _ = _qsp_append_basis_vector(basis_vectors, psi[idx], safe_eps)
            if not added:
                available[idx] = False
                candidate_mask[idx] = False
                continue
            core_indices.append(idx)
            available[idx] = False
            candidate_mask[idx] = False
            delta_history.append(float(delta[idx].item()))

    if len(core_indices) < core_num:
        fallback_order = qfi_core_order if torch.is_tensor(qfi_core_order) else torch.argsort(p, descending=True)
        fallback_order = _qsp_unique_order(fallback_order, num_tokens, device)
        for idx in fallback_order.tolist():
            if len(core_indices) >= core_num:
                break
            idx = int(idx)
            if bool(available[idx].item()):
                core_indices.append(idx)
                available[idx] = False
                _qsp_append_basis_vector(basis_vectors, psi[idx], safe_eps)

    recovery_indices = []
    trace_res = torch.zeros((), dtype=torch.float32, device=device)
    comp_diag = {
        "trace_residual": 0.0,
        "entropy_g_perp": 0.0,
        "mean_residual_norm_core": 0.0,
    }
    if recovery_mode != "none" and len(core_indices) < select_num:
        rec_score, comp_diag = _qsp_complement_recovery_scores(
            psi,
            p,
            core_indices,
            safe_eps,
            use_spatial_local_prior=use_spatial_local_prior,
            lambda_b=lambda_b,
            prior_scores=p,
            complement_bonus_eta=complement_bonus_eta,
            mode=recovery_mode,
        )
        trace_res = torch.tensor(float(comp_diag["trace_residual"]), dtype=torch.float32, device=device)
        rec_score = torch.nan_to_num(rec_score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        rec_score = rec_score.masked_fill(~available, float("-inf"))
        need = select_num - len(core_indices)
        if need > 0:
            order = torch.argsort(rec_score, descending=True)
            for idx in order.tolist():
                if len(recovery_indices) >= need:
                    break
                idx = int(idx)
                if bool(available[idx].item()):
                    recovery_indices.append(idx)
                    available[idx] = False

    selected = core_indices + recovery_indices
    keep_idx = _qsp_finalize_indices(selected, select_num, num_tokens, device)
    selected_set = {int(x) for x in keep_idx.tolist()}
    core_set = {int(x) for x in core_indices}
    recovery_indices = [idx for idx in recovery_indices if int(idx) in selected_set]
    if len(recovery_indices) < max(0, select_num - len(core_set & selected_set)):
        recovery_indices = [int(x) for x in keep_idx.tolist() if int(x) not in core_set]

    top_p_idx = set(torch.argsort(p, descending=True)[:select_num].tolist())
    core_idx_final = [int(x) for x in core_indices if int(x) in selected_set]
    final_idx = [int(x) for x in keep_idx.tolist()]
    p_median = float(p.median().item()) if p.numel() else 0.0
    delta_threshold = float(torch.tensor(delta_history, device=device).median().item()) if delta_history else 0.0
    num_low_p_high_delta_selected = 0
    for idx, delta_value in zip(core_indices, delta_history):
        if float(p[int(idx)].item()) < p_median and float(delta_value) > delta_threshold:
            num_low_p_high_delta_selected += 1

    info = {
        "qsp_enabled": True,
        "qsp_fallback": False,
        "qsp_fallback_reason": "",
        "qsp_core_selection": core_selection,
        "qsp_recovery": recovery_mode,
        "qsp_use_spatial_local_prior": bool(use_spatial_local_prior),
        "qsp_lambda_b": float(lambda_b),
        "qsp_candidate_topm": int(candidate_topm),
        "qsp_candidate_size": int(candidate_size),
        "qsp_k_core": int(len([idx for idx in core_indices if idx in selected_set])),
        "qsp_k_recover": int(len(recovery_indices)),
        "qsp_trace_res": float(trace_res.item()) if torch.is_tensor(trace_res) else 0.0,
        "qsp_delta_max": float(max(delta_history)) if delta_history else 0.0,
        "qsp_delta_mean": float(sum(delta_history) / len(delta_history)) if delta_history else 0.0,
        "qsp_entropy_g_perp": float(comp_diag["entropy_g_perp"]),
        "qsp_mean_residual_norm_core": float(comp_diag["mean_residual_norm_core"]),
        "qsp_task_anchor_alpha": float(task_anchor_alpha),
        "qsp_complement_bonus_eta": float(complement_bonus_eta),
        "qsp_selector_label": selector_label,
        "mean_p_core": float(p[core_idx_final].mean().item()) if core_idx_final else 0.0,
        "mean_p_recovery": float(p[recovery_indices].mean().item()) if recovery_indices else 0.0,
        "mean_p_final": float(p[keep_idx].mean().item()) if keep_idx.numel() else 0.0,
        "overlap_core_top_p": float(len(set(core_idx_final) & top_p_idx) / max(1, len(core_idx_final))),
        "overlap_final_top_p": float(len(set(final_idx) & top_p_idx) / max(1, len(final_idx))),
        "num_low_p_high_delta_selected": int(num_low_p_high_delta_selected),
        "core_indices": core_idx_final,
        "recovery_indices": [int(x) for x in recovery_indices],
        "selected_indices": final_idx,
    }
    return keep_idx.long(), info


class ECPruner:
    def __init__(self, debug=None):
        if debug is None:
            self.debug = get_env_bool("EC_DEBUG", default=False)
        else:
            self.debug = bool(debug)
        self.score_source = os.environ.get("EC_SCORE_SOURCE", "norm").strip().lower()
        self.last_score_source = self.score_source
        self.last_qmo_info = {}
        self.last_qfid_info = {}
        self.solver = os.environ.get("EC_SOLVER", "greedy").strip().lower()
        self.candidate_ratio = float(os.environ.get("EC_CANDIDATE_RATIO", "2.0"))
        self.global_candidate_ratio = float(os.environ.get("EC_GLOBAL_CANDIDATE_RATIO", "1.0"))
        self.beta_a = float(os.environ.get("EC_BETA_A", "0.5"))
        self.beta_semantic = float(os.environ.get("EC_BETA_SEM", "0.3"))
        self.beta_spatial = float(os.environ.get("EC_BETA_SPATIAL", "0.2"))
        self.qmo_w_rel = float(os.environ.get("EC_W_REL", "0.40"))
        self.qmo_w_sem = float(os.environ.get("EC_W_SEM", "0.25"))
        self.qmo_w_spatial = float(os.environ.get("EC_W_SPATIAL", "0.15"))
        self.qmo_w_chain = float(os.environ.get("EC_W_CHAIN", "0.15"))
        self.qmo_w_ent = float(os.environ.get("EC_W_ENT", "0.05"))
        self.qmo_init_source = os.environ.get("EC_QMO_INIT_SOURCE", "semantic").strip().lower()
        self.qmo_debug = get_env_bool("EC_QMO_DEBUG", default=False)
        self.qfid_tau = float(os.environ.get("EC_QFID_TAU", "0.50"))
        self.qfid_eps = float(os.environ.get("EC_QFID_EPS", "1e-6"))
        self.qfid_prob_source = os.environ.get("EC_QFID_PROB_SOURCE", "semantic").strip().lower()
        self.qfid_select_mode = os.environ.get("EC_QFID_SELECT_MODE", "qf").strip().lower()
        if self.qfid_select_mode not in {"qf", "cls_topk", "semantic_topk", "visual_kcenter"}:
            raise ValueError(f"Unknown EC_QFID_SELECT_MODE: {self.qfid_select_mode}")
        self.qfid_selector = os.environ.get("EC_QFID_SELECTOR", "qfi_residual").strip().lower()
        if self.qfid_selector not in {
            "qfi_residual",
            "evidence_recover",
            "core_then_recover",
            "adaptive_core_recover",
            "qmo_cr",
            "qmo_gated",
            "qsp_cr_density_projective",
            "density_projective_core_recover",
            "qfi_core_complement_recovery",
            "qsp_density_projective_task_anchor",
            "qsp_density_projective_task_anchor_a05",
            "qsp_density_projective_complement_bonus",
            "qfi_zeno_stabilized",
            "qfi_spatial_buffer_recovery",
            "qfi_progressive_128_to_64",
            "qfi_coarse_to_fine_pruning",
        }:
            raise ValueError(f"Unknown EC_QFID_SELECTOR: {self.qfid_selector}")
        self.qfid_core_ratio = min(max(float(os.environ.get("EC_QFID_CORE_RATIO", "0.5")), 0.0), 1.0)
        self.qfid_cls_mix_beta = float(os.environ.get("EC_QFID_CLS_MIX_BETA", "0.05"))
        self.qfid_cls_mix_mode = os.environ.get("EC_QFID_CLS_MIX_MODE", "linear").strip().lower()
        if self.qfid_cls_mix_mode not in {"linear", "geometric"}:
            raise ValueError(f"Unknown EC_QFID_CLS_MIX_MODE: {self.qfid_cls_mix_mode}")
        self.qfid_cls_gate = get_env_bool("EC_QFID_CLS_GATE", default=False)
        self.qfid_cls_gate_mode = os.environ.get("EC_QFID_CLS_GATE_MODE", "agreement").strip().lower()
        if self.qfid_cls_gate and self.qfid_cls_gate_mode != "agreement":
            raise ValueError(f"Unknown EC_QFID_CLS_GATE_MODE: {self.qfid_cls_gate_mode}")
        if self.qfid_cls_gate and self.qfid_prob_source in {"clsmix", "vtpmix"} and self.qfid_cls_mix_mode != "linear":
            raise ValueError("EC_QFID_CLS_GATE requires EC_QFID_CLS_MIX_MODE=linear")
        self.qfid_cls_gate_beta_base = float(os.environ.get("EC_QFID_CLS_GATE_BETA_BASE", "0.105"))
        self.qfid_cls_gate_min = float(os.environ.get("EC_QFID_CLS_GATE_MIN", "0.5"))
        self.qfid_cls_gate_max = float(os.environ.get("EC_QFID_CLS_GATE_MAX", "1.5"))
        if self.qfid_cls_gate and self.qfid_cls_gate_min > self.qfid_cls_gate_max:
            raise ValueError("EC_QFID_CLS_GATE_MIN must be <= EC_QFID_CLS_GATE_MAX")
        self.qfid_cls_attn_layer = os.environ.get("EC_QFID_CLS_ATTN_LAYER", "last").strip().lower()
        if self.qfid_cls_attn_layer not in {"last", "-2", "-4", "last4mean"}:
            raise ValueError(f"Unknown EC_QFID_CLS_ATTN_LAYER: {self.qfid_cls_attn_layer}")
        self.qficr_cls_prior_layers = os.environ.get("EC_QFICR_CLS_PRIOR_LAYERS", "").strip().lower()
        if not self.qficr_cls_prior_layers:
            if self.qfid_cls_attn_layer == "last":
                self.qficr_cls_prior_layers = "last1"
            elif self.qfid_cls_attn_layer == "last4mean":
                self.qficr_cls_prior_layers = "last4"
            else:
                self.qficr_cls_prior_layers = "last2"
        if self.qficr_cls_prior_layers not in {"last1", "last2", "last4"}:
            raise ValueError(f"Unknown EC_QFICR_CLS_PRIOR_LAYERS: {self.qficr_cls_prior_layers}")
        self.qfid_cls_head_reduce = os.environ.get("EC_QFID_CLS_HEAD_REDUCE", "mean").strip().lower()
        if self.qfid_cls_head_reduce not in {"mean", "entropy_weighted"}:
            raise ValueError(f"Unknown EC_QFID_CLS_HEAD_REDUCE: {self.qfid_cls_head_reduce}")
        self.qfid_kernel = os.environ.get("EC_QFID_KERNEL", "amplitude").strip().lower()
        self.qfid_overlap_kernel = get_qfid_overlap_kernel(default="relu_square")
        self.qfid_depolarize = float(os.environ.get("EC_QFID_DEPOLARIZE", "0.0"))
        self.qfid_depolarize_mode = os.environ.get("EC_QFID_DEPOLARIZE_MODE", "fixed").strip().lower()
        self.qfid_depolarize_min = float(os.environ.get("EC_QFID_DEPOLARIZE_MIN", "0.05"))
        self.qfid_depolarize_max = float(os.environ.get("EC_QFID_DEPOLARIZE_MAX", "0.25"))
        self.qfid_spatial_state = get_env_bool("EC_QFID_SPATIAL_STATE", default=False)
        self.qfid_spatial_lambda = float(os.environ.get("EC_QFID_SPATIAL_LAMBDA", "0.10"))
        self.qfid_spatial_sigma = float(os.environ.get("EC_QFID_SPATIAL_SIGMA", "0.20"))
        self.qfid_measure_prior_mode = os.environ.get("EC_QFID_MEASURE_PRIOR_MODE", "none").strip().lower()
        self.qfid_measure_prior_lambda = float(os.environ.get("EC_QFID_MEASURE_PRIOR_LAMBDA", "0.10"))
        self.qfid_anchor_mode = os.environ.get("EC_QFID_ANCHOR_MODE", "none").strip().lower()
        self.qfid_measure_anchor_ratio = float(os.environ.get("EC_QFID_MEASURE_ANCHOR_RATIO", "0.125"))
        self.qfid_measure_anchor_min = int(os.environ.get("EC_QFID_MEASURE_ANCHOR_MIN", "0"))
        self.qfid_budget_calib = get_env_bool("EC_QFID_BUDGET_CALIB", default=False)
        self.qfid_budget_alpha = float(os.environ.get("EC_QFID_BUDGET_ALPHA", "3.0"))
        self.qfid_budget_gamma_min = float(os.environ.get("EC_QFID_BUDGET_GAMMA_MIN", "0.05"))
        self.qfid_budget_gamma_max = float(os.environ.get("EC_QFID_BUDGET_GAMMA_MAX", "20.0"))
        self.qfid_budget_iters = int(os.environ.get("EC_QFID_BUDGET_ITERS", "30"))
        self.qfid_budget_eps = float(os.environ.get("EC_QFID_BUDGET_EPS", "1e-12"))
        self.qfid_spectral_filter = get_env_bool("EC_QFID_SPECTRAL_FILTER", default=False)
        self.qfid_spectral_gamma = float(os.environ.get("EC_QFID_SPECTRAL_GAMMA", "1.0"))
        self.qfid_spectral_eps = float(os.environ.get("EC_QFID_SPECTRAL_EPS", "1e-12"))
        self.qfid_spectral_trace_norm = get_env_bool("EC_QFID_SPECTRAL_TRACE_NORM", default=True)
        self.qfid_debug = get_env_bool("EC_QFID_DEBUG", default=False)
        self.qfid_debug_stats_jsonl = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
        self.per_unit_topl = int(os.environ.get("EC_PER_UNIT_TOPL", "8"))
        self.use_semantic_candidate = get_env_bool("EC_USE_SEMANTIC_CANDIDATE", default=True)
        self.use_spatial_candidate = get_env_bool("EC_USE_SPATIAL_CANDIDATE", default=False)
        self.spatial_exclude_global = get_env_bool("EC_SPATIAL_EXCLUDE_GLOBAL", default=True)
        self.spatial_grid_size = int(os.environ.get("EC_SPATIAL_GRID_SIZE", "4"))
        self.spatial_topl = int(os.environ.get("EC_SPATIAL_TOPL", "1"))
        self.use_chain_coverage = get_env_bool("EC_USE_CHAIN_COVERAGE", default=True)
        self.gamma_chain = float(os.environ.get("EC_GAMMA_CHAIN", "0.1"))
        self.use_repulsion = get_env_bool("EC_USE_REPULSION", default=True)
        self.use_complement = get_env_bool("EC_USE_COMPLEMENT", default=True)
        self.use_phi = get_env_bool("EC_USE_PHI", default=True)
        self.topk_r = int(os.environ.get("EC_TOPK_R", "8"))
        self.topk_c = int(os.environ.get("EC_TOPK_C", "8"))
        self.sigma = float(os.environ.get("EC_SIGMA", "2.0"))
        self.lambda_repulsion = float(os.environ.get("EC_LAMBDA", "0.1"))
        self.delta_complement = float(os.environ.get("EC_DELTA", "0.1"))
        self.qa_steps = int(os.environ.get("EC_QA_STEPS", "50"))
        self.qa_t0 = float(os.environ.get("EC_QA_T0", "1.0"))
        self.qa_tend = float(os.environ.get("EC_QA_TEND", "0.01"))
        self.qa_seed = int(os.environ.get("EC_QA_SEED", "42"))
        self.qa_max_swap_ratio = float(os.environ.get("EC_QA_MAX_SWAP_RATIO", "0.15"))
        self.qa_min_improve = float(os.environ.get("EC_QA_MIN_IMPROVE", "1e-6"))

    @property
    def needs_semantics(self):
        return (
            self.score_source == "semantic"
            or self.score_source == "qmo"
            or (
                self.score_source == "qfid"
                and self.qfid_prob_source in {"semantic", "measurement", "clsmix", "vtpmix"}
            )
            or self.use_semantic_candidate
            or self.use_spatial_candidate
            or self.use_chain_coverage
            or self.use_repulsion
            or self.use_complement
        )

    @property
    def needs_relevance(self):
        return (
            self.score_source == "relevance"
            or self.score_source == "qmo"
            or (self.score_source == "qfid" and self.qfid_prob_source == "relevance")
        )

    def _debug(self, message):
        global _EC_DEBUG_COUNT
        if not self.debug:
            return
        limit = int(os.environ.get("EC_DEBUG_LIMIT", "5"))
        if _EC_DEBUG_COUNT >= limit:
            return
        print(f"[EC-Pruner] {message}", flush=True)
        _EC_DEBUG_COUNT += 1

    def _debug_qec(self, message):
        global _EC_DEBUG_COUNT
        if not self.debug:
            return
        limit = int(os.environ.get("EC_DEBUG_LIMIT", "5"))
        if _EC_DEBUG_COUNT >= limit:
            return
        print(f"[QEC-Pruner] {message}", flush=True)
        _EC_DEBUG_COUNT += 1

    def _qmo_debug_enabled(self):
        return self.debug or self.qmo_debug

    def _debug_qmo_pruner(self, message):
        global _EC_DEBUG_COUNT
        if not self._qmo_debug_enabled():
            return
        limit = int(os.environ.get("EC_DEBUG_LIMIT", "5"))
        if _EC_DEBUG_COUNT >= limit:
            return
        print(f"[QMO-Pruner] {message}", flush=True)
        _EC_DEBUG_COUNT += 1

    def _qfid_debug_enabled(self):
        return self.debug or self.qfid_debug

    def _debug_qf_pruner(self, message):
        global _EC_DEBUG_COUNT
        if not self._qfid_debug_enabled():
            return
        limit = int(os.environ.get("EC_DEBUG_LIMIT", "5"))
        if _EC_DEBUG_COUNT >= limit:
            return
        print(f"[QF-Pruner] {message}", flush=True)
        _EC_DEBUG_COUNT += 1

    def _write_qfid_debug_stats(self, info):
        """Append optional QF diagnostics without changing the selection path."""
        global _QFID_STATS_COUNT
        if not self.qfid_debug_stats_jsonl:
            return

        stats_path = self.qfid_debug_stats_jsonl
        if stats_path.lower() in {"1", "true", "yes", "on"}:
            stats_path = os.path.abspath("debug_qficr_stats.jsonl")
        output_dir = os.path.dirname(os.path.abspath(stats_path))
        os.makedirs(output_dir, exist_ok=True)

        def _float_or_none(value):
            if value is None:
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        def _int_or_none(value):
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        question_type = info.get("question_type", "")
        record = {
            "sample_index": _QFID_STATS_COUNT,
            "process_id": os.getpid(),
            "benchmark": os.environ.get("EC_QFID_DEBUG_BENCHMARK", ""),
            "question_id": os.environ.get("EC_QFID_DEBUG_QUESTION_ID", ""),
            "method": info.get("qficr_method", ""),
            "local_residual_mode": info.get("qficr_local_residual_mode", ""),
            "dual_residual_mode": info.get("qficr_dual_residual_mode", ""),
            "alloc_mode": info.get("qficr_alloc_mode", "frozen"),
            "core_mode": info.get("qficr_core_mode", ""),
            "qficr_role_core_gap": bool(info.get("qficr_role_core_gap", False)),
            "qficr_m2_dual_quality": bool(info.get("qficr_m2_dual_quality", False)),
            "qficr_role_core_gap_info": info.get("qficr_role_core_gap_info", {}),
            "complement_quality_source": info.get("qficr_complement_quality_source", ""),
            "native_cd_quality_min": _float_or_none(info.get("qficr_native_cd_quality_min")),
            "native_cd_quality_max": _float_or_none(info.get("qficr_native_cd_quality_max")),
            "native_cd_quality_mean": _float_or_none(info.get("qficr_native_cd_quality_mean")),
            "native_cd_quality_median": _float_or_none(info.get("qficr_native_cd_quality_median")),
            "K_total": _int_or_none(info.get("final_keep_size", info.get("keep_size"))),
            "r_gap": _int_or_none(info.get("qficr_role_core_r_gap")),
            "K_sem": _int_or_none(info.get("qficr_role_core_K_sem")),
            "K_comp": _int_or_none(info.get("qficr_role_core_K_comp")),
            "num_visual_tokens_before": _int_or_none(info.get("qficr_num_visual_tokens_before", info.get("num_tokens"))),
            "num_visual_tokens_after": _int_or_none(info.get("qficr_num_visual_tokens_after")),
            "semantic_indices": info.get("qficr_role_core_semantic_indices", []),
            "complement_indices": info.get("qficr_role_core_complement_indices", []),
            "rolecore_selected_indices": info.get("qficr_rolecore_selected_indices", []),
            "actual_forward_selected_indices": info.get("qficr_actual_forward_selected_indices", []),
            "actual_forward_match": bool(info.get("qficr_role_core_exact_match_forward", False)),
            "role_core_jaccard_to_v0": _float_or_none(info.get("qficr_role_core_jaccard_to_v0")),
            "role_core_num_diff_tokens": _int_or_none(info.get("qficr_role_core_num_diff_tokens")),
            "recovery_score_mode": info.get("qficr_recovery_score_mode", "frozen"),
            "question_type": question_type,
            "is_yes_no": bool(question_type == "binary"),
            "K": _int_or_none(info.get("final_keep_size", info.get("keep_size"))),
            "N": _int_or_none(info.get("num_tokens")),
            "token_norm_values": info.get("token_norm_values", []),
            "budget_mode": info.get("qficr_budget_mode"),
            "rho_min": _float_or_none(info.get("qficr_rho_min")),
            "rho_max": _float_or_none(info.get("qficr_rho_max")),
            "entropy": _float_or_none(info.get("qficr_entropy", info.get("entropy"))),
            "entropy_norm": _float_or_none(info.get("qficr_entropy_norm", info.get("entropy_norm"))),
            "entropy_hat": _float_or_none(info.get("qficr_entropy_hat")),
            "entropy_gamma": _float_or_none(info.get("qficr_entropy_gamma")),
            "rho": _float_or_none(info.get("qficr_rho", info.get("adapt_recover_ratio"))),
            "raw_k_rest": _int_or_none(info.get("qficr_raw_k_rest")),
            "final_k_rest": _int_or_none(info.get("qficr_final_k_rest", info.get("adapt_k_recover"))),
            "final_k_red": _int_or_none(info.get("qficr_final_k_red", info.get("adapt_k_core"))),
            "abs_cap": _int_or_none(info.get("qficr_abs_cap")),
            "ratio_cap": _float_or_none(info.get("qficr_ratio_cap")),
            "cap_applied": bool(info.get("qficr_cap_applied", False)),
            "yes_no_gate_applied": bool(info.get("qficr_yes_no_gate_applied", False)),
            "residual_alloc_mode": info.get("qficr_residual_alloc_mode", "legacy"),
            "compete_margin": _float_or_none(info.get("qficr_compete_margin", 0.0)),
            "max_flexible_slots": _int_or_none(info.get("qficr_max_flexible_slots", 0)),
            "K_base": _int_or_none(info.get("qficr_K_base", info.get("qficr_fixed_core_count", 0))),
            "K_rest_max": _int_or_none(info.get("qficr_K_rest_max", info.get("qficr_max_flexible_slots", 0))),
            "fixed_core_count": _int_or_none(info.get("qficr_fixed_core_count", 0)),
            "actual_residual_count": _int_or_none(info.get("qficr_actual_residual_count", 0)),
            "actual_core_count": _int_or_none(info.get("qficr_actual_core_count", 0)),
            "candidate_pool_enabled": bool(info.get("qficr_candidate_pool_enabled", False)),
            "candidate_pool_size": _int_or_none(info.get("qficr_candidate_pool_size", 0)),
            "candidate_pool_multiplier": _float_or_none(info.get("qficr_candidate_pool_multiplier", 0.0)),
            "external_candidate_pool_enabled": bool(
                info.get("qficr_external_candidate_pool_enabled", False)
            ),
            "external_candidate_pool_size": _int_or_none(
                info.get("qficr_external_candidate_pool_size", 0)
            ),
            "external_candidate_pool_multiplier": _float_or_none(
                info.get("qficr_external_candidate_pool_multiplier", 0.0)
            ),
            "balanced_pool_overrides_external_pool": bool(
                info.get("qficr_balanced_pool_overrides_external_pool", False)
            ),
            "balanced_pool_size_requested": _int_or_none(
                info.get("qficr_balanced_pool_size_requested", 0)
            ),
            "initial_core_pool_size": _int_or_none(info.get("qficr_initial_core_pool_size", 0)),
            "initial_residual_pool_size": _int_or_none(
                info.get("qficr_initial_residual_pool_size", 0)
            ),
            "remaining_core_pool_size_by_step": info.get(
                "qficr_remaining_core_pool_size_by_step", []
            ),
            "remaining_residual_pool_size_by_step": info.get(
                "qficr_remaining_residual_pool_size_by_step", []
            ),
            "balanced_core_pool_indices": info.get("qficr_balanced_core_pool_indices", []),
            "balanced_residual_pool_indices": info.get(
                "qficr_balanced_residual_pool_indices", []
            ),
            "full_core_indices": info.get("qficr_full_core_indices", []),
            "fixed_core_indices": info.get("qficr_fixed_core_indices", []),
            "base_core_indices": info.get("qficr_base_core_indices", info.get("qficr_fixed_core_indices", [])),
            "reserved_core_indices": info.get("qficr_reserved_core_indices", []),
            "selected_core_indices": info.get("qficr_selected_core_indices", []),
            "selected_residual_indices": info.get("qficr_selected_residual_indices", []),
            "fallback_core_indices": info.get("qficr_fallback_core_indices", []),
            "final_selected_indices": info.get("qficr_final_selected_indices", []),
            "p3_final_indices": info.get("qficr_p3_final_indices", []),
            "p3_semantic_channel_pivot_count": _int_or_none(
                info.get("qficr_p3_semantic_channel_pivot_count")
            ),
            "p3_local_channel_pivot_count": _int_or_none(
                info.get("qficr_p3_local_channel_pivot_count")
            ),
            "p3_channel_sequence": info.get("qficr_p3_channel_sequence", []),
            "p3_trace": info.get("qficr_p3_trace", []),
            "p3_semantic_only_reproduction": bool(
                info.get("qficr_p3_semantic_only_reproduction", False)
            ),
            "p3_semantic_only_first_mismatch_step": _int_or_none(
                info.get("qficr_p3_semantic_only_first_mismatch_step")
            ),
            "p3_semantic_only_order": info.get("qficr_p3_semantic_only_order", []),
            "p3_semantic_only_reference_order": info.get(
                "qficr_p3_semantic_only_reference_order", []
            ),
            "p3_local_degenerate": bool(info.get("qficr_p3_local_degenerate", False)),
            "p3_p_finite": bool(info.get("qficr_p3_p_finite", False)),
            "p3_l_finite": bool(info.get("qficr_p3_l_finite", False)),
            "p3_q_finite": bool(info.get("qficr_p3_q_finite", False)),
            "p3_semantic_residual_finite": bool(
                info.get("qficr_p3_semantic_residual_finite", False)
            ),
            "p3_local_residual_finite": bool(
                info.get("qficr_p3_local_residual_finite", False)
            ),
            "p3_q_sum": _float_or_none(info.get("qficr_p3_q_sum")),
            "p3_l_min": _float_or_none(info.get("qficr_p3_l_min")),
            "p3_l_max": _float_or_none(info.get("qficr_p3_l_max")),
            "p3_l_mean": _float_or_none(info.get("qficr_p3_l_mean")),
            "p3_actual_forward_match": bool(info.get("qficr_p3_actual_forward_match", False)),
            "p3_canonical_ordering_preserved": bool(
                info.get("qficr_p3_canonical_ordering_preserved", False)
            ),
            "p1tc_final_indices": info.get("qficr_p1tc_final_indices", []),
            "p1tc_semantic_indices": info.get("qficr_p1tc_semantic_indices", []),
            "p1tc_local_indices": info.get("qficr_p1tc_local_indices", []),
            "p1tc_semantic_first26_reference": info.get("qficr_p1tc_semantic_first26_reference", []),
            "p1tc_semantic_first26_reproduction": bool(
                info.get("qficr_p1tc_semantic_first26_reproduction", False)
            ),
            "p1tc_conditioning_order": info.get("qficr_p1tc_conditioning_order", []),
            "p1lr_conditioning_mode": info.get("qficr_p1lr_conditioning_mode", "all"),
            "p1lr_conditioning_top_m": _int_or_none(info.get("qficr_p1lr_conditioning_top_m")),
            "p1lr_conditioning_threshold": _float_or_none(info.get("qficr_p1lr_conditioning_threshold")),
            "p1lr_weak_semantic_alpha": _float_or_none(info.get("qficr_p1lr_weak_semantic_alpha")),
            "p1lr_conditioning_diag_values": info.get("qficr_p1lr_conditioning_diag_values", []),
            "p1lr_conditioning_residual_before": info.get("qficr_p1lr_conditioning_residual_before", []),
            "p1tc_local_trace": info.get("qficr_p1tc_local_trace", []),
            "p1tc_k_semantic": _int_or_none(info.get("qficr_p1tc_k_semantic")),
            "p1tc_k_local": _int_or_none(info.get("qficr_p1tc_k_local")),
            "p1tc_local_degenerate": bool(info.get("qficr_p1tc_local_degenerate", False)),
            "p1tc_local_score_mean": _float_or_none(info.get("qficr_p1tc_local_score_mean")),
            "p1tc_local_p_mean": _float_or_none(info.get("qficr_p1tc_local_p_mean")),
            "p1tc_local_q_mean": _float_or_none(info.get("qficr_p1tc_local_q_mean")),
            "p1tc_local_residual_mean": _float_or_none(info.get("qficr_p1tc_local_residual_mean")),
            "p1tc_p_finite": bool(info.get("qficr_p1tc_p_finite", False)),
            "p1tc_l_finite": bool(info.get("qficr_p1tc_l_finite", False)),
            "p1tc_q_finite": bool(info.get("qficr_p1tc_q_finite", False)),
            "p1tc_local_residual_finite": bool(
                info.get("qficr_p1tc_local_residual_finite", False)
            ),
            "p1tc_actual_forward_match": bool(info.get("qficr_p1tc_actual_forward_match", False)),
            "p1tc_canonical_ordering_preserved": bool(
                info.get("qficr_p1tc_canonical_ordering_preserved", False)
            ),
            "p1tc_prob_values": info.get("qficr_p1tc_prob_values", []),
            "p1tc_lcenter_values": info.get("qficr_p1tc_lcenter_values", []),
            "p1tc_q_values": info.get("qficr_p1tc_q_values", []),
            "p1lr_q_is_lcenter": bool(info.get("qficr_p1lr_q_is_lcenter", False)),
            "p1lr_p_used_in_selector": bool(info.get("qficr_p1lr_p_used_in_selector", False)),
            "p1lr_initial_local_residual_mass": _float_or_none(info.get("qficr_p1lr_initial_local_residual_mass")),
            "p1lr_after_core_local_residual_mass": _float_or_none(info.get("qficr_p1lr_after_core_local_residual_mass")),
            "p1lr_core_conditioning_reduction": _float_or_none(info.get("qficr_p1lr_core_conditioning_reduction")),
            "p1lr_initial_local_kernel_mass": _float_or_none(info.get("qficr_p1lr_initial_local_kernel_mass")),
            "p1lr_core_conditioning_reduction_normalized": _float_or_none(info.get("qficr_p1lr_core_conditioning_reduction_normalized")),
            "p1lr_semantic_residual_values": info.get("qficr_p1lr_semantic_residual_values", []),
            "p1lr_actual_forward_match": bool(info.get("qficr_p1lr_actual_forward_match", False)),
            "p1lr_exact_k32": bool(info.get("qficr_p1lr_exact_k32", False)),
            "p1lr_canonical_ordering_preserved": bool(info.get("qficr_p1lr_canonical_ordering_preserved", False)),
            "p1lr_entropy_budget": bool(info.get("qficr_entropy_budget", False)),
            "p1lr_entropy_question_type": info.get("qficr_entropy_question_type", ""),
            "p1lr_entropy_ratio_base": _float_or_none(info.get("qficr_entropy_ratio_base")),
            "p1lr_entropy_raw_recover_ratio": _float_or_none(info.get("qficr_entropy_raw_recover_ratio")),
            "p1lr_entropy_raw_k_rest": _int_or_none(info.get("qficr_entropy_raw_k_rest")),
            "p1lr_entropy_ratio_cap_k": _int_or_none(info.get("qficr_entropy_ratio_cap_k")),
            "p1lr_entropy_ratio_cap_applied": bool(info.get("qficr_entropy_ratio_cap_applied", False)),
            "p1lr_entropy_k_rest": _int_or_none(info.get("qficr_entropy_k_rest")),
            "p1lr_entropy_k_sem": _int_or_none(info.get("qficr_entropy_k_sem")),
            "p1lr_entropy_actual_forward_match": bool(info.get("qficr_p1lr_entropy_actual_forward_match", False)),
            "p1lr_entropy_exact_k": bool(info.get("qficr_p1lr_entropy_exact_k", False)),
            "p1lr_entropy_canonical_ordering_preserved": bool(info.get("qficr_p1lr_entropy_canonical_ordering_preserved", False)),
            "semantic_prefix_reproduction": bool(info.get("qficr_semantic_prefix_reproduction", False)),
            "objective_audit_mode": info.get("qficr_objective_audit_mode", "legacy_current"),
            "relation_audit_mode": info.get("qficr_relation_audit_mode", "legacy_current"),
            "objective_target_weight": info.get("qficr_objective_target_weight", "task_conditioned"),
            "objective_overlap_kernel": info.get("qficr_objective_overlap_kernel", "relu_square"),
            "ratio_cap_active": bool(info.get("qficr_ratio_cap_active", True)),
            "coverage_monotonic": info.get("qficr_coverage_monotonic"),
            "coverage_trace": info.get("qficr_coverage_trace", []),
            "task_weighted_coverage": _float_or_none(info.get("qficr_task_weighted_coverage")),
            "uniform_coverage": _float_or_none(info.get("qficr_uniform_coverage")),
            "mean_selected_p": _float_or_none(info.get("qficr_mean_selected_p")),
            "affinity_latency_ms": _float_or_none(info.get("qficr_affinity_latency_ms")),
            "greedy_latency_ms": _float_or_none(info.get("qficr_greedy_latency_ms")),
            "total_pruning_latency_ms": _float_or_none(info.get("qficr_total_pruning_latency_ms")),
            "peak_gpu_memory_mb": _float_or_none(info.get("qficr_peak_gpu_memory_mb")),
            "relation_residual_trace": info.get("qficr_relation_residual_trace", []),
            "relation_residual_monotonic": info.get("qficr_relation_residual_monotonic"),
            "relation_reconstruction_residual": _float_or_none(
                info.get("qficr_relation_reconstruction_residual")
            ),
            "relation_reconstruction_residual_raw": _float_or_none(
                info.get("qficr_relation_reconstruction_residual_raw")
            ),
            "legacy_relation_reconstruction_residual": _float_or_none(
                info.get("qficr_legacy_relation_reconstruction_residual")
            ),
            "relation_qr_orthogonality_error": _float_or_none(
                info.get("qficr_relation_qr_orthogonality_error")
            ),
            "relation_nan_inf_count": _int_or_none(info.get("qficr_relation_nan_inf_count", 0)),
            "legacy_final_selected_indices": info.get("qficr_legacy_final_selected_indices", []),
            "residual_utilization": _float_or_none(info.get("qficr_residual_utilization", 0.0)),
            "fallback_to_core_rate": _float_or_none(
                info.get("qficr_fallback_to_core_rate", 0.0)
            ),
            "competition_steps": info.get("qficr_competition_steps", []),
            "trainfree_restoration_mode": info.get("qficr_trainfree_restoration_mode", "legacy"),
            "K_rest_cap": _int_or_none(info.get("qficr_K_rest_cap", 0)),
            "actual_restoration_count": _int_or_none(info.get("qficr_actual_restoration_count", 0)),
            "core_continuation_count": _int_or_none(info.get("qficr_core_continuation_count", 0)),
            "restoration_step": _int_or_none(info.get("qficr_restoration_step", 0)),
            "restoration_candidate_index": _int_or_none(info.get("qficr_restoration_candidate_index")),
            "core_candidate_index": _int_or_none(info.get("qficr_core_candidate_index")),
            "duplicate_count": _int_or_none(info.get("qficr_duplicate_count", 0)),
            "nan_inf_count": _int_or_none(info.get("qficr_nan_inf_count", 0)),
            "anchor_alpha": _float_or_none(info.get("qficr_anchor_alpha")),
            "support_beta": _float_or_none(info.get("qficr_support_beta")),
            "self_discount": _float_or_none(info.get("qficr_self_discount")),
            "mean_p_core": _float_or_none(info.get("qficr_mean_p_core")),
            "mean_p_rest": _float_or_none(info.get("qficr_mean_p_rest")),
            "mean_residual_gain_rest": _float_or_none(info.get("qficr_mean_residual_gain_rest")),
            "mean_support_rest": _float_or_none(info.get("qficr_mean_support_rest")),
            "mean_self_gain_rest": _float_or_none(info.get("qficr_mean_self_gain_rest")),
            "mean_restoration_score_rest": _float_or_none(info.get("qficr_mean_restoration_score_rest")),
            "agreement": _float_or_none(info.get("agreement")),
            "js_div": _float_or_none(info.get("js_div")),
            "gate": _float_or_none(info.get("gate")),
            "beta_eff": _float_or_none(info.get("beta_eff")),
            "beta_base": _float_or_none(info.get("beta_base")),
            "p_sem_entropy": _float_or_none(info.get("p_sem_entropy")),
            "p_cls_entropy": _float_or_none(info.get("p_cls_entropy")),
            "spatial_semantic_v1": bool(info.get("qficr_spatial_semantic_v1", False)),
            "spatial_v1_temperature": _float_or_none(info.get("qficr_spatial_v1_temperature")),
            "spatial_v1_whole_weight": _float_or_none(info.get("qficr_spatial_v1_whole_weight")),
            "spatial_v1_pre_entropy": _float_or_none(info.get("qficr_spatial_v1_pre_entropy")),
            "spatial_v1_pre_entropy_norm": _float_or_none(info.get("qficr_spatial_v1_pre_entropy_norm")),
            "spatial_v1_selection_entropy": _float_or_none(info.get("qficr_spatial_v1_selection_entropy")),
            "spatial_v1_selection_entropy_norm": _float_or_none(info.get("qficr_spatial_v1_selection_entropy_norm")),
            "spatial_diverse_recovery_v3": bool(info.get("qficr_spatial_diverse_recovery_v3", False)),
            "v3_proposal_count": _int_or_none(info.get("qficr_v3_proposal_count")),
            "v3_chosen_proposal_index": _int_or_none(info.get("qficr_v3_chosen_proposal_index")),
            "v3_chosen_base_seed": _int_or_none(info.get("qficr_v3_chosen_base_seed")),
            "v3_spatial_score": _float_or_none(info.get("qficr_v3_spatial_score")),
            "v3_proposal_spatial_scores": info.get("qficr_v3_proposal_spatial_scores", []),
            "relation_routed_recovery_v4": bool(info.get("qficr_relation_routed_recovery_v4", False)),
            "v4_route": info.get("qficr_v4_route", ""),
            "v4_explicit_spatial_relation": bool(info.get("qficr_v4_explicit_spatial_relation", False)),
            "v4_pattern_version": info.get("qficr_v4_pattern_version", ""),
            "p_vtp_entropy": _float_or_none(info.get("p_vtp_entropy")),
            "visual_prior_source": info.get("visual_prior_source", "none"),
            "vtp_available": bool(info.get("vtp_available", False)),
            "vtp_fallback": bool(info.get("vtp_fallback", False)),
            "p_vtp_min": _float_or_none(info.get("p_vtp_min")),
            "p_vtp_max": _float_or_none(info.get("p_vtp_max")),
            "p_vtp_mean": _float_or_none(info.get("p_vtp_mean")),
            "p_vtp_sum": _float_or_none(info.get("p_vtp_sum")),
            "p_sem_values": info.get("p_sem_values", []),
            "p_cls_values": info.get("p_cls_values", []),
            "p_vtp_values": info.get("p_vtp_values", []),
            "p_mix_values": info.get("p_mix_values", []),
            "prob_values": info.get("prob_values", []),
            "keep_size": _int_or_none(info.get("keep_size")),
            "budget_calib": bool(info.get("budget_calib", False)),
            "budget_alpha": _float_or_none(info.get("budget_alpha", self.qfid_budget_alpha)),
            "budget_target_neff": _float_or_none(info.get("budget_target_neff", 0.0)),
            "budget_neff_before": _float_or_none(info.get("budget_neff_before", 0.0)),
            "budget_neff_after": _float_or_none(info.get("budget_neff_after", 0.0)),
            "budget_gamma": _float_or_none(info.get("budget_gamma", 1.0)),
        }
        with open(stats_path, "a", encoding="utf-8") as stats_file:
            stats_file.write(json.dumps(record, sort_keys=True) + "\n")
        _QFID_STATS_COUNT += 1

    def _debug_config(self):
        self._debug(
            f"use_repulsion={self.use_repulsion}, "
            f"use_complement={self.use_complement}, "
            f"use_phi={self.use_phi}, "
            f"use_semantic_candidate={self.use_semantic_candidate}, "
            f"use_spatial_candidate={self.use_spatial_candidate}, "
            f"spatial_exclude_global={self.spatial_exclude_global}, "
            f"use_chain_coverage={self.use_chain_coverage}, "
            f"gamma_chain={self.gamma_chain}, "
            f"solver={self.solver}"
        )

    @staticmethod
    def aggregate_text_to_visual_attention(text_attn):
        if not torch.is_tensor(text_attn):
            return None
        if text_attn.ndim == 1:
            return torch.nan_to_num(text_attn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if text_attn.ndim == 4:
            return text_attn.float().mean(dim=(0, 1, 2))
        if text_attn.ndim == 3:
            return text_attn.float().mean(dim=(0, 1))
        return None

    def _extract_relevance_score(self, num_tokens, device, relevance=None, text_attn=None):
        score = None
        source = "none"

        if torch.is_tensor(relevance):
            relevance = torch.nan_to_num(relevance.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if relevance.ndim == 1:
                score = relevance
            elif relevance.ndim >= 2:
                score = relevance.mean(dim=tuple(range(relevance.ndim - 1)))
            if torch.is_tensor(score) and score.numel() == num_tokens:
                source = "relevance"
            else:
                score = None

        if score is None:
            attn_score = self.aggregate_text_to_visual_attention(text_attn)
            if torch.is_tensor(attn_score) and attn_score.numel() == num_tokens:
                score = attn_score
                source = "attn"

        if score is None or score.numel() != num_tokens:
            return None, "none"
        return safe_normalize(score.to(device=device).reshape(-1)), source

    def _softmax_probability_from_score(self, score, tau, device):
        score = safe_normalize(score.to(device=device).reshape(-1))
        logits = score / tau
        probs = torch.softmax(logits, dim=0)
        return torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _normalized_probability_from_attention(self, score, device):
        if not torch.is_tensor(score):
            return None
        score = torch.nan_to_num(score.float().to(device=device).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if score.numel() == 0:
            return None
        score = safe_normalize(score).clamp_min(0.0)
        score_sum = score.sum()
        if not torch.isfinite(score_sum) or score_sum.item() <= float(self.qfid_eps):
            return None
        return score / score_sum.clamp_min(float(self.qfid_eps))

    def extract_semantic_units(self, question):
        if isinstance(question, (list, tuple)):
            question = question[0] if question else None
        if not isinstance(question, str) or not question.strip():
            units = ["object"]
        else:
            words = re.findall(r"[a-z]+(?:'[a-z]+)?", question.lower())
            units = []
            seen = set()
            for word in words:
                if word in _QUESTION_STOP_WORDS or word in seen:
                    continue
                units.append(word)
                seen.add(word)
                if len(units) == 8:
                    break
            if not units:
                units = ["object"]
        self._debug(f"semantic_units={units}")
        return units

    @staticmethod
    def build_relation_matrix(units, device=None):
        num_units = len(units)
        relation = torch.zeros(num_units, num_units, dtype=torch.float32, device=device)
        relation_words = _SPATIAL_WORDS | _ACTION_WORDS
        relation_indices = [i for i, unit in enumerate(units) if unit in relation_words]

        if relation_indices:
            for relation_idx in relation_indices:
                for other_idx in range(num_units):
                    if other_idx == relation_idx:
                        continue
                    relation[relation_idx, other_idx] = 1.0
                    relation[other_idx, relation_idx] = 1.0
        elif num_units > 1:
            relation.fill_(0.1)
            relation.fill_diagonal_(0.0)
        return relation

    def compute_semantic_response(self, features, units, text_embeds=None, temperature=1.0):
        if features.ndim == 3:
            features = features.float().mean(dim=0)
        elif features.ndim == 2:
            features = features.float()
        else:
            return None, "invalid"

        num_tokens = features.shape[0]
        num_units = max(1, len(units))
        device = features.device
        uniform = torch.full(
            (num_tokens, num_units),
            1.0 / num_units,
            dtype=torch.float32,
            device=device,
        )
        if not torch.is_tensor(text_embeds):
            self._debug("semantic text_embeds unavailable, use uniform response")
            return uniform, "uniform"

        if text_embeds.ndim == 3 and text_embeds.shape[0] == 1:
            text_embeds = text_embeds.squeeze(0)
        if text_embeds.ndim != 2 or text_embeds.shape[0] != num_units:
            self._debug("semantic text_embeds shape mismatch, use uniform response")
            return uniform, "uniform"
        if text_embeds.shape[-1] != features.shape[-1]:
            self._debug("semantic feature dimension mismatch, use uniform response")
            return uniform, "uniform"

        text_embeds = text_embeds.to(device=device, dtype=torch.float32)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = torch.matmul(features, text_embeds.t()) / max(float(temperature), 1e-6)
        response = torch.softmax(logits, dim=-1)
        response = torch.nan_to_num(response, nan=1.0 / num_units, posinf=0.0, neginf=0.0)
        if not torch.isfinite(response).all():
            self._debug("semantic response is non-finite, use uniform response")
            return uniform, "uniform"
        return response, "clip"

    @staticmethod
    def compute_semantic_node_score(b, eps=1e-8):
        if not torch.is_tensor(b) or b.ndim != 2 or b.shape[1] == 0:
            return None

        b = torch.nan_to_num(b.float(), nan=0.0, posinf=0.0, neginf=0.0)
        num_units = b.shape[1]
        max_response = b.max(dim=-1).values
        if num_units == 1:
            confidence = torch.ones_like(max_response)
        else:
            entropy = -(b.clamp_min(eps) * b.clamp_min(eps).log()).sum(dim=-1)
            confidence = 1.0 - entropy / torch.log(
                torch.tensor(float(num_units), device=b.device, dtype=b.dtype)
            )
            confidence = confidence.clamp(0.0, 1.0)
        return max_response * confidence

    def compute_measurement_node_score(self, b, num_tokens, device):
        if not torch.is_tensor(b) or b.ndim != 2 or b.shape[0] != num_tokens or b.shape[1] == 0:
            return None, "invalid"

        response = torch.nan_to_num(b.float().to(device=device), nan=0.0, posinf=0.0, neginf=0.0)
        response = response.clamp_min(0.0)
        response = response / response.sum(dim=-1, keepdim=True).clamp_min(float(self.qfid_eps))
        alpha = torch.full(
            (response.shape[1],),
            1.0 / max(1, response.shape[1]),
            dtype=response.dtype,
            device=device,
        )
        score = (response.pow(2) * alpha.unsqueeze(0)).sum(dim=-1)
        score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return score, "uniform_alpha"

    def build_qfid_spatial_factor(self, num_tokens, device):
        if num_tokens <= 0:
            return None, {
                "spatial_state_enabled": bool(self.qfid_spatial_state),
                "spatial_state_applied": False,
                "spatial_state_fallback": True,
                "spatial_lambda": float(max(0.0, min(float(self.qfid_spatial_lambda), 1.0))),
                "spatial_sigma": float(max(float(self.qfid_spatial_sigma), float(self.qfid_eps))),
                "grid_size": 0,
                "spatial_factor_min": 1.0,
                "spatial_factor_max": 1.0,
                "spatial_factor_mean": 1.0,
            }

        grid_size = int(round(math.sqrt(num_tokens)))
        if grid_size * grid_size != num_tokens:
            return None, {
                "spatial_state_enabled": bool(self.qfid_spatial_state),
                "spatial_state_applied": False,
                "spatial_state_fallback": True,
                "spatial_lambda": float(max(0.0, min(float(self.qfid_spatial_lambda), 1.0))),
                "spatial_sigma": float(max(float(self.qfid_spatial_sigma), float(self.qfid_eps))),
                "grid_size": 0,
                "spatial_factor_min": 1.0,
                "spatial_factor_max": 1.0,
                "spatial_factor_mean": 1.0,
            }

        lambda_coef = max(0.0, min(float(self.qfid_spatial_lambda), 1.0))
        sigma = max(float(self.qfid_spatial_sigma), float(self.qfid_eps))
        coord_1d = torch.linspace(0.0, 1.0, steps=grid_size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coord_1d, coord_1d, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        dist2 = torch.cdist(coords, coords, p=2).pow(2)
        spatial_state = torch.exp(-dist2 / max(2.0 * sigma * sigma, float(self.qfid_eps)))
        spatial_state = torch.nan_to_num(spatial_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        spatial_state.fill_diagonal_(1.0)
        spatial_factor = (1.0 - lambda_coef) + lambda_coef * spatial_state
        spatial_factor = torch.nan_to_num(spatial_factor.float(), nan=1.0, posinf=1.0, neginf=1.0)
        spatial_factor.fill_diagonal_(1.0)
        return spatial_factor, {
            "spatial_state_enabled": bool(self.qfid_spatial_state),
            "spatial_state_applied": True,
            "spatial_state_fallback": False,
            "spatial_lambda": lambda_coef,
            "spatial_sigma": sigma,
            "grid_size": grid_size,
            "spatial_factor_min": float(spatial_factor.min().item()),
            "spatial_factor_max": float(spatial_factor.max().item()),
            "spatial_factor_mean": float(spatial_factor.mean().item()),
        }

    def compute_qfid_cls_probability(self, cls_attn, num_tokens, device):
        info = {
            "cls_attn_available": False,
            "cls_attn_layer": self.qfid_cls_attn_layer,
            "cls_prior_layers": self.qficr_cls_prior_layers,
            "cls_head_reduce": self.qfid_cls_head_reduce,
            "cls_head_entropy_min": 0.0,
            "cls_head_entropy_max": 0.0,
            "cls_head_entropy_mean": 0.0,
            "cls_head_weight_min": 0.0,
            "cls_head_weight_max": 0.0,
        }
        if not torch.is_tensor(cls_attn):
            return None, info

        attention = torch.nan_to_num(
            cls_attn.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if attention.ndim == 1:
            attention = attention.unsqueeze(0)

        if attention.ndim == 2:
            cls_score = attention.mean(dim=0)
        elif attention.ndim == 3:
            if self.qfid_cls_head_reduce == "mean":
                cls_score = attention.mean(dim=1).mean(dim=0)
            else:
                head_probs = attention / attention.sum(dim=-1, keepdim=True).clamp_min(self.qfid_eps)
                safe_head_probs = head_probs.clamp_min(self.qfid_eps)
                head_entropy = -(safe_head_probs * safe_head_probs.log()).sum(dim=-1)
                head_weights = 1.0 / (head_entropy + self.qfid_eps)
                head_weights = head_weights / head_weights.sum(dim=-1, keepdim=True).clamp_min(self.qfid_eps)
                cls_score = (head_weights.unsqueeze(-1) * head_probs).sum(dim=1).mean(dim=0)
                info.update({
                    "cls_head_entropy_min": float(head_entropy.min().item()),
                    "cls_head_entropy_max": float(head_entropy.max().item()),
                    "cls_head_entropy_mean": float(head_entropy.mean().item()),
                    "cls_head_weight_min": float(head_weights.min().item()),
                    "cls_head_weight_max": float(head_weights.max().item()),
                })
        else:
            return None, info

        cls_score = cls_score.reshape(-1)
        if cls_score.numel() != num_tokens or cls_score.sum().item() <= float(self.qfid_eps):
            return None, info
        p_cls = cls_score / cls_score.sum().clamp_min(float(self.qfid_eps))
        info["cls_attn_available"] = True
        return p_cls, info

    def compute_qfid_vtp_probability(self, transition_prior, num_tokens, device):
        info = {
            "vtp_available": False,
            "vtp_sum": 0.0,
            "vtp_min": 0.0,
            "vtp_max": 0.0,
            "vtp_mean": 0.0,
        }
        if not torch.is_tensor(transition_prior):
            return None, info

        prior = torch.nan_to_num(
            transition_prior.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if prior.ndim == 2:
            prior = prior[0]
        prior = prior.reshape(-1)
        if prior.numel() != num_tokens or prior.sum().item() <= float(self.qfid_eps):
            return None, info
        p_vtp = prior / prior.sum().clamp_min(float(self.qfid_eps))
        info.update({
            "vtp_available": True,
            "vtp_sum": float(p_vtp.sum().item()),
            "vtp_min": float(p_vtp.min().item()),
            "vtp_max": float(p_vtp.max().item()),
            "vtp_mean": float(p_vtp.mean().item()),
        })
        return p_vtp, info

    def compute_qfid_probability(
        self,
        num_tokens,
        device,
        b=None,
        relevance=None,
        text_attn=None,
        cls_attn=None,
        transition_prior=None,
        semantic_response_source=None,
    ):
        tau = max(float(self.qfid_tau), float(self.qfid_eps))
        source = self.qfid_prob_source
        probs = None
        source_used = source
        source_fallback = False
        measurement_score = None
        measurement_alpha_mode = "none"
        p_sem = None
        p_cls = None
        p_vtp = None
        p_visual = None
        p_mix = None
        agreement = 0.0
        js_div = 0.0
        gate = 1.0
        beta_eff = float(max(0.0, min(float(self.qfid_cls_mix_beta), 1.0)))
        p_sem_entropy = 0.0
        p_cls_entropy = 0.0
        p_vtp_entropy = 0.0
        cls_attn_available = False
        vtp_available = False
        cls_fallback = False
        vtp_fallback = False
        visual_prior_source = "none"
        observation_mode = os.environ.get("EC_QFICR_OBSERVATION_MODE", "full").strip().lower()
        if observation_mode not in {"full", "sem_only", "cls_only", "fixed_fusion"}:
            observation_mode = "full"
        cls_reduce_info = {
            "cls_attn_layer": self.qfid_cls_attn_layer,
            "cls_prior_layers": self.qficr_cls_prior_layers,
            "cls_head_reduce": self.qfid_cls_head_reduce,
            "cls_head_entropy_min": 0.0,
            "cls_head_entropy_max": 0.0,
            "cls_head_entropy_mean": 0.0,
            "cls_head_weight_min": 0.0,
            "cls_head_weight_max": 0.0,
        }
        vtp_info = {
            "vtp_available": False,
            "vtp_sum": 0.0,
            "vtp_min": 0.0,
            "vtp_max": 0.0,
            "vtp_mean": 0.0,
        }
        depolarize_mode = self.qfid_depolarize_mode
        if depolarize_mode not in {"fixed", "adaptive"}:
            depolarize_mode = "fixed"

        def build_semantic_probs():
            semantic_score = self.compute_semantic_node_score(b)
            if semantic_score is None and torch.is_tensor(b) and b.ndim == 2 and b.shape[0] == num_tokens:
                semantic_score = torch.nan_to_num(
                    b.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).max(dim=-1).values
            if (
                semantic_score is None
                or semantic_score.numel() != num_tokens
                or semantic_response_source == "uniform"
                or torch.max(torch.abs(semantic_score)).item() <= float(self.qfid_eps)
            ):
                return None
            return self._softmax_probability_from_score(semantic_score, tau, device)

        if source == "semantic":
            p_sem = build_semantic_probs()
            if p_sem is None:
                source_used = "uniform"
            else:
                probs = p_sem
        elif source == "measurement":
            measurement_score, measurement_alpha_mode = self.compute_measurement_node_score(
                b,
                num_tokens,
                device,
            )
            if (
                measurement_score is None
                or measurement_score.numel() != num_tokens
                or semantic_response_source == "uniform"
                or torch.max(torch.abs(measurement_score)).item() <= float(self.qfid_eps)
            ):
                source_fallback = True
                source_used = "semantic_fallback"
                score = self.compute_semantic_node_score(b)
                if score is None and torch.is_tensor(b) and b.ndim == 2 and b.shape[0] == num_tokens:
                    score = torch.nan_to_num(b.float(), nan=0.0, posinf=0.0, neginf=0.0).max(dim=-1).values
                if (
                    score is None
                    or score.numel() != num_tokens
                    or semantic_response_source == "uniform"
                    or torch.max(torch.abs(score)).item() <= float(self.qfid_eps)
                ):
                    source_used = "uniform"
                else:
                    probs = self._softmax_probability_from_score(score, tau, device)
            else:
                probs = self._softmax_probability_from_score(measurement_score, tau, device)
        elif source in {"clsmix", "cls_only_qf"}:
            p_cls, cls_reduce_info = self.compute_qfid_cls_probability(
                cls_attn,
                num_tokens,
                device,
            )
            cls_attn_available = p_cls is not None
        elif source in {"vtpmix", "vtp_only_qf"}:
            p_vtp, vtp_info = self.compute_qfid_vtp_probability(
                transition_prior,
                num_tokens,
                device,
            )
            vtp_available = p_vtp is not None

        if source in {"clsmix", "vtpmix"}:
            p_sem = build_semantic_probs()
            p_visual = p_cls if source == "clsmix" else p_vtp
            visual_prior_source = "cls" if source == "clsmix" else "vtp"
            if p_sem is not None and p_visual is not None:
                beta = max(0.0, min(float(self.qfid_cls_mix_beta), 1.0))
                safe_p_sem = p_sem.clamp_min(float(self.qfid_eps))
                safe_p_visual = p_visual.clamp_min(float(self.qfid_eps))
                p_sem_entropy = float((-(p_sem * safe_p_sem.log()).sum()).item())
                p_cls_entropy = float((-(p_visual * safe_p_visual.log()).sum()).item())
                if source == "vtpmix":
                    p_vtp_entropy = p_cls_entropy
                if self.qfid_cls_gate:
                    midpoint = 0.5 * (p_sem + p_visual)
                    safe_midpoint = midpoint.clamp_min(float(self.qfid_eps))
                    js_divergence = 0.5 * (
                        (p_sem * (safe_p_sem.log() - safe_midpoint.log())).sum()
                        + (p_visual * (safe_p_visual.log() - safe_midpoint.log())).sum()
                    )
                    js_value = max(0.0, float(js_divergence.item()))
                    js_div = js_value
                    agreement = 1.0 - js_value / math.log(2.0)
                    agreement = max(0.0, min(agreement, 1.0))
                    gate = max(
                        float(self.qfid_cls_gate_min),
                        min(0.5 + agreement, float(self.qfid_cls_gate_max)),
                    )
                    beta_base = max(0.0, min(float(self.qfid_cls_gate_beta_base), 1.0))
                    beta_eff = max(0.0, min(beta_base * gate, 1.0))
                    beta = beta_eff
                else:
                    beta_eff = beta
                if self.qfid_cls_mix_mode == "linear":
                    p_mix = (1.0 - beta) * p_sem + beta * p_visual
                else:
                    log_mix = (
                        (1.0 - beta) * torch.log(p_sem + float(self.qfid_eps))
                        + beta * torch.log(p_visual + float(self.qfid_eps))
                    )
                    log_mix = log_mix - log_mix.max()
                    p_mix = torch.exp(log_mix)
                p_mix = p_mix / p_mix.sum().clamp_min(float(self.qfid_eps))
                probs = p_mix
                source_used = source
            else:
                if source == "clsmix":
                    cls_fallback = True
                else:
                    vtp_fallback = True
                source_fallback = True
                if p_sem is not None:
                    probs = p_sem
                    source_used = "semantic_fallback"
                else:
                    source_used = "uniform"
        elif source == "cls_only_qf":
            if p_cls is not None:
                probs = p_cls
                source_used = "cls_only_qf"
            else:
                cls_fallback = True
                source_fallback = True
                source_used = "uniform"
        elif source == "vtp_only_qf":
            if p_vtp is not None:
                probs = p_vtp
                p_vtp_entropy = float((-(p_vtp * p_vtp.clamp_min(float(self.qfid_eps)).log()).sum()).item())
                source_used = "vtp_only_qf"
                visual_prior_source = "vtp"
            else:
                vtp_fallback = True
                source_fallback = True
                source_used = "uniform"
        elif source == "relevance":
            score, rel_source = self._extract_relevance_score(
                num_tokens,
                device,
                relevance=relevance,
                text_attn=text_attn,
            )
            if score is None:
                source_used = "uniform"
            else:
                source_used = rel_source
                probs = self._softmax_probability_from_score(score, tau, device)
        elif source not in {"semantic", "measurement"}:
            source_used = "uniform"

        if observation_mode != "full" and source == "clsmix":
            if p_sem is None:
                p_sem = build_semantic_probs()
            if p_cls is None:
                p_cls, cls_reduce_info = self.compute_qfid_cls_probability(
                    cls_attn,
                    num_tokens,
                    device,
                )
                cls_attn_available = p_cls is not None
            if observation_mode == "sem_only":
                if p_sem is not None:
                    probs = p_sem
                    source_used = "sem_only"
                else:
                    source_fallback = True
                    source_used = "uniform"
            elif observation_mode == "cls_only":
                if p_cls is not None:
                    probs = p_cls
                    source_used = "cls_only"
                else:
                    cls_fallback = True
                    source_fallback = True
                    source_used = "uniform"
            elif observation_mode == "fixed_fusion":
                if p_sem is not None and p_cls is not None:
                    p_mix = 0.5 * p_sem + 0.5 * p_cls
                    p_mix = p_mix / p_mix.sum().clamp_min(float(self.qfid_eps))
                    probs = p_mix
                    source_used = "fixed_fusion"
                elif p_sem is not None:
                    probs = p_sem
                    source_fallback = True
                    source_used = "semantic_fallback"
                elif p_cls is not None:
                    probs = p_cls
                    cls_fallback = False
                    source_fallback = True
                    source_used = "cls_fallback"
                else:
                    cls_fallback = True
                    source_fallback = True
                    source_used = "uniform"

        if probs is None or probs.numel() != num_tokens:
            probs = torch.full(
                (num_tokens,),
                1.0 / max(1, num_tokens),
                dtype=torch.float32,
                device=device,
            )

        probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum().clamp_min(float(self.qfid_eps))

        entropy = 0.0
        entropy_norm = 1.0
        if probs.numel() > 0:
            safe_probs = probs.clamp_min(float(self.qfid_eps))
            entropy = float((-(safe_probs * safe_probs.log()).sum()).item())
            max_entropy = max(float(math.log(max(2, probs.numel()))), float(self.qfid_eps))
            entropy_norm = max(0.0, min(1.0, entropy / max_entropy))

        if depolarize_mode == "adaptive":
            eps_min = max(0.0, min(float(self.qfid_depolarize_min), 1.0))
            eps_max = max(0.0, min(float(self.qfid_depolarize_max), 1.0))
            if eps_max < eps_min:
                eps_min, eps_max = eps_max, eps_min
            eps_dep = eps_min + (eps_max - eps_min) * (1.0 - entropy_norm)
        else:
            eps_dep = float(self.qfid_depolarize)
        eps_dep = max(0.0, min(eps_dep, 1.0))

        if eps_dep > 0.0:
            uniform = torch.full_like(probs, 1.0 / max(1, probs.numel()))
            probs = (1.0 - eps_dep) * probs + eps_dep * uniform
            probs = probs / probs.sum().clamp_min(float(self.qfid_eps))

        prob_info = {
            "mode": depolarize_mode,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "adaptive_depolarize": float(eps_dep),
            "p_min": float(probs.min().item()) if probs.numel() > 0 else 0.0,
            "p_max": float(probs.max().item()) if probs.numel() > 0 else 0.0,
            "p_mean": float(probs.mean().item()) if probs.numel() > 0 else 0.0,
            "p_sum": float(probs.sum().item()) if probs.numel() > 0 else 0.0,
            "measurement_score_min": float(measurement_score.min().item()) if measurement_score is not None and measurement_score.numel() > 0 else 0.0,
            "measurement_score_max": float(measurement_score.max().item()) if measurement_score is not None and measurement_score.numel() > 0 else 0.0,
            "measurement_score_mean": float(measurement_score.mean().item()) if measurement_score is not None and measurement_score.numel() > 0 else 0.0,
            "measurement_prob_min": float(probs.min().item()) if source == "measurement" and probs.numel() > 0 else 0.0,
            "measurement_prob_max": float(probs.max().item()) if source == "measurement" and probs.numel() > 0 else 0.0,
            "measurement_prob_mean": float(probs.mean().item()) if source == "measurement" and probs.numel() > 0 else 0.0,
            "measurement_alpha_mode": measurement_alpha_mode,
            "source_fallback": source_fallback,
            "cls_attn_available": cls_attn_available,
            "vtp_available": vtp_available,
            "cls_fallback": cls_fallback,
            "vtp_fallback": vtp_fallback,
            "visual_prior_source": visual_prior_source,
            "cls_mix_mode": self.qfid_cls_mix_mode,
            "observation_mode": observation_mode,
            "cls_mix_beta": float(max(0.0, min(float(self.qfid_cls_mix_beta), 1.0))),
            "cls_gate_enabled": bool(self.qfid_cls_gate),
            "cls_gate_mode": self.qfid_cls_gate_mode,
            "agreement": agreement,
            "js_div": js_div,
            "gate": gate,
            "beta_eff": beta_eff,
            "beta_base": float(max(0.0, min(float(self.qfid_cls_gate_beta_base), 1.0))),
            "gate_min": float(self.qfid_cls_gate_min),
            "gate_max": float(self.qfid_cls_gate_max),
            "p_sem_entropy": p_sem_entropy,
            "p_cls_entropy": p_cls_entropy,
            "p_vtp_entropy": p_vtp_entropy,
            "cls_attn_layer": cls_reduce_info["cls_attn_layer"],
            "cls_prior_layers": cls_reduce_info["cls_prior_layers"],
            "cls_head_reduce": cls_reduce_info["cls_head_reduce"],
            "cls_head_entropy_min": cls_reduce_info["cls_head_entropy_min"],
            "cls_head_entropy_max": cls_reduce_info["cls_head_entropy_max"],
            "cls_head_entropy_mean": cls_reduce_info["cls_head_entropy_mean"],
            "cls_head_weight_min": cls_reduce_info["cls_head_weight_min"],
            "cls_head_weight_max": cls_reduce_info["cls_head_weight_max"],
            "p_cls_min": float(p_cls.min().item()) if p_cls is not None and p_cls.numel() > 0 else 0.0,
            "p_cls_max": float(p_cls.max().item()) if p_cls is not None and p_cls.numel() > 0 else 0.0,
            "p_cls_mean": float(p_cls.mean().item()) if p_cls is not None and p_cls.numel() > 0 else 0.0,
            "p_vtp_min": float(p_vtp.min().item()) if p_vtp is not None and p_vtp.numel() > 0 else 0.0,
            "p_vtp_max": float(p_vtp.max().item()) if p_vtp is not None and p_vtp.numel() > 0 else 0.0,
            "p_vtp_mean": float(p_vtp.mean().item()) if p_vtp is not None and p_vtp.numel() > 0 else 0.0,
            "p_vtp_sum": float(p_vtp.sum().item()) if p_vtp is not None and p_vtp.numel() > 0 else 0.0,
            "p_sem_min": float(p_sem.min().item()) if p_sem is not None and p_sem.numel() > 0 else 0.0,
            "p_sem_max": float(p_sem.max().item()) if p_sem is not None and p_sem.numel() > 0 else 0.0,
            "p_sem_mean": float(p_sem.mean().item()) if p_sem is not None and p_sem.numel() > 0 else 0.0,
            "p_mix_min": float(p_mix.min().item()) if p_mix is not None and p_mix.numel() > 0 else 0.0,
            "p_mix_max": float(p_mix.max().item()) if p_mix is not None and p_mix.numel() > 0 else 0.0,
            "p_mix_mean": float(p_mix.mean().item()) if p_mix is not None and p_mix.numel() > 0 else 0.0,
            "p_mix_sum": float(p_mix.sum().item()) if p_mix is not None and p_mix.numel() > 0 else 0.0,
        }
        if get_env_bool("EC_QFID_DEBUG_DUMP_PROBS", default=False):
            prob_info.update({
                "p_sem_values": [float(x) for x in p_sem.detach().cpu().tolist()] if p_sem is not None and p_sem.numel() > 0 else [],
                "p_cls_values": [float(x) for x in p_cls.detach().cpu().tolist()] if p_cls is not None and p_cls.numel() > 0 else [],
                "p_vtp_values": [float(x) for x in p_vtp.detach().cpu().tolist()] if p_vtp is not None and p_vtp.numel() > 0 else [],
                "p_mix_values": [float(x) for x in p_mix.detach().cpu().tolist()] if p_mix is not None and p_mix.numel() > 0 else [],
                "prob_values": [float(x) for x in probs.detach().cpu().tolist()] if probs is not None and probs.numel() > 0 else [],
            })
        return probs, source_used, prob_info

    @torch.no_grad()
    def compute_task_probability(
        self,
        visual_tokens,
        question=None,
        relevance=None,
        cls_attn=None,
        text_embeds=None,
        semantic_features=None,
        semantic_units=None,
        transition_prior=None,
    ):
        """Return the frozen QFi-CR task prior without performing selection."""
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")
        _, num_tokens, _ = visual_tokens.shape
        units = semantic_units or self.extract_semantic_units(question)
        semantic_response = None
        semantic_response_source = None
        if self.needs_semantics:
            response_features = semantic_features if semantic_features is not None else visual_tokens
            semantic_response, semantic_response_source = self.compute_semantic_response(
                response_features,
                units,
                text_embeds=text_embeds,
            )
        probs, source_used, prob_info = self.compute_qfid_probability(
            num_tokens,
            visual_tokens.device,
            b=semantic_response,
            relevance=relevance,
            cls_attn=cls_attn,
            transition_prior=transition_prior,
            semantic_response_source=semantic_response_source,
        )
        return probs, {
            "source": source_used,
            "semantic_response_source": semantic_response_source,
            "semantic_units": units,
            **prob_info,
        }

    def select_by_quantum_fidelity(
        self,
        visual_tokens,
        K,
        semantic_response=None,
        relevance=None,
        text_attn=None,
        cls_attn=None,
        transition_prior=None,
        question=None,
        semantic_response_source=None,
        core_indices_override=None,
    ):
        """Select a fixed-size subset that preserves the instruction-conditioned visual mixed state.

        QF-Pruner models each visual token as a pure state psi_i and constructs an
        instruction-conditioned mixed state rho_q = sum_i p_i^q |psi_i><psi_i|.
        The amplitude kernel uses sqrt(p_i p_j) <psi_i, psi_j>. The density kernel
        uses sqrt(p_i p_j) |<psi_i, psi_j>|^2, matching Hilbert-Schmidt inner
        products between pure-state density matrices. Pivoted-Cholesky residual
        greedy selection chooses a subset that maximizes quantum kernel fidelity /
        minimizes mixed-state approximation residual under a fixed K budget.
        """
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        global _QFICR_RUNTIME_PROFILE_COUNT
        pruning_start = time.perf_counter()
        profile_limit = max(_env_int("EC_QFICR_RUNTIME_PROFILE_SAMPLES", "55"), 0)
        profile_this_sample = _QFICR_RUNTIME_PROFILE_COUNT < profile_limit
        _QFICR_RUNTIME_PROFILE_COUNT += 1
        if profile_this_sample and visual_tokens.is_cuda:
            torch.cuda.reset_peak_memory_stats(visual_tokens.device)

        features = visual_tokens.float().mean(dim=0)
        num_tokens = features.shape[0]
        device = features.device
        eps = max(float(self.qfid_eps), 1e-12)

        psi = features / features.norm(dim=-1, keepdim=True).clamp_min(eps)
        probs, prob_source_used, prob_info = self.compute_qfid_probability(
            num_tokens,
            device,
            b=semantic_response,
            relevance=relevance,
            text_attn=text_attn,
            cls_attn=cls_attn,
            transition_prior=transition_prior,
            semantic_response_source=semantic_response_source,
        )
        cls_prior_for_restore = None
        if self.qfid_prob_source in {"clsmix", "cls_only_qf"} or os.environ.get(
            "EC_QFICR_RESTORE_EXTRA_PRIOR", "none"
        ).strip().lower() == "uncovered_cls":
            cls_prior_for_restore, _ = self.compute_qfid_cls_probability(cls_attn, num_tokens, device)
        safe_probs = probs.clamp_min(max(float(self.qfid_budget_eps), eps))
        safe_probs = safe_probs / safe_probs.sum().clamp_min(max(float(self.qfid_budget_eps), eps))
        neff_before = float(torch.exp(-(safe_probs * safe_probs.log()).sum()).item())
        budget_info = {
            "budget_alpha": float(self.qfid_budget_alpha),
            "budget_target_neff": float(min(num_tokens, max(min(int(K), num_tokens), self.qfid_budget_alpha * int(K)))),
            "budget_neff_before": neff_before,
            "budget_neff_after": neff_before,
            "budget_gamma": 1.0,
        }
        if self.qfid_budget_calib:
            probs, budget_info = budget_calibrate_prob(
                probs,
                keep_num=K,
                alpha=self.qfid_budget_alpha,
                gamma_min=self.qfid_budget_gamma_min,
                gamma_max=self.qfid_budget_gamma_max,
                iters=self.qfid_budget_iters,
                eps=self.qfid_budget_eps,
            )
        sqrt_p = torch.sqrt(probs.clamp_min(eps))
        sim = torch.matmul(psi, psi.t())
        weight = sqrt_p[:, None] * sqrt_p[None, :]
        if self.qfid_kernel == "amplitude":
            kernel_q = weight * sim
        elif self.qfid_kernel == "density":
            kernel_q = weight * compute_projection_overlap_kernel(
                psi,
                eps=eps,
                overlap_kernel=self.qfid_overlap_kernel,
                assume_normalized=True,
            )
        else:
            raise ValueError(f"Unknown EC_QFID_KERNEL: {self.qfid_kernel}")
        spectral_info = {
            "spectral_filter_enabled": bool(self.qfid_spectral_filter),
            "spectral_filter_applied": False,
            "spectral_filter_fallback": False,
            "spectral_gamma": float(self.qfid_spectral_gamma),
            "spectral_eps": float(self.qfid_spectral_eps),
            "spectral_trace_norm": bool(self.qfid_spectral_trace_norm),
            "spectral_trace_before": float(torch.trace(0.5 * (kernel_q + kernel_q.t())).item()),
            "spectral_trace_after": float(torch.trace(0.5 * (kernel_q + kernel_q.t())).item()),
            "spectral_rank_pos": 0,
            "spectral_min_eval": 0.0,
            "spectral_max_eval": 0.0,
        }
        if self.qfid_spectral_filter:
            if self.qfid_kernel != "density":
                spectral_info["spectral_filter_fallback"] = True
            else:
                try:
                    kernel_q, filtered_info = spectral_filter_kernel(
                        kernel_q,
                        gamma=self.qfid_spectral_gamma,
                        eps=self.qfid_spectral_eps,
                        trace_norm=self.qfid_spectral_trace_norm,
                    )
                    spectral_info.update(filtered_info)
                    spectral_info["spectral_filter_applied"] = True
                except RuntimeError as exc:
                    global _QFID_SPECTRAL_WARNED
                    spectral_info["spectral_filter_fallback"] = True
                    if not _QFID_SPECTRAL_WARNED:
                        print(
                            f"[QF-Pruner] warning: spectral filter fallback to raw density kernel "
                            f"because eigh failed: {exc}",
                            flush=True,
                        )
                        _QFID_SPECTRAL_WARNED = True
        spatial_info = {
            "spatial_state_enabled": bool(self.qfid_spatial_state),
            "spatial_state_applied": False,
            "spatial_state_fallback": False,
            "spatial_lambda": float(max(0.0, min(float(self.qfid_spatial_lambda), 1.0))),
            "spatial_sigma": float(max(float(self.qfid_spatial_sigma), float(self.qfid_eps))),
            "grid_size": 0,
            "spatial_factor_min": 1.0,
            "spatial_factor_max": 1.0,
            "spatial_factor_mean": 1.0,
        }
        if self.qfid_spatial_state:
            spatial_factor, spatial_info = self.build_qfid_spatial_factor(num_tokens, device)
            if spatial_factor is not None:
                kernel_q = kernel_q * spatial_factor
        kernel_q = 0.5 * (kernel_q + kernel_q.t())
        kernel_q = torch.nan_to_num(kernel_q.float(), nan=0.0, posinf=0.0, neginf=0.0)
        kernel_diag_initial = torch.diag(kernel_q).clone().clamp_min(0.0)
        residual = kernel_diag_initial.clone()
        selected = []
        chol_columns = []
        measure_prior_mode = self.qfid_measure_prior_mode
        if measure_prior_mode not in {"none", "soft"}:
            measure_prior_mode = "none"
        measure_prior_lambda = max(0.0, float(self.qfid_measure_prior_lambda))
        p_norm = probs - probs.min()
        p_norm = p_norm / p_norm.max().clamp_min(eps)
        p_norm = torch.nan_to_num(p_norm.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        p_norm_min = float(p_norm.min().item()) if p_norm.numel() > 0 else 0.0
        p_norm_max = float(p_norm.max().item()) if p_norm.numel() > 0 else 0.0
        score_min = float("inf")
        score_max = float("-inf")
        anchor_mode = self.qfid_anchor_mode
        if anchor_mode not in {"none", "measurement"}:
            anchor_mode = "none"
        anchor_idx = torch.empty(0, dtype=torch.long, device=device)

        if anchor_mode == "measurement" and K > 0 and num_tokens > 0:
            anchor_num = int(round(float(K) * float(self.qfid_measure_anchor_ratio)))
            anchor_num = max(int(self.qfid_measure_anchor_min), anchor_num)
            anchor_num = max(0, min(anchor_num, int(K), int(num_tokens)))
            if anchor_num > 0:
                anchor_idx = torch.topk(probs, k=anchor_num).indices

        def apply_qf_pivot(pivot):
            pivot_residual = residual[pivot].clamp_min(eps)
            if not chol_columns:
                new_col = kernel_q[:, pivot] / pivot_residual.sqrt()
            else:
                prev = torch.stack(chol_columns, dim=1)
                correction = prev @ prev[pivot, :]
                new_col = (kernel_q[:, pivot] - correction) / pivot_residual.sqrt()
            new_col_local = torch.nan_to_num(new_col.float(), nan=0.0, posinf=0.0, neginf=0.0)
            chol_columns.append(new_col_local)
            selected.append(int(pivot))
            return (residual - new_col_local.pow(2)).clamp_min(0.0)

        for pivot in anchor_idx.tolist():
            if pivot in selected:
                continue
            residual = apply_qf_pivot(int(pivot))

        objective_audit_mode = os.environ.get(
            "EC_QFICR_OBJECTIVE_AUDIT_MODE", "legacy_current"
        ).strip().lower()
        valid_objective_audit_modes = {
            "legacy_current",
            "direct_task_relu_square",
            "direct_task_relu_cosine",
            "direct_uniform_relu_cosine",
        }
        if objective_audit_mode not in valid_objective_audit_modes:
            raise ValueError(f"unsupported EC_QFICR_OBJECTIVE_AUDIT_MODE={objective_audit_mode}")
        direct_objective_audit = objective_audit_mode != "legacy_current"

        relation_audit_mode = os.environ.get(
            "EC_QFICR_RELATION_AUDIT_MODE", "legacy_current"
        ).strip().lower()
        valid_relation_audit_modes = {
            "legacy_current",
            "relation_direct",
            "relation_restore",
        }
        if relation_audit_mode not in valid_relation_audit_modes:
            raise ValueError(f"unsupported EC_QFICR_RELATION_AUDIT_MODE={relation_audit_mode}")
        if direct_objective_audit and relation_audit_mode != "legacy_current":
            raise ValueError("objective and relation audit modes cannot both be active")

        relation_direct_audit = relation_audit_mode == "relation_direct"
        qfi_order_target = 0 if direct_objective_audit or relation_direct_audit else max(0, int(K))
        if (
            self.qfid_selector == "adaptive_core_recover"
            and os.environ.get("EC_QFICR_RESIDUAL_ALLOC_MODE", "legacy").strip().lower()
            == "competitive_balanced"
        ):
            balanced_ratio_cap = min(
                max(_env_float("EC_QFICR_RECOVER_RATIO_CAP", "0.25"), 0.0),
                1.0,
            )
            qfi_order_target += max(int(round(float(K) * balanced_ratio_cap)), 0)
        steps = min(qfi_order_target, num_tokens)
        while len(selected) < steps:
            masked_residual = residual.clone()
            if selected:
                masked_residual[torch.tensor(selected, device=device, dtype=torch.long)] = float("-inf")
            effective_score = masked_residual
            if measure_prior_mode == "soft":
                effective_score = masked_residual * (1.0 + measure_prior_lambda * p_norm)
            finite_score = effective_score[torch.isfinite(effective_score)]
            if finite_score.numel() > 0:
                score_min = min(score_min, float(finite_score.min().item()))
                score_max = max(score_max, float(finite_score.max().item()))
            pivot = int(torch.argmax(effective_score).item())
            pivot_value = float(effective_score[pivot].item())
            if not math.isfinite(pivot_value) or pivot_value <= eps:
                break

            residual = apply_qf_pivot(pivot)

        if len(selected) < steps:
            remaining_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            if selected:
                remaining_mask[torch.tensor(selected, device=device, dtype=torch.long)] = False
            remaining_idx = torch.nonzero(remaining_mask, as_tuple=False).reshape(-1)
            if remaining_idx.numel() > 0:
                filler_scores = kernel_diag_initial[remaining_idx]
                filler_order = torch.argsort(filler_scores, descending=True)
                filler_idx = remaining_idx[filler_order]
                needed = steps - len(selected)
                selected.extend(filler_idx[:needed].tolist())

        if len(selected) < steps:
            all_idx = torch.arange(num_tokens, device=device)
            remaining_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            if selected:
                remaining_mask[torch.tensor(selected, device=device, dtype=torch.long)] = False
            filler_idx = all_idx[remaining_mask]
            needed = steps - len(selected)
            selected.extend(filler_idx[:needed].tolist())

        keep_idx = torch.tensor(selected, dtype=torch.long, device=device)
        if keep_idx.numel() > K:
            keep_idx = keep_idx[:K]
        keep_idx = torch.unique(keep_idx, sorted=True)
        if keep_idx.numel() < K:
            remaining_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            remaining_mask[keep_idx] = False
            filler_idx = torch.nonzero(remaining_mask, as_tuple=False).reshape(-1)
            filler_idx = filler_idx[: K - keep_idx.numel()]
            keep_idx = torch.unique(torch.cat([keep_idx, filler_idx]), sorted=True)
        if keep_idx.numel() > K:
            keep_idx = keep_idx[:K].sort().values

        selector_used = self.qfid_selector
        if selector_used == "density_projective_core_recover":
            selector_used = "qsp_cr_density_projective"
        qfi_residual_order = torch.tensor(selected, dtype=torch.long, device=device)
        core_ratio_used = float(self.qfid_core_ratio)
        core_size = 0
        recover_size = 0
        adaptive_info = {
            "adapt_mode": "none",
            "question_type": "none",
            "entropy_norm": 0.0,
            "recover_ratio": 0.0,
            "k_core": 0,
            "k_recover": 0,
            "recover_prior_anchor": False,
            "recover_prior_gamma": 0.0,
            "anchor_alpha": 0.0,
            "recover_cand_pool": False,
            "recover_cand_mult": 0.0,
            "recover_cand_size": 0,
            "question_text_available": False,
            "restoration_mode": "full",
            "fixed_restoration_ratio": 0.2,
            "use_spatial_local_prior": False,
            "prior_ablation": "none",
            "prior_lambda": 0.0,
            "random_seed": 42,
            "spatial_prior_mean": 0.0,
            "local_prior_mean": 0.0,
            "spatial_local_prior_mean": 0.0,
            "restore_extra_prior": "none",
            "restore_extra_lambda": 0.0,
            "subspace_residual_mode": "fast",
            "restore_extra_enabled": False,
            "restore_extra_mean": 0.0,
            "restore_extra_std": 0.0,
            "restore_extra_min": 0.0,
            "restore_extra_max": 0.0,
            "restore_extra_norm_mean": 0.0,
            "restore_extra_base_corr": 0.0,
            "restore_extra_mean_g_res": 0.0,
        }
        qmo_cr_info = {
            "qmo_cr_enabled": False,
            "question_type": "none",
            "entropy": 0.0,
            "purity": 0.0,
            "D_q": 0.0,
            "use_purity": False,
            "lambda_h": 0.0,
            "lambda_p": 0.0,
            "d_tau": 0.0,
            "d_eta": 0.0,
            "recover_ratio": 0.0,
            "k_core": 0,
            "k_recover": 0,
            "w_cov": 0.0,
            "w_h": 0.0,
            "lambda_j": 0.0,
            "coupling_reduce": "mean",
            "recover_alpha_prior": 0.0,
            "recover_gamma": 0.0,
            "h_min": 0.0,
            "h_max": 0.0,
            "h_mean": 0.0,
            "J_min": 0.0,
            "J_max": 0.0,
            "J_mean": 0.0,
            "core_indices": [],
            "recovery_indices": [],
            "selected_indices": [],
        }
        qmo_gated_info = {
            "gate_enabled": False,
            "gate_type": "default",
            "gate_mode": "none",
            "gate_types": [],
            "gate_beta": 0.0,
            "baseline_indices": [],
            "qmo_cr_indices": [],
            "selected_indices": [],
            "baseline_overlap": 1.0,
            "qmo_cr_overlap": 0.0,
            "core_indices": [],
            "recovery_indices": [],
        }
        qsp_info = {
            "qsp_enabled": False,
            "qsp_fallback": False,
            "qsp_fallback_reason": "",
            "qsp_core_selection": "none",
            "qsp_recovery": "none",
            "qsp_use_spatial_local_prior": False,
            "qsp_lambda_b": 0.0,
            "qsp_candidate_topm": 0,
            "qsp_candidate_size": 0,
            "qsp_k_core": 0,
            "qsp_k_recover": 0,
            "qsp_trace_res": 0.0,
            "qsp_delta_max": 0.0,
            "qsp_delta_mean": 0.0,
            "qsp_entropy_g_perp": 0.0,
            "qsp_mean_residual_norm_core": 0.0,
            "qsp_task_anchor_alpha": 0.0,
            "qsp_complement_bonus_eta": 0.0,
            "qsp_selector_label": "none",
            "mean_p_core": 0.0,
            "mean_p_recovery": 0.0,
            "mean_p_final": 0.0,
            "overlap_core_top_p": 0.0,
            "overlap_final_top_p": 0.0,
            "num_low_p_high_delta_selected": 0,
            "core_indices": [],
            "recovery_indices": [],
            "selected_indices": [],
        }
        qfi_cards_info = {
            "profile": "none",
            "k": int(min(max(0, int(K)), num_tokens)),
            "num_tokens": int(num_tokens),
            "k_core": 0,
            "k_rec": 0,
            "final_indices": [],
            "core_indices": [],
            "rec_indices": [],
            "mean_p_final": 0.0,
            "mean_p_core": 0.0,
            "mean_p_rec": 0.0,
            "fallback_used": False,
            "fallback_reason": "",
        }
        evidence_coverage_min = 0.0
        evidence_coverage_max = 0.0
        evidence_coverage_mean = 0.0
        evidence_uncovered_mass = 0.0
        if selector_used == "evidence_recover":
            keep_idx = select_by_evidence_recovery(features, probs, K, eps=eps)
            core_ratio_used = 0.0
            recover_size = int(keep_idx.numel())
        elif selector_used == "core_then_recover":
            keep_idx = select_by_core_then_recover(
                states=features,
                probs=probs,
                k=K,
                qfi_core_indices=qfi_residual_order,
                core_ratio=core_ratio_used,
                eps=eps,
            )
            core_size = min(max(int(round(min(K, num_tokens) * core_ratio_used)), 0), min(K, num_tokens))
            core_size = min(core_size, int(keep_idx.numel()))
            recover_size = max(0, int(keep_idx.numel()) - core_size)
        elif selector_used == "adaptive_core_recover":
            if direct_objective_audit:
                keep_idx, adaptive_info = select_by_direct_coverage_audit(
                    states=features,
                    probs=probs,
                    k=K,
                    audit_mode=objective_audit_mode,
                    eps=eps,
                )
            elif relation_audit_mode == "relation_direct":
                keep_idx, adaptive_info, _ = select_by_task_relation_structure(
                    states=features,
                    probs=probs,
                    k=K,
                    mode="relation_direct",
                    initial_indices=None,
                    eps=eps,
                )
            elif relation_audit_mode == "relation_restore":
                _, baseline_info = select_by_adaptive_core_recover(
                    states=features,
                    probs=probs,
                    k=K,
                    qfi_core_order=qfi_residual_order,
                    question_text=question,
                    cls_prior=cls_prior_for_restore,
                    eps=eps,
                )
                baseline_core = baseline_info.get("fixed_core_indices", [])
                keep_idx, relation_info, relation_gram = select_by_task_relation_structure(
                    states=features,
                    probs=probs,
                    k=K,
                    mode="relation_restore",
                    initial_indices=baseline_core,
                    eps=eps,
                )
                _, baseline_relation_trace = _relation_residual_trace_from_gram(
                    relation_gram,
                    baseline_info.get("final_selected_indices", []),
                    eps=eps,
                )
                baseline_initial = baseline_relation_trace[0] if baseline_relation_trace else 0.0
                baseline_final = baseline_relation_trace[-1] if baseline_relation_trace else 0.0
                adaptive_info = dict(baseline_info)
                adaptive_info.update(relation_info)
                adaptive_info.update({
                    "recover_ratio": baseline_info["recover_ratio"],
                    "recover_ratio_requested": baseline_info.get("recover_ratio_requested", baseline_info["recover_ratio"]),
                    "recover_ratio_min": baseline_info.get("recover_ratio_min", 0.0),
                    "recover_ratio_max": baseline_info.get("recover_ratio_max", 0.0),
                    "recover_ratio_cap": baseline_info.get("recover_ratio_cap", 0.0),
                    "recover_abs_cap": baseline_info.get("recover_abs_cap", 0),
                    "recover_raw_k": baseline_info.get("recover_raw_k", len(relation_info["selected_residual_indices"])),
                    "recover_cap_applied": baseline_info.get("recover_cap_applied", False),
                    "recover_entropy_hat": baseline_info.get("recover_entropy_hat", 0.0),
                    "recover_entropy_gamma": baseline_info.get("recover_entropy_gamma", 1.0),
                    "recover_budget_mode": baseline_info.get("recover_budget_mode", "ratio_cap"),
                    "ratio_cap_active": True,
                    "legacy_relation_reconstruction_residual": float(
                        baseline_final / max(baseline_initial, max(float(eps), 1e-12))
                    ),
                    "legacy_relation_reconstruction_residual_raw": baseline_final,
                    "legacy_final_selected_indices": baseline_info.get("final_selected_indices", []),
                })
            else:
                keep_idx, adaptive_info = select_by_adaptive_core_recover(
                    states=features,
                    probs=probs,
                    k=K,
                    qfi_core_order=qfi_residual_order,
                    question_text=question,
                    cls_prior=cls_prior_for_restore,
                    eps=eps,
                    core_indices_override=core_indices_override,
                )
            core_ratio_used = 1.0 - float(adaptive_info["recover_ratio"])
            core_size = int(adaptive_info["k_core"])
            recover_size = int(adaptive_info["k_recover"])
        elif selector_used in {
            "qfi_zeno_stabilized",
            "qfi_spatial_buffer_recovery",
            "qfi_progressive_128_to_64",
            "qfi_coarse_to_fine_pruning",
        }:
            try:
                if selector_used == "qfi_zeno_stabilized":
                    keep_idx, adaptive_info, qfi_cards_info = select_by_qfi_zeno_stabilized(
                        states=features,
                        probs=probs,
                        k=K,
                        qfi_core_order=qfi_residual_order,
                        question_text=question,
                        attn_obs=cls_attn,
                        eps=eps,
                    )
                elif selector_used == "qfi_spatial_buffer_recovery":
                    keep_idx, adaptive_info, qfi_cards_info = select_by_qfi_spatial_buffer_recovery(
                        states=features,
                        probs=probs,
                        k=K,
                        qfi_core_order=qfi_residual_order,
                        question_text=question,
                        eps=eps,
                    )
                elif selector_used == "qfi_progressive_128_to_64":
                    keep_idx, adaptive_info, qfi_cards_info = select_by_qfi_progressive(
                        states=features,
                        probs=probs,
                        k=K,
                        qfi_core_order=qfi_residual_order,
                        question_text=question,
                        eps=eps,
                    )
                else:
                    keep_idx, adaptive_info, qfi_cards_info = select_by_qfi_coarse_to_fine(
                        states=features,
                        probs=probs,
                        k=K,
                        qfi_core_order=qfi_residual_order,
                        question_text=question,
                        coarse_score=None,
                        eps=eps,
                    )
            except (RuntimeError, ValueError, TypeError) as exc:
                keep_idx, adaptive_info = select_by_adaptive_core_recover(
                    states=features,
                    probs=probs,
                    k=K,
                    qfi_core_order=qfi_residual_order,
                    question_text=question,
                    eps=eps,
                )
                qfi_cards_info = _qfi_cards_base_record(
                    selector_used,
                    keep_idx,
                    adaptive_info,
                    probs,
                    num_tokens,
                    fallback_used=True,
                    fallback_reason=str(exc),
                )
            qfi_cards_info["profile"] = selector_used
            core_ratio_used = 1.0 - float(adaptive_info["recover_ratio"])
            core_size = int(adaptive_info["k_core"])
            recover_size = int(adaptive_info["k_recover"])
        elif selector_used == "qmo_cr":
            if visual_tokens.shape[0] != 1:
                raise ValueError(
                    "EC_QFID_SELECTOR=qmo_cr currently requires batch size 1 to avoid mixing "
                    "batch-mean QFi probabilities with per-sample visual states."
                )
            qmo_states = visual_tokens[0].float()
            keep_idx, qmo_cr_info = select_by_qmo_cr(
                states=qmo_states,
                probs=probs,
                k=K,
                question_text=question,
                eps=eps,
            )
            core_ratio_used = 1.0 - float(qmo_cr_info["recover_ratio"])
            core_size = int(qmo_cr_info["k_core"])
            recover_size = int(qmo_cr_info["k_recover"])
            adaptive_info.update({
                "adapt_mode": "qmo_cr",
                "question_type": qmo_cr_info["question_type"],
                "entropy_norm": qmo_cr_info["entropy"],
                "recover_ratio": qmo_cr_info["recover_ratio"],
                "k_core": qmo_cr_info["k_core"],
                "k_recover": qmo_cr_info["k_recover"],
                "recover_prior_anchor": True,
                "recover_prior_gamma": qmo_cr_info["recover_gamma"],
                "recover_cand_pool": False,
                "recover_cand_mult": 0.0,
                "recover_cand_size": max(0, num_tokens - qmo_cr_info["k_core"]),
                "question_text_available": isinstance(question, str) and bool(question.strip()),
            })
        elif selector_used == "qmo_gated":
            if visual_tokens.shape[0] != 1:
                raise ValueError(
                    "EC_QFID_SELECTOR=qmo_gated currently requires batch size 1 to avoid mixing "
                    "batch-mean QFi probabilities with per-sample visual states."
                )
            qmo_states = visual_tokens[0].float()
            keep_idx, qmo_gated_info = select_by_qmo_gated(
                states=features,
                probs=probs,
                k=K,
                qfi_core_order=qfi_residual_order,
                question_text=question,
                qmo_states=qmo_states,
                eps=eps,
            )
            adaptive_info = dict(qmo_gated_info["baseline_info"])
            qmo_cr_info = dict(qmo_gated_info["qmo_cr_info"])
            core_ratio_used = 1.0 - float(adaptive_info["recover_ratio"])
            core_size = int(adaptive_info["k_core"])
            recover_size = int(adaptive_info["k_recover"])
        elif selector_used in {
            "qsp_cr_density_projective",
            "qsp_density_projective_task_anchor",
            "qsp_density_projective_task_anchor_a05",
            "qsp_density_projective_complement_bonus",
        }:
            try:
                qsp_states = visual_tokens[0].float() if visual_tokens.shape[0] == 1 else features
                os.environ["EC_QSP_SELECTOR_LABEL"] = selector_used
                keep_idx, qsp_info = select_by_qsp_density_projective(
                    states=qsp_states,
                    probs=probs,
                    k=K,
                    qfi_core_order=qfi_residual_order,
                    eps=eps,
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                qsp_info = dict(qsp_info)
                qsp_info.update({
                    "qsp_enabled": True,
                    "qsp_fallback": True,
                    "qsp_fallback_reason": str(exc)[:240],
                    "selected_indices": [int(x) for x in keep_idx.tolist()],
                    "qsp_selector_label": selector_used,
                })
                keep_idx = _qsp_finalize_indices([int(x) for x in qfi_residual_order.tolist()], K, num_tokens, device)
            core_size = int(qsp_info["qsp_k_core"])
            recover_size = int(qsp_info["qsp_k_recover"])
            core_ratio_used = float(core_size / max(1, int(keep_idx.numel())))
            adaptive_info.update({
                "adapt_mode": "qsp_density_projective",
                "question_type": "none",
                "entropy_norm": prob_info["entropy_norm"],
                "recover_ratio": float(recover_size / max(1, int(keep_idx.numel()))),
                "k_core": core_size,
                "k_recover": recover_size,
                "recover_prior_anchor": qsp_info["qsp_recovery"] == "prior_recovery",
                "recover_prior_gamma": 0.0,
                "recover_cand_pool": qsp_info["qsp_candidate_topm"] > 0,
                "recover_cand_mult": 0.0,
                "recover_cand_size": qsp_info["qsp_candidate_size"],
                "question_text_available": isinstance(question, str) and bool(question.strip()),
            })
        elif selector_used == "qfi_core_complement_recovery":
            keep_idx, qsp_info = select_by_qfi_core_complement_recovery(
                states=features,
                probs=probs,
                k=K,
                qfi_core_order=qfi_residual_order,
                question_text=question,
                eps=eps,
            )
            core_size = int(qsp_info["qsp_k_core"])
            recover_size = int(qsp_info["qsp_k_recover"])
            core_ratio_used = float(core_size / max(1, int(keep_idx.numel())))
            adaptive_info.update({
                "adapt_mode": "qfi_core_complement_recovery",
                "question_type": "none",
                "entropy_norm": prob_info["entropy_norm"],
                "recover_ratio": float(recover_size / max(1, int(keep_idx.numel()))),
                "k_core": core_size,
                "k_recover": recover_size,
                "recover_prior_anchor": False,
                "recover_prior_gamma": 0.0,
                "recover_cand_pool": False,
                "recover_cand_mult": 0.0,
                "recover_cand_size": qsp_info["qsp_candidate_size"],
                "question_text_available": isinstance(question, str) and bool(question.strip()),
            })
        if selector_used in {
            "evidence_recover",
            "core_then_recover",
            "adaptive_core_recover",
            "qmo_cr",
            "qmo_gated",
            "qsp_cr_density_projective",
            "qsp_density_projective_task_anchor",
            "qsp_density_projective_task_anchor_a05",
            "qsp_density_projective_complement_bonus",
            "qfi_zeno_stabilized",
            "qfi_spatial_buffer_recovery",
            "qfi_progressive_128_to_64",
            "qfi_coarse_to_fine_pruning",
            "qfi_core_complement_recovery",
        }:
            selected = [int(index) for index in keep_idx.tolist()]
            if keep_idx.numel() > 0:
                evidence_kernel = compute_projection_overlap_kernel(
                    psi,
                    eps=eps,
                    overlap_kernel=self.qfid_overlap_kernel,
                    assume_normalized=True,
                )
                coverage = evidence_kernel[:, keep_idx].max(dim=1).values
                coverage = torch.nan_to_num(coverage.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
                uncovered = probs.float() * (1.0 - coverage)
                evidence_coverage_min = float(coverage.min().item())
                evidence_coverage_max = float(coverage.max().item())
                evidence_coverage_mean = float(coverage.mean().item())
                evidence_uncovered_mass = float(uncovered.sum().item())
            keep_idx = torch.unique(keep_idx, sorted=True)
            if keep_idx.numel() > K:
                keep_idx = keep_idx[:K].sort().values

        residual_initial_sum = float(kernel_diag_initial.sum().item())
        residual_final_sum = float(residual.sum().item())
        if selector_used in {
            "evidence_recover",
            "core_then_recover",
            "adaptive_core_recover",
            "qmo_cr",
            "qmo_gated",
            "qsp_cr_density_projective",
            "qsp_density_projective_task_anchor",
            "qsp_density_projective_task_anchor_a05",
            "qsp_density_projective_complement_bonus",
            "qfi_zeno_stabilized",
            "qfi_spatial_buffer_recovery",
            "qfi_progressive_128_to_64",
            "qfi_coarse_to_fine_pruning",
            "qfi_core_complement_recovery",
        }:
            residual_initial_sum = float(probs.sum().item())
            residual_final_sum = evidence_uncovered_mass
        explained_ratio = 0.0
        if residual_initial_sum > eps:
            explained_ratio = max(0.0, min(1.0, 1.0 - residual_final_sum / residual_initial_sum))
        if not math.isfinite(score_min):
            score_min = 0.0
        if not math.isfinite(score_max):
            score_max = 0.0

        anchor_prob_stats = {
            "anchor_prob_min": 0.0,
            "anchor_prob_max": 0.0,
            "anchor_prob_mean": 0.0,
        }
        if anchor_idx.numel() > 0:
            anchor_probs = probs[anchor_idx]
            anchor_prob_stats = {
                "anchor_prob_min": float(anchor_probs.min().item()),
                "anchor_prob_max": float(anchor_probs.max().item()),
                "anchor_prob_mean": float(anchor_probs.mean().item()),
            }
        anchor_num = int(anchor_idx.numel())
        qf_fill_count = max(0, int(keep_idx.numel()) - anchor_num)

        # Optional, disabled-by-default diagnostic/modulation input for P1-LR.
        # This replays only the frozen QFi pivoted-Cholesky residual through the
        # requested semantic prefix; it does not alter Frozen V0 ordering.
        p1lr_semantic_residual_values = None
        p1lr_alpha = float(os.environ.get("EC_QFICR_WEAK_SEMANTIC_ALPHA", "0.0"))
        p1lr_mode = os.environ.get("EC_QFICR_LOCAL_RESIDUAL_MODE", "").strip().lower()
        if p1lr_alpha > 0.0 and p1lr_mode in {
            "p1_lr_semantic_core_conditioned_local_residual",
            "p1_tc_semantic_core_conditioned_local_residual",
            "p1_lr_entropy_core_conditioned_local_residual",
        }:
            sem_count = max(
                0,
                min(
                    int(num_tokens),
                    int(os.environ.get("EC_QFICR_P1TC_K_SEMANTIC", "26")),
                ),
            )
            sem_order = [int(x) for x in qfi_order[:sem_count].tolist()]
            sem_residual = kernel_diag_initial.clone()
            sem_columns = []
            for pivot in sem_order:
                denom = sem_residual[pivot].clamp_min(eps).sqrt()
                if sem_columns:
                    previous = torch.stack(sem_columns, dim=1)
                    sem_column = (kernel_q[:, pivot] - previous @ previous[pivot, :]) / denom
                else:
                    sem_column = kernel_q[:, pivot] / denom
                sem_column = torch.nan_to_num(sem_column.float(), nan=0.0, posinf=0.0, neginf=0.0)
                sem_columns.append(sem_column)
                sem_residual = (sem_residual - sem_column.square()).clamp_min(0.0)
            p1lr_semantic_residual_values = [float(x) for x in sem_residual.detach().cpu().tolist()]

        self.last_qfid_info = {
            "enabled": True,
            "num_tokens": int(num_tokens),
            "token_norm_values": [float(x) for x in features.norm(dim=-1).detach().cpu().tolist()] if get_env_bool("EC_QFID_DEBUG_DUMP_PROBS", default=False) else [],
            "select_mode": "qf",
            "selector": selector_used,
            "core_ratio": core_ratio_used,
            "core_size": core_size,
            "recover_size": recover_size,
            "adapt_mode": adaptive_info["adapt_mode"],
            "qficr_alloc_mode": adaptive_info.get("alloc_mode", "frozen"),
            "question_type": adaptive_info["question_type"],
            "adapt_entropy_norm": adaptive_info["entropy_norm"],
            "adapt_recover_ratio": adaptive_info["recover_ratio"],
            "adapt_k_core": adaptive_info["k_core"],
            "adapt_k_recover": adaptive_info["k_recover"],
            "qficr_budget_mode": adaptive_info.get("recover_budget_mode", "abs_cap"),
            "qficr_rho_min": adaptive_info.get("recover_ratio_min", adaptive_info.get("ratio_min", 0.0)),
            "qficr_rho_max": adaptive_info.get("recover_ratio_max", adaptive_info.get("ratio_max", 0.25)),
            "qficr_entropy": adaptive_info.get("entropy", prob_info["entropy"]),
            "qficr_entropy_norm": adaptive_info.get("entropy_norm", prob_info["entropy_norm"]),
            "qficr_entropy_hat": adaptive_info.get("recover_entropy_hat", adaptive_info.get("entropy_norm", 0.0)),
            "qficr_entropy_gamma": adaptive_info.get("recover_entropy_gamma", 1.0),
            "qficr_rho": adaptive_info.get("recover_ratio_requested", adaptive_info["recover_ratio"]),
            "qficr_raw_k_rest": adaptive_info.get("recover_raw_k", adaptive_info["k_recover"]),
            "qficr_final_k_rest": adaptive_info["k_recover"],
            "qficr_final_k_red": adaptive_info["k_core"],
            "qficr_abs_cap": adaptive_info.get("recover_abs_cap", 16),
            "qficr_ratio_cap": adaptive_info.get("recover_ratio_cap", 0.25),
            "qficr_cap_applied": adaptive_info.get("recover_cap_applied", False),
            "qficr_yes_no_gate_applied": adaptive_info.get("yn_gate_applied", False),
            "qficr_recovery_score_mode": adaptive_info.get("recovery_score_mode", "frozen"),
            "qficr_residual_alloc_mode": adaptive_info.get("residual_alloc_mode", "legacy"),
            "qficr_compete_margin": adaptive_info.get("compete_margin", 0.0),
            "qficr_max_flexible_slots": adaptive_info.get("max_flexible_slots", adaptive_info["k_recover"]),
            "qficr_K_base": adaptive_info.get("K_base", adaptive_info.get("fixed_core_count", adaptive_info["k_core"])),
            "qficr_K_rest_max": adaptive_info.get("K_rest_max", adaptive_info.get("max_flexible_slots", adaptive_info["k_recover"])),
            "qficr_fixed_core_count": adaptive_info.get("fixed_core_count", adaptive_info["k_core"]),
            "qficr_actual_residual_count": adaptive_info.get("actual_residual_count", adaptive_info["k_recover"]),
            "qficr_actual_core_count": adaptive_info.get("actual_core_count", adaptive_info["k_core"]),
            "qficr_candidate_pool_enabled": adaptive_info.get("candidate_pool_enabled", adaptive_info["recover_cand_pool"]),
            "qficr_candidate_pool_size": adaptive_info.get("candidate_pool_size", adaptive_info["recover_cand_size"]),
            "qficr_candidate_pool_multiplier": adaptive_info.get("candidate_pool_multiplier", adaptive_info["recover_cand_mult"]),
            "qficr_external_candidate_pool_enabled": adaptive_info.get(
                "external_candidate_pool_enabled", adaptive_info["recover_cand_pool"]
            ),
            "qficr_external_candidate_pool_size": adaptive_info.get(
                "external_candidate_pool_size", adaptive_info["recover_cand_size"]
            ),
            "qficr_external_candidate_pool_multiplier": adaptive_info.get(
                "external_candidate_pool_multiplier", adaptive_info["recover_cand_mult"]
            ),
            "qficr_balanced_pool_overrides_external_pool": adaptive_info.get(
                "balanced_pool_overrides_external_pool", False
            ),
            "qficr_balanced_pool_size_requested": adaptive_info.get(
                "balanced_pool_size_requested", 0
            ),
            "qficr_initial_core_pool_size": adaptive_info.get("initial_core_pool_size", 0),
            "qficr_initial_residual_pool_size": adaptive_info.get("initial_residual_pool_size", 0),
            "qficr_remaining_core_pool_size_by_step": adaptive_info.get(
                "remaining_core_pool_size_by_step", []
            ),
            "qficr_remaining_residual_pool_size_by_step": adaptive_info.get(
                "remaining_residual_pool_size_by_step", []
            ),
            "qficr_balanced_core_pool_indices": adaptive_info.get(
                "balanced_core_pool_indices", []
            ),
            "qficr_balanced_residual_pool_indices": adaptive_info.get(
                "balanced_residual_pool_indices", []
            ),
            "qficr_full_core_indices": adaptive_info.get("full_core_indices", []),
            "qficr_fixed_core_indices": adaptive_info.get("fixed_core_indices", []),
            "qficr_base_core_indices": adaptive_info.get("base_core_indices", adaptive_info.get("fixed_core_indices", [])),
            "qficr_reserved_core_indices": adaptive_info.get("reserved_core_indices", []),
            "qficr_selected_core_indices": adaptive_info.get("selected_core_indices", []),
            "qficr_selected_residual_indices": adaptive_info.get("selected_residual_indices", []),
            "qficr_fallback_core_indices": adaptive_info.get("fallback_core_indices", []),
            "qficr_final_selected_indices": adaptive_info.get("final_selected_indices", []),
            "qficr_residual_utilization": adaptive_info.get("residual_utilization", 0.0),
            "qficr_fallback_to_core_rate": adaptive_info.get("fallback_to_core_rate", 0.0),
            "qficr_competition_steps": adaptive_info.get("competition_steps", []),
            "qficr_trainfree_restoration_mode": adaptive_info.get("trainfree_restoration_mode", "legacy"),
            "qficr_K_rest_cap": adaptive_info.get("K_rest_cap", adaptive_info.get("k_recover", 0)),
            "qficr_actual_restoration_count": adaptive_info.get("actual_restoration_count", adaptive_info.get("actual_residual_count", adaptive_info.get("k_recover", 0))),
            "qficr_core_continuation_count": adaptive_info.get("core_continuation_count", 0),
            "qficr_restoration_step": adaptive_info.get("restoration_step", 0),
            "qficr_restoration_candidate_index": adaptive_info.get("restoration_candidate_index"),
            "qficr_core_candidate_index": adaptive_info.get("core_candidate_index"),
            "qficr_duplicate_count": adaptive_info.get("duplicate_count", 0),
            "qficr_nan_inf_count": adaptive_info.get("nan_inf_count", 0),
            "recover_prior_anchor": adaptive_info["recover_prior_anchor"],
            "recover_prior_gamma": adaptive_info["recover_prior_gamma"],
            "recover_cand_pool": adaptive_info["recover_cand_pool"],
            "recover_cand_mult": adaptive_info["recover_cand_mult"],
            "recover_cand_size": adaptive_info["recover_cand_size"],
            "question_text_available": adaptive_info["question_text_available"],
            "qficr_restoration_mode": adaptive_info.get("restoration_mode", "full"),
            "qficr_fixed_restoration_ratio": adaptive_info.get("fixed_restoration_ratio", 0.2),
            "qficr_use_spatial_local_prior": adaptive_info.get("use_spatial_local_prior", False),
            "qficr_prior_ablation": adaptive_info.get("prior_ablation", "none"),
            "qficr_prior_lambda": adaptive_info.get("prior_lambda", 0.0),
            "qficr_anchor_alpha": adaptive_info.get("anchor_alpha", adaptive_info.get("recover_prior_gamma", 0.0)),
            "qficr_support_beta": adaptive_info.get("support_beta", 0.0),
            "qficr_self_discount": adaptive_info.get("self_discount", 0.0),
            "qficr_mean_p_core": adaptive_info.get("mean_p_core"),
            "qficr_mean_p_rest": adaptive_info.get("mean_p_rest"),
            "qficr_median_p_rest": adaptive_info.get("median_p_rest"),
            "qficr_mean_residual_gain_rest": adaptive_info.get("mean_residual_gain_rest"),
            "qficr_median_residual_gain_rest": adaptive_info.get("median_residual_gain_rest"),
            "qficr_mean_max_overlap_to_core_rest": adaptive_info.get("mean_max_overlap_to_core_rest"),
            "qficr_median_max_overlap_to_core_rest": adaptive_info.get("median_max_overlap_to_core_rest"),
            "qficr_mean_qfi_rank_rest": adaptive_info.get("mean_qfi_rank_rest"),
            "qficr_median_qfi_rank_rest": adaptive_info.get("median_qfi_rank_rest"),
            "qficr_near_boundary_rank_threshold": adaptive_info.get("near_boundary_rank_threshold"),
            "qficr_near_boundary_rank_hit_rate": adaptive_info.get("near_boundary_rank_hit_rate"),
            "qficr_mean_p_discarded": adaptive_info.get("mean_p_discarded"),
            "qficr_median_p_discarded": adaptive_info.get("median_p_discarded"),
            "qficr_mean_residual_gain_discarded": adaptive_info.get("mean_residual_gain_discarded"),
            "qficr_median_residual_gain_discarded": adaptive_info.get("median_residual_gain_discarded"),
            "qficr_mean_max_overlap_to_core_discarded": adaptive_info.get("mean_max_overlap_to_core_discarded"),
            "qficr_median_max_overlap_to_core_discarded": adaptive_info.get("median_max_overlap_to_core_discarded"),
            "qficr_mean_qfi_rank_discarded": adaptive_info.get("mean_qfi_rank_discarded"),
            "qficr_median_qfi_rank_discarded": adaptive_info.get("median_qfi_rank_discarded"),
            "qficr_mean_support_rest": adaptive_info.get("mean_support_rest"),
            "qficr_mean_self_gain_rest": adaptive_info.get("mean_self_gain_rest"),
            "qficr_mean_restoration_score_rest": adaptive_info.get("mean_restoration_score_rest"),
            "qficr_comp_score_mode": adaptive_info.get("comp_score_mode", "current_anchor"),
            "qficr_objective_audit_mode": adaptive_info.get(
                "objective_audit_mode", objective_audit_mode
            ),
            "qficr_relation_audit_mode": adaptive_info.get(
                "relation_audit_mode", relation_audit_mode
            ),
            "qficr_objective_target_weight": adaptive_info.get(
                "objective_target_weight", "task_conditioned"
            ),
            "qficr_objective_overlap_kernel": adaptive_info.get(
                "objective_overlap_kernel", self.qfid_overlap_kernel
            ),
            "qficr_ratio_cap_active": adaptive_info.get(
                "ratio_cap_active", objective_audit_mode == "legacy_current"
            ),
            "qficr_coverage_monotonic": adaptive_info.get("coverage_monotonic"),
            "qficr_coverage_trace": adaptive_info.get("coverage_trace", []),
            "qficr_task_weighted_coverage": adaptive_info.get(
                "task_weighted_coverage", explained_ratio
            ),
            "qficr_uniform_coverage": adaptive_info.get(
                "uniform_coverage", evidence_coverage_mean
            ),
            "qficr_mean_selected_p": adaptive_info.get(
                "mean_selected_p",
                float(probs[keep_idx].mean().item()) if keep_idx.numel() else 0.0,
            ),
            "qficr_affinity_latency_ms": adaptive_info.get("affinity_latency_ms"),
            "qficr_greedy_latency_ms": adaptive_info.get("greedy_latency_ms"),
            "qficr_relation_residual_trace": adaptive_info.get("relation_residual_trace", []),
            "qficr_relation_residual_monotonic": adaptive_info.get("relation_residual_monotonic"),
            "qficr_relation_reconstruction_residual": adaptive_info.get("relation_reconstruction_residual"),
            "qficr_relation_reconstruction_residual_raw": adaptive_info.get("relation_reconstruction_residual_raw"),
            "qficr_legacy_relation_reconstruction_residual": adaptive_info.get("legacy_relation_reconstruction_residual"),
            "qficr_relation_qr_orthogonality_error": adaptive_info.get("relation_qr_orthogonality_error"),
            "qficr_relation_nan_inf_count": adaptive_info.get("relation_nan_inf_count", 0),
            "qficr_legacy_final_selected_indices": adaptive_info.get("legacy_final_selected_indices", []),
            "qficr_random_seed": adaptive_info.get("random_seed", 42),
            "qficr_spatial_prior_mean": adaptive_info.get("spatial_prior_mean", 0.0),
            "qficr_local_prior_mean": adaptive_info.get("local_prior_mean", 0.0),
            "qficr_spatial_local_prior_mean": adaptive_info.get("spatial_local_prior_mean", 0.0),
            "qficr_restore_extra_prior": adaptive_info.get("restore_extra_prior", "none"),
            "qficr_restore_extra_lambda": adaptive_info.get("restore_extra_lambda", 0.0),
            "qficr_subspace_residual_mode": adaptive_info.get("subspace_residual_mode", "fast"),
            "qficr_restore_extra_enabled": adaptive_info.get("restore_extra_enabled", False),
            "qficr_restore_extra_mean": adaptive_info.get("restore_extra_mean", 0.0),
            "qficr_restore_extra_std": adaptive_info.get("restore_extra_std", 0.0),
            "qficr_restore_extra_min": adaptive_info.get("restore_extra_min", 0.0),
            "qficr_restore_extra_max": adaptive_info.get("restore_extra_max", 0.0),
            "qficr_restore_extra_norm_mean": adaptive_info.get("restore_extra_norm_mean", 0.0),
            "qficr_restore_extra_base_corr": adaptive_info.get("restore_extra_base_corr", 0.0),
            "qficr_restore_extra_mean_g_res": adaptive_info.get("restore_extra_mean_g_res", 0.0),
            "qmo_cr_enabled": qmo_cr_info["qmo_cr_enabled"],
            "qmo_entropy": qmo_cr_info["entropy"],
            "qmo_purity": qmo_cr_info["purity"],
            "qmo_D_q": qmo_cr_info["D_q"],
            "qmo_use_purity": qmo_cr_info["use_purity"],
            "qmo_lambda_h": qmo_cr_info["lambda_h"],
            "qmo_lambda_p": qmo_cr_info["lambda_p"],
            "qmo_d_tau": qmo_cr_info["d_tau"],
            "qmo_d_eta": qmo_cr_info["d_eta"],
            "qmo_w_cov": qmo_cr_info["w_cov"],
            "qmo_w_h": qmo_cr_info["w_h"],
            "qmo_lambda_j": qmo_cr_info["lambda_j"],
            "qmo_coupling_reduce": qmo_cr_info["coupling_reduce"],
            "qmo_recover_alpha_prior": qmo_cr_info["recover_alpha_prior"],
            "qmo_recover_gamma": qmo_cr_info["recover_gamma"],
            "qmo_h_min": qmo_cr_info["h_min"],
            "qmo_h_max": qmo_cr_info["h_max"],
            "qmo_h_mean": qmo_cr_info["h_mean"],
            "qmo_J_min": qmo_cr_info["J_min"],
            "qmo_J_max": qmo_cr_info["J_max"],
            "qmo_J_mean": qmo_cr_info["J_mean"],
            "qmo_gate_enabled": qmo_gated_info["gate_enabled"],
            "qmo_gate_type": qmo_gated_info["gate_type"],
            "qmo_gate_mode": qmo_gated_info["gate_mode"],
            "qmo_gate_beta": qmo_gated_info["gate_beta"],
            "qmo_gate_baseline_overlap": qmo_gated_info["baseline_overlap"],
            "qmo_gate_qmo_cr_overlap": qmo_gated_info["qmo_cr_overlap"],
            "qsp_enabled": qsp_info["qsp_enabled"],
            "qsp_fallback": qsp_info["qsp_fallback"],
            "qsp_fallback_reason": qsp_info["qsp_fallback_reason"],
            "qsp_core_selection": qsp_info["qsp_core_selection"],
            "qsp_recovery": qsp_info["qsp_recovery"],
            "qsp_use_spatial_local_prior": qsp_info["qsp_use_spatial_local_prior"],
            "qsp_lambda_b": qsp_info["qsp_lambda_b"],
            "qsp_candidate_topm": qsp_info["qsp_candidate_topm"],
            "qsp_candidate_size": qsp_info["qsp_candidate_size"],
            "qsp_k_core": qsp_info["qsp_k_core"],
            "qsp_k_recover": qsp_info["qsp_k_recover"],
            "qsp_trace_res": qsp_info["qsp_trace_res"],
            "qsp_delta_max": qsp_info["qsp_delta_max"],
            "qsp_delta_mean": qsp_info["qsp_delta_mean"],
            "qsp_entropy_g_perp": qsp_info["qsp_entropy_g_perp"],
            "qsp_mean_residual_norm_core": qsp_info["qsp_mean_residual_norm_core"],
            "qsp_task_anchor_alpha": qsp_info["qsp_task_anchor_alpha"],
            "qsp_complement_bonus_eta": qsp_info["qsp_complement_bonus_eta"],
            "qfi_cards_profile": qfi_cards_info.get("profile", "none"),
            "qfi_cards_fallback": bool(qfi_cards_info.get("fallback_used", False)),
            "qfi_cards_fallback_reason": qfi_cards_info.get("fallback_reason", ""),
            "qfi_cards_mean_p_final": float(qfi_cards_info.get("mean_p_final", 0.0)),
            "qfi_cards_mean_p_core": float(qfi_cards_info.get("mean_p_core", 0.0)),
            "qfi_cards_mean_p_rec": float(qfi_cards_info.get("mean_p_rec", 0.0)),
            "qfi_cards_num_buffer_tokens": int(qfi_cards_info.get("num_buffer_tokens", 0)),
            "qfi_cards_zeno_beta": float(qfi_cards_info.get("zeno_beta", 0.0)),
            "qfi_cards_progressive_mid_k": int(qfi_cards_info.get("mid_k", 0)),
            "qfi_cards_coarse_k": int(qfi_cards_info.get("coarse_k", 0)),
            "prob_source": self.qfid_prob_source,
            "prob_source_used": prob_source_used,
            "tau": self.qfid_tau,
            "kernel": self.qfid_kernel,
            "overlap_kernel": self.qfid_overlap_kernel,
            "budget_calib": bool(self.qfid_budget_calib),
            "budget_alpha": budget_info["budget_alpha"],
            "budget_target_neff": budget_info["budget_target_neff"],
            "budget_neff_before": budget_info["budget_neff_before"],
            "budget_neff_after": budget_info["budget_neff_after"],
            "budget_gamma": budget_info["budget_gamma"],
            "spectral_filter_enabled": spectral_info["spectral_filter_enabled"],
            "spectral_filter_applied": spectral_info["spectral_filter_applied"],
            "spectral_filter_fallback": spectral_info["spectral_filter_fallback"],
            "spectral_gamma": spectral_info["spectral_gamma"],
            "spectral_eps": spectral_info["spectral_eps"],
            "spectral_trace_norm": spectral_info["spectral_trace_norm"],
            "spectral_trace_before": spectral_info["spectral_trace_before"],
            "spectral_trace_after": spectral_info["spectral_trace_after"],
            "spectral_rank_pos": spectral_info["spectral_rank_pos"],
            "spectral_min_eval": spectral_info["spectral_min_eval"],
            "spectral_max_eval": spectral_info["spectral_max_eval"],
            "spatial_state_enabled": spatial_info["spatial_state_enabled"],
            "spatial_state_applied": spatial_info["spatial_state_applied"],
            "spatial_state_fallback": spatial_info["spatial_state_fallback"],
            "spatial_lambda": spatial_info["spatial_lambda"],
            "spatial_sigma": spatial_info["spatial_sigma"],
            "grid_size": spatial_info["grid_size"],
            "spatial_factor_min": spatial_info["spatial_factor_min"],
            "spatial_factor_max": spatial_info["spatial_factor_max"],
            "spatial_factor_mean": spatial_info["spatial_factor_mean"],
            "measure_prior_mode": measure_prior_mode,
            "measure_prior_lambda": measure_prior_lambda,
            "anchor_mode": anchor_mode,
            "anchor_num": anchor_num,
            "qf_fill_count": qf_fill_count,
            "prob_fallback": bool(prob_info["source_fallback"]),
            "measurement_alpha_mode": prob_info["measurement_alpha_mode"],
            "cls_attn_available": prob_info["cls_attn_available"],
            "cls_fallback": prob_info["cls_fallback"],
            "cls_mix_mode": prob_info["cls_mix_mode"],
            "observation_mode": prob_info["observation_mode"],
            "cls_mix_beta": prob_info["cls_mix_beta"],
            "cls_gate_enabled": prob_info["cls_gate_enabled"],
            "cls_gate_mode": prob_info["cls_gate_mode"],
            "agreement": prob_info["agreement"],
            "js_div": prob_info["js_div"],
            "gate": prob_info["gate"],
            "beta_eff": prob_info["beta_eff"],
            "beta_base": prob_info["beta_base"],
            "gate_min": prob_info["gate_min"],
            "gate_max": prob_info["gate_max"],
            "p_sem_entropy": prob_info["p_sem_entropy"],
            "p_cls_entropy": prob_info["p_cls_entropy"],
            "cls_attn_layer": prob_info["cls_attn_layer"],
            "cls_prior_layers": prob_info["cls_prior_layers"],
            "cls_head_reduce": prob_info["cls_head_reduce"],
            "cls_head_entropy_min": prob_info["cls_head_entropy_min"],
            "cls_head_entropy_max": prob_info["cls_head_entropy_max"],
            "cls_head_entropy_mean": prob_info["cls_head_entropy_mean"],
            "cls_head_weight_min": prob_info["cls_head_weight_min"],
            "cls_head_weight_max": prob_info["cls_head_weight_max"],
            "depolarize_mode": prob_info["mode"],
            "depolarize": float(prob_info["adaptive_depolarize"]),
            "depolarize_fixed": float(max(0.0, min(float(self.qfid_depolarize), 1.0))),
            "depolarize_min": float(max(0.0, min(float(self.qfid_depolarize_min), 1.0))),
            "depolarize_max": float(max(0.0, min(float(self.qfid_depolarize_max), 1.0))),
            "candidate_source": "all",
            "candidate_ratio": 1.0,
            "candidate_rel_source": "none",
            "candidate_fallback": "none",
            "candidate_size_before_truncation": int(num_tokens),
            "candidate_size_after_truncation": int(num_tokens),
            "candidate_padded_count": 0,
            "entropy": prob_info["entropy"],
            "entropy_norm": prob_info["entropy_norm"],
            "adaptive_depolarize": prob_info["adaptive_depolarize"],
            "p_min": prob_info["p_min"],
            "p_max": prob_info["p_max"],
            "p_mean": prob_info["p_mean"],
            "p_sum": prob_info["p_sum"],
            "measurement_score_min": prob_info["measurement_score_min"],
            "measurement_score_max": prob_info["measurement_score_max"],
            "measurement_score_mean": prob_info["measurement_score_mean"],
            "measurement_prob_min": prob_info["measurement_prob_min"],
            "measurement_prob_max": prob_info["measurement_prob_max"],
            "measurement_prob_mean": prob_info["measurement_prob_mean"],
            "p_sem_min": prob_info["p_sem_min"],
            "p_sem_max": prob_info["p_sem_max"],
            "p_sem_mean": prob_info["p_sem_mean"],
            "p_cls_min": prob_info["p_cls_min"],
            "p_cls_max": prob_info["p_cls_max"],
            "p_cls_mean": prob_info["p_cls_mean"],
            "p_mix_min": prob_info["p_mix_min"],
            "p_mix_max": prob_info["p_mix_max"],
            "p_mix_mean": prob_info["p_mix_mean"],
            "p_mix_sum": prob_info["p_mix_sum"],
            "p_norm_min": p_norm_min,
            "p_norm_max": p_norm_max,
            "score_min": score_min,
            "score_max": score_max,
            "anchor_prob_min": anchor_prob_stats["anchor_prob_min"],
            "anchor_prob_max": anchor_prob_stats["anchor_prob_max"],
            "anchor_prob_mean": anchor_prob_stats["anchor_prob_mean"],
            "kernel_diag_min": float(kernel_diag_initial.min().item()),
            "kernel_diag_max": float(kernel_diag_initial.max().item()),
            "kernel_diag_mean": float(kernel_diag_initial.mean().item()),
            "residual_initial_sum": residual_initial_sum,
            "residual_final_sum": residual_final_sum,
            "explained_ratio": explained_ratio,
            "evidence_coverage_min": evidence_coverage_min,
            "evidence_coverage_max": evidence_coverage_max,
            "evidence_coverage_mean": evidence_coverage_mean,
            "evidence_uncovered_mass": evidence_uncovered_mass,
            "selected_size": int(len(selected)),
            "final_keep_size": int(keep_idx.numel()),
            "keep_size": int(keep_idx.numel()),
            "qficr_p1lr_semantic_residual_values": p1lr_semantic_residual_values or [],
        }
        # Non-serialized tensor handle for post-selection feature-only transforms
        # (e.g. TC-ORT in llava_arch after the multimodal projector). The debug
        # writer below uses an explicit allow-list, so this does not alter the
        # existing QFi-CR debug JSONL schema.
        self.last_qfid_info["_prob_tensor"] = probs.detach()

        self.last_qfid_info["qficr_total_pruning_latency_ms"] = (
            (time.perf_counter() - pruning_start) * 1000.0 if profile_this_sample else None
        )
        self.last_qfid_info["qficr_peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(visual_tokens.device)) / (1024.0 * 1024.0)
            if profile_this_sample and visual_tokens.is_cuda
            else None
        )

        self._write_qfid_debug_stats(self.last_qfid_info)

        if selector_used in {
            "qfi_zeno_stabilized",
            "qfi_spatial_buffer_recovery",
            "qfi_progressive_128_to_64",
            "qfi_coarse_to_fine_pruning",
        } and get_env_bool("EC_QFI_CARDS_DUMP_DIAG", default=False):
            diag_path = os.environ.get("EC_QFI_CARDS_DIAG_PATH", "").strip()
            if diag_path:
                try:
                    diag_dir = os.path.dirname(os.path.expanduser(diag_path))
                    if diag_dir:
                        os.makedirs(diag_dir, exist_ok=True)
                    with open(os.path.expanduser(diag_path), "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(qfi_cards_info) + "\n")
                except OSError as exc:
                    if self.qmo_debug:
                        print(f"[QFI-Cards] warning: failed to dump diag to {diag_path}: {exc}", flush=True)

        if selector_used in {
            "qsp_cr_density_projective",
            "qsp_density_projective_task_anchor",
            "qsp_density_projective_task_anchor_a05",
            "qsp_density_projective_complement_bonus",
            "qfi_core_complement_recovery",
        }:
            dump_path = os.environ.get("EC_QSP_DUMP_INDICES", "").strip()
            if dump_path:
                try:
                    dump_dir = os.path.dirname(os.path.expanduser(dump_path))
                    if dump_dir:
                        os.makedirs(dump_dir, exist_ok=True)
                    with open(os.path.expanduser(dump_path), "a", encoding="utf-8") as fp:
                        record = {
                            "question_id": "",
                            "selector": qsp_info.get("qsp_selector_label", selector_used),
                            "K": int(min(max(0, int(K)), num_tokens)),
                            "K_core": int(qsp_info["qsp_k_core"]),
                            "K_rec": int(qsp_info["qsp_k_recover"]),
                            "core_selection": qsp_info["qsp_core_selection"],
                            "recovery": qsp_info["qsp_recovery"],
                            "use_spatial_local_prior": bool(qsp_info["qsp_use_spatial_local_prior"]),
                            "lambda_b": float(qsp_info["qsp_lambda_b"]),
                            "candidate_topm": int(qsp_info["qsp_candidate_topm"]),
                            "candidate_size": int(qsp_info["qsp_candidate_size"]),
                            "fallback": bool(qsp_info["qsp_fallback"]),
                            "fallback_reason": qsp_info["qsp_fallback_reason"],
                            "trace_res": float(qsp_info["qsp_trace_res"]),
                            "delta_max": float(qsp_info["qsp_delta_max"]),
                            "delta_mean": float(qsp_info["qsp_delta_mean"]),
                            "core_indices": qsp_info["core_indices"],
                            "recovery_indices": qsp_info["recovery_indices"],
                            "selected_indices": [int(x) for x in keep_idx.tolist()],
                        }
                        fp.write(json.dumps(record) + "\n")
                except OSError as exc:
                    if self.qmo_debug:
                        print(f"[QSP-CR] warning: failed to dump indices to {dump_path}: {exc}", flush=True)
            if get_env_bool("EC_QSP_DUMP_DIAG", default=False):
                diag_path = os.environ.get("EC_QSP_DIAG_PATH", "").strip()
                if diag_path:
                    try:
                        diag_dir = os.path.dirname(os.path.expanduser(diag_path))
                        if diag_dir:
                            os.makedirs(diag_dir, exist_ok=True)
                        p_float = probs.float()
                        p_float = p_float / p_float.sum().clamp_min(eps)
                        final_indices = [int(x) for x in keep_idx.tolist()]
                        core_indices = qsp_info.get("core_indices", [])
                        rec_indices = qsp_info.get("recovery_indices", [])
                        with open(os.path.expanduser(diag_path), "a", encoding="utf-8") as fp:
                            record = {
                                "profile": qsp_info.get("qsp_selector_label", selector_used),
                                "k": int(min(max(0, int(K)), num_tokens)),
                                "k_core": int(qsp_info["qsp_k_core"]),
                                "k_rec": int(qsp_info["qsp_k_recover"]),
                                "num_tokens": int(num_tokens),
                                "mean_p_core": float(qsp_info.get("mean_p_core", 0.0)),
                                "mean_p_recovery": float(qsp_info.get("mean_p_recovery", 0.0)),
                                "mean_p_final": float(qsp_info.get("mean_p_final", 0.0)),
                                "mean_delta_core": float(qsp_info.get("qsp_delta_mean", 0.0)),
                                "mean_residual_norm_core": float(qsp_info.get("qsp_mean_residual_norm_core", 0.0)),
                                "trace_residual": float(qsp_info.get("qsp_trace_res", 0.0)),
                                "entropy_p": _qsp_entropy_from_prob(p_float, max(float(eps), 1e-12)),
                                "entropy_g_perp": float(qsp_info.get("qsp_entropy_g_perp", 0.0)),
                                "overlap_core_top_p": float(qsp_info.get("overlap_core_top_p", 0.0)),
                                "overlap_final_top_p": float(qsp_info.get("overlap_final_top_p", 0.0)),
                                "num_low_p_high_delta_selected": int(qsp_info.get("num_low_p_high_delta_selected", 0)),
                                "fallback_used": bool(qsp_info.get("qsp_fallback", False)),
                                "fallback_reason": qsp_info.get("qsp_fallback_reason", ""),
                                "core_indices": core_indices,
                                "rec_indices": rec_indices,
                                "final_indices": final_indices,
                            }
                            fp.write(json.dumps(record) + "\n")
                    except OSError as exc:
                        if self.qmo_debug:
                            print(f"[QSP-CR] warning: failed to dump diag to {diag_path}: {exc}", flush=True)

        if selector_used in {"qmo_cr", "qmo_gated"}:
            dump_path = os.environ.get("EC_QMO_DUMP_INDICES", "").strip()
            if dump_path:
                try:
                    dump_dir = os.path.dirname(os.path.expanduser(dump_path))
                    if dump_dir:
                        os.makedirs(dump_dir, exist_ok=True)
                    with open(os.path.expanduser(dump_path), "a", encoding="utf-8") as fp:
                        record = {
                            "question_id": "",
                            "selector": selector_used,
                            "qtype": qmo_cr_info["question_type"],
                            "gate_type": qmo_gated_info["gate_type"],
                            "gate_enabled": bool(qmo_gated_info["gate_enabled"]),
                            "gate_mode": qmo_gated_info["gate_mode"],
                            "gate_beta": float(qmo_gated_info["gate_beta"]),
                            "K": int(min(max(0, int(K)), num_tokens)),
                            "K_core": int(qmo_cr_info["k_core"]),
                            "K_rec": int(qmo_cr_info["k_recover"]),
                            "entropy": float(qmo_cr_info["entropy"]),
                            "purity": float(qmo_cr_info["purity"]),
                            "D_q": float(qmo_cr_info["D_q"]),
                            "core_indices": qmo_cr_info["core_indices"],
                            "recovery_indices": qmo_cr_info["recovery_indices"],
                            "baseline_indices": qmo_gated_info["baseline_indices"],
                            "qmo_cr_indices": qmo_gated_info["qmo_cr_indices"],
                            "baseline_overlap": float(qmo_gated_info["baseline_overlap"]),
                            "qmo_cr_overlap": float(qmo_gated_info["qmo_cr_overlap"]),
                            "selected_indices": [int(x) for x in keep_idx.tolist()],
                        }
                        if selector_used == "qmo_gated":
                            record["K_core"] = int(adaptive_info["k_core"])
                            record["K_rec"] = int(adaptive_info["k_recover"])
                            record["core_indices"] = qmo_gated_info.get("core_indices", [])
                            record["recovery_indices"] = qmo_gated_info.get("recovery_indices", [])
                        fp.write(json.dumps(record) + "\n")
                except OSError as exc:
                    if self.qmo_debug:
                        print(f"[QMO-CR] warning: failed to dump indices to {dump_path}: {exc}", flush=True)
            if self.qmo_debug:
                print(
                    "[QMO-CR] "
                    f"qtype={qmo_cr_info['question_type']} "
                    f"entropy={qmo_cr_info['entropy']:.6f} "
                    f"purity={qmo_cr_info['purity']:.6f} "
                    f"D_q={qmo_cr_info['D_q']:.6f} "
                    f"recover_ratio={qmo_cr_info['recover_ratio']:.6f} "
                    f"K_core={qmo_cr_info['k_core']} "
                    f"K_rec={qmo_cr_info['k_recover']}",
                    flush=True,
                )

        self._debug_qf_pruner("enabled=True")
        self._debug_qf_pruner(f"selector={self.last_qfid_info['selector']}")
        self._debug_qf_pruner(
            f"core_ratio={self.last_qfid_info['core_ratio']:.6f}, "
            f"core_size={self.last_qfid_info['core_size']}, "
            f"recover_size={self.last_qfid_info['recover_size']}"
        )
        self._debug_qf_pruner(
            f"adapt_mode={self.last_qfid_info['adapt_mode']}, "
            f"question_type={self.last_qfid_info['question_type']}, "
            f"adapt_entropy_norm={self.last_qfid_info['adapt_entropy_norm']:.6f}, "
            f"adapt_recover_ratio={self.last_qfid_info['adapt_recover_ratio']:.6f}, "
            f"adapt_k_core={self.last_qfid_info['adapt_k_core']}, "
            f"adapt_k_recover={self.last_qfid_info['adapt_k_recover']}, "
            f"question_text_available={self.last_qfid_info['question_text_available']}"
        )
        self._debug_qf_pruner(
            f"budget_mode={self.last_qfid_info['qficr_budget_mode']}, "
            f"K={self.last_qfid_info['final_keep_size']}, "
            f"rho_min={self.last_qfid_info['qficr_rho_min']:.6f}, "
            f"rho_max={self.last_qfid_info['qficr_rho_max']:.6f}, "
            f"H_hat={self.last_qfid_info['qficr_entropy_hat']:.6f}, "
            f"entropy_gamma={self.last_qfid_info['qficr_entropy_gamma']:.6f}, "
            f"rho={self.last_qfid_info['qficr_rho']:.6f}, "
            f"raw_K_rest={self.last_qfid_info['qficr_raw_k_rest']}, "
            f"final_K_rest={self.last_qfid_info['qficr_final_k_rest']}, "
            f"final_K_red={self.last_qfid_info['qficr_final_k_red']}, "
            f"abs_cap={self.last_qfid_info['qficr_abs_cap']}, "
            f"ratio_cap={self.last_qfid_info['qficr_ratio_cap']:.6f}, "
            f"cap_applied={self.last_qfid_info['qficr_cap_applied']}, "
            f"yes_no_gate_applied={self.last_qfid_info['qficr_yes_no_gate_applied']}"
        )
        self._debug_qf_pruner(
            f"recover_prior_anchor={self.last_qfid_info['recover_prior_anchor']}, "
            f"recover_prior_gamma={self.last_qfid_info['recover_prior_gamma']:.6f}, "
            f"recover_cand_pool={self.last_qfid_info['recover_cand_pool']}, "
            f"recover_cand_mult={self.last_qfid_info['recover_cand_mult']:.6f}, "
            f"recover_cand_size={self.last_qfid_info['recover_cand_size']}"
        )
        if selector_used == "adaptive_core_recover" and not adaptive_info["question_text_available"]:
            self._debug_qf_pruner("question text unavailable; adaptive recovery uses entropy-only mode")
        self._debug_qf_pruner(f"kernel={self.qfid_kernel}")
        self._debug_qf_pruner(
            f"budget_calib={self.last_qfid_info['budget_calib']}, "
            f"budget_alpha={self.last_qfid_info['budget_alpha']:.6f}, "
            f"budget_target_neff={self.last_qfid_info['budget_target_neff']:.6f}, "
            f"budget_neff_before={self.last_qfid_info['budget_neff_before']:.6f}, "
            f"budget_neff_after={self.last_qfid_info['budget_neff_after']:.6f}, "
            f"budget_gamma={self.last_qfid_info['budget_gamma']:.6f}"
        )
        self._debug_qf_pruner(
            f"spectral_filter_enabled={self.last_qfid_info['spectral_filter_enabled']}, "
            f"spectral_filter_applied={self.last_qfid_info['spectral_filter_applied']}, "
            f"spectral_filter_fallback={self.last_qfid_info['spectral_filter_fallback']}, "
            f"spectral_gamma={self.last_qfid_info['spectral_gamma']:.6f}"
        )
        self._debug_qf_pruner(
            f"spectral_trace_before={self.last_qfid_info['spectral_trace_before']:.6f}, "
            f"spectral_trace_after={self.last_qfid_info['spectral_trace_after']:.6f}, "
            f"spectral_rank_pos={self.last_qfid_info['spectral_rank_pos']}, "
            f"spectral_min_eval={self.last_qfid_info['spectral_min_eval']:.6f}, "
            f"spectral_max_eval={self.last_qfid_info['spectral_max_eval']:.6f}"
        )
        if self.qfid_prob_source in {"clsmix", "cls_only_qf"}:
            self._debug_qf_pruner(
                f"cls_gate_enabled={self.last_qfid_info['cls_gate_enabled']}, "
                f"cls_gate_mode={self.last_qfid_info['cls_gate_mode']}, "
                f"agreement={self.last_qfid_info['agreement']:.6f}, "
                f"beta_eff={self.last_qfid_info['beta_eff']:.6f}, "
                f"beta_base={self.last_qfid_info['beta_base']:.6f}, "
                f"gate_min={self.last_qfid_info['gate_min']:.6f}, "
                f"gate_max={self.last_qfid_info['gate_max']:.6f}, "
                f"p_sem_entropy={self.last_qfid_info['p_sem_entropy']:.6f}, "
                f"p_cls_entropy={self.last_qfid_info['p_cls_entropy']:.6f}, "
                f"keep_size={self.last_qfid_info['keep_size']}"
            )
            self._debug_qf_pruner(
                f"prob_source={self.qfid_prob_source}, select_mode=qf, "
                f"cls_prior_layers={self.last_qfid_info['cls_prior_layers']}, "
                f"cls_attn_layer={self.last_qfid_info['cls_attn_layer']}, "
                f"cls_head_reduce={self.last_qfid_info['cls_head_reduce']}, "
                f"cls_attn_available={self.last_qfid_info['cls_attn_available']}, "
                f"cls_fallback={self.last_qfid_info['cls_fallback']}, "
                f"cls_mix_mode={self.last_qfid_info['cls_mix_mode']}, "
                f"cls_mix_beta={self.last_qfid_info['cls_mix_beta']:.6f}, "
                f"p_cls_min={self.last_qfid_info['p_cls_min']:.6f}, "
                f"p_cls_max={self.last_qfid_info['p_cls_max']:.6f}, "
                f"p_cls_mean={self.last_qfid_info['p_cls_mean']:.6f}, "
                f"p_sem_min={self.last_qfid_info['p_sem_min']:.6f}, "
                f"p_sem_max={self.last_qfid_info['p_sem_max']:.6f}, "
                f"p_sem_mean={self.last_qfid_info['p_sem_mean']:.6f}, "
                f"p_mix_min={self.last_qfid_info['p_mix_min']:.6f}, "
                f"p_mix_max={self.last_qfid_info['p_mix_max']:.6f}, "
                f"p_mix_mean={self.last_qfid_info['p_mix_mean']:.6f}"
            )
        self._debug_qf_pruner(
            f"spatial_state_enabled={self.last_qfid_info['spatial_state_enabled']}, "
            f"spatial_state_applied={self.last_qfid_info['spatial_state_applied']}, "
            f"spatial_state_fallback={self.last_qfid_info['spatial_state_fallback']}"
        )
        self._debug_qf_pruner(
            f"spatial_lambda={self.last_qfid_info['spatial_lambda']:.6f}, "
            f"spatial_sigma={self.last_qfid_info['spatial_sigma']:.6f}, "
            f"grid_size={self.last_qfid_info['grid_size']}"
        )
        self._debug_qf_pruner(
            f"spatial_factor_min={self.last_qfid_info['spatial_factor_min']:.6f}, "
            f"spatial_factor_max={self.last_qfid_info['spatial_factor_max']:.6f}, "
            f"spatial_factor_mean={self.last_qfid_info['spatial_factor_mean']:.6f}"
        )
        self._debug_qf_pruner(
            f"measure_prior_mode={measure_prior_mode}, measure_prior_lambda={measure_prior_lambda:.6f}"
        )
        self._debug_qf_pruner(
            f"anchor_mode={anchor_mode}, anchor_num={anchor_num}, qf_fill_count={qf_fill_count}"
        )
        self._debug_qf_pruner(
            f"depolarize_mode={self.last_qfid_info['depolarize_mode']}, "
            f"depolarize={self.last_qfid_info['depolarize']:.6f}"
        )
        self._debug_qf_pruner(
            f"prob_source={self.qfid_prob_source}, prob_source_used={prob_source_used}, tau={self.qfid_tau}"
        )
        self._debug_qf_pruner(
            f"prob_fallback={self.last_qfid_info['prob_fallback']}, "
            f"measurement_alpha_mode={self.last_qfid_info['measurement_alpha_mode']}"
        )
        if self.qfid_prob_source == "measurement":
            self._debug_qf_pruner(
                f"measurement_score_min={self.last_qfid_info['measurement_score_min']:.6f}, "
                f"measurement_score_max={self.last_qfid_info['measurement_score_max']:.6f}, "
                f"measurement_score_mean={self.last_qfid_info['measurement_score_mean']:.6f}"
            )
            self._debug_qf_pruner(
                f"measurement_prob_min={self.last_qfid_info['measurement_prob_min']:.6f}, "
                f"measurement_prob_max={self.last_qfid_info['measurement_prob_max']:.6f}, "
                f"measurement_prob_mean={self.last_qfid_info['measurement_prob_mean']:.6f}"
            )
        self._debug_qf_pruner(
            f"entropy={self.last_qfid_info['entropy']:.6f}, "
            f"entropy_norm={self.last_qfid_info['entropy_norm']:.6f}, "
            f"adaptive_depolarize={self.last_qfid_info['adaptive_depolarize']:.6f}"
        )
        self._debug_qf_pruner(
            f"score_min={self.last_qfid_info['score_min']:.6f}, "
            f"score_max={self.last_qfid_info['score_max']:.6f}, "
            f"p_norm_min={self.last_qfid_info['p_norm_min']:.6f}, "
            f"p_norm_max={self.last_qfid_info['p_norm_max']:.6f}"
        )
        self._debug_qf_pruner(
            f"anchor_prob_min={self.last_qfid_info['anchor_prob_min']:.6f}, "
            f"anchor_prob_max={self.last_qfid_info['anchor_prob_max']:.6f}, "
            f"anchor_prob_mean={self.last_qfid_info['anchor_prob_mean']:.6f}"
        )
        self._debug_qf_pruner(
            f"candidate_source=all, candidate_ratio=1.0, "
            f"candidate_size_before={num_tokens}, candidate_size_after={num_tokens}, "
            f"candidate_padded=0, candidate_fallback=none, candidate_rel_source=none"
        )
        self._debug_qf_pruner(
            f"p_min={self.last_qfid_info['p_min']:.6f}, "
            f"p_max={self.last_qfid_info['p_max']:.6f}, "
            f"p_mean={self.last_qfid_info['p_mean']:.6f}"
        )
        self._debug_qf_pruner(
            f"kernel_diag_min={self.last_qfid_info['kernel_diag_min']:.6f}, "
            f"kernel_diag_max={self.last_qfid_info['kernel_diag_max']:.6f}, "
            f"kernel_diag_mean={self.last_qfid_info['kernel_diag_mean']:.6f}"
        )
        self._debug_qf_pruner(
            f"residual_initial_sum={self.last_qfid_info['residual_initial_sum']:.6f}, "
            f"residual_final_sum={self.last_qfid_info['residual_final_sum']:.6f}, "
            f"explained_ratio={self.last_qfid_info['explained_ratio']:.6f}"
        )
        self._debug_qf_pruner(
            f"evidence_coverage_min={self.last_qfid_info['evidence_coverage_min']:.6f}, "
            f"evidence_coverage_max={self.last_qfid_info['evidence_coverage_max']:.6f}, "
            f"evidence_coverage_mean={self.last_qfid_info['evidence_coverage_mean']:.6f}, "
            f"evidence_uncovered_mass={self.last_qfid_info['evidence_uncovered_mass']:.6f}"
        )
        self._debug_qf_pruner(
            f"selected_size={self.last_qfid_info['selected_size']}, "
            f"final_keep_size={self.last_qfid_info['final_keep_size']}"
        )
        return keep_idx.sort().values, self.last_qfid_info

    def select_by_cls_topk(self, visual_tokens, K, cls_attn=None):
        """Direct CLS-attention ranking baseline without QF residual selection."""
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        num_tokens = visual_tokens.shape[1]
        device = visual_tokens.device
        select_num = min(max(0, int(K)), num_tokens)
        p_cls, cls_info = self.compute_qfid_cls_probability(cls_attn, num_tokens, device)
        cls_fallback = p_cls is None
        if p_cls is None:
            p_cls = torch.full(
                (num_tokens,),
                1.0 / max(1, num_tokens),
                dtype=torch.float32,
                device=device,
            )

        keep_idx = torch.topk(p_cls, k=select_num, largest=True, sorted=False).indices
        keep_idx = keep_idx.sort().values
        self.last_qfid_info = {
            "enabled": True,
            "select_mode": "cls_topk",
            "prob_source": "cls_only_qf",
            "prob_source_used": "cls_topk" if not cls_fallback else "uniform_fallback",
            "prob_fallback": cls_fallback,
            "cls_attn_available": bool(cls_info["cls_attn_available"]),
            "cls_fallback": cls_fallback,
            "cls_attn_layer": cls_info["cls_attn_layer"],
            "cls_head_reduce": cls_info["cls_head_reduce"],
            "cls_head_entropy_min": cls_info["cls_head_entropy_min"],
            "cls_head_entropy_max": cls_info["cls_head_entropy_max"],
            "cls_head_entropy_mean": cls_info["cls_head_entropy_mean"],
            "cls_head_weight_min": cls_info["cls_head_weight_min"],
            "cls_head_weight_max": cls_info["cls_head_weight_max"],
            "cls_mix_mode": self.qfid_cls_mix_mode,
            "cls_mix_beta": float(max(0.0, min(float(self.qfid_cls_mix_beta), 1.0))),
            "candidate_source": "all",
            "candidate_ratio": 1.0,
            "candidate_rel_source": "none",
            "candidate_fallback": "none",
            "candidate_size_before_truncation": int(num_tokens),
            "candidate_size_after_truncation": int(num_tokens),
            "candidate_padded_count": 0,
            "p_cls_min": float(p_cls.min().item()) if p_cls.numel() else 0.0,
            "p_cls_max": float(p_cls.max().item()) if p_cls.numel() else 0.0,
            "p_cls_mean": float(p_cls.mean().item()) if p_cls.numel() else 0.0,
            "p_sum": float(p_cls.sum().item()) if p_cls.numel() else 0.0,
            "selected_size": int(keep_idx.numel()),
            "final_keep_size": int(keep_idx.numel()),
        }
        self._debug_qf_pruner(
            f"prob_source=cls_only_qf, select_mode=cls_topk, "
            f"cls_attn_layer={self.last_qfid_info['cls_attn_layer']}, "
            f"cls_head_reduce={self.last_qfid_info['cls_head_reduce']}, "
            f"cls_mix_mode={self.last_qfid_info['cls_mix_mode']}, "
            f"cls_mix_beta={self.last_qfid_info['cls_mix_beta']:.6f}, "
            f"cls_attn_available={self.last_qfid_info['cls_attn_available']}, "
            f"cls_fallback={self.last_qfid_info['cls_fallback']}, "
            f"final_keep_size={self.last_qfid_info['final_keep_size']}"
        )
        return keep_idx, self.last_qfid_info

    def select_by_semantic_topk(
        self,
        visual_tokens,
        K,
        semantic_response=None,
        semantic_response_source=None,
    ):
        """Direct instruction-conditioned relevance ranking without QF residual selection."""
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        num_tokens = visual_tokens.shape[1]
        device = visual_tokens.device
        select_num = min(max(0, int(K)), num_tokens)
        probs, source_used, prob_info = self.compute_qfid_probability(
            num_tokens,
            device,
            b=semantic_response,
            semantic_response_source=semantic_response_source,
        )
        keep_idx = torch.topk(probs, k=select_num, largest=True, sorted=False).indices.sort().values
        self.last_qfid_info = {
            "enabled": True,
            "select_mode": "semantic_topk",
            "prob_source": "semantic",
            "prob_source_used": source_used,
            "prob_fallback": source_used == "uniform",
            "candidate_source": "all",
            "candidate_ratio": 1.0,
            "candidate_rel_source": "semantic_probability",
            "candidate_fallback": "uniform" if source_used == "uniform" else "none",
            "candidate_size_before_truncation": int(num_tokens),
            "candidate_size_after_truncation": int(num_tokens),
            "candidate_padded_count": 0,
            "p_min": prob_info["p_min"],
            "p_max": prob_info["p_max"],
            "p_mean": prob_info["p_mean"],
            "p_sum": prob_info["p_sum"],
            "selected_size": int(keep_idx.numel()),
            "final_keep_size": int(keep_idx.numel()),
            "keep_size": int(keep_idx.numel()),
        }
        self._debug_qf_pruner(
            f"select_mode=semantic_topk, prob_source_used={source_used}, "
            f"final_keep_size={keep_idx.numel()}"
        )
        return keep_idx, self.last_qfid_info

    def select_by_visual_kcenter(self, visual_tokens, K):
        """Deterministic cosine k-center baseline over visual token states only."""
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        features = visual_tokens.float().mean(dim=0)
        num_tokens = features.shape[0]
        device = features.device
        select_num = min(max(0, int(K)), num_tokens)
        initial_pivot = -1
        if select_num == 0:
            keep_idx = torch.empty(0, dtype=torch.long, device=device)
        else:
            norms = features.norm(dim=-1)
            states = features / norms.unsqueeze(-1).clamp_min(max(float(self.qfid_eps), 1e-12))
            first = int(torch.argmax(norms).item())
            initial_pivot = first
            selected = [first]
            min_distance = 1.0 - torch.matmul(states, states[first])
            min_distance[first] = -1.0
            while len(selected) < select_num:
                pivot = int(torch.argmax(min_distance).item())
                selected.append(pivot)
                distance = 1.0 - torch.matmul(states, states[pivot])
                min_distance = torch.minimum(min_distance, distance)
                min_distance[torch.tensor(selected, device=device, dtype=torch.long)] = -1.0
            keep_idx = torch.tensor(selected, dtype=torch.long, device=device).sort().values

        self.last_qfid_info = {
            "enabled": True,
            "select_mode": "visual_kcenter",
            "prob_source": "none",
            "prob_source_used": "none",
            "prob_fallback": False,
            "candidate_source": "all",
            "candidate_ratio": 1.0,
            "candidate_rel_source": "none",
            "candidate_fallback": "none",
            "candidate_size_before_truncation": int(num_tokens),
            "candidate_size_after_truncation": int(num_tokens),
            "candidate_padded_count": 0,
            "first_pivot": initial_pivot,
            "selected_size": int(keep_idx.numel()),
            "final_keep_size": int(keep_idx.numel()),
            "keep_size": int(keep_idx.numel()),
        }
        self._debug_qf_pruner(
            f"select_mode=visual_kcenter, first_pivot={self.last_qfid_info['first_pivot']}, "
            f"final_keep_size={keep_idx.numel()}"
        )
        return keep_idx, self.last_qfid_info

    def compute_qmo_relevance_score(self, norm_score, relevance=None, text_attn=None):
        num_tokens = norm_score.numel()
        device = norm_score.device
        score = None
        source = "norm"

        if torch.is_tensor(relevance):
            relevance = torch.nan_to_num(relevance.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if relevance.ndim == 1:
                score = relevance
            elif relevance.ndim >= 2:
                score = relevance.mean(dim=tuple(range(relevance.ndim - 1)))
            if torch.is_tensor(score) and score.numel() == num_tokens:
                source = "relevance"
            else:
                score = None

        if score is None:
            attn_score = self.aggregate_text_to_visual_attention(text_attn)
            if torch.is_tensor(attn_score) and attn_score.numel() == num_tokens:
                score = attn_score
                source = "attn"

        if score is None:
            score = norm_score

        return safe_normalize(score.to(device=device).reshape(-1)), source

    def compute_qmo_semantic_score(self, norm_score, b=None, semantic_response_source=None):
        num_tokens = norm_score.numel()
        device = norm_score.device
        score = self.compute_semantic_node_score(b)
        source = "semantic"
        if score is None or score.numel() != num_tokens:
            score = norm_score
            source = "norm"
        elif semantic_response_source == "uniform" and torch.max(score).item() <= 1e-8:
            score = norm_score
            source = "norm"
        return safe_normalize(score.to(device=device).reshape(-1)), source

    @staticmethod
    def compute_qmo_entropy_score(b, eps=1e-8):
        if not torch.is_tensor(b) or b.ndim != 2 or b.shape[0] == 0:
            return None
        if b.shape[1] <= 1:
            return torch.zeros(b.shape[0], dtype=torch.float32, device=b.device)

        probs = torch.nan_to_num(b.float(), nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
        entropy = entropy / math.log(float(b.shape[1]))
        return safe_normalize(entropy)

    def compute_qmo_spatial_deficit_score(self, base_score, initial_topk, num_tokens):
        device = base_score.device
        side = int(round(math.sqrt(num_tokens)))
        if side * side != num_tokens:
            return torch.zeros(num_tokens, dtype=torch.float32, device=device)

        grid_size = max(1, int(self.spatial_grid_size))
        cell_extent = max(1, int(math.ceil(side / grid_size)))
        positions = self.build_patch_positions(num_tokens, device)
        x_cell = torch.div(positions[:, 0].long(), cell_extent, rounding_mode="floor").clamp(0, grid_size - 1)
        y_cell = torch.div(positions[:, 1].long(), cell_extent, rounding_mode="floor").clamp(0, grid_size - 1)
        cell_id = y_cell * grid_size + x_cell
        cell_count = grid_size * grid_size
        coverage = torch.zeros(cell_count, dtype=torch.float32, device=device)

        if torch.is_tensor(initial_topk) and initial_topk.numel() > 0:
            topk_cells = cell_id[initial_topk.reshape(-1)]
            coverage.scatter_add_(
                0,
                topk_cells,
                torch.ones_like(topk_cells, dtype=coverage.dtype),
            )

        cell_deficit = 1.0 / (1.0 + coverage)
        spatial_score = cell_deficit[cell_id] * safe_normalize(base_score)
        return safe_normalize(spatial_score)

    def compute_qmo_chain_deficit_score(self, b, relation_matrix, initial_topk):
        if not torch.is_tensor(b) or b.ndim != 2 or b.shape[0] == 0:
            return None

        prepared_relation = self._prepare_chain_relation_matrix(relation_matrix)
        if prepared_relation is None or prepared_relation.numel() == 0:
            return torch.zeros(b.shape[0], dtype=torch.float32, device=b.device)
        if prepared_relation.shape[0] != b.shape[1]:
            return torch.zeros(b.shape[0], dtype=torch.float32, device=b.device)

        active_edges = torch.count_nonzero(torch.triu(prepared_relation, diagonal=1)).item()
        if active_edges == 0:
            return torch.zeros(b.shape[0], dtype=torch.float32, device=b.device)

        probs = torch.nan_to_num(b.float(), nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if torch.is_tensor(initial_topk) and initial_topk.numel() > 0:
            cov = probs[initial_topk.reshape(-1)].max(dim=0).values
        else:
            cov = torch.zeros(probs.shape[1], dtype=torch.float32, device=b.device)
        degree = (prepared_relation > 0).float().sum(dim=-1)
        deficit = (1.0 - cov).clamp_min(0.0) * degree
        chain_score = torch.matmul(probs, deficit)
        return safe_normalize(chain_score)

    def compute_qmo_node_score(
        self,
        visual_tokens,
        relevance=None,
        text_attn=None,
        b=None,
        relation_matrix=None,
        semantic_response_source=None,
        K=None,
    ):
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        num_tokens = visual_tokens.shape[1]
        device = visual_tokens.device
        norm_score = visual_tokens.norm(dim=-1).mean(dim=0).to(device=device).reshape(-1)
        rel_score, rel_source = self.compute_qmo_relevance_score(
            norm_score,
            relevance=relevance,
            text_attn=text_attn,
        )
        sem_score, sem_source = self.compute_qmo_semantic_score(
            norm_score,
            b=b,
            semantic_response_source=semantic_response_source,
        )
        entropy_score = self.compute_qmo_entropy_score(b)
        if entropy_score is None or entropy_score.numel() != num_tokens:
            entropy_score = torch.zeros(num_tokens, dtype=torch.float32, device=device)

        init_source = self.qmo_init_source
        if init_source == "relevance":
            init_score = rel_score
        elif init_source == "attn":
            init_score = rel_score if rel_source == "attn" else safe_normalize(norm_score)
        elif init_source == "norm":
            init_score = safe_normalize(norm_score)
        else:
            init_score = sem_score
            init_source = "semantic"

        if init_score.numel() != num_tokens:
            init_score = safe_normalize(norm_score)
            init_source = "norm"

        topk = min(max(1, int(K or num_tokens)), num_tokens)
        initial_topk = torch.topk(init_score, topk, largest=True, sorted=False).indices
        spatial_score = self.compute_qmo_spatial_deficit_score(init_score, initial_topk, num_tokens)
        chain_score = self.compute_qmo_chain_deficit_score(b, relation_matrix, initial_topk)
        if chain_score is None or chain_score.numel() != num_tokens:
            chain_score = torch.zeros(num_tokens, dtype=torch.float32, device=device)

        qmo_score = (
            self.qmo_w_rel * rel_score
            + self.qmo_w_sem * sem_score
            + self.qmo_w_spatial * spatial_score
            + self.qmo_w_chain * chain_score
            - self.qmo_w_ent * entropy_score
        )
        qmo_score = safe_normalize(qmo_score.to(device=device).reshape(-1))
        qmo_score = torch.nan_to_num(qmo_score, nan=0.0, posinf=1.0, neginf=0.0)

        prepared_relation = self._prepare_chain_relation_matrix(relation_matrix)
        chain_edges = int(
            0
            if prepared_relation is None
            else torch.count_nonzero(torch.triu(prepared_relation, diagonal=1)).item()
        )
        self.last_qmo_info = {
            "qmo_enabled": True,
            "weights": {
                "rel": self.qmo_w_rel,
                "sem": self.qmo_w_sem,
                "spatial": self.qmo_w_spatial,
                "chain": self.qmo_w_chain,
                "ent": self.qmo_w_ent,
            },
            "rel_mean": float(rel_score.mean().item()),
            "sem_mean": float(sem_score.mean().item()),
            "spatial_mean": float(spatial_score.mean().item()),
            "chain_mean": float(chain_score.mean().item()),
            "entropy_mean": float(entropy_score.mean().item()),
            "qmo_min": float(qmo_score.min().item()),
            "qmo_max": float(qmo_score.max().item()),
            "qmo_mean": float(qmo_score.mean().item()),
            "initial_topk_size": int(initial_topk.numel()),
            "spatial_nonzero_ratio": float((spatial_score > 0).float().mean().item()),
            "chain_edges": chain_edges,
            "chain_nonzero_ratio": float((chain_score > 0).float().mean().item()),
            "qmo_init_source": init_source,
            "rel_source": rel_source,
            "sem_source": sem_source,
        }

        self._debug_qmo_pruner("qmo_enabled=True")
        self._debug_qmo_pruner(
            "weights: "
            f"rel={self.qmo_w_rel}, sem={self.qmo_w_sem}, spatial={self.qmo_w_spatial}, "
            f"chain={self.qmo_w_chain}, ent={self.qmo_w_ent}"
        )
        self._debug_qmo_pruner(
            f"rel_mean={self.last_qmo_info['rel_mean']:.6f}, "
            f"sem_mean={self.last_qmo_info['sem_mean']:.6f}, "
            f"spatial_mean={self.last_qmo_info['spatial_mean']:.6f}, "
            f"chain_mean={self.last_qmo_info['chain_mean']:.6f}, "
            f"entropy_mean={self.last_qmo_info['entropy_mean']:.6f}"
        )
        self._debug_qmo_pruner(
            f"qmo_min={self.last_qmo_info['qmo_min']:.6f}, "
            f"qmo_max={self.last_qmo_info['qmo_max']:.6f}, "
            f"qmo_mean={self.last_qmo_info['qmo_mean']:.6f}"
        )
        self._debug_qmo_pruner(
            f"initial_topk_size={self.last_qmo_info['initial_topk_size']}, "
            f"qmo_init_source={self.last_qmo_info['qmo_init_source']}, "
            f"rel_source={self.last_qmo_info['rel_source']}, "
            f"sem_source={self.last_qmo_info['sem_source']}"
        )
        self._debug_qmo_pruner(
            f"spatial_nonzero_ratio={self.last_qmo_info['spatial_nonzero_ratio']:.6f}, "
            f"chain_edges={self.last_qmo_info['chain_edges']}, "
            f"chain_nonzero_ratio={self.last_qmo_info['chain_nonzero_ratio']:.6f}"
        )
        return qmo_score

    @staticmethod
    def _unique_count(index):
        if not torch.is_tensor(index) or index.numel() == 0:
            return 0
        return torch.unique(index).numel()

    @staticmethod
    def _overlap_ratio(left, right):
        if (
            not torch.is_tensor(left)
            or not torch.is_tensor(right)
            or left.numel() == 0
            or right.numel() == 0
        ):
            return 0.0
        left = torch.unique(left)
        right = torch.unique(right)
        overlap = torch.isin(left, right).sum().item()
        return overlap / max(1, min(left.numel(), right.numel()))

    @staticmethod
    def _selected_to_mask(selected, size, device):
        if torch.is_tensor(selected) and selected.dtype == torch.bool:
            mask = selected.to(device=device).reshape(-1).clone()
            if mask.numel() != size:
                raise ValueError(f"selected bool mask size mismatch: {mask.numel()} vs {size}")
            return mask

        if not torch.is_tensor(selected):
            selected = torch.tensor(selected, dtype=torch.long, device=device)
        selected = selected.to(device=device, dtype=torch.long).reshape(-1)
        if selected.numel() == 0:
            return torch.zeros(size, dtype=torch.bool, device=device)

        if torch.unique(selected).numel() != selected.numel():
            raise ValueError("selected indices contain duplicates")
        if selected.min().item() < 0 or selected.max().item() >= size:
            raise ValueError("selected indices out of local candidate range")

        mask = torch.zeros(size, dtype=torch.bool, device=device)
        mask[selected] = True
        return mask

    @staticmethod
    def _validate_keep_idx(keep_idx, num_tokens, keep_num, device):
        if keep_idx is None:
            raise ValueError("keep_idx is None")

        keep_idx = keep_idx.to(device=device, dtype=torch.long).reshape(-1)
        if keep_idx.numel() != keep_num:
            raise ValueError(f"keep_idx size mismatch: expected {keep_num}, got {keep_idx.numel()}")
        if keep_idx.numel() == 0:
            return keep_idx

        keep_idx = keep_idx.sort().values
        keep_idx_float = keep_idx.float()
        if not torch.isfinite(keep_idx_float).all():
            raise ValueError("keep_idx contains NaN or Inf")
        if keep_idx.min().item() < 0 or keep_idx.max().item() >= num_tokens:
            raise ValueError(f"keep_idx out of range [0, {num_tokens})")
        if torch.unique(keep_idx).numel() != keep_num:
            raise ValueError("keep_idx contains duplicates")
        return keep_idx

    @staticmethod
    def _selection_overlap_ratio(left_idx, right_idx):
        if (
            not torch.is_tensor(left_idx)
            or not torch.is_tensor(right_idx)
            or left_idx.numel() == 0
            or right_idx.numel() == 0
        ):
            return 0.0
        left_idx = torch.unique(left_idx.reshape(-1))
        right_idx = torch.unique(right_idx.reshape(-1))
        overlap = torch.isin(left_idx, right_idx).sum().item()
        return overlap / max(1, min(left_idx.numel(), right_idx.numel()))

    @staticmethod
    def _selection_hamming_distance(left_idx, right_idx):
        if (
            not torch.is_tensor(left_idx)
            or not torch.is_tensor(right_idx)
            or left_idx.numel() == 0
            or right_idx.numel() == 0
        ):
            return 0
        left_idx = torch.unique(left_idx.reshape(-1))
        right_idx = torch.unique(right_idx.reshape(-1))
        overlap = torch.isin(left_idx, right_idx).sum().item()
        return int(left_idx.numel() + right_idx.numel() - 2 * overlap)

    def _append_spatial_candidates(self, candidate_parts, spatial_coverage, a, positions, global_idx=None):
        if not self.use_spatial_candidate:
            return torch.empty(0, dtype=torch.long, device=a.device)
        if not torch.is_tensor(positions) or positions.ndim != 2 or positions.shape[0] != a.numel():
            self._debug("spatial candidate positions unavailable, skip")
            return torch.empty(0, dtype=torch.long, device=a.device)

        grid_size = max(1, int(self.spatial_grid_size))
        topl = max(0, int(self.spatial_topl))
        if topl == 0:
            return torch.empty(0, dtype=torch.long, device=a.device)

        x = positions[:, 0]
        y = positions[:, 1]
        x_span = (x.max() - x.min()).clamp_min(1e-6)
        y_span = (y.max() - y.min()).clamp_min(1e-6)
        x_bin = (((x - x.min()) / x_span) * grid_size).long().clamp(0, grid_size - 1)
        y_bin = (((y - y.min()) / y_span) * grid_size).long().clamp(0, grid_size - 1)
        spatial_parts = []

        for gy in range(grid_size):
            for gx in range(grid_size):
                cell_mask = (x_bin == gx) & (y_bin == gy)
                if not cell_mask.any():
                    continue
                cell_idx = torch.nonzero(cell_mask, as_tuple=False).reshape(-1)
                if self.spatial_exclude_global and torch.is_tensor(global_idx) and global_idx.numel() > 0:
                    non_global_mask = ~torch.isin(cell_idx, global_idx)
                    if non_global_mask.any():
                        cell_idx = cell_idx[non_global_mask]
                local_topl = min(topl, cell_idx.numel())
                selected = cell_idx[
                    torch.topk(a[cell_idx], local_topl, largest=True, sorted=False).indices
                ]
                spatial_parts.append(selected)
                candidate_parts.append(selected)
                spatial_coverage.scatter_add_(
                    0,
                    selected,
                    torch.ones_like(selected, dtype=spatial_coverage.dtype),
                )
        if not spatial_parts:
            return torch.empty(0, dtype=torch.long, device=a.device)
        return torch.unique(torch.cat(spatial_parts), sorted=True)

    def build_semantic_candidate_pool(self, a, b, keep_num, positions=None):
        num_tokens = a.numel()
        max_candidates = max(keep_num, int(round(self.candidate_ratio * keep_num)))
        max_candidates = min(num_tokens, max_candidates)
        global_budget = int(round(self.global_candidate_ratio * keep_num))
        global_budget = min(num_tokens, max(1, global_budget))

        semantic_coverage = torch.zeros(num_tokens, dtype=torch.float32, device=a.device)
        spatial_coverage = torch.zeros(num_tokens, dtype=torch.float32, device=a.device)
        global_idx = torch.topk(a, global_budget, largest=True, sorted=False).indices
        candidate_parts = [global_idx]
        semantic_idx = torch.empty(0, dtype=torch.long, device=a.device)
        spatial_idx = torch.empty(0, dtype=torch.long, device=a.device)

        if self.use_semantic_candidate and torch.is_tensor(b) and b.ndim == 2 and b.shape[0] == num_tokens:
            topl = min(num_tokens, max(0, self.per_unit_topl))
            if topl > 0:
                semantic_parts = []
                for unit_idx in range(b.shape[1]):
                    unit_idxes = torch.topk(b[:, unit_idx], topl, largest=True, sorted=False).indices
                    semantic_parts.append(unit_idxes)
                    candidate_parts.append(unit_idxes)
                    semantic_coverage.scatter_add_(
                        0,
                        unit_idxes,
                        torch.ones_like(unit_idxes, dtype=semantic_coverage.dtype),
                    )
                if semantic_parts:
                    semantic_idx = torch.unique(torch.cat(semantic_parts), sorted=True)

        spatial_idx = self._append_spatial_candidates(
            candidate_parts,
            spatial_coverage,
            a,
            positions,
            global_idx=global_idx,
        )
        candidate_idx = torch.unique(torch.cat(candidate_parts), sorted=True)
        padded_count = 0
        if candidate_idx.numel() < keep_num:
            missing = keep_num - candidate_idx.numel()
            available = torch.ones(num_tokens, dtype=torch.bool, device=a.device)
            available[candidate_idx] = False
            filler_pool = torch.nonzero(available, as_tuple=False).reshape(-1)
            if filler_pool.numel() > 0:
                filler_count = min(missing, filler_pool.numel())
                filler_idx = filler_pool[
                    torch.topk(a[filler_pool], filler_count, largest=True, sorted=False).indices
                ]
                candidate_idx = torch.unique(torch.cat([candidate_idx, filler_idx]), sorted=True)
                padded_count = filler_idx.numel()
        before_truncation = candidate_idx.numel()
        if candidate_idx.numel() > max_candidates:
            truncate_score = (
                self.beta_a * normalize(a[candidate_idx])
                + self.beta_semantic * normalize(semantic_coverage[candidate_idx])
                + self.beta_spatial * normalize(spatial_coverage[candidate_idx])
            )
            keep_local = torch.topk(
                truncate_score,
                max_candidates,
                largest=True,
                sorted=False,
            ).indices
            candidate_idx = candidate_idx[keep_local].sort().values
        meta = {
            "K": int(keep_num),
            "max_candidates": int(max_candidates),
            "global_budget": int(global_budget),
            "candidate_size_before_truncation": int(before_truncation),
            "candidate_size_after_truncation": int(candidate_idx.numel()),
            "candidate_padded_count": int(padded_count),
            "global_candidate_count": int(self._unique_count(global_idx)),
            "semantic_candidate_count": int(self._unique_count(semantic_idx)),
            "spatial_candidate_count": int(self._unique_count(spatial_idx)),
            "spatial_exclude_global": self.spatial_exclude_global,
            "global_semantic_overlap": self._overlap_ratio(global_idx, semantic_idx),
            "global_spatial_overlap": self._overlap_ratio(global_idx, spatial_idx),
            "semantic_spatial_overlap": self._overlap_ratio(semantic_idx, spatial_idx),
        }
        return candidate_idx.to(device=a.device), meta

    @staticmethod
    def build_patch_positions(num_tokens, device):
        width = max(1, int(math.ceil(math.sqrt(num_tokens))))
        token_idx = torch.arange(num_tokens, device=device)
        y = torch.div(token_idx, width, rounding_mode="floor")
        x = token_idx.remainder(width)
        return torch.stack((x, y), dim=-1).float()

    @staticmethod
    def _prepare_positions(positions, num_tokens, device):
        if torch.is_tensor(positions):
            positions = positions.to(device=device, dtype=torch.float32)
            if positions.ndim == 3:
                positions = positions[0]
            if positions.ndim == 2 and positions.shape == (num_tokens, 2):
                return positions
        return ECPruner.build_patch_positions(num_tokens, device)

    @staticmethod
    def _normalize_matrix(matrix, eps=1e-8):
        matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
        matrix.fill_diagonal_(0.0)
        max_value = matrix.max()
        if not torch.isfinite(max_value) or max_value.item() <= eps:
            return torch.zeros_like(matrix)
        matrix = matrix / max_value
        matrix.fill_diagonal_(0.0)
        return matrix

    @staticmethod
    def _symmetrize_matrix(matrix):
        matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
        matrix = torch.maximum(matrix, matrix.t())
        matrix.fill_diagonal_(0.0)
        return matrix

    @staticmethod
    def _row_topk_sparse(matrix, topk):
        size = matrix.shape[0]
        topk = min(max(0, int(topk)), max(0, size - 1))
        if topk == 0 or size <= 1:
            return torch.zeros_like(matrix)

        values, indices = torch.topk(matrix, topk, dim=-1, largest=True, sorted=False)
        sparse = torch.zeros_like(matrix)
        sparse.scatter_(1, indices, values)
        sparse.fill_diagonal_(0.0)
        return sparse

    @staticmethod
    def _prepare_chain_relation_matrix(relation_matrix, eps=1e-8):
        if (
            not torch.is_tensor(relation_matrix)
            or relation_matrix.ndim != 2
            or relation_matrix.shape[0] != relation_matrix.shape[1]
        ):
            return None

        relation = torch.nan_to_num(relation_matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
        relation = torch.maximum(relation, relation.t())
        relation.fill_diagonal_(0.0)
        max_value = relation.max()
        if not torch.isfinite(max_value) or max_value.item() <= eps:
            return torch.zeros_like(relation)
        relation = relation / max_value
        relation.fill_diagonal_(0.0)
        return relation

    @staticmethod
    def compute_chain_coverage_from_presence(unit_presence, chain_relation_matrix):
        if (
            not torch.is_tensor(unit_presence)
            or not torch.is_tensor(chain_relation_matrix)
            or unit_presence.ndim != 1
            or chain_relation_matrix.shape != (unit_presence.numel(), unit_presence.numel())
        ):
            device = None
            if torch.is_tensor(unit_presence):
                device = unit_presence.device
            elif torch.is_tensor(chain_relation_matrix):
                device = chain_relation_matrix.device
            return torch.tensor(0.0, device=device)

        unit_presence = torch.nan_to_num(
            unit_presence.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_(0.0, 1.0)
        relation = torch.triu(
            torch.nan_to_num(chain_relation_matrix.float(), nan=0.0, posinf=0.0, neginf=0.0),
            diagonal=1,
        )
        pair_presence = unit_presence[:, None] * unit_presence[None, :]
        return (pair_presence * relation).sum()

    @staticmethod
    def compute_chain_coverage_energy(selected_mask, b_c, chain_relation_matrix):
        if (
            not torch.is_tensor(b_c)
            or b_c.ndim != 2
            or b_c.shape[1] == 0
            or not torch.is_tensor(chain_relation_matrix)
            or chain_relation_matrix.shape != (b_c.shape[1], b_c.shape[1])
        ):
            device = b_c.device if torch.is_tensor(b_c) else None
            return torch.tensor(0.0, device=device)

        selected_mask = selected_mask.to(device=b_c.device, dtype=torch.bool).reshape(-1)
        if selected_mask.numel() != b_c.shape[0] or not selected_mask.any():
            return torch.zeros((), dtype=torch.float32, device=b_c.device)

        unit_presence = torch.nan_to_num(
            b_c[selected_mask].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).amax(dim=0)
        return ECPruner.compute_chain_coverage_from_presence(unit_presence, chain_relation_matrix)

    @staticmethod
    def compute_chain_gain_batch(current_presence, b_c, chain_relation_matrix):
        if (
            not torch.is_tensor(current_presence)
            or not torch.is_tensor(b_c)
            or b_c.ndim != 2
            or b_c.shape[1] == 0
            or current_presence.ndim != 1
            or current_presence.numel() != b_c.shape[1]
            or not torch.is_tensor(chain_relation_matrix)
            or chain_relation_matrix.shape != (b_c.shape[1], b_c.shape[1])
        ):
            device = None
            size = 0
            if torch.is_tensor(b_c):
                device = b_c.device
                size = b_c.shape[0]
            elif torch.is_tensor(current_presence):
                device = current_presence.device
            return torch.zeros(size, dtype=torch.float32, device=device)

        current_presence = torch.nan_to_num(
            current_presence.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_(0.0, 1.0)
        b_c = torch.nan_to_num(b_c.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)
        current_score = ECPruner.compute_chain_coverage_from_presence(
            current_presence,
            chain_relation_matrix,
        )
        new_presence = torch.maximum(current_presence.unsqueeze(0), b_c)
        relation = torch.triu(chain_relation_matrix.float(), diagonal=1).unsqueeze(0)
        pair_presence = new_presence[:, :, None] * new_presence[:, None, :]
        new_scores = (pair_presence * relation).sum(dim=(1, 2))
        return new_scores - current_score

    def compute_repulsion(self, features_c, b_c, positions_c):
        candidate_size = features_c.shape[0]
        zeros = torch.zeros(
            candidate_size,
            candidate_size,
            dtype=torch.float32,
            device=features_c.device,
        )
        if not self.use_repulsion or candidate_size <= 1:
            return zeros
        if b_c.ndim != 2 or b_c.shape[0] != candidate_size:
            self._debug("repulsion semantic response unavailable, return zeros")
            return zeros

        features_c = F.normalize(features_c.float(), dim=-1, eps=1e-8)
        visual_similarity = torch.relu(torch.matmul(features_c, features_c.t()))
        semantic_overlap = torch.matmul(b_c.float(), b_c.float().t()).clamp_min(0.0)
        distances_sq = torch.cdist(positions_c.float(), positions_c.float()).square()
        sigma = max(self.sigma, 1e-6)
        spatial_weight = torch.exp(-distances_sq / (2.0 * sigma * sigma))
        repulsion = visual_similarity * semantic_overlap * spatial_weight
        repulsion = self._normalize_matrix(repulsion)
        repulsion = self._row_topk_sparse(repulsion, self.topk_r)
        return self._symmetrize_matrix(repulsion)

    def compute_phi(self, positions_c, question):
        candidate_size = positions_c.shape[0]
        phi = torch.ones(
            candidate_size,
            candidate_size,
            dtype=torch.float32,
            device=positions_c.device,
        )
        if not self.use_phi:
            return phi

        if isinstance(question, (list, tuple)):
            question = question[0] if question else ""
        words = set(re.findall(r"[a-z]+", question.lower())) if isinstance(question, str) else set()
        x = positions_c[:, 0]
        y = positions_c[:, 1]
        rules = []
        if "left" in words:
            rules.append((x[:, None] < x[None, :]).float())
        if "right" in words:
            rules.append((x[:, None] > x[None, :]).float())
        if words & {"above", "over"}:
            rules.append((y[:, None] < y[None, :]).float())
        if words & {"below", "under"}:
            rules.append((y[:, None] > y[None, :]).float())
        if words & {"near", "next", "beside"}:
            distances = torch.cdist(positions_c.float(), positions_c.float())
            rules.append(torch.exp(-distances / max(self.sigma, 1e-6)))

        if rules:
            phi = torch.stack(rules, dim=0).mean(dim=0)
        return phi

    def compute_complementarity(self, a_c, b_c, relation_matrix, positions_c, question):
        candidate_size = a_c.numel()
        zeros = torch.zeros(
            candidate_size,
            candidate_size,
            dtype=torch.float32,
            device=a_c.device,
        )
        if not self.use_complement or candidate_size <= 1:
            return zeros
        if b_c.ndim != 2 or b_c.shape[0] != candidate_size:
            self._debug("complement semantic response unavailable, return zeros")
            return zeros
        if relation_matrix.shape != (b_c.shape[1], b_c.shape[1]):
            self._debug("complement relation matrix shape mismatch, return zeros")
            return zeros

        node_pair_score = torch.sqrt(
            (a_c[:, None] * a_c[None, :]).clamp_min(0.0)
        )
        semantic_relation = torch.matmul(
            torch.matmul(b_c.float(), relation_matrix.float()),
            b_c.float().t(),
        ).clamp_min(0.0)
        phi = self.compute_phi(positions_c, question)
        complementarity = node_pair_score * semantic_relation * phi
        complementarity = self._normalize_matrix(complementarity)
        complementarity = self._row_topk_sparse(complementarity, self.topk_c)
        return self._symmetrize_matrix(complementarity)

    def greedy_select(
        self,
        a_c,
        repulsion_c,
        complementarity_c,
        keep_num,
        b_c=None,
        chain_relation_matrix=None,
        gamma_chain=0.0,
    ):
        candidate_size = a_c.numel()
        keep_num = min(max(0, int(keep_num)), candidate_size)
        if keep_num == 0:
            return torch.empty(0, dtype=torch.long, device=a_c.device)

        selected = []
        available = torch.ones(candidate_size, dtype=torch.bool, device=a_c.device)
        current_presence = None
        if (
            float(gamma_chain) > 0.0
            and torch.is_tensor(b_c)
            and b_c.ndim == 2
            and b_c.shape[0] == candidate_size
            and torch.is_tensor(chain_relation_matrix)
            and chain_relation_matrix.shape == (b_c.shape[1], b_c.shape[1])
        ):
            current_presence = torch.zeros(
                b_c.shape[1],
                dtype=torch.float32,
                device=a_c.device,
            )

        while len(selected) < keep_num:
            energy_score = a_c.clone()
            if selected:
                selected_idx = torch.stack(selected)
                energy_score -= self.lambda_repulsion * repulsion_c[:, selected_idx].sum(dim=-1)
                energy_score += self.delta_complement * complementarity_c[:, selected_idx].sum(dim=-1)
            if current_presence is not None:
                chain_gain = self.compute_chain_gain_batch(
                    current_presence,
                    b_c,
                    chain_relation_matrix,
                )
                energy_score += float(gamma_chain) * chain_gain
            energy_score = energy_score.masked_fill(~available, -float("inf"))
            next_idx = torch.argmax(energy_score)
            selected.append(next_idx)
            available[next_idx] = False
            if current_presence is not None:
                current_presence = torch.maximum(current_presence, b_c[next_idx].float())

        return torch.stack(selected)

    @staticmethod
    def compute_energy(
        selected_mask,
        a_c,
        R_c,
        C_c,
        lambda_coef,
        delta_coef,
        b_c=None,
        chain_relation_matrix=None,
        gamma_chain=0.0,
    ):
        if not torch.is_tensor(selected_mask):
            raise ValueError("selected_mask must be a tensor")

        selected_mask = selected_mask.to(device=a_c.device, dtype=torch.bool).reshape(-1)
        a_c = torch.nan_to_num(a_c.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        if selected_mask.numel() != a_c.numel():
            raise ValueError(
                f"selected_mask size mismatch: {selected_mask.numel()} vs {a_c.numel()}"
            )

        selected_idx = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
        energy = -a_c[selected_idx].sum() if selected_idx.numel() > 0 else a_c.new_zeros(())
        if selected_idx.numel() <= 1:
            return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.is_tensor(R_c):
            repulsion = torch.nan_to_num(R_c.float(), nan=0.0, posinf=0.0, neginf=0.0)
            repulsion = repulsion.index_select(0, selected_idx).index_select(1, selected_idx)
            energy = energy + float(lambda_coef) * torch.triu(repulsion, diagonal=1).sum()

        if torch.is_tensor(C_c):
            complement = torch.nan_to_num(C_c.float(), nan=0.0, posinf=0.0, neginf=0.0)
            complement = complement.index_select(0, selected_idx).index_select(1, selected_idx)
            energy = energy - float(delta_coef) * torch.triu(complement, diagonal=1).sum()

        if float(gamma_chain) > 0.0:
            chain_coverage = ECPruner.compute_chain_coverage_energy(
                selected_mask,
                b_c,
                chain_relation_matrix,
            )
            energy = energy - float(gamma_chain) * chain_coverage

        return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

    def qanneal_select(
        self,
        a_c,
        R_c,
        C_c,
        K,
        candidate_idx=None,
        init_selected=None,
        lambda_coef=0.1,
        delta_coef=0.1,
        b_c=None,
        chain_relation_matrix=None,
        gamma_chain=0.0,
    ):
        start_time = time.perf_counter()
        a_c = torch.nan_to_num(a_c.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        candidate_size = a_c.numel()
        K = max(0, int(K))

        if candidate_idx is None:
            candidate_idx = torch.arange(candidate_size, device=a_c.device)
        candidate_idx = candidate_idx.to(device=a_c.device, dtype=torch.long).reshape(-1)
        if candidate_idx.numel() != candidate_size:
            raise ValueError(
                f"candidate_idx size mismatch: {candidate_idx.numel()} vs {candidate_size}"
            )
        if candidate_size < K:
            raise ValueError(
                f"candidate pool too small for fixed-K selection: candidate_size={candidate_size}, K={K}"
            )

        if candidate_size == 0 or K == 0:
            return candidate_idx[:0], {
                "solver_requested": self.solver,
                "solver_used": "qanneal",
                "greedy_energy": 0.0,
                "best_energy": 0.0,
                "energy_improvement": 0.0,
                "accepted_swaps": 0,
                "improved_swaps": 0,
                "rejected_by_hamming": 0,
                "best_hamming_distance": 0,
                "max_hamming": 0,
                "final_keep_size": 0,
                "qanneal_time": time.perf_counter() - start_time,
            }

        if init_selected is None:
            greedy_local_idx = self.greedy_select(
                a_c,
                R_c if torch.is_tensor(R_c) else torch.zeros(candidate_size, candidate_size, device=a_c.device),
                C_c if torch.is_tensor(C_c) else torch.zeros(candidate_size, candidate_size, device=a_c.device),
                K,
                b_c=b_c,
                chain_relation_matrix=chain_relation_matrix,
                gamma_chain=gamma_chain,
            )
            greedy_mask = self._selected_to_mask(greedy_local_idx, candidate_size, a_c.device)
        else:
            greedy_mask = self._selected_to_mask(init_selected, candidate_size, a_c.device)

        if greedy_mask.sum().item() != K:
            raise ValueError(
                f"greedy init must keep exactly K={K} tokens, got {greedy_mask.sum().item()}"
            )

        greedy_energy = float(
            self.compute_energy(
                greedy_mask,
                a_c,
                R_c,
                C_c,
                lambda_coef,
                delta_coef,
                b_c=b_c,
                chain_relation_matrix=chain_relation_matrix,
                gamma_chain=gamma_chain,
            ).item()
        )
        greedy_chain_coverage = float(
            self.compute_chain_coverage_energy(
                greedy_mask,
                b_c,
                chain_relation_matrix,
            ).item()
        )
        debug_info = {
            "solver_requested": self.solver,
            "solver_used": "qanneal",
            "greedy_energy": greedy_energy,
            "best_energy": greedy_energy,
            "greedy_chain_coverage": greedy_chain_coverage,
            "best_chain_coverage": greedy_chain_coverage,
            "energy_improvement": 0.0,
            "accepted_swaps": 0,
            "improved_swaps": 0,
            "rejected_by_hamming": 0,
            "best_hamming_distance": 0,
            "max_hamming": int(2 * K * self.qa_max_swap_ratio),
            "final_keep_size": int(K),
            "qanneal_time": 0.0,
        }

        if candidate_size <= K:
            keep_idx = candidate_idx[greedy_mask].sort().values
            if keep_idx.numel() != K:
                raise ValueError(
                    f"candidate_size<=K but keep size is {keep_idx.numel()} instead of {K}"
                )
            debug_info["qanneal_time"] = time.perf_counter() - start_time
            return keep_idx, debug_info

        current_mask = greedy_mask.clone()
        best_mask = greedy_mask.clone()
        current_energy = greedy_energy
        best_energy = greedy_energy
        best_hamming_distance = 0

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.qa_seed)
        steps = max(0, int(self.qa_steps))
        max_hamming = debug_info["max_hamming"]

        for step in range(steps):
            selected_idx = torch.nonzero(current_mask, as_tuple=False).reshape(-1)
            unselected_idx = torch.nonzero(~current_mask, as_tuple=False).reshape(-1)
            if selected_idx.numel() == 0 or unselected_idx.numel() == 0:
                break

            out_pos = torch.randint(
                selected_idx.numel(),
                (1,),
                generator=generator,
                device="cpu",
            ).item()
            in_pos = torch.randint(
                unselected_idx.numel(),
                (1,),
                generator=generator,
                device="cpu",
            ).item()
            out_idx = selected_idx[out_pos]
            in_idx = unselected_idx[in_pos]

            proposed_mask = current_mask.clone()
            proposed_mask[out_idx] = False
            proposed_mask[in_idx] = True

            hamming = int((proposed_mask != greedy_mask).sum().item())
            if hamming > max_hamming:
                debug_info["rejected_by_hamming"] += 1
                continue

            new_energy = float(
                self.compute_energy(
                    proposed_mask,
                    a_c,
                    R_c,
                    C_c,
                    lambda_coef,
                    delta_coef,
                    b_c=b_c,
                    chain_relation_matrix=chain_relation_matrix,
                    gamma_chain=gamma_chain,
                ).item()
            )
            delta_energy = new_energy - current_energy
            if steps <= 1:
                temperature = max(self.qa_tend, 1e-8)
            else:
                progress = step / max(steps - 1, 1)
                if self.qa_t0 > 0.0 and self.qa_tend > 0.0:
                    temperature = self.qa_t0 * ((self.qa_tend / self.qa_t0) ** progress)
                else:
                    temperature = max(self.qa_tend, 1e-8)

            accept = delta_energy < 0.0
            if not accept and temperature > 0.0:
                accept_prob = math.exp(max(-60.0, -delta_energy / max(temperature, 1e-8)))
                rand_value = torch.rand(1, generator=generator, device="cpu").item()
                accept = rand_value < accept_prob

            if not accept:
                continue

            current_mask = proposed_mask
            current_energy = new_energy
            debug_info["accepted_swaps"] += 1

            if new_energy < best_energy - self.qa_min_improve:
                best_mask = proposed_mask.clone()
                best_energy = new_energy
                best_hamming_distance = hamming
                debug_info["improved_swaps"] += 1

        keep_idx = candidate_idx[best_mask].sort().values
        if keep_idx.numel() != K:
            raise ValueError(f"qanneal keep size mismatch: expected {K}, got {keep_idx.numel()}")

        debug_info["best_energy"] = best_energy
        debug_info["best_chain_coverage"] = float(
            self.compute_chain_coverage_energy(
                best_mask,
                b_c,
                chain_relation_matrix,
            ).item()
        )
        debug_info["energy_improvement"] = greedy_energy - best_energy
        debug_info["best_hamming_distance"] = best_hamming_distance
        debug_info["final_keep_size"] = int(keep_idx.numel())
        debug_info["qanneal_time"] = time.perf_counter() - start_time
        return keep_idx, debug_info

    def _solve_candidate_subset(
        self,
        a_c,
        R_c,
        C_c,
        candidate_idx,
        keep_num,
        b_c=None,
        chain_relation_matrix=None,
    ):
        if a_c.numel() < keep_num:
            raise ValueError(
                f"candidate subset too small for fixed-K selection: {a_c.numel()} < {keep_num}"
            )
        gamma_chain = self.gamma_chain if self.use_chain_coverage else 0.0
        greedy_local_idx = self.greedy_select(
            a_c,
            R_c,
            C_c,
            keep_num,
            b_c=b_c,
            chain_relation_matrix=chain_relation_matrix,
            gamma_chain=gamma_chain,
        )
        greedy_mask = self._selected_to_mask(greedy_local_idx, a_c.numel(), a_c.device)
        greedy_keep_idx = candidate_idx[greedy_local_idx].sort().values
        greedy_energy = float(
            self.compute_energy(
                greedy_mask,
                a_c,
                R_c,
                C_c,
                self.lambda_repulsion,
                self.delta_complement,
                b_c=b_c,
                chain_relation_matrix=chain_relation_matrix,
                gamma_chain=gamma_chain,
            ).item()
        )
        greedy_chain_coverage = float(
            self.compute_chain_coverage_energy(
                greedy_mask,
                b_c,
                chain_relation_matrix,
            ).item()
        )

        debug_info = {
            "solver_requested": self.solver,
            "solver_used": "greedy",
            "greedy_energy": greedy_energy,
            "best_energy": greedy_energy,
            "greedy_chain_coverage": greedy_chain_coverage,
            "best_chain_coverage": greedy_chain_coverage,
            "energy_improvement": 0.0,
            "accepted_swaps": 0,
            "improved_swaps": 0,
            "rejected_by_hamming": 0,
            "best_hamming_distance": 0,
            "max_hamming": int(2 * keep_num * self.qa_max_swap_ratio),
            "final_keep_size": int(greedy_keep_idx.numel()),
            "qanneal_time": 0.0,
        }

        if self.solver == "greedy":
            return greedy_keep_idx, debug_info

        if self.solver != "qanneal":
            self._debug_qec(f"unknown solver '{self.solver}', fallback to greedy")
            debug_info["solver_used"] = "greedy_fallback"
            return greedy_keep_idx, debug_info

        try:
            keep_idx, qa_debug = self.qanneal_select(
                a_c=a_c,
                R_c=R_c,
                C_c=C_c,
                K=keep_num,
                candidate_idx=candidate_idx,
                init_selected=greedy_mask,
                lambda_coef=self.lambda_repulsion,
                delta_coef=self.delta_complement,
                b_c=b_c,
                chain_relation_matrix=chain_relation_matrix,
                gamma_chain=gamma_chain,
            )
        except Exception as exc:
            self._debug_qec(f"qanneal failed, fallback to greedy: {exc}")
            debug_info["solver_used"] = "greedy_fallback"
            debug_info["fallback_reason"] = str(exc)
            return greedy_keep_idx, debug_info

        if qa_debug["best_energy"] > greedy_energy + max(self.qa_min_improve, 1e-8):
            self._debug_qec(
                "qanneal returned worse energy than greedy, fallback to greedy"
            )
            debug_info["solver_used"] = "greedy_fallback"
            debug_info["fallback_reason"] = "worse_than_greedy"
            return greedy_keep_idx, debug_info

        return keep_idx, qa_debug

    def compute_node_score(
        self,
        visual_tokens,
        relevance=None,
        text_attn=None,
        b=None,
        semantic_response_source=None,
        relation_matrix=None,
        K=None,
    ):
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        num_tokens = visual_tokens.shape[1]
        device = visual_tokens.device
        norm_score = visual_tokens.norm(dim=-1).mean(dim=0)
        source = self.score_source
        actual_source = source
        score = None
        self.last_qmo_info = {}
        self.last_qfid_info = {}

        if source == "norm":
            score = norm_score
        elif source == "relevance":
            if torch.is_tensor(relevance):
                relevance = relevance.float()
                if relevance.ndim == 1:
                    score = relevance
                elif relevance.ndim >= 2:
                    score = relevance.mean(dim=tuple(range(relevance.ndim - 1)))
            if score is None:
                self._debug("relevance unavailable, fallback to norm")
        elif source == "attn":
            score = self.aggregate_text_to_visual_attention(text_attn)
            if score is None:
                self._debug("text_attn unavailable, fallback to norm")
        elif source == "semantic":
            score = self.compute_semantic_node_score(b)
            if score is None:
                self._debug("semantic b unavailable, fallback to norm")
            elif semantic_response_source == "uniform" and score.max().item() <= 1e-8:
                self._debug("semantic response is uniform, fallback to norm")
                score = None
        elif source == "qmo":
            score = self.compute_qmo_node_score(
                visual_tokens,
                relevance=relevance,
                text_attn=text_attn,
                b=b,
                relation_matrix=relation_matrix,
                semantic_response_source=semantic_response_source,
                K=K,
            )
        else:
            self._debug(f"unknown score source '{source}', fallback to norm")

        if score is None or score.numel() != num_tokens:
            if score is not None:
                self._debug(
                    f"{source} score shape mismatch, expected {num_tokens} values, "
                    f"got {score.numel()}; fallback to norm"
                )
            score = norm_score
            actual_source = "norm"

        score = normalize(score.to(device=device)).reshape(-1)
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
        self.last_score_source = actual_source
        return score

    @torch.no_grad()
    def select(
        self,
        visual_tokens,
        keep_num=None,
        keep_ratio=None,
        question=None,
        relevance=None,
        text_attn=None,
        cls_attn=None,
        text_embeds=None,
        positions=None,
        b=None,
        semantic_features=None,
        semantic_units=None,
        transition_prior=None,
        **kwargs
    ):
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")

        _, num_tokens, _ = visual_tokens.shape
        self._debug_config()
        keep_num = self._resolve_keep_num(num_tokens, keep_num, keep_ratio)

        units = semantic_units or self.extract_semantic_units(question)
        relation_matrix = self.build_relation_matrix(units, device=visual_tokens.device)
        chain_relation_matrix = self._prepare_chain_relation_matrix(relation_matrix)
        semantic_response_source = None
        if self.needs_semantics and b is None:
            response_features = semantic_features if semantic_features is not None else visual_tokens
            b, semantic_response_source = self.compute_semantic_response(
                response_features,
                units,
                text_embeds=text_embeds,
            )
            if semantic_response_source == "uniform":
                self._debug("semantic response is uniform; semantic score/coupling may be weak.")

        if self.score_source == "qfid":
            if self.qfid_select_mode == "cls_topk":
                keep_idx, qfid_info = self.select_by_cls_topk(
                    visual_tokens,
                    keep_num,
                    cls_attn=cls_attn,
                )
                selection_formulation = "cls_attention_ranking"
                solver_strategy = "direct_cls_topk"
                solver_used = "cls_topk"
            elif self.qfid_select_mode == "semantic_topk":
                keep_idx, qfid_info = self.select_by_semantic_topk(
                    visual_tokens,
                    keep_num,
                    semantic_response=b,
                    semantic_response_source=semantic_response_source,
                )
                selection_formulation = "semantic_probability_ranking"
                solver_strategy = "direct_semantic_topk"
                solver_used = "semantic_topk"
            elif self.qfid_select_mode == "visual_kcenter":
                keep_idx, qfid_info = self.select_by_visual_kcenter(
                    visual_tokens,
                    keep_num,
                )
                selection_formulation = "visual_cosine_kcenter"
                solver_strategy = "deterministic_farthest_point"
                solver_used = "visual_kcenter"
            else:
                keep_idx, qfid_info = self.select_by_quantum_fidelity(
                    visual_tokens,
                    keep_num,
                    semantic_response=b,
                    relevance=relevance,
                    text_attn=text_attn,
                    cls_attn=cls_attn,
                    transition_prior=transition_prior,
                    question=question,
                    semantic_response_source=semantic_response_source,
                    core_indices_override=kwargs.get("core_indices_override"),
                )
                if qfid_info.get("selector") == "evidence_recover":
                    selection_formulation = "uncovered_evidence_recovery"
                    solver_strategy = "greedy_evidence_coverage"
                    solver_used = "evidence_recover_greedy"
                elif qfid_info.get("selector") == "core_then_recover":
                    selection_formulation = "core_then_uncovered_evidence_recovery"
                    solver_strategy = "qfi_core_plus_greedy_evidence_coverage"
                    solver_used = "core_then_recover_greedy"
                elif qfid_info.get("selector") == "adaptive_core_recover":
                    selection_formulation = "adaptive_core_then_uncovered_evidence_recovery"
                    solver_strategy = "adaptive_qfi_core_plus_prior_anchored_evidence_coverage"
                    solver_used = "adaptive_core_recover_greedy"
                elif qfid_info.get("selector") == "qmo_cr":
                    selection_formulation = "qmo_cr_coupled_recovery"
                    solver_strategy = "multi_observable_coupled_core_recovery"
                    solver_used = "qmo_cr_greedy"
                elif qfid_info.get("selector") == "qmo_gated":
                    selection_formulation = "qmo_gated_adaptive_recovery"
                    solver_strategy = "text_gated_qmo_recovery_with_baseline_fallback"
                    solver_used = "qmo_gated_greedy"
                elif qfid_info.get("selector") in {
                    "qsp_cr_density_projective",
                    "qsp_density_projective_task_anchor",
                    "qsp_density_projective_task_anchor_a05",
                    "qsp_density_projective_complement_bonus",
                }:
                    selection_formulation = "density_projective_core_recover"
                    solver_strategy = "task_conditioned_density_projection_plus_complement_recovery"
                    solver_used = "qsp_density_projective_greedy"
                elif qfid_info.get("selector") == "qfi_core_complement_recovery":
                    selection_formulation = "qfi_core_complement_recovery"
                    solver_strategy = "adaptive_qfi_core_plus_complement_projected_recovery"
                    solver_used = "qfi_core_complement_recovery_greedy"
                elif qfid_info.get("selector") in {
                    "qfi_zeno_stabilized",
                    "qfi_spatial_buffer_recovery",
                    "qfi_progressive_128_to_64",
                    "qfi_coarse_to_fine_pruning",
                }:
                    selection_formulation = qfid_info.get("selector")
                    solver_strategy = "qfi_adaptive_recover_prior_plus_card_variant"
                    solver_used = "qfi_cards_greedy"
                else:
                    selection_formulation = "quantum_fidelity_kernel"
                    solver_strategy = "pivoted_cholesky_residual_greedy"
                    solver_used = "qfid_residual_greedy"
            keep_idx = self._validate_keep_idx(
                keep_idx,
                num_tokens,
                keep_num,
                visual_tokens.device,
            )
            self.last_score_source = "qfid"
            debug_info = {
                "method": "ec_pruner",
                "selection_formulation": selection_formulation,
                "solver_strategy": solver_strategy,
                "requested_score_source": self.score_source,
                "actual_score_source": "qfid",
                "score_source": "qfid",
                "num_tokens": num_tokens,
                "keep_num": keep_num,
                "semantic_units": units,
                "semantic_response_source": semantic_response_source,
                "qfid_bypassed_energy_pipeline": True,
                "use_semantic_candidate": False,
                "use_spatial_candidate": False,
                "spatial_exclude_global": False,
                "use_chain_coverage": False,
                "use_repulsion": False,
                "use_complement": False,
                "use_phi": False,
                "candidate_size": int(qfid_info["candidate_size_after_truncation"]),
                "candidate_meta": {
                    "candidate_source": qfid_info["candidate_source"],
                    "candidate_ratio": qfid_info["candidate_ratio"],
                    "candidate_rel_source": qfid_info["candidate_rel_source"],
                    "candidate_fallback": qfid_info["candidate_fallback"],
                    "candidate_size_before_truncation": qfid_info["candidate_size_before_truncation"],
                    "candidate_size_after_truncation": qfid_info["candidate_size_after_truncation"],
                },
                "candidate_padded_count": int(qfid_info["candidate_padded_count"]),
                "repulsion_nnz": 0,
                "complement_nnz": 0,
                "solver_used": solver_used,
                "solver_debug": {
                    "solver_used": solver_used,
                    "final_keep_size": int(keep_idx.numel()),
                },
                "qfid_info": qfid_info,
                "qmo_info": {},
            }
            return keep_idx, debug_info

        scores = self.compute_node_score(
            visual_tokens,
            relevance=relevance,
            text_attn=text_attn,
            b=b,
            semantic_response_source=semantic_response_source,
            relation_matrix=relation_matrix,
            K=keep_num,
        )
        positions = self._prepare_positions(positions, num_tokens, visual_tokens.device)
        candidate_idx = torch.arange(num_tokens, device=visual_tokens.device)
        candidate_meta = {
            "K": int(keep_num),
            "max_candidates": int(num_tokens),
            "global_budget": int(num_tokens),
            "candidate_size_before_truncation": int(num_tokens),
            "candidate_size_after_truncation": int(num_tokens),
            "candidate_padded_count": 0,
            "global_candidate_count": int(num_tokens),
            "semantic_candidate_count": 0,
            "spatial_candidate_count": 0,
            "global_semantic_overlap": 0.0,
            "global_spatial_overlap": 0.0,
            "semantic_spatial_overlap": 0.0,
        }
        if self.needs_semantics and (self.use_semantic_candidate or self.use_spatial_candidate):
            candidate_idx, candidate_meta = self.build_semantic_candidate_pool(
                scores,
                b,
                keep_num,
                positions=positions,
            )

        candidate_scores = scores[candidate_idx]
        response_features = semantic_features if semantic_features is not None else visual_tokens
        if response_features.ndim == 3:
            response_features = response_features.float().mean(dim=0)
        features_c = response_features[candidate_idx]
        positions_c = positions[candidate_idx]
        if torch.is_tensor(b) and b.ndim == 2 and b.shape[0] == num_tokens:
            b_c = b[candidate_idx]
        else:
            b_c = torch.empty(candidate_idx.numel(), 0, device=visual_tokens.device)

        repulsion_c = self.compute_repulsion(features_c, b_c, positions_c)
        complementarity_c = self.compute_complementarity(
            candidate_scores,
            b_c,
            relation_matrix,
            positions_c,
            question,
        )
        keep_idx, solver_debug = self._solve_candidate_subset(
            candidate_scores,
            repulsion_c,
            complementarity_c,
            candidate_idx,
            keep_num,
            b_c=b_c,
            chain_relation_matrix=chain_relation_matrix,
        )
        full_candidate_hamming_distance = None
        full_candidate_overlap_ratio = None
        if self.debug:
            zero_repulsion = torch.zeros_like(repulsion_c)
            zero_complementarity = torch.zeros_like(complementarity_c)
            candidate_only_keep_idx, _ = self._solve_candidate_subset(
                candidate_scores,
                zero_repulsion,
                zero_complementarity,
                candidate_idx,
                keep_num,
                b_c=b_c,
                chain_relation_matrix=chain_relation_matrix,
            )
            full_candidate_hamming_distance = self._selection_hamming_distance(
                keep_idx,
                candidate_only_keep_idx,
            )
            full_candidate_overlap_ratio = self._selection_overlap_ratio(
                keep_idx,
                candidate_only_keep_idx,
            )
        keep_idx_differs_from_spatial0 = None
        if self.debug and self.use_spatial_candidate:
            old_use_spatial = self.use_spatial_candidate
            self.use_spatial_candidate = False
            try:
                alt_candidate_idx, _ = self.build_semantic_candidate_pool(
                    scores,
                    b,
                    keep_num,
                    positions=positions,
                )
                alt_candidate_scores = scores[alt_candidate_idx]
                alt_features_c = response_features[alt_candidate_idx]
                alt_positions_c = positions[alt_candidate_idx]
                if torch.is_tensor(b) and b.ndim == 2 and b.shape[0] == num_tokens:
                    alt_b_c = b[alt_candidate_idx]
                else:
                    alt_b_c = torch.empty(alt_candidate_idx.numel(), 0, device=visual_tokens.device)
                alt_repulsion_c = self.compute_repulsion(alt_features_c, alt_b_c, alt_positions_c)
                alt_complementarity_c = self.compute_complementarity(
                    alt_candidate_scores,
                    alt_b_c,
                    relation_matrix,
                    alt_positions_c,
                    question,
                )
                alt_keep_idx, _ = self._solve_candidate_subset(
                    alt_candidate_scores,
                    alt_repulsion_c,
                    alt_complementarity_c,
                    alt_candidate_idx,
                    keep_num,
                    b_c=alt_b_c,
                    chain_relation_matrix=chain_relation_matrix,
                )
                keep_idx_differs_from_spatial0 = not torch.equal(keep_idx, alt_keep_idx)
            finally:
                self.use_spatial_candidate = old_use_spatial

        debug_info = {
            "method": "ec_pruner",
            "selection_formulation": "fixed_budget_spin_qubo",
            "solver_strategy": "budget_preserving_spin_exchange"
            if solver_debug["solver_used"] == "qanneal"
            else "fixed_budget_greedy",
            "requested_score_source": self.score_source,
            "actual_score_source": self.last_score_source,
            "score_source": self.last_score_source,
            "num_tokens": num_tokens,
            "keep_num": keep_num,
            "semantic_units": units,
            "relation_shape": tuple(relation_matrix.shape),
            "semantic_response_source": semantic_response_source,
            "use_semantic_candidate": self.use_semantic_candidate,
            "use_spatial_candidate": self.use_spatial_candidate,
            "spatial_exclude_global": self.spatial_exclude_global,
            "use_chain_coverage": self.use_chain_coverage,
            "gamma_chain": self.gamma_chain,
            "chain_active_edges": int(
                0
                if chain_relation_matrix is None
                else torch.count_nonzero(torch.triu(chain_relation_matrix, diagonal=1)).item()
            ),
            "candidate_size": candidate_idx.numel(),
            "candidate_meta": candidate_meta,
            "candidate_padded_count": candidate_meta["candidate_padded_count"],
            "keep_idx_differs_from_spatial0": keep_idx_differs_from_spatial0,
            "full_candidate_hamming_distance": full_candidate_hamming_distance,
            "full_candidate_overlap_ratio": full_candidate_overlap_ratio,
            "use_repulsion": self.use_repulsion,
            "use_complement": self.use_complement,
            "use_phi": self.use_phi,
            "repulsion_nnz": torch.count_nonzero(repulsion_c).item(),
            "complement_nnz": torch.count_nonzero(complementarity_c).item(),
            "solver_used": solver_debug["solver_used"],
            "greedy_chain_coverage": solver_debug["greedy_chain_coverage"],
            "best_chain_coverage": solver_debug["best_chain_coverage"],
            "energy_improvement": solver_debug["energy_improvement"],
            "accepted_swaps": solver_debug["accepted_swaps"],
            "improved_swaps": solver_debug["improved_swaps"],
            "best_hamming_distance": solver_debug["best_hamming_distance"],
            "solver_debug": solver_debug,
            "qmo_info": self.last_qmo_info,
        }
        if self.needs_semantics:
            self._debug(
                f"G_q.shape={tuple(relation_matrix.shape)}, "
                f"b.shape={tuple(b.shape) if torch.is_tensor(b) else None}, "
                f"K={candidate_meta['K']}, "
                f"max_candidates={candidate_meta['max_candidates']}, "
                f"global_budget={candidate_meta['global_budget']}, "
                f"candidate_size_before={candidate_meta['candidate_size_before_truncation']}, "
                f"candidate_size_after={candidate_meta['candidate_size_after_truncation']}, "
                f"candidate_padded={candidate_meta['candidate_padded_count']}, "
                f"global/semantic/spatial_counts="
                f"{candidate_meta['global_candidate_count']}/"
                f"{candidate_meta['semantic_candidate_count']}/"
                f"{candidate_meta['spatial_candidate_count']}, "
                f"spatial_exclude_global={candidate_meta['spatial_exclude_global']}, "
                f"overlap_gs={candidate_meta['global_semantic_overlap']:.3f}, "
                f"overlap_gp={candidate_meta['global_spatial_overlap']:.3f}, "
                f"overlap_sp={candidate_meta['semantic_spatial_overlap']:.3f}, "
                f"semantic_response_source={semantic_response_source}, "
                f"keep_diff_spatial0={keep_idx_differs_from_spatial0}, "
                f"full_candidate_hamming_distance={full_candidate_hamming_distance}, "
                f"full_candidate_overlap_ratio={full_candidate_overlap_ratio}, "
                f"use_chain_coverage={self.use_chain_coverage}, "
                f"gamma_chain={self.gamma_chain}, "
                f"chain_edges={debug_info['chain_active_edges']}, "
                f"best_chain_coverage={debug_info['best_chain_coverage']:.6f}, "
                f"R.nnz={debug_info['repulsion_nnz']}, "
                f"C.nnz={debug_info['complement_nnz']}"
            )
        self._debug_qec(
            f"solver={solver_debug['solver_used']}, "
            f"formulation=fixed_budget_spin_qubo, "
            f"search=budget_preserving_spin_exchange, "
            f"candidate_size={candidate_idx.numel()}, "
            f"K={keep_num}, "
            f"qa_steps={self.qa_steps}, "
            f"qa_T0={self.qa_t0}, "
            f"qa_Tend={self.qa_tend}, "
            f"qa_max_swap_ratio={self.qa_max_swap_ratio}, "
            f"greedy_energy={solver_debug['greedy_energy']:.6f}, "
            f"best_energy={solver_debug['best_energy']:.6f}, "
            f"greedy_chain_coverage={solver_debug['greedy_chain_coverage']:.6f}, "
            f"best_chain_coverage={solver_debug['best_chain_coverage']:.6f}, "
            f"energy_improvement={solver_debug['energy_improvement']:.6f}, "
            f"accepted_swaps={solver_debug['accepted_swaps']}, "
            f"improved_swaps={solver_debug['improved_swaps']}, "
            f"rejected_by_hamming={solver_debug['rejected_by_hamming']}, "
            f"best_hamming_distance={solver_debug['best_hamming_distance']}, "
            f"final_keep_size={solver_debug['final_keep_size']}, "
            f"qanneal_time={solver_debug['qanneal_time']:.6f}"
        )
        keep_idx = self._validate_keep_idx(
            keep_idx,
            num_tokens,
            keep_num,
            visual_tokens.device,
        )
        return keep_idx, debug_info

    @staticmethod
    def _resolve_keep_num(num_tokens, keep_num, keep_ratio):
        if keep_num is None:
            if keep_ratio is None:
                keep_num = num_tokens
            else:
                keep_num = int(round(num_tokens * keep_ratio))

        keep_num = int(keep_num)
        keep_num = max(0, min(keep_num, num_tokens))
        if keep_num == 0 and num_tokens > 0:
            keep_num = 1
        return keep_num
