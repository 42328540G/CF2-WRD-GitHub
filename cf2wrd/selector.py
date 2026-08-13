from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from tqdm import tqdm


@dataclass
class SelectorResult:
    """Training-set Pearson feature-activity result."""

    selected_local_indices: List[int]
    max_abs_corr_local: torch.Tensor
    correlation_matrix: torch.Tensor
    mean_abs_correlation_all: float
    mean_abs_correlation_selected: float


@torch.no_grad()
def fit_local_complementary_selector(
    encoder,
    loader: Iterable,
    device: torch.device,
    tau: float = 0.7,
) -> SelectorResult:
    """Fit the final CF2-WRD Pearson activity mask on the training split only.

    Final manuscript rule
    ---------------------
    1. Retain every global feature dimension.
    2. For each local feature dimension j, compute
           m_j = max_i |rho_ij|
       over all global dimensions i.
    3. Activate local dimension j iff m_j <= tau.
    4. Freeze this activity mask and reuse it unchanged for validation/test.

    No minimum-dimension fallback is applied. The selected indices control sparse
    projection activity; they do not change the stored 1024->512 parameterization.
    """
    encoder.eval()

    n = 0
    sum_g = sum_l = sum_g2 = sum_l2 = cross = None

    for batch in tqdm(loader, desc="Fitting Pearson selector"):
        gx = batch["global_image"].to(device, non_blocking=True)
        lx = batch["local_image"].to(device, non_blocking=True)
        g, l = encoder(gx, lx)

        # Accumulate sufficient statistics on CPU in float64 for numerical stability.
        g = g.detach().double().cpu()
        l = l.detach().double().cpu()

        if sum_g is None:
            sum_g = torch.zeros(g.shape[1], dtype=torch.float64)
            sum_l = torch.zeros(l.shape[1], dtype=torch.float64)
            sum_g2 = torch.zeros_like(sum_g)
            sum_l2 = torch.zeros_like(sum_l)
            cross = torch.zeros(g.shape[1], l.shape[1], dtype=torch.float64)

        n += g.shape[0]
        sum_g += g.sum(dim=0)
        sum_l += l.sum(dim=0)
        sum_g2 += (g * g).sum(dim=0)
        sum_l2 += (l * l).sum(dim=0)
        cross += g.T @ l

    if n < 2:
        raise ValueError("At least two training samples are required to fit the Pearson selector.")

    cov_num = cross - torch.outer(sum_g, sum_l) / n
    var_g_num = (sum_g2 - (sum_g * sum_g) / n).clamp_min(1e-12)
    var_l_num = (sum_l2 - (sum_l * sum_l) / n).clamp_min(1e-12)
    denom = torch.sqrt(torch.outer(var_g_num, var_l_num)).clamp_min(1e-12)
    corr = (cov_num / denom).clamp(-1.0, 1.0)

    max_abs_local = corr.abs().max(dim=0).values
    selected = torch.where(max_abs_local <= tau)[0]

    if selected.numel() == 0:
        raise RuntimeError(
            "Pearson masking retained zero active local feature dimensions. "
            "The final manuscript rule does not define a fallback; inspect the "
            "training data/features or revise tau explicitly before training."
        )

    selected_corr = corr[:, selected]
    return SelectorResult(
        selected_local_indices=selected.tolist(),
        max_abs_corr_local=max_abs_local.float(),
        correlation_matrix=corr.float(),
        mean_abs_correlation_all=float(corr.abs().mean().item()),
        mean_abs_correlation_selected=float(selected_corr.abs().mean().item()),
    )
