import torch
from torch import nn

from cf2wrd.selector import fit_local_complementary_selector


class IdentityEncoder(nn.Module):
    def forward(self, global_x, local_x):
        return global_x, local_x


def test_strict_pearson_rule_has_no_fallback():
    # local dim 0 is perfectly correlated with global dim 0 -> mask/deactivate.
    # local dim 1 is below tau against both global dims -> keep.
    g = torch.tensor(
        [
            [-3.0, 0.0],
            [-2.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [3.0, 0.0],
        ]
    )
    l = torch.stack(
        [
            g[:, 0],
            torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0]),
        ],
        dim=1,
    )
    loader = [{"global_image": g, "local_image": l}]
    result = fit_local_complementary_selector(
        IdentityEncoder(), loader, device=torch.device("cpu"), tau=0.7
    )
    assert result.selected_local_indices == [1]
    assert result.max_abs_corr_local[0] > 0.99
    assert result.max_abs_corr_local[1] <= 0.7


def test_zero_selected_dimensions_raises():
    g = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    l = g.clone()  # perfect correlation -> all masked/deactivated
    loader = [{"global_image": g, "local_image": l}]
    try:
        fit_local_complementary_selector(
            IdentityEncoder(), loader, device=torch.device("cpu"), tau=0.7
        )
    except RuntimeError as exc:
        assert "zero active local feature dimensions" in str(exc)
    else:
        raise AssertionError("Expected strict selector to raise when no local dims survive")
