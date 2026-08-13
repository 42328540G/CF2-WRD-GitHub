import torch

from cf2wrd.model import CF2WRD


def test_forward_smoke():
    model = CF2WRD(
        pretrained=False,
        num_classes=2,
        global_dim=512,
        local_dim=1024,
        fusion_dim=128,
        dropout=0.25,
        selected_local_indices=list(range(64)),
    )
    model.eval()
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        y = model(x, x)
    assert y.shape == (1, 2)


def test_final_vector_level_dimensions():
    model = CF2WRD(
        pretrained=False,
        num_classes=2,
        global_dim=512,
        local_dim=1024,
        fusion_dim=512,
        dropout=0.25,
        selected_local_indices=list(range(64)),
    )
    assert model.encoder.global_proj.out_features == 512
    assert model.encoder.local_proj.out_features == 1024
    assert model.acffm.global_align.out_features == 512
    assert model.acffm.local_align.out_features == 512
    assert model.classifier.out_features == 2
