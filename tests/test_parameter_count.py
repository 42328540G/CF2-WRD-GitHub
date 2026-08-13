from cf2wrd.model import CF2WRD


def nparams(model):
    return sum(p.numel() for p in model.parameters())


def test_parameter_count_is_fixed_across_activity_masks():
    counts=[]
    for k in (1, 64, 512, 1024):
        model=CF2WRD(pretrained=False, selected_local_indices=range(k))
        counts.append(nparams(model))
    assert counts == [8_989_568] * 4
