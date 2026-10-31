import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from progression_demo import simulate, stack, exp_fusion, exp_longitudinal, exp_causal, exp_missing


def test_simulate_shapes_and_labels():
    D = simulate(500, seed=1)
    assert stack(D["X"]).shape == (500, 13)
    assert set(np.unique(D["y"])) <= {0, 1}


def test_fusion_beats_every_single_modality():
    r = exp_fusion(seed=0)
    best_single = max(r[k][0] for k in ("imaging only", "wearable only", "tabular only"))
    assert r["early fusion (one model, all features)"][0] > best_single


def test_longitudinal_rate_is_recovered():
    r = exp_longitudinal(seed=0, visits=4)
    assert abs(r["est_rate"] - r["true_rate"]) < 0.1 and r["offset_corr"] > 0.9


def test_confounding_biases_naive_but_not_oracle():
    c = exp_causal(reps=30)
    assert abs(c["naive difference"][0]) > 0.3
    assert abs(c["regression adjusted (true z, oracle)"][0]) < 0.05


def test_more_missing_data_lowers_auc():
    rates, out = exp_missing(seed=0)
    for v in out.values():
        assert v[0] > v[-1]
