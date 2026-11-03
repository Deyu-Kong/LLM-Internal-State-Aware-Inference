import pytest
torch=pytest.importorskip('torch')
from state_inference.training.calibration import fit_calibration, apply_platt, metrics_at_threshold


def test_calibrator_is_monotone_and_uses_fitted_threshold():
    labels=[0,0,1,1]
    scores=[0.1,0.3,0.7,0.9]
    fitted=fit_calibration(labels,scores)
    transformed=apply_platt(scores,fitted)
    assert fitted['scale']>0
    assert transformed==sorted(transformed)
    assert metrics_at_threshold(labels,transformed,fitted['threshold'])['f1_at_calibrated_threshold']==1
    with pytest.raises(ValueError):fit_calibration([1,1],[0.5,0.6])
