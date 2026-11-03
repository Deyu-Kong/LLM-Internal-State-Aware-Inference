import runpy
from pathlib import Path
from state_inference.metrics.binary import binary_metrics


def test_fast_auc_matches_reference_with_ties():
    fn=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/analyze_expanded.py'))['fast_auc']
    labels=[1,0,1,0,1,0]
    scores=[.8,.8,.6,.2,.2,.1]
    assert fn(labels,scores)==binary_metrics(labels,scores)['auroc']
    assert fn([1,1],[.2,.3]) is None
