import runpy
from pathlib import Path
from state_inference.metrics.binary import binary_metrics


def test_fast_auc_matches_reference_with_ties():
    fn=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/analyze_expanded.py'))['fast_auc']
    labels=[1,0,1,0,1,0]
    scores=[.8,.8,.6,.2,.2,.1]
    assert fn(labels,scores)==binary_metrics(labels,scores)['auroc']
    assert fn([1,1],[.2,.3]) is None


def test_data_size_analysis_requires_identical_test_samples(tmp_path):
    import json
    import pytest
    m=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/analyze_expanded.py'))
    dirs=[]
    for name,ids in [('prm800k-expanded-seed42',['a','b']),('prm800k-expanded-train200-control',['a','other'])]:
        path=tmp_path/name;path.mkdir();dirs.append(path)
        samples=[{'sample_id':v,'root_id':v,'prefix_steps':[],'label':i} for i,v in enumerate(ids)]
        predictions=[{'sample_id':v,'label':i,'raw_score':.2+.5*i,'length_position_score':.3+.4*i} for i,v in enumerate(ids)]
        for file,rows in [('test_samples.jsonl',samples),('test_predictions.jsonl',predictions)]:
            (path/file).write_text(''.join(json.dumps(x)+'\n' for x in rows))
        metric={'auroc':1}
        (path/'result.json').write_text(json.dumps({'test_raw':metric,'test_calibrated':metric,'length_position_baseline':metric,'matched_pairs':{}}))
    with pytest.raises(ValueError,match='differ'):
        m['analyze'](dirs,iterations=2)
