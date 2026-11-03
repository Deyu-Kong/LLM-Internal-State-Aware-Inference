"""Descriptive, root-aware diagnostics for repeated pilot runs; CPU only."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time
from datetime import datetime, timezone
from state_inference.manifest import git_state
from state_inference.cli import write_json
from state_inference.data.preparation import read_jsonl
from state_inference.metrics.binary import binary_metrics, undo_positive_class_weight


def summarize(run_dirs, iterations=500):
    runs = []
    reference_ids = None
    for directory in map(Path, run_dirs):
        samples = read_jsonl(directory / 'validation_samples.jsonl')
        predictions = {r['sample_id']: r for r in read_jsonl(directory / 'validation_predictions.jsonl')}
        result = json.loads((directory / 'result.json').read_text())
        ids = [r['sample_id'] for r in samples]
        if set(ids) != set(predictions) or (reference_ids is not None and ids != reference_ids):
            raise ValueError('Validation samples differ across runs')
        reference_ids = ids
        for row in samples:
            if row['label'] != predictions[row['sample_id']]['label']:
                raise ValueError('Prediction label mismatch')
        runs.append((directory, samples, predictions, result))
    if not runs:
        raise ValueError('At least one run required')
    samples = runs[0][1]
    groups = defaultdict(list)
    for row in samples:
        groups[(row['root_id'], tuple(row['prefix_steps']))].append(row)
    pair_groups = [g for g in groups.values() if {x['label'] for x in g} == {0, 1}]
    by_label = {}
    for label in (0, 1):
        rows = [r for r in samples if r['label'] == label]
        by_label[str(label)] = {'samples': len(rows), 'median_input_tokens': statistics.median(len(r['input_ids']) for r in rows),
                                'mean_input_tokens': statistics.mean(len(r['input_ids']) for r in rows),
                                'median_step_index': statistics.median(r['step_index'] for r in rows)}
    root_map = defaultdict(list)
    for i,row in enumerate(samples):root_map[row['root_id']].append(i)
    seed_results = []
    for directory, rows, predictions, result in runs:
        scores = [predictions[r['sample_id']]['trained_score'] for r in rows]
        labels = [r['label'] for r in rows]
        fit = result['baselines']['length_position_fit']
        baseline = []
        for r in rows:
            x = [math.log1p(len(r['input_ids'])), math.log1p(r['step_index'])]
            z = fit['bias'] + sum(w * (v-m)/s for w,v,m,s in zip(fit['weight'],x,fit['mean'],fit['scale']))
            baseline.append(1/(1+math.exp(-z)))
        pairs = []
        pair_map = {r['sample_id']: predictions[r['sample_id']]['trained_score'] for r in rows}
        for group in pair_groups:
            for pos in (r for r in group if r['label'] == 1):
                for neg in (r for r in group if r['label'] == 0):
                    a,b=pair_map[pos['sample_id']],pair_map[neg['sample_id']]
                    pairs.append(float(a>b)+0.5*float(a==b))
        rng = random.Random(731)
        roots = sorted(root_map)
        deltas = []
        for _ in range(iterations):
            indexes = [i for _ in roots for i in root_map[rng.choice(roots)]]
            ys = [labels[i] for i in indexes]
            if len(set(ys)) != 2:continue
            p = binary_metrics(ys,[scores[i] for i in indexes])['auroc']
            q = binary_metrics(ys,[baseline[i] for i in indexes])['auroc']
            deltas.append(p-q)
        deltas.sort()
        corrected = [undo_positive_class_weight(p, result['class_pos_weight']) for p in scores]
        seed_results.append({'run': directory.name, 'validation': result['after']['validation'],
                             'weight_adjusted_validation': binary_metrics(labels, corrected),
                             'weight_adjustment_note': 'Analytical inversion of training class weight; assumes ideal weighted-risk optimum and does not establish calibration',
                             'baseline': result['baselines']['length_position'],
                             'paired_candidate_count': len(pairs), 'paired_candidate_accuracy': statistics.mean(pairs) if pairs else None,
                             'auroc_delta_vs_length': result['after']['validation']['auroc']-result['baselines']['length_position']['auroc'],
                             'root_bootstrap_95pct_delta': [deltas[int(.025*(len(deltas)-1))],deltas[int(.975*(len(deltas)-1))]] if deltas else None,
                             'bootstrap_replicates': len(deltas), 'note': 'Exploratory root-cluster bootstrap, not a formal test'})
    control = None
    indexed = {directory.name: (rows, preds) for directory, rows, preds, _ in runs}
    pre = indexed.get('prm800k-control-pretrained-42')
    rand = indexed.get('prm800k-control-random-42')
    if pre and rand:
        labels = [r['label'] for r in samples]
        pre_scores = [pre[1][r['sample_id']]['trained_score'] for r in samples]
        rand_scores = [rand[1][r['sample_id']]['trained_score'] for r in samples]
        rng = random.Random(732)
        roots = sorted(root_map)
        deltas = []
        for _ in range(iterations):
            indexes = [i for _ in roots for i in root_map[rng.choice(roots)]]
            ys = [labels[i] for i in indexes]
            if len(set(ys)) < 2: continue
            deltas.append(binary_metrics(ys,[pre_scores[i] for i in indexes])['auroc'] -
                          binary_metrics(ys,[rand_scores[i] for i in indexes])['auroc'])
        deltas.sort()
        control = {'metric':'pretrained_minus_random_auroc',
                   'delta':binary_metrics(labels,pre_scores)['auroc'] - binary_metrics(labels,rand_scores)['auroc'],
                   'root_bootstrap_95pct_delta':[deltas[int(.025*(len(deltas)-1))],deltas[int(.975*(len(deltas)-1))]],
                   'replicates':len(deltas), 'note':'One initialization seed; exploratory root-cluster bootstrap'}
    return {'validation_samples':len(samples),'validation_roots':len(root_map),'by_label':by_label,
            'prefix_groups':len(groups),'mixed_label_prefix_groups':len(pair_groups), 'runs':seed_results,
            'backbone_control':control} 


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    start=time.perf_counter()
    result=summarize(args.run)
    result.update(git=git_state(),timestamp=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-start,
                  reproduce_argv=["python","scripts/analyze_probe_pilot.py"] + sum((["--run",r] for r in args.run),[]) + ["--output",args.output])
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
