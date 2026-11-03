"""Evaluate held-out rankings with root-cluster bootstrap; no fitting or tuning."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import statistics
import time
from state_inference.cli import write_json
from state_inference.data.preparation import read_jsonl
from state_inference.metrics.binary import binary_metrics
from state_inference.manifest import git_state


def analyze(run_dirs, iterations=500):
    references=None
    output=[]
    for directory in map(Path,run_dirs):
        rows=read_jsonl(directory/'test_samples.jsonl')
        predictions={x['sample_id']:x for x in read_jsonl(directory/'test_predictions.jsonl')}
        result=json.loads((directory/'result.json').read_text())
        ids=[r['sample_id'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(predictions) or (references is not None and references!=ids):
            raise ValueError('Test samples differ or predictions missing')
        references=ids
        if any(r['label']!=predictions[r['sample_id']]['label'] for r in rows):raise ValueError('Labels differ')
        root_indexes=defaultdict(list);groups=defaultdict(list)
        for i,row in enumerate(rows):
            root_indexes[row['root_id']].append(i)
            groups[(row['root_id'],tuple(row['prefix_steps']))].append(i)
        pairs_by_root=defaultdict(list)
        for (root,_),indexes in groups.items():
            for a in indexes:
                for b in indexes:
                    if rows[a]['label']==1 and rows[b]['label']==0:pairs_by_root[root].append((a,b))
        scores=[predictions[r['sample_id']]['raw_score'] for r in rows]
        base=[predictions[r['sample_id']]['length_position_score'] for r in rows]
        labels=[r['label'] for r in rows]
        def pair_value(p,q):return float(p>q)+0.5*float(p==q)
        rng=random.Random(994)
        roots=sorted(root_indexes);auc_deltas=[];pair_deltas=[]
        for _ in range(iterations):
            drawn=[rng.choice(roots) for _ in roots]
            indexes=[i for root in drawn for i in root_indexes[root]]
            ys=[labels[i] for i in indexes]
            if len(set(ys))==2:
                auc_deltas.append(binary_metrics(ys,[scores[i] for i in indexes])['auroc']-binary_metrics(ys,[base[i] for i in indexes])['auroc'])
            pairs=[pair for root in drawn for pair in pairs_by_root[root]]
            if pairs:pair_deltas.append(statistics.mean(pair_value(scores[a],scores[b])-pair_value(base[a],base[b]) for a,b in pairs))
        def interval(values):
            values.sort()
            return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None
        output.append({'run':directory.name,'test':result['test_raw'],'test_calibrated':result['test_calibrated'],
                       'baseline':result['length_position_baseline'],'matched_pairs':result['matched_pairs'],
                       'auroc_delta_vs_baseline':result['test_raw']['auroc']-result['length_position_baseline']['auroc'],
                       'root_bootstrap_95pct_auroc_delta':interval(auc_deltas),
                       'root_bootstrap_95pct_pair_accuracy_delta':interval(pair_deltas),
                       'matched_pair_roots':len(pairs_by_root),'bootstrap_replicates':len(auc_deltas)})
    return {'runs':output,'note':'Exploratory root-cluster bootstrap on the same held-out roots; not independent across model seeds'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();start=time.perf_counter()
    result=analyze(args.run)
    result.update(git=git_state(),timestamp=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-start,
                  reproduce_argv=['python','scripts/analyze_expanded.py']+sum((['--run',r] for r in args.run),[])+['--output',args.output])
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);write_json(out,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
