"""Reconstruct training sample exposure from saved order, seed and actual update count."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import time
from state_inference.cli import write_json
from state_inference.data.preparation import read_jsonl
from state_inference.data.sources import sha256_file
from state_inference.manifest import git_state


def exposure_indices(count,updates,accumulation,seed):
    if count<1 or updates<0 or accumulation<1:raise ValueError('Invalid exposure counts')
    rng=random.Random(seed);order=[];offset=0;indices=[]
    for _ in range(updates*accumulation):
        if offset>=len(order):
            order=list(range(count));rng.shuffle(order);offset=0
        indices.append(order[offset]);offset+=1
    return indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();start=time.perf_counter();runs=[]
    for directory in map(Path,args.run):
        rows=read_jsonl(directory/'train_samples.jsonl')
        manifest=json.loads((directory/'manifest.json').read_text())
        result=json.loads((directory/'result.json').read_text())
        cfg=manifest['config']
        indices=exposure_indices(len(rows),result['optimizer_steps'],cfg['gradient_accumulation'],cfg['model_seed'])
        seen=[rows[i] for i in indices]
        runs.append({'run':directory.name,'seed':cfg['model_seed'],'optimizer_steps':result['optimizer_steps'],
                     'gradient_accumulation':cfg['gradient_accumulation'],'training_pool_samples':len(rows),
                     'training_pool_roots':len({r['root_id'] for r in rows}),
                     'sample_exposures':len(seen),'unique_samples':len({r['sample_id'] for r in seen}),
                     'unique_roots':len({r['root_id'] for r in seen}),
                     'positive_exposures':sum(r['label'] for r in seen),'negative_exposures':sum(1-r['label'] for r in seen),
                     'train_samples_sha256':sha256_file(directory/'train_samples.jsonl')})
    result={'status':'success','runs':runs,'git':git_state(),'timestamp':datetime.now(timezone.utc).isoformat(),
            'runtime_seconds':time.perf_counter()-start,
            'method':'Replay Python Random(seed) shuffles using actual optimizer steps and accumulation; no text or score changes',
            'reproduce_argv':['python','scripts/audit_training_exposure.py']+sum((['--run',x] for x in args.run),[])+['--output',args.output]}
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);write_json(out,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
