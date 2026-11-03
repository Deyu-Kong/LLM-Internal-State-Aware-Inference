"""Compare question fingerprints only; no benchmark labels or scores are used."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import yaml
from state_inference.cli import write_json
from state_inference.data.preparation import read_jsonl
from state_inference.data.sources import sha256_file
from state_inference.data.splits import question_id
from state_inference.manifest import git_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--selected',required=True)
    p.add_argument('--raw-dir',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    start=time.perf_counter()
    cfg=yaml.safe_load(Path(args.config).read_text())
    selected=defaultdict(set)
    for row in read_jsonl(Path(args.selected)):
        selected[row['split']].add(row['root_id'].split(':',1)[1])
    partitions={}
    for name,expected in cfg['files'].items():
        path=Path(args.raw_dir)/name
        if sha256_file(path)!=expected:raise ValueError(f'Hash mismatch: {name}')
        rows=json.loads(path.read_text())
        roots={question_id(row['problem']) for row in rows}
        partitions[name]={'records':len(rows),'unique_questions':len(roots),'sha256':expected,
                          'overlap_roots':{k:len(v & roots) for k,v in selected.items()}}
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'success','timestamp':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-start,
            'git':git_state(),'config':cfg,'selected_sha256':sha256_file(args.selected),
            'selected_roots':{k:len(v) for k,v in selected.items()},'partitions':partitions,
            'method':'NFKC + whitespace collapse + casefold + SHA256; exact normalized text only',
            'benchmark_labels_used':False,'limitations':'Does not detect semantic paraphrases or pretraining contamination',
            'reproduce_argv':['python','scripts/audit_processbench_overlap.py','--config',args.config,'--selected',args.selected,'--raw-dir',args.raw_dir,'--output',args.output]}
    write_json(out,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
