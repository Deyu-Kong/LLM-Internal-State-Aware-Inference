"""Freeze final split membership and record exclusions before fitting probes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--supplement',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    splits={}
    source_map={'train':[(args.initial,'train'),(args.supplement,'train')],
                'validation':[(args.initial,'validation'),(args.supplement,'validation')],
                'test':[(args.initial,'test')],
                'test_in_domain':[(args.supplement,'test')]}
    excluded=Counter()
    provenance=[]
    for split,sources in source_map.items():
        splits[split]=[]
        for directory,source_split in sources:
            path=directory/f'{source_split}-all.jsonl'
            provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for row in read(path):
                if row['status']!='usable':
                    excluded[f'{split}/{row["status"]}']+=1
                    continue
                if row.get('source_type') in ('GSM_SV','GSM_FOBAR','MATH_SV','MATH_FOBAR'):
                    excluded[f'{split}/synthetic_inverse_question_type']+=1
                    continue
                splits[split].append(row)
        with (args.output/f'{split}.jsonl').open('w') as stream:
            for row in splits[split]:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    names=list(splits)
    for index,a in enumerate(names):
        ids={r['question_id'] for r in splits[a]}
        assert len({r['trace_id'] for r in splits[a]})==len(splits[a])
        for b in names[index+1:]:
            assert not ids.intersection(r['question_id'] for r in splits[b]),(a,b)
    sizes={split:dict(traces=len(rows),questions=len({r['question_id'] for r in rows}),
                      correct=sum(r['correct'] for r in rows),incorrect=sum(1-r['correct'] for r in rows))
           for split,rows in splits.items()}
    manifest=dict(sizes=sizes,excluded=dict(excluded),sources=provenance,
                  decision='Frozen before training; checkpoint selection uses validation MSE only',
                  limitations=['Rational-answer-only completed, extractable traces.',
                               'Initial and supplementary generation have different token caps.',
                               'Exact reference matching is not human-verified process correctness.',
                               'Synthetic inverse-question types excluded after malformed-question audit.'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
