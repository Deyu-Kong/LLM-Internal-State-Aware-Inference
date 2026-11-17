"""Post-hoc conservative numeric-literal leakage diagnostic, not semantic proof."""
import argparse
import json
from pathlib import Path
import re

from run_probe_pilot import read, rational, auc


def numeric_values(text):
    candidates=re.findall(r'[+-]?\\(?:dfrac|tfrac|frac)\s*\{[+-]?\d+\}\s*\{[+-]?\d+\}',text)
    candidates+=re.findall(r'(?<![A-Za-z0-9])[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?',text)
    return {value for candidate in candidates if (value:=rational(candidate)) is not None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--traces',required=True)
    p.add_argument('--otv',required=True)
    p.add_argument('--no-lora',required=True)
    p.add_argument('--hidden',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    predictions={name:read(Path(directory)/'predictions.jsonl') for name,directory in
                 [('otv',args.otv),('no_lora',args.no_lora),('hidden',args.hidden)]}
    result=[]
    for split in ('test','test_in_domain'):
        rows=read(Path(args.traces)/f'{split}.jsonl')
        for length in (32,64):
            eligible=[]
            for row in rows:
                ids=tokenizer.encode(row['response'],add_special_tokens=False)
                if len(ids)<length:
                    continue
                text=tokenizer.decode(ids[:length])
                if numeric_values(text).intersection({row['reference_answer'],row['prediction']}):
                    continue
                eligible.append(row)
            keys={r['trace_id'] for r in eligible}
            metrics={}
            for name,records in predictions.items():
                selected=[r for r in records if r['split']==split and r['prefix']==f'fixed{length}' and r['trace_id'] in keys]
                metrics[name]=auc([r['correct'] for r in selected],[r['score'] for r in selected])
            result.append(dict(split=split,prefix=f'fixed{length}',total=len(rows),eligible=len(eligible),
                questions=len({r['question_id'] for r in eligible}),correct=sum(r['correct'] for r in eligible),
                auroc=metrics,trace_ids=sorted(keys)))
    report=dict(scope='post-hoc diagnostic; no numeric literal equal to predicted or gold answer in the observed response prefix',
                limitations=['May miss number words and semantic equivalents.',
                             'May conservatively exclude a given number or step index equal to the answer.',
                             'Selection uses completed responses and references; not a deployable stopping rule.'],results=result)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
