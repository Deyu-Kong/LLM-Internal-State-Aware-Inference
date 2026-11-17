"""Compare saved predictions with paired question-level uncertainty estimates."""
import argparse
import importlib.util
import json
from pathlib import Path
import random

spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('run_probe_pilot.py'))
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--otv',required=True)
    p.add_argument('--no-lora',required=True)
    p.add_argument('--hidden',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    models={name:pilot.read(Path(path)/'predictions.jsonl') for name,path in
            [('otv',args.otv),('no_lora',args.no_lora),('hidden',args.hidden)]}
    result=[]
    for split in sorted({r['split'] for r in models['otv']}):
        for prefix in sorted({r['prefix'] for r in models['otv']}):
            rows={m:{r['trace_id']:r for r in data if r['split']==split and r['prefix']==prefix}
                  for m,data in models.items()}
            keys=sorted(set.intersection(*(set(r) for r in rows.values())))
            qids=sorted({rows['otv'][k]['question_id'] for k in keys})
            groups={q:[k for k in keys if rows['otv'][k]['question_id']==q] for q in qids}
            if not keys:
                continue
            y=[rows['otv'][k]['correct'] for k in keys]
            record=dict(split=split,prefix=prefix,n=len(keys),questions=len(qids),positives=sum(y),
                        auroc={m:pilot.auc(y,[rows[m][k]['score'] for k in keys]) for m in rows})
            enough_groups=min(len({rows['otv'][k]['question_id'] for k in keys if rows['otv'][k]['correct']}),
                              len({rows['otv'][k]['question_id'] for k in keys if not rows['otv'][k]['correct']}))>=5
            record['ci_status']='computed' if enough_groups else 'too_few_independent_class_groups'
            paired={m:[] for m in ('no_lora','hidden')}
            rng=random.Random(42)
            for _ in range(2000 if enough_groups else 0):
                sampled=[k for _ in qids for k in groups[rng.choice(qids)]]
                labels=[rows['otv'][k]['correct'] for k in sampled]
                a=pilot.auc(labels,[rows['otv'][k]['score'] for k in sampled])
                for m in paired:
                    b=pilot.auc(labels,[rows[m][k]['score'] for k in sampled])
                    if a is not None and b is not None:
                        paired[m].append(a-b)
            record['otv_minus_baseline_95ci']={}
            for m,values in paired.items():
                values.sort()
                record['otv_minus_baseline_95ci'][m]=[
                    values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None
            # Within-question metric removes between-question difficulty differences.
            within={m:[] for m in rows}
            for q,group in groups.items():
                labels=[rows['otv'][k]['correct'] for k in group]
                for m in rows:
                    value=pilot.auc(labels,[rows[m][k]['score'] for k in group])
                    if value is not None:
                        within[m].append(value)
            record['mixed_outcome_questions']=len(within['otv'])
            record['mean_within_question_auroc']={m:sum(v)/len(v) if v else None for m,v in within.items()}
            record['within_question_auroc_95ci']={}
            for m,values in within.items():
                if len(values)<5:
                    record['within_question_auroc_95ci'][m]=None
                    continue
                rng=random.Random(42)
                estimates=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(2000))
                record['within_question_auroc_95ci'][m]=[estimates[49],estimates[1949]]
            result.append(record)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
