"""Reload the selected checkpoint and compare a saved trace's prefix scores."""
import argparse
import json
from pathlib import Path

from run_probe_pilot import guard, prefix_offsets, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--traces',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    guard()
    import torch
    from transformers import AutoTokenizer
    from otv import OTVVerifier
    from otv.data import encode_trace
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    model=OTVVerifier.from_pretrained(checkpoint=args.checkpoint,device='cuda:0',dtype=torch.bfloat16).eval()
    row=read(args.traces)[0]
    encoded=encode_trace(tokenizer,row['question'],row['response'])
    _,offsets=prefix_offsets(tokenizer,row)
    positions=[encoded['response_start']+n for _,n in offsets]
    records=[r for r in read(args.predictions) if r['split']=='test' and r['trace_id']==row['trace_id']]
    expected=torch.tensor([next(r['score'] for r in records if r['prefix']==key) for key,_ in offsets])
    with torch.no_grad():
        ids=torch.tensor([encoded['input_ids']],device=model.device)
        actual=model(model.prefill(ids),positions)[0].float().cpu()
    torch.testing.assert_close(actual,expected,atol=3e-3,rtol=3e-3)
    result=dict(status='passed',trace_id=row['trace_id'],prefixes=[key for key,_ in offsets],
                max_absolute_difference=(actual-expected).abs().max().item(),
                expected=expected.tolist(),actual=actual.tolist())
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
