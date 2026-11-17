"""Small, outcome-supervised active-probe pilot; not a paper-number reproduction."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def boxed(text):
    starts = list(re.finditer(r'\\boxed\s*\{', text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return text[start:i]
    return None


def rational(text):
    """Strict arithmetic-only parsing; never evaluate model-generated code."""
    if text is None:
        return None
    text = text.strip().replace('$', '').replace(',', '').replace(' ', '')
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    match = re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', text)
    try:
        if match:
            value = Fraction(int(match[2]), int(match[3]))
            return str(-value if match[1] == '-' else value)
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', text):
            return str(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass
    return None


def gold(row):
    if 'reference_answer' in row:
        return rational(row['reference_answer'])
    solution = row['reference_solution']
    marker = re.split(r'The answer is:\s*', solution)
    return rational(marker[-1]) if len(marker) > 1 else rational(boxed(solution))


def guard():
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
    selected = [line.split(',') for line in devices.splitlines() if uuid in line]
    if len(selected) != 1 or selected[0][0].strip() not in ('0', '1'):
        raise RuntimeError('Only physical GPU 0/1 selected by UUID is permitted.')
    if int(selected[0][2]) > 100:
        raise RuntimeError('Selected GPU is occupied.')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                         '--format=csv,noheader'], text=True)
    if uuid in processes:
        raise RuntimeError('Selected GPU has a compute process.')


def generate(args):
    guard()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    manifest = dict(vars(args), scope='rational-answer-only pilot',
                    labeling='strict final boxed rational, exact Fraction equality; failures excluded',
                    prompt_suffix='Solve the problem. Put your final answer in \\boxed{...}.')
    (out / 'generation-config.json').write_text(json.dumps(manifest, indent=2))
    began = time.perf_counter()
    counts = {}
    split_limits=[('train',args.train_questions),('validation',args.val_questions),('test',args.test_questions)]
    domain_path=Path(args.questions)/'test_in_domain.jsonl'
    if domain_path.exists():
        split_limits.append(('test_in_domain',len(read(domain_path))))
    for split, limit in split_limits:
        eligible = [r for r in read(Path(args.questions) / f'{split}.jsonl') if gold(r) is not None]
        selected = eligible[:limit]
        if len(selected) != limit:
            raise ValueError(f'Not enough eligible questions for {split}')
        jobs = [(q, i) for q in selected for i in range(args.samples)]
        counts[split] = dict(questions=len(selected), total=0, usable=0, correct=0, truncated=0, extraction_failed=0)
        write(out / f'{split}-questions.jsonl', selected)
        with (out / f'{split}-all.jsonl').open('w') as all_stream, (out / f'{split}.jsonl').open('w') as usable:
            for start in range(0, len(jobs), args.batch_size):
                batch = jobs[start:start + args.batch_size]
                questions = [q['question'] + '\n\n' + manifest['prompt_suffix'] for q, _ in batch]
                prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}],
                           tokenize=False, add_generation_prompt=True) for q in questions]
                inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda:0')
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=0.95,
                                            max_new_tokens=args.max_new_tokens,
                                            pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                for (question, index), prompt_question, tokens in zip(batch, questions, result[:, inputs.input_ids.shape[1]:]):
                    ids = tokens.tolist()
                    eos = tokenizer.eos_token_id
                    complete = eos in ids
                    if complete:
                        ids = ids[:ids.index(eos) + 1]
                    response = tokenizer.decode(ids, skip_special_tokens=True)
                    prediction = rational(boxed(response))
                    status = 'truncated' if not complete else ('extraction_failed' if prediction is None else 'usable')
                    record = dict(question_id=question['question_id'], sample_index=index,
                                  trace_id=f"{question['question_id']}-{index}", question=prompt_question,
                                  raw_question=question['question'], response=response,
                                  source=question['source'], source_type=question.get('source_type'),
                                  reference_answer=gold(question), prediction=prediction, status=status,
                                  generated_tokens=len(ids), correct=int(prediction == gold(question)) if status == 'usable' else None)
                    all_stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    counts[split]['total'] += 1
                    counts[split][status] += 1
                    if status == 'usable':
                        usable.write(json.dumps(record, ensure_ascii=False) + '\n')
                        counts[split]['correct'] += record['correct']
                all_stream.flush()
                usable.flush()
                print(json.dumps(dict(stage='generation', split=split, batch=start // args.batch_size,
                                      seconds=elapsed, counts=counts[split])), flush=True)
                (out / 'generation-summary.json').write_text(json.dumps(dict(counts=counts,
                    seconds=time.perf_counter()-began, peak_bytes=torch.cuda.max_memory_allocated()), indent=2))


def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(y for _, y in pairs[i:j]) * (i + 1 + j) / 2
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores):
    if not sum(labels):
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = seen = 0
    total = 0.0
    i = 0
    while i < len(pairs):
        j = i+1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        positive = sum(y for _, y in pairs[i:j])
        tp += positive
        seen += j-i
        total += positive*tp/seen
        i = j
    return total/sum(labels)


def summarize(predictions, output):
    summary = []
    for split in sorted({r['split'] for r in predictions}):
        for prefix in ('25%','50%','75%','100%','before_final_box','fixed32','fixed64'):
            rows = [r for r in predictions if r['split']==split and r['prefix']==prefix]
            labels = [r['correct'] for r in rows]
            scores = [r['score'] for r in rows]
            groups = sorted({r['question_id'] for r in rows})
            by_question = {q:[r for r in rows if r['question_id']==q] for q in groups}
            positive_questions={r['question_id'] for r in rows if r['correct']}
            negative_questions={r['question_id'] for r in rows if not r['correct']}
            enough_groups=min(len(positive_questions),len(negative_questions))>=5
            rng = random.Random(42)
            bootstrap = []
            for _ in range(1000 if enough_groups else 0):
                sampled = [r for _ in groups for r in by_question[rng.choice(groups)]]
                value = auc([r['correct'] for r in sampled],[r['score'] for r in sampled])
                if value is not None:
                    bootstrap.append(value)
            bootstrap.sort()
            ci = [bootstrap[int(.025*(len(bootstrap)-1))],bootstrap[int(.975*(len(bootstrap)-1))]] if bootstrap else None
            summary.append(dict(split=split,prefix=prefix,n=len(rows),questions=len(groups),positives=sum(labels),
                positive_questions=len(positive_questions),negative_questions=len(negative_questions),
                ci_status='computed' if enough_groups else 'too_few_independent_class_groups',
                auroc=auc(labels,scores),auroc_question_bootstrap_95ci=ci,
                average_precision=average_precision(labels,scores),
                error_average_precision=average_precision([1-y for y in labels],[-s for s in scores]),
                length_auroc=auc(labels,[-r['response_tokens'] for r in rows])))
    (output/'evaluation.json').write_text(json.dumps(summary,indent=2))
    return summary


def prefix_offsets(tokenizer,row):
    length=len(tokenizer.encode(row['response'],add_special_tokens=False))
    offsets=[(f'{int(100*f)}%',max(1,round(f*length))) for f in (.25,.5,.75,1)]
    markers=list(re.finditer(r'\\boxed\s*\{',row['response']))
    if markers:
        text=row['response'][:markers[-1].start()]
        offsets.append(('before_final_box',min(length,len(tokenizer.encode(text,add_special_tokens=False)))))
    offsets.extend((f'fixed{n}',n) for n in (32,64) if length>=n)
    return length,offsets


def train(args):
    guard()
    import torch
    from otv import OTVVerifier
    from otv.data import TraceDataset, TraceCollator
    from otv.train import backward_batch, evaluate
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = OTVVerifier.from_pretrained(args.model, device='cuda:0', dtype=torch.bfloat16,
                                      rank=8, alpha=16, head_dims=(128, 64), head_dropout=0)
    if args.mode == 'no_lora':
        for name, parameter in model.named_parameters():
            if 'lora_' in name:
                parameter.requires_grad_(False)
    collate = TraceCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    datasets = {split: TraceDataset(Path(args.traces) / f'{split}.jsonl', tokenizer, 2048)
                for split in ('train', 'validation')}
    for split in ('train',):
        if len({r['correct'] for r in datasets[split].rows}) < 2:
            raise RuntimeError(f'{split} has only one class; pilot cannot assess discrimination.')
    validation = DataLoader(datasets['validation'], batch_size=1, collate_fn=collate)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    config = dict(vars(args), trainable_parameters=sum(p.numel() for p in parameters),
                  target='linear outcome-derived MSE', head_dims=[128,64], gradient_accumulation=4,
                  limitations='single seed pilot, not parameter-matched to hidden-state baseline')
    (output / 'pilot-config.json').write_text(json.dumps(config, indent=2))
    best = float('inf')
    began = time.perf_counter()
    (output/'initial').mkdir()
    score_model(model,tokenizer,args,output/'initial')
    for epoch in range(args.epochs):
        loader = DataLoader(datasets['train'], batch_size=1, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(args.seed+epoch))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, batch in enumerate(loader):
            model.train()
            group_start = index // 4 * 4
            divisor = min(4, len(loader)-group_start)
            loss = backward_batch(model, batch, chunk_size=args.probe_chunk_size, loss_scale=1/divisor)
            losses.append(loss)
            if (index+1) % 4 == 0 or index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if index % 8 == 0:
                print(json.dumps(dict(stage='train', mode=args.mode, epoch=epoch+1,
                     trace=index+1, total=len(loader), train_mse=loss,
                     seconds=time.perf_counter()-began)), flush=True)
        val = evaluate(model, validation, args.probe_chunk_size, 'linear')
        row = dict(epoch=epoch+1, train_mse=sum(losses)/len(losses), validation_mse=val,
                   seconds=time.perf_counter()-began, peak_bytes=torch.cuda.max_memory_allocated())
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        if val < best:
            best = val
            model.save_pretrained(output/'best', tokenizer)
    score_model(model, tokenizer, args, output, checkpoint=output/'best')


def score_model(model, tokenizer, args, output, checkpoint=None):
    import torch
    from otv.data import encode_trace
    from safetensors.torch import load_file
    if checkpoint is not None:
        # Restore selected checkpoint into existing model to avoid another 4B load.
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.lm, load_file(str(checkpoint/'adapter_model.safetensors')))
        state = load_file(str(checkpoint/'otv_head.safetensors'))
        with torch.no_grad():
            model.truth_embedding.copy_(state.pop('truth_embedding').to(model.device))
        model.head.load_state_dict({k.removeprefix('head.'):v for k,v in state.items()})
    model.eval()
    predictions = []
    with torch.no_grad():
        for split in ('validation','test','test_in_domain'):
            if not (Path(args.traces)/f'{split}.jsonl').exists():
                continue
            for row in read(Path(args.traces)/f'{split}.jsonl'):
                encoded = encode_trace(tokenizer,row['question'],row['response'])
                ids = torch.tensor([encoded['input_ids']],device=model.device)
                start = encoded['response_start']
                # Exclude appended EOS for offline percentage-prefix evaluation.
                length,offsets=prefix_offsets(tokenizer,row)
                positions=[start+n for _,n in offsets]
                scores = model(model.prefill(ids),positions).float().cpu()[0].tolist()
                for (key,_),pos,score in zip(offsets,positions,scores):
                    predictions.append(dict(split=split,trace_id=row['trace_id'],question_id=row['question_id'],
                        correct=row['correct'],prefix=key,response_tokens=pos-start,score=score,
                        full_response_tokens=length))
    write(output/'predictions.jsonl',predictions)
    summary = summarize(predictions,output)
    print(json.dumps(dict(stage='evaluation',mode=args.mode,metrics=summary)),flush=True)
    if checkpoint is not None and args.mode=='otv':
        cache_pairs(model,tokenizer,args,output)


def cache_pairs(model,tokenizer,args,output):
    """Content-use diagnostic at fixed positions within the same question.

    Zeroing response K/V is an out-of-distribution stress test, not a causal
    proof of semantically correct reasoning. Pair selection never uses scores.
    """
    import torch
    from otv import FrozenCache
    from otv.data import encode_trace
    results=[]
    with torch.no_grad():
        for split in ('test','test_in_domain'):
            path=Path(args.traces)/f'{split}.jsonl'
            if not path.exists():
                continue
            records=read(path)
            groups={q:[r for r in records if r['question_id']==q] for q in sorted({r['question_id'] for r in records})}
            for question_id,rows in groups.items():
                positive=sorted([r for r in rows if r['correct']],key=lambda r:r['sample_index'])
                negative=sorted([r for r in rows if not r['correct']],key=lambda r:r['sample_index'])
                if not positive or not negative:
                    continue
                pair=[positive[0],negative[0]]
                assert pair[0]['question']==pair[1]['question']
                lengths=[len(tokenizer.encode(r['response'],add_special_tokens=False)) for r in pair]
                for offset in (32,64):
                    if min(lengths)<offset:
                        continue
                    scores=[]
                    erased=[]
                    for row in pair:
                        encoded=encode_trace(tokenizer,row['question'],row['response'])
                        start=encoded['response_start']
                        position=start+offset
                        ids=torch.tensor([encoded['input_ids'][:position]],device=model.device)
                        cache=model.prefill(ids)
                        scores.append(model(cache,[position]).item())
                        layers=[]
                        for keys,values in cache.layers:
                            k,v=keys.clone(),values.clone()
                            k[:,:,start:]=0
                            v[:,:,start:]=0
                            layers.append((k,v))
                        erased_cache=FrozenCache(tuple(layers),cache.attention_mask)
                        erased.append(model(erased_cache,[position]).item())
                    results.append(dict(split=split,question_id=question_id,response_tokens=offset,
                        correct_trace=pair[0]['trace_id'],incorrect_trace=pair[1]['trace_id'],
                        correct_score=scores[0],incorrect_score=scores[1],margin=scores[0]-scores[1],
                        erased_correct_score=erased[0],erased_incorrect_score=erased[1],
                        erased_margin=erased[0]-erased[1]))
    write(output/'cache-pairs.jsonl',results)
    print(json.dumps(dict(stage='cache_pairs',pairs=len(results))),flush=True)


def hidden_baseline(args):
    guard()
    import torch
    from otv import OTVVerifier
    from otv.model import RegressionHead
    from otv.data import encode_trace
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model = OTVVerifier.from_pretrained(args.model,device='cuda:0',dtype=torch.bfloat16,
                                      rank=8,alpha=16,head_dims=(128,64),head_dropout=0).eval()
    otv_budget=sum(p.numel() for p in model.parameters() if p.requires_grad)
    width=max(1,round((otv_budget-129)/(model.lm.config.hidden_size+65)))
    # Match the total OTV trainable budget, including its LoRA and truth vector.
    model.head=RegressionHead(model.lm.config.hidden_size,(width,64),dropout=0).to(model.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    features = {}
    began = time.perf_counter()
    with torch.no_grad(), model.lm.disable_adapter():
        for split in ('train','validation','test','test_in_domain'):
            if not (Path(args.traces)/f'{split}.jsonl').exists():
                continue
            features[split] = []
            for index,row in enumerate(read(Path(args.traces)/f'{split}.jsonl')):
                encoded = encode_trace(tokenizer,row['question'],row['response'])
                ids = torch.tensor([encoded['input_ids']],device=model.device)
                start = encoded['response_start']
                hidden = model.decoder(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state
                dense_hidden = hidden[0,start-1:].detach().cpu()
                length = dense_hidden.shape[0]-1
                target = .5+(row['correct']-.5)*torch.arange(length+1)/length
                features[split].append(dict(row=row,hidden=dense_hidden,target=target))
                if index%16==0:
                    print(json.dumps(dict(stage='hidden_features',split=split,trace=index+1,
                                          seconds=time.perf_counter()-began)),flush=True)
    config = dict(vars(args),head_dims=[width,64],gradient_accumulation=4,
                  trainable_parameters=sum(p.numel() for p in model.head.parameters()),
                  otv_trainable_parameter_budget=otv_budget,
                  targets='same dense linear targets as OTV',feature='last token final-layer hidden state',
                  scope='single seed pilot; three-layer head; approximately parameter matched to full OTV')
    (output/'pilot-config.json').write_text(json.dumps(config,indent=2))
    optimizer = torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=0)
    best = float('inf')
    for epoch in range(args.epochs):
        indices = list(DataLoader(range(len(features['train'])),batch_size=None,shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed+epoch)))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index,trace_index in enumerate(indices):
            feature = features['train'][trace_index]
            scores = model.head(feature['hidden'].to(model.device))
            loss = (scores-feature['target'].to(model.device)).square().mean()
            divisor = min(4,len(indices)-index//4*4)
            (loss/divisor).backward()
            losses.append(loss.item())
            if (index+1)%4==0 or index+1==len(indices):
                torch.nn.utils.clip_grad_norm_(model.head.parameters(),1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            squared=count=0
            for feature in features['validation']:
                scores=model.head(feature['hidden'].to(model.device))
                squared+=(scores-feature['target'].to(model.device)).square().sum().item()
                count+=len(scores)
        val=squared/count
        metrics=dict(epoch=epoch+1,train_mse=sum(losses)/len(losses),validation_mse=val,
                     seconds=time.perf_counter()-began,peak_bytes=torch.cuda.max_memory_allocated())
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics)+'\n')
        if val<best:
            best=val
            torch.save(model.head.state_dict(),output/'best-head.pt')
        print(json.dumps(metrics),flush=True)
    model.head.load_state_dict(torch.load(output/'best-head.pt',weights_only=True))
    predictions=[]
    with torch.no_grad():
        for split in features:
            if split=='train':
                continue
            for feature in features[split]:
                row=feature['row']
                length,offsets=prefix_offsets(tokenizer,row)
                positions=[n for _,n in offsets]
                scores=model.head(feature['hidden'][positions].to(model.device)).cpu().tolist()
                for (key,_),pos,score in zip(offsets,positions,scores):
                    predictions.append(dict(split=split,trace_id=row['trace_id'],question_id=row['question_id'],
                        correct=row['correct'],prefix=key,response_tokens=pos,score=score,full_response_tokens=length))
    write(output/'predictions.jsonl',predictions)
    print(json.dumps(dict(stage='evaluation',mode='hidden',metrics=summarize(predictions,output))),flush=True)


def benchmark(args):
    guard()
    import torch
    from otv import OTVVerifier
    from otv.data import TraceDataset, TraceCollator
    from otv.train import backward_batch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=OTVVerifier.from_pretrained(args.model,device='cuda:0',dtype=torch.bfloat16,
                                    rank=8,alpha=16,head_dims=(128,64),head_dropout=0)
    dataset=TraceDataset(Path(args.traces)/'train.jsonl',tokenizer,2048)
    indices=sorted(range(len(dataset)),key=lambda i:len(dataset.rows[i]['input_ids']))
    selected=[indices[len(indices)//4],indices[len(indices)//2],indices[-1]]
    collate=TraceCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    results=[]
    for i in selected:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        began=time.perf_counter()
        loss=backward_batch(model,collate([dataset[i]]),chunk_size=args.probe_chunk_size)
        torch.cuda.synchronize()
        row=dict(tokens=len(dataset.rows[i]['input_ids']),loss=loss,chunk_size=args.probe_chunk_size,
                 seconds=time.perf_counter()-began,peak_bytes=torch.cuda.max_memory_allocated())
        results.append(row)
        print(json.dumps(row),flush=True)
    (output/'benchmark.json').write_text(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['generate','train','hidden','benchmark'])
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--questions')
    p.add_argument('--traces')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-questions',type=int,default=32)
    p.add_argument('--val-questions',type=int,default=8)
    p.add_argument('--test-questions',type=int,default=32)
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-new-tokens',type=int,default=768)
    p.add_argument('--mode',choices=['otv','no_lora'],default='otv')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--probe-chunk-size',type=int,default=128)
    args=p.parse_args()
    {'generate':generate,'train':train,'hidden':hidden_baseline,'benchmark':benchmark}[args.stage](args)


if __name__=='__main__':
    main()
