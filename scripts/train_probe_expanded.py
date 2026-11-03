"""Fixed fresh-root PRM800K expansion; CPU preflight then bounded single-GPU training."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import yaml
from state_inference.cli import write_json
from state_inference.data.preparation import read_jsonl,write_jsonl
from state_inference.data.prm800k import extract_candidates
from state_inference.data.sources import sha256_file
from state_inference.data.splits import question_id
from state_inference.manifest import environment,git_state
from state_inference.runtime.devices import select_idle_gpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--gpu',type=int,choices=(0,1))
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    fields={'experiment_id','seed','model_seed','source_revision','source_file','source_sha256','previous_selected',
            'benchmark_config','benchmark_dir','model_config','output_dir','calibration_roots','test_roots',
            'max_per_root','max_tokens','rank','optimizer_steps','gradient_accumulation','learning_rate','max_wall_seconds'}
    import re
    if not isinstance(cfg,dict) or not fields <= set(cfg) or set(cfg)-fields-{'train_root_limit'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',cfg['experiment_id']):
        raise ValueError('Invalid expanded config')
    limits={'seed':(0,2**32-1),'model_seed':(0,2**32-1),'calibration_roots':(1,200),'test_roots':(1,200),
            'max_per_root':(2,16),'max_tokens':(16,1024),'rank':(1,16),'optimizer_steps':(1,300),
            'gradient_accumulation':(1,8),'max_wall_seconds':(1,1800)}
    for k,(lo,hi) in limits.items():
        if type(cfg[k]) is not int or not lo<=cfg[k]<=hi:raise ValueError(f'Invalid {k}')
    if 'train_root_limit' in cfg and (type(cfg['train_root_limit']) is not int or not 1<=cfg['train_root_limit']<=2000):raise ValueError('Invalid train_root_limit')
    if type(cfg['learning_rate']) not in (int,float) or not 0<cfg['learning_rate']<=0.01:raise ValueError('Invalid learning rate')
    if args.preflight and args.gpu is not None:parser.error('Preflight is CPU only')
    gpu=select_idle_gpu(args.gpu) if args.gpu is not None else None
    if gpu is None:os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from state_inference.data.expanded import select_expanded,limit_training_roots
    from state_inference.kv_probe import SingleKVProbe
    from state_inference.kv_probe.single_probe import fork_dynamic_cache
    from state_inference.training.pilot import frozen_digest,prefix_cache,example_logits,evaluate,length_position_baseline,trainable_state,load_trainable_state
    from state_inference.training.calibration import fit_calibration,apply_platt,metrics_at_threshold
    from state_inference.metrics.binary import binary_metrics
    from fetch_pilot_assets import verify
    torch.set_num_threads(2)
    torch.manual_seed(cfg['model_seed'])
    device='cuda:0' if gpu else 'cpu'
    if gpu:
        assert torch.cuda.device_count()==1
        torch.cuda.reset_peak_memory_stats()
    run_name=cfg['experiment_id']+('-preflight' if args.preflight else '')
    output=Path(cfg['output_dir'])/run_name
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=True))
    env=environment();env.update(cuda_version=torch.version.cuda,gpu_information=gpu,gpu_status='single_authorized_gpu' if gpu else 'cpu_only')
    manifest={'schema_version':1,'experiment_id':run_name,'git':git_state(),'seed':cfg['seed'],'model_seed':cfg['model_seed'],
              'environment':env,'config':cfg,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
              'mode':'preflight' if args.preflight else 'training','started_at':datetime.now(timezone.utc).isoformat(),
              'dataset_manifest':'dataset_manifest.json','dtype':'float32','batch_size':1,'effective_batch_size':cfg['gradient_accumulation'],
              'input_length_limit':cfg['max_tokens'],'output_length':1,'sampling_parameters':{},
              'reproduce_argv':['python','scripts/train_probe_expanded.py','--config',args.config]+(['--preflight'] if args.preflight else [])+(['--gpu',str(args.gpu)] if gpu else [])}
    def sync():
        if gpu:torch.cuda.synchronize()
    with (output/'events.jsonl').open('w') as log:
        def emit(event,**fields):
            item={'timestamp':datetime.now(timezone.utc).isoformat(),'event':event,**fields}
            log.write(json.dumps(item)+'\n');log.flush();print(json.dumps(item),flush=True)
        start=time.perf_counter();emit('run_started',seed=cfg['seed'],model_seed=cfg['model_seed'],environment=env)
        try:
            if sha256_file(cfg['source_file'])!=cfg['source_sha256']:raise ValueError('source checksum mismatch')
            model_cfg=json.loads(Path(cfg['model_config']).read_text())
            assets=[]
            for name,expected in model_cfg['files'].items():
                path=Path(model_cfg['local_dir'])/name;verify(path,expected)
                assets.append({'file':name,'sha256':sha256_file(path),'bytes':path.stat().st_size})
            write_json(output/'model_assets.json',{**model_cfg,'verified_files':assets})
            tokenizer=AutoTokenizer.from_pretrained(model_cfg['local_dir'],local_files_only=True)
            previous=read_jsonl(Path(cfg['previous_selected']))
            excluded_roots={r['root_id'] for r in previous}
            benchmark_cfg=yaml.safe_load(Path(cfg['benchmark_config']).read_text())
            benchmark=set()
            for name,digest in benchmark_cfg['files'].items():
                path=Path(cfg['benchmark_dir'])/name
                if sha256_file(path)!=digest:raise ValueError('benchmark checksum mismatch')
                benchmark.update(question_id(row['problem']) for row in json.loads(path.read_text()))
            raw=read_jsonl(Path(cfg['source_file']))
            candidates,audit=extract_candidates(raw)
            rows,removed,assignments=select_expanded(candidates,tokenizer,excluded_roots,benchmark,cfg['seed'],
                cfg['calibration_roots'],cfg['test_roots'],cfg['max_per_root'],cfg['max_tokens'])
            full_train=[r for r in rows if r['split']=='train']
            full_positive=sum(r['label'] for r in full_train)
            full_negative=len(full_train)-full_positive
            if not full_positive or not full_negative:raise ValueError('Both labels required in full training cohort')
            full_pos_weight=full_negative/full_positive
            if 'train_root_limit' in cfg:
                rows=limit_training_roots(rows,cfg['train_root_limit'],cfg['seed'])
            splits={name:[r for r in rows if r['split']==name] for name in ('train','calibration','test')}
            roots={name:{r['root_id'] for r in group} for name,group in splits.items()}
            assert not roots['train'] & roots['calibration'] and not roots['train'] & roots['test'] and not roots['calibration'] & roots['test']
            assert not set(assignments)&excluded_roots
            if any({r['label'] for r in group}!={0,1} for group in splits.values()):raise ValueError('Both labels required in each split')
            for name,group in splits.items():write_jsonl(output/f'{name}_samples.jsonl',group)
            write_jsonl(output/'excluded_overlength.jsonl',removed)
            write_jsonl(output/'root_assignments.jsonl',[{'root_id':r,'split':assignments[r]} for r in sorted(assignments)])
            summary={name:{'samples':len(group),'roots':len(roots[name]),'positive':sum(r['label'] for r in group),
                           'negative':sum(1-r['label'] for r in group)} for name,group in splits.items()}
            dm={'source_revision':cfg['source_revision'],'source_sha256':cfg['source_sha256'],'audit':audit,'counts':summary,
                'previous_selected_sha256':sha256_file(cfg['previous_selected']),'excluded_previous_roots':len(excluded_roots),
                'benchmark_revision':benchmark_cfg['revision'],'benchmark_files':benchmark_cfg['files'],
                'benchmark_usage':'question fingerprints only; no labels or scores','overlength_removed':len(removed),
                'root_assignments':len(assignments),'train_root_limit':cfg.get('train_root_limit'),'full_training_class_pos_weight':full_pos_weight,'selection':'fresh roots, deterministic pair-first capped by root, no truncation',
                'model_input_fields':['problem','prefix_steps','candidate_step'],
                'files':{p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in output.glob('*.jsonl') if p.name!='events.jsonl'}}
            write_json(output/'dataset_manifest.json',dm);emit('dataset_ready',counts=summary,overlength_removed=len(removed))
            backbone=AutoModelForCausalLM.from_pretrained(model_cfg['local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
            torch.manual_seed(cfg['model_seed'])
            probe=SingleKVProbe(backbone,rank=cfg['rank'],init_token_id=tokenizer.eos_token_id,output_logits=True).to(device)
            backbone.config.to_json_file(output/'model_config.json')
            manifest.update(model={'name':model_cfg['model_id'],'revision':model_cfg['revision']},tokenizer_revision=model_cfg['revision'],adapter_revision='checkpoint.pt after training')
            frozen=frozen_digest(probe);initial=trainable_state(probe)
            cache=prefix_cache(probe,splits['train'][0]['input_ids'],device)
            original=[(l.keys.clone(),l.values.clone()) for l in cache.layers]
            with torch.no_grad():
                baseline_logits=backbone(torch.tensor([[tokenizer.eos_token_id]],device=device),past_key_values=fork_dynamic_cache(cache,backbone.config),use_cache=True).logits.clone()
            if args.preflight:
                loss=torch.nn.functional.binary_cross_entropy_with_logits(probe(cache),torch.tensor([splits['train'][0]['label']],device=device,dtype=torch.float32));loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters() if p.requires_grad)
                result={'status':'success','optimizer_steps':0,'loss':loss.item()}
            else:
                train=splits['train'];cal=splits['calibration'];test=splits['test']
                pos=sum(r['label'] for r in train);neg=len(train)-pos
                pos_weight=torch.tensor([full_pos_weight],device=device)
                optimizer=torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=0)
                rng=random.Random(cfg['model_seed']);order=[];offset=0;updates=0;sync();training_start=time.perf_counter()
                for _ in range(cfg['optimizer_steps']):
                    if time.perf_counter()-start>cfg['max_wall_seconds']:
                        emit('wall_time_limit',completed_steps=updates);break
                    optimizer.zero_grad(set_to_none=True);probe.train();loss_total=0
                    for _ in range(cfg['gradient_accumulation']):
                        if offset>=len(order):order=list(range(len(train)));rng.shuffle(order);offset=0
                        row=train[order[offset]];offset+=1
                        logits=example_logits(probe,row,device)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([row['label']],device=device,dtype=torch.float32),pos_weight=pos_weight)
                        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                        (loss/cfg['gradient_accumulation']).backward();loss_total+=loss.item()/cfg['gradient_accumulation']
                    norm=torch.nn.utils.clip_grad_norm_([p for p in probe.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
                    optimizer.step();updates+=1
                    if updates==1 or updates%25==0:emit('optimizer_step',step=updates,weighted_bce=loss_total,gradient_norm=float(norm.item()))
                sync();training_seconds=time.perf_counter()-training_start
                cal_metrics,cal_scores=evaluate(probe,cal,device)
                calibration=fit_calibration([r['label'] for r in cal],cal_scores)
                # Only after all fitting is complete do we inspect the held-out test labels/scores.
                test_metrics,test_scores=evaluate(probe,test,device)
                calibrated_scores=apply_platt(test_scores,calibration)
                calibrated_metrics=metrics_at_threshold([r['label'] for r in test],calibrated_scores,calibration['threshold'])
                length_baseline,length_fit=length_position_baseline(train,test)
                constant=binary_metrics([r['label'] for r in test],[pos/len(train)]*len(test))
                baseline_scores=[]
                for row in test:
                    values=[math.log1p(len(row['input_ids'])),math.log1p(row['step_index'])]
                    z=length_fit['bias']+sum(w*(v-m)/sd for w,v,m,sd in zip(length_fit['weight'],values,length_fit['mean'],length_fit['scale']))
                    baseline_scores.append(1/(1+math.exp(-z)))
                pairgroups=defaultdict(list)
                for row,score,base_score in zip(test,test_scores,baseline_scores):
                    pairgroups[(row['root_id'],tuple(row['prefix_steps']))].append((row['label'],score,base_score))
                pairs=[float(p>n)+0.5*float(p==n) for group in pairgroups.values() for y,p,_ in group if y==1 for z,n,_ in group if z==0]
                baseline_pairs=[float(p>n)+0.5*float(p==n) for group in pairgroups.values() for y,_,p in group if y==1 for z,_,n in group if z==0]
                state=trainable_state(probe);changed=sum(not torch.equal(initial[k],v) for k,v in state.items())
                assert updates>0 and changed>0
                torch.save({'trainable':state,'model':model_cfg['model_id'],'revision':model_cfg['revision'],'rank':cfg['rank'],'output_logits':True,'seed':cfg['model_seed'],'optimizer_steps':updates,'calibration':calibration},output/'checkpoint.pt')
                loaded=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=True);load_trainable_state(probe,loaded['trainable'])
                with torch.no_grad():restored=float(probe(prefix_cache(probe,test[0]['input_ids'],device)).sigmoid().item())
                assert abs(restored-test_scores[0])<1e-6
                write_jsonl(output/'test_predictions.jsonl',[{'sample_id':r['sample_id'],'root_id':r['root_id'],'label':r['label'],'raw_score':a,'calibrated_score':b,'length_position_score':c} for r,a,b,c in zip(test,test_scores,calibrated_scores,baseline_scores)])
                result={'status':'success','optimizer_steps':updates,'training_seconds':training_seconds,
                    'trainable_parameters':sum(p.numel() for p in probe.parameters() if p.requires_grad),'changed_trainable_tensors':changed,
                    'class_pos_weight':float(pos_weight.item()),'class_pos_weight_source':'full_fresh_training_cohort','calibration_metrics':cal_metrics,'calibration':calibration,
                    'test_raw':test_metrics,'test_calibrated':calibrated_metrics,'length_position_baseline':length_baseline,
                    'length_position_fit':length_fit,'constant_baseline':constant,
                    'matched_pairs':{'groups':sum({y for y,_,_ in g}=={0,1} for g in pairgroups.values()),'pairs':len(pairs),'accuracy':sum(pairs)/len(pairs) if pairs else None,'length_position_accuracy':sum(baseline_pairs)/len(baseline_pairs) if baseline_pairs else None},
                    'checkpoint':{'file':'checkpoint.pt','sha256':sha256_file(output/'checkpoint.pt'),'bytes':(output/'checkpoint.pt').stat().st_size},
                    'checkpoint_reload_verified':True,'test_used_for_tuning':False}
            assert frozen==frozen_digest(probe)
            for l,(k,v) in zip(cache.layers,original):torch.testing.assert_close(l.keys,k,rtol=0,atol=0);torch.testing.assert_close(l.values,v,rtol=0,atol=0)
            with torch.no_grad():final_logits=backbone(torch.tensor([[tokenizer.eos_token_id]],device=device),past_key_values=fork_dynamic_cache(cache,backbone.config),use_cache=True).logits
            torch.testing.assert_close(baseline_logits,final_logits,rtol=0,atol=0)
            result.update(frozen_backbone_unchanged=True,generator_cache_unchanged=True,generator_logits_unchanged=True)
            write_json(output/'result.json',result);manifest['status']='success';emit('run_succeeded',optimizer_steps=result['optimizer_steps'])
        except Exception as exc:
            manifest['status']='failed';emit('run_failed',error_type=type(exc).__name__,message=str(exc));raise
        finally:
            sync();manifest.update(runtime_seconds=time.perf_counter()-start,finished_at=datetime.now(timezone.utc).isoformat(),peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None)
            write_json(output/'manifest.json',manifest);emit('run_finished',status=manifest['status'],runtime_seconds=manifest['runtime_seconds'])


if __name__=='__main__':main()
