"""Fresh-root, pair-preserving selection for the expanded PRM800K study."""
from collections import defaultdict
import hashlib


def rank(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def select_expanded(rows, tokenizer, excluded_roots, benchmark_fingerprints, seed=42,
                    calibration_roots=100, test_roots=100, max_per_root=8, max_tokens=512):
    eligible = [r for r in rows if r['root_id'] not in excluded_roots and r['root_id'].split(':',1)[1] not in benchmark_fingerprints]
    roots = sorted({r['root_id'] for r in eligible}, key=lambda x:(rank(seed,x),x))
    if len(roots) <= calibration_roots + test_roots:
        raise ValueError('Insufficient fresh roots')
    assignments = {root: ('test' if i < test_roots else 'calibration' if i < test_roots + calibration_roots else 'train')
                   for i,root in enumerate(roots)}
    from .prm800k import model_text
    groups = defaultdict(list)
    removed = []
    for row in eligible:
        tokens = tokenizer.encode(model_text(row), add_special_tokens=True)
        if len(tokens)>max_tokens:
            removed.append({'sample_id':row['sample_id'],'root_id':row['root_id'],'split':assignments[row['root_id']],
                            'tokens':len(tokens),'label':row['label'],'reason':'over_length_no_truncation'})
            continue
        groups[row['root_id']].append({**row,'input_ids':tokens,'split':assignments[row['root_id']]})
    selected=[]
    for root in roots:
        rows_for_root=groups[root]
        prefixes=defaultdict(list)
        for row in rows_for_root:prefixes[tuple(row['prefix_steps'])].append(row)
        pairs=[]
        for prefix,group in prefixes.items():
            positives=sorted((r for r in group if r['label']==1),key=lambda x:rank(seed,x['sample_id']))
            negatives=sorted((r for r in group if r['label']==0),key=lambda x:rank(seed,x['sample_id']))
            pairs.extend(zip(positives,negatives))
        pairs.sort(key=lambda pair:rank(seed,pair[0]['sample_id']+pair[1]['sample_id']))
        chosen=[]
        for positive,negative in pairs:
            if len(chosen)+2>max_per_root:break
            chosen.extend((positive,negative))
        used={r['sample_id'] for r in chosen}
        rest=sorted((r for r in rows_for_root if r['sample_id'] not in used),key=lambda x:rank(seed,x['sample_id']))
        chosen.extend(rest[:max_per_root-len(chosen)])
        selected.extend(chosen)
    return selected, removed, assignments


def limit_training_roots(rows, limit, seed):
    """Limit training roots only, preserving calibration and test byte-for-byte."""
    roots=sorted({r['root_id'] for r in rows if r['split']=='train'},key=lambda root:(rank(seed,root),root))
    if type(limit) is not int or not 1<=limit<len(roots):
        raise ValueError('Train root limit must select a strict nonempty subset')
    keep=set(roots[:limit])
    return [r for r in rows if r['split']!='train' or r['root_id'] in keep]
