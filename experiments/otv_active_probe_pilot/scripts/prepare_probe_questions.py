"""Prepare grouped pilot questions, without using supplied solutions as model traces."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metamath', required=True)
    p.add_argument('--gsm8k', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    test = [json.loads(line) for line in Path(args.gsm8k).read_text().splitlines() if line.strip()]
    forbidden = {normalize(row['question']) for row in test}
    # One representative per original-question group prevents rewritten variants
    # from appearing in different splits. Also remove exact query overlap.
    groups = {}
    excluded = 0
    for row in json.loads(Path(args.metamath).read_text()):
        key = normalize(row.get('original_question') or row['query'])
        if key in forbidden or normalize(row['query']) in forbidden:
            excluded += 1
            continue
        if key not in groups:
            groups[key] = row
    keys = sorted(groups)
    rng = random.Random(42)
    rng.shuffle(keys)
    rows = []
    seen_queries = set()
    for key in keys:
        row = groups[key]
        query = normalize(row['query'])
        if query in seen_queries:
            continue
        seen_queries.add(query)
        rows.append(dict(question_id=hashlib.sha256(key.encode()).hexdigest(),
                         question=row['query'], reference_solution=row['response'],
                         source='meta-math/MetaMathQA', source_type=row.get('type'),
                         original_question=row.get('original_question')))
    assert len(rows) >= 2300
    splits = {'train': rows[:2000], 'validation': rows[2000:2300],
              'smoke_train': rows[:100]}
    rng.shuffle(test)
    splits['test'] = [dict(question_id=hashlib.sha256(normalize(r['question']).encode()).hexdigest(),
                           question=r['question'], reference_solution=r['answer'],
                           reference_answer=r['answer'].rsplit('####', 1)[1].strip(),
                           source='openai/grade-school-math/test') for r in test[:500]]
    for split, records in splits.items():
        with (out / f'{split}.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    manifest = dict(seed=42, status='questions_only_no_generated_traces',
                    metamath_sha256=sha(args.metamath), gsm8k_sha256=sha(args.gsm8k),
                    excluded_overlap_rows=excluded, available_groups=len(rows),
                    sizes={k: len(v) for k, v in splits.items()},
                    limitations=['Exact normalization is not semantic decontamination.',
                                 'Smoke train is a subset of train, not a held-out split.',
                                 'Reference solutions must not be included in model prompts.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
