"""Conservative PRM800K branch reconstruction for local step classification."""
from collections import Counter
import hashlib
import json
from .splits import question_id

REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"
SOURCE_SHA256 = "e9da6a73f827ffb9a8c0dc644c541d34ed76b3d4d1e4896ff5f7b37ddf5ae34d"


def sample_key(problem, prefix, candidate):
    value = [question_id(problem), prefix, candidate]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def extract_candidates(records):
    """Score each candidate on its own chosen/human-corrected history, never a sibling.

    Neutral chosen steps may remain in history, but are not supervised as positive
    or negative. Never continue a chosen erroneous or uncertain history.
    """
    audit, samples = Counter(), {}
    for row_index, row in enumerate(records):
        audit["source_records"] += 1
        if row.get("is_quality_control_question") or row.get("is_initial_screening_question"):
            audit["excluded_qc_or_screening"] += 1
            continue
        if row.get("label", {}).get("finish_reason") in ("bad_problem", "give_up"):
            audit["excluded_bad_problem_or_give_up"] += 1
            continue
        problem = row.get("question", {}).get("problem")
        if not isinstance(problem, str) or not problem.strip():
            audit["excluded_missing_problem"] += 1
            continue
        prefix = []
        for step_index, step in enumerate(row.get("label", {}).get("steps", [])):
            completions = step.get("completions") or []
            for candidate_index, completion in enumerate(completions):
                rating = completion.get("rating")
                text = completion.get("text")
                if completion.get("flagged") not in (None, False):
                    audit["excluded_flagged_candidates"] += 1
                    continue
                if type(rating) is not int or rating not in (-1, 1):
                    audit["excluded_neutral_or_unknown_candidates"] += 1
                    continue
                if not isinstance(text, str) or not text.strip():
                    audit["excluded_empty_candidates"] += 1
                    continue
                identity = sample_key(problem, prefix, text)
                item = samples.setdefault(identity, {
                    "sample_id": identity, "root_id": "prm800k:" + question_id(problem),
                    "problem": problem, "prefix_steps": list(prefix), "candidate_step": text,
                    "step_index": step_index, "ratings": set(), "source_locations": [],
                })
                item["ratings"].add(rating)
                item["source_locations"].append([row_index, step_index, candidate_index])
            chosen = step.get("chosen_completion")
            if chosen is None:
                human = step.get("human_completion")
                if isinstance(human, dict):
                    review = human.get("corrected_rating")
                    human = (human.get("text") if human.get("source") == "human"
                             and human.get("flagged") in (None, False)
                             and (review is None or (type(review) is int and review in (0, 1))) else None)
                if isinstance(human, str) and human.strip():
                    prefix.append(human)
                    audit["human_corrected_prefix_steps"] += 1
                else:
                    audit["stopped_missing_chosen_history"] += 1
                    break
            elif type(chosen) is int and 0 <= chosen < len(completions):
                completion = completions[chosen]
                text = completion.get("text")
                if (type(completion.get("rating")) is int and completion["rating"] in (0, 1)
                        and completion.get("flagged") in (None, False) and isinstance(text, str) and text.strip()):
                    prefix.append(text)
                else:
                    audit["stopped_erroneous_or_uncertain_history"] += 1
                    break
            else:
                audit["stopped_invalid_chosen_index"] += 1
                break
    result = []
    for key in sorted(samples):
        row = samples[key]
        ratings = row.pop("ratings")
        if len(ratings) != 1:
            audit["excluded_conflicting_labels"] += 1
            continue
        row["label"] = int(next(iter(ratings)) == 1)
        row["label_provenance"] = "prm800k_human_step_rating_plus1_vs_minus1"
        result.append(row)
    audit["eligible_candidates"] = len(result)
    audit["eligible_roots"] = len({row["root_id"] for row in result})
    return result, dict(audit)


def model_text(row):
    """Explicit observation-field allowlist; excludes all labels and future text."""
    steps = row["prefix_steps"] + [row["candidate_step"]]
    return "Problem:\n" + row["problem"] + "\n\nSolution:\n" + "\n\n".join(steps)


def choose_subset(rows, seed=42, train_roots=200, validation_roots=50, per_root=8,
                  max_train=1200, max_validation=300):
    roots = sorted({r["root_id"] for r in rows}, key=lambda x: (hashlib.sha256(f"{seed}:{x}".encode()).hexdigest(), x))
    if len(roots) < train_roots + validation_roots:
        raise ValueError(f"Requested {train_roots + validation_roots} roots but only {len(roots)} eligible")
    selected = {root: "train" for root in roots[:train_roots]}
    selected.update({root: "validation" for root in roots[train_roots:train_roots + validation_roots]})
    counts, result = Counter(), []
    for row in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest()):
        root = row["root_id"]
        if root not in selected or counts[root] >= per_root:
            continue
        split = selected[root]
        limit = max_train if split == "train" else max_validation
        if counts[split] >= limit:
            continue
        result.append({**row, "split": split})
        counts[root] += 1
        counts[split] += 1
    return result
