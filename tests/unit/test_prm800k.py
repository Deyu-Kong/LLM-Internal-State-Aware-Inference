import copy
import pytest
from state_inference.data.prm800k import extract_candidates, model_text, choose_subset
from state_inference.metrics.binary import binary_metrics
from state_inference.training.config import load_training_config


def completion(text, rating, **extra):
    return {"text": text, "rating": rating, "flagged": False, **extra}


def record(steps):
    return {"question": {"problem": "Find the value of a small expression", "ground_truth_answer": "DO_NOT_INPUT"},
            "label": {"steps": steps, "finish_reason": "solution"}}


def test_candidates_share_chosen_history_not_sibling_or_future():
    row = record([
        {"completions": [completion("chosen good", 1), completion("wrong sibling", -1)], "chosen_completion": 0},
        {"completions": [completion("neutral", 0)], "chosen_completion": 0},
        {"completions": [completion("target good", 1), completion("target bad", -1)], "chosen_completion": 0},
        {"completions": [completion("future text", 1)], "chosen_completion": 0}])
    examples, _ = extract_candidates([row])
    targets = [e for e in examples if e["candidate_step"].startswith("target")]
    assert len(targets) == 2
    for example in targets:
        assert example["prefix_steps"] == ["chosen good", "neutral"]
        text = model_text(example)
        assert "wrong sibling" not in text and "future text" not in text
        assert "DO_NOT_INPUT" not in text
        assert model_text({**example, "label": 99, "ground_truth_answer": "LEAK", "first_error_step": 9}) == text


def test_human_correction_and_erroneous_prefix_stop():
    first = {"completions": [completion("bad", -1)], "chosen_completion": None,
             "human_completion": {"text": "human repair", "source": "human", "rating": None, "flagged": False}}
    second = {"completions": [completion("next", 1)], "chosen_completion": 0}
    examples, _ = extract_candidates([record([first, second])])
    assert next(e for e in examples if e["candidate_step"] == "next")["prefix_steps"] == ["human repair"]
    first["human_completion"]["corrected_rating"] = -1
    examples, _ = extract_candidates([record([first, second])])
    assert len(examples) == 1
    first = {"completions": [completion("bad", -1)], "chosen_completion": 0}
    examples, _ = extract_candidates([record([first, second])])
    assert len(examples) == 1


def test_uncertain_qc_giveup_and_conflicts_removed():
    row = record([{"completions": [completion("good", 1), completion("flagged", -1, flagged=True), completion("neutral", 0)], "chosen_completion": 0}])
    conflict = copy.deepcopy(row)
    conflict["label"]["steps"][0]["completions"][0]["rating"] = -1
    qc = {**row, "is_quality_control_question": True}
    bad = copy.deepcopy(row)
    bad["label"]["finish_reason"] = "give_up"
    samples, audit = extract_candidates([row, conflict, qc, bad])
    assert samples == []
    assert audit["excluded_conflicting_labels"] == 1
    assert audit["excluded_qc_or_screening"] == 1
    assert audit["excluded_bad_problem_or_give_up"] == 1


def test_group_selection_deterministic_disjoint_and_bounded():
    examples = [{"sample_id": f"s{i}-{j}", "root_id": f"q{i}", "label": j % 2} for i in range(30) for j in range(6)]
    a = choose_subset(examples, train_roots=10, validation_roots=5, per_root=2, max_train=20, max_validation=10)
    b = choose_subset(list(reversed(examples)), train_roots=10, validation_roots=5, per_root=2, max_train=20, max_validation=10)
    assert a == b
    train = {r["root_id"] for r in a if r["split"] == "train"}
    val = {r["root_id"] for r in a if r["split"] == "validation"}
    assert len(train) == 10 and len(val) == 5 and not train & val
    assert len(a) == 30
    with pytest.raises(ValueError):
        choose_subset(examples, train_roots=30, validation_roots=1)


def test_metrics_rank_ties_and_degenerate_classes():
    perfect = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert perfect["auroc"] == perfect["average_precision"] == perfect["f1_at_0_5"] == 1
    tied = binary_metrics([0, 1], [0.5, 0.5])
    assert tied["auroc"] == tied["average_precision"] == 0.5
    assert tied["brier"] == 0.25
    assert binary_metrics([0], [0.1])["auroc"] is None
    with pytest.raises(ValueError):
        binary_metrics([1], [float("nan")])


def test_pilot_config_bounded(tmp_path):
    import yaml
    cfg = load_training_config("configs/wp1_probe/prm800k_pilot.yaml")
    assert cfg["optimizer_steps"] == 100
    cfg["optimizer_steps"] = 100000
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_training_config(p)


def test_control_config_keeps_data_seed_and_changes_backbone():
    a = load_training_config('configs/wp1_probe/prm800k_control_pretrained.yaml')
    b = load_training_config('configs/wp1_probe/prm800k_control_random.yaml')
    assert a['seed'] == b['seed'] == a['model_seed'] == b['model_seed'] == 42
    assert a['backbone_mode'] == 'pretrained'
    assert b['backbone_mode'] == 'random'


def test_model_seed_is_independent_of_data_seed():
    a = load_training_config('configs/wp1_probe/prm800k_modelseed_43.yaml')
    b = load_training_config('configs/wp1_probe/prm800k_modelseed_44.yaml')
    assert a['seed'] == b['seed'] == 42
    assert (a['model_seed'], b['model_seed']) == (43, 44)


def test_weighted_bce_odds_inversion():
    from state_inference.metrics.binary import undo_positive_class_weight
    assert undo_positive_class_weight(0.5, 0.5) == pytest.approx(2/3)
    assert undo_positive_class_weight(0.0, 0.5) == 0
    assert undo_positive_class_weight(1.0, 0.5) == 1
    with pytest.raises(ValueError):
        undo_positive_class_weight(0.5, 0)


def test_expanded_selection_excludes_prior_and_keeps_pairs():
    from state_inference.data.expanded import select_expanded
    class Tokenizer:
        def encode(self, text, add_special_tokens=True): return list(range(len(text.split())))
    rows=[]
    for i in range(9):
        for label in (0,1):
            rows.append({'root_id':f'prm800k:{i}', 'sample_id':f'{i}-{label}', 'problem':f'problem {i}',
                         'prefix_steps':['shared'], 'candidate_step':f'candidate {label}', 'label':label, 'step_index':1})
    selected, removed, assignments=select_expanded(rows,Tokenizer(),{'prm800k:0'},{'1'},calibration_roots=2,test_roots=2)
    assert not removed and len(assignments)==7
    assert 'prm800k:0' not in assignments and 'prm800k:1' not in assignments
    for root in assignments:
        group=[r for r in selected if r['root_id']==root]
        assert {r['label'] for r in group}=={0,1}
        assert len({r['split'] for r in group})==1
