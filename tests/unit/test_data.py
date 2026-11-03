import copy
import pytest
from state_inference.data.gsm8k import canonical_number, parse_reference, prepare_gsm8k
from state_inference.data.splits import assert_no_leakage, grouped_split, question_id
from state_inference.data.synthetic import generate_synthetic, validate_trace


def test_synthetic_labels_replay_and_splits():
    rows = generate_synthetic()
    assert len(rows) == 1000
    assert assert_no_leakage(rows)["unique_roots"] == 250
    assert {split: sum(r["split"] == split for r in rows) for split in ("train", "validation", "test_id", "test_ood")} == {
        "train": 640, "validation": 80, "test_id": 80, "test_ood": 200}
    for row in rows:
        validate_trace(row)
        if row["variant"] == "correct":
            assert row["first_error_step"] is None
            assert row["final_answer_correct"]
        else:
            assert row["first_error_step"] == 2
            assert row["final_answer_correct"] == (row["variant"] == "recovered_error")
        if row["variant"] == "dependency_error":
            assert row["steps"][1]["arithmetic_valid"]
            assert not row["steps"][1]["dependency_valid"]
        if row["variant"] == "arithmetic_error_propagated":
            assert row["steps"][2]["local_process_valid"]
            assert not row["steps"][2]["reference_value_match"]
    assert rows == generate_synthetic(seed=42)
    assert rows != generate_synthetic(seed=43)


def test_label_validator_rejects_corruption():
    rows = generate_synthetic(id_count=10, ood_count=2)
    row = copy.deepcopy(rows[1])
    row["steps"][1]["arithmetic_valid"] = True
    with pytest.raises(ValueError, match="label"):
        validate_trace(row)
    row = copy.deepcopy(rows[3])
    row["final_answer_correct"] = False
    with pytest.raises(ValueError, match="outcome"):
        validate_trace(row)


def test_independent_reference_arithmetic():
    from state_inference.data.synthetic import build_trace
    root = {"root_id": "test", "domain": "id", "initial": 10,
            "operations": [{"operation": "add", "operand": 3}, {"operation": "mul", "operand": 2}, {"operation": "sub", "operand": 4}]}
    expected = {"correct": ("22", None), "arithmetic_error_propagated": ("23", 2),
                "dependency_error": ("16", 2), "recovered_error": ("22", 2)}
    for variant, (answer, first) in expected.items():
        row = build_trace(root, variant)
        assert row["final_answer"] == answer
        assert row["reference_answer"] == "22"
        assert row["first_error_step"] == first
        validate_trace(row)


def test_group_assignment_stable_and_leakage_caught():
    roots = [f"q{i}" for i in range(20)]
    assert grouped_split(roots, 42) == grouped_split(list(reversed(roots)) + roots, 42)
    base = {"sample_id": "a", "root_id": "root", "question": "How MANY?", "split": "train"}
    other = {"sample_id": "b", "root_id": "other", "question": " how   many? ", "split": "test_id"}
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([base, other])
    other.update(question="Different question", root_id="root")
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([base, other])
    with pytest.raises(ValueError, match="test data"):
        assert_no_leakage([{**base, "source_split": "test"}])
    with pytest.raises(ValueError, match="Duplicate"):
        assert_no_leakage([base, base])
    assert question_id("Ａ  b") == question_id("a b")


@pytest.mark.parametrize("raw,expected", [("1,200.50", "1200.5"), ("-0.00", "0"), ("-3", "-3"), ("20.000", "20")])
def test_numeric_answers(raw, expected):
    assert canonical_number(raw) == expected


@pytest.mark.parametrize("raw", ["nan", "1,00", "2 dollars", "2+3", "1e2"])
def test_ambiguous_answers_rejected(raw):
    with pytest.raises(ValueError):
        canonical_number(raw)


def test_gsm_reference_and_unknown_process_labels():
    answer, steps = parse_reference("First 2+3=<<2+3=5>>5.\nThen 5*2=10.\n#### 10")
    assert answer == "10"
    assert len(steps) == 2
    assert steps[0]["text_without_calculator_annotations"] == "First 2+3=5."
    assert all(s["local_process_valid"] is None for s in steps)
    with pytest.raises(ValueError):
        parse_reference("No final marker")
    with pytest.raises(ValueError):
        parse_reference("x #### 1 #### 2")


def test_gsm_official_test_preserved_and_overlap_quarantined():
    train = [{"question": f"Question {i}?", "answer": "Reference line.\n#### 3"} for i in range(20)]
    test = [{"question": " QUESTION 0? ", "answer": "Reference line.\n#### 3"}]
    rows, excluded = prepare_gsm8k(train, test, seed=42)
    assert len(excluded) == 1
    assert excluded[0]["source_row"] == 0
    assert len(rows) == 20
    assert sum(r["source_split"] == "test" for r in rows) == 1
    assert all(r["split"] == "test_id" for r in rows if r["source_split"] == "test")
    assert all(r["split"] in ("train", "validation") for r in rows if r["source_split"] == "train")
    assert all(r["final_answer_correct"] is None and r["pseudo_confidence_targets"] is None for r in rows)
    assert_no_leakage(rows)


def test_correction_contract_checked():
    row = generate_synthetic(id_count=2, ood_count=1)[3]
    row["steps"][2]["corrects_step"] = None
    with pytest.raises(ValueError, match="correction"):
        validate_trace(row)
