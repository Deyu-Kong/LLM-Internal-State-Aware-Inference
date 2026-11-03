"""Lossless source tracking and conservative parsing of GSM8K references."""
from decimal import Decimal
import re
from .splits import grouped_split, question_id

_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def canonical_number(text):
    text = text.strip()
    if not _NUMBER.fullmatch(text):
        raise ValueError("Expected one finite numeric answer, without units or extra text")
    value = Decimal(text.replace(",", ""))
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def parse_reference(answer):
    if not isinstance(answer, str) or answer.count("####") != 1:
        raise ValueError("Reference answer requires exactly one #### separator")
    reasoning, numeric = answer.split("####")
    result = canonical_number(numeric)
    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Reference solution has no reasoning lines")
    steps = [{"index": i, "text": line,
              "text_without_calculator_annotations": re.sub(r"<<[^<>]*>>", "", line),
              "arithmetic_valid": None, "dependency_valid": None, "local_process_valid": None,
              "reference_value_match": None}
             for i, line in enumerate(lines, 1)]
    return result, steps


def prepare_gsm8k(train, test, seed):
    """Preserve official test; quarantine training copies of normalized test questions."""
    test_questions = {question_id(row["question"]) for row in test}
    excluded, remaining = [], []
    for index, row in enumerate(train):
        if question_id(row["question"]) in test_questions:
            excluded.append({"source_split": "train", "source_row": index,
                             "root_id": "gsm8k:" + question_id(row["question"]),
                             "reason": "normalized_question_also_in_official_test"})
        else:
            remaining.append((index, row))
    assignments = grouped_split([question_id(row["question"]) for _, row in remaining], seed,
                                validation_fraction=0.1, test_fraction=0.0)
    records = []
    for source_split, rows in (("train", remaining), ("test", list(enumerate(test)))):
        for index, raw in rows:
            if not isinstance(raw.get("question"), str) or not raw["question"].strip():
                raise ValueError("Missing GSM8K question")
            root = question_id(raw["question"])
            answer, steps = parse_reference(raw["answer"])
            records.append({"schema_version": 1, "sample_id": f"gsm8k:{source_split}:{index:05d}",
                            "root_id": "gsm8k:" + root, "source": "gsm8k", "source_split": source_split,
                            "source_row": index, "split": "test_id" if source_split == "test" else assignments[root],
                            "question": raw["question"], "steps": steps, "reference_answer": answer,
                            "reference_solution": raw["answer"], "trace_origin": "dataset_reference_solution",
                            "final_answer": None, "final_answer_correct": None, "first_error_step": None,
                            "first_error_status": "unknown",
                            "label_provenance": "reference_answer_only_process_labels_unavailable",
                            "pseudo_confidence_targets": None})
    return records, excluded
