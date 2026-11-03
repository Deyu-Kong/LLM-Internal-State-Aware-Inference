"""Structured integer arithmetic with separate local and outcome labels."""
import hashlib
import json
import random
from .splits import grouped_split

VERSION = "synthetic-arithmetic-v1"
VARIANTS = ("correct", "arithmetic_error_propagated", "dependency_error", "recovered_error")


def apply(operation, left, right):
    if operation == "add":
        return left + right
    if operation == "sub":
        return left - right
    if operation == "mul":
        return left * right
    raise ValueError("Unsupported arithmetic operation")


def make_roots(id_count, ood_count, seed):
    if type(id_count) is not int or type(ood_count) is not int or not 1 <= id_count <= 10000 or not 1 <= ood_count <= 10000:
        raise ValueError("Root counts must be integers in [1, 10000]")
    rng, roots, seen = random.Random(seed), [], set()
    for domain, count in (("id", id_count), ("ood", ood_count)):
        made = 0
        while made < count:
            initial = rng.randint(10, 50) if domain == "id" else rng.randint(100, 200)
            operations = [{"operation": "add", "operand": rng.randint(1, 9)},
                          {"operation": "mul", "operand": rng.randint(2, 5)},
                          {"operation": "sub", "operand": rng.randint(1, 9)}]
            if domain == "ood":
                operations += [{"operation": "mul", "operand": rng.randint(2, 5)},
                               {"operation": "add", "operand": rng.randint(1, 9)}]
            identity = json.dumps({"initial": initial, "operations": operations}, sort_keys=True)
            root_id = "synthetic:" + hashlib.sha256(identity.encode()).hexdigest()
            if root_id in seen:
                continue
            seen.add(root_id)
            made += 1
            roots.append({"root_id": root_id, "domain": domain, "initial": initial, "operations": operations})
    return roots


def build_trace(root, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown trace variant")
    # Choose an interior error, leaving a later step to expose propagation/recovery.
    error_at = 2
    initial, operations = root["initial"], root["operations"]
    state, gold, steps = {"x0": initial}, {"x0": initial}, []
    question = f"Start with x0={initial}. " + " ".join(
        f"Set x{i}={op['operation']}(x{i-1},{op['operand']})." for i, op in enumerate(operations, 1)) + f" Find x{len(operations)}."

    def append(i, ref, claimed, correction=None):
        op = operations[i - 1]
        target = f"x{i}"
        local_expected = apply(op["operation"], state[ref], op["operand"])
        arithmetic_valid = claimed == local_expected
        dependency_valid = ref == f"x{i-1}"
        index = len(steps) + 1
        step = {"index": index, "output": target, "input_ref": ref,
                "input_value": state[ref], "operation": op["operation"], "operand": op["operand"],
                "claimed_result": claimed, "local_expected_result": local_expected,
                "arithmetic_valid": arithmetic_valid, "dependency_valid": dependency_valid,
                "local_process_valid": arithmetic_valid and dependency_valid,
                "reference_value_match": claimed == gold[target], "corrects_step": correction,
                "text": (f"Correction of step {correction}: " if correction else "") +
                        f"{target}={op['operation']}({ref}={state[ref]},{op['operand']})={claimed}."}
        steps.append(step)
        state[target] = claimed

    for i, op in enumerate(operations, 1):
        target, ref = f"x{i}", f"x{i-1}"
        gold[target] = apply(op["operation"], gold[ref], op["operand"])
        if variant == "dependency_error" and i == error_at:
            ref = "x0"
        claimed = apply(op["operation"], state[ref], op["operand"])
        if variant in ("arithmetic_error_propagated", "recovered_error") and i == error_at:
            claimed += 1
        append(i, ref, claimed)
        if variant == "recovered_error" and i == error_at:
            bad_index = len(steps)
            append(i, f"x{i-1}", gold[target], correction=bad_index)
    first_error = next((step["index"] for step in steps if not step["local_process_valid"]), None)
    final = state[f"x{len(operations)}"]
    return {"schema_version": 1, "sample_id": root["root_id"] + ":" + variant,
            "root_id": root["root_id"], "source": VERSION, "source_split": None,
            "domain": root["domain"], "question": question, "variant": variant,
            "initial_value": initial, "operations": operations, "steps": steps,
            "trace_text": "\n".join(step["text"] for step in steps),
            "reference_answer": str(gold[f"x{len(operations)}"]), "final_answer": str(final),
            "final_answer_correct": final == gold[f"x{len(operations)}"],
            "first_error_step": first_error, "first_error_status": "known_error" if first_error else "known_no_error",
            "label_provenance": "exact_structured_execution_v1",
            "trace_origin": "synthetic_controlled", "pseudo_confidence_targets": None}


def generate_synthetic(id_count=200, ood_count=50, seed=42):
    roots = make_roots(id_count, ood_count, seed)
    assignments = grouped_split([r["root_id"] for r in roots if r["domain"] == "id"], seed)
    result = []
    for root in roots:
        for variant in VARIANTS:
            row = build_trace(root, variant)
            row["split"] = "test_ood" if root["domain"] == "ood" else assignments[root["root_id"]]
            result.append(row)
    return result


def validate_trace(row):
    """Replay declared references, verify all labels; independent of variant names."""
    state, reference = {"x0": row["initial_value"]}, {"x0": row["initial_value"]}
    for i, operation in enumerate(row["operations"], 1):
        reference[f"x{i}"] = apply(operation["operation"], reference[f"x{i-1}"], operation["operand"])
    first = None
    next_operation = 1
    for expected_index, step in enumerate(row["steps"], 1):
        if step["index"] != expected_index:
            raise ValueError("Step indices must be consecutive, 1-based")
        i = int(step["output"][1:])
        if not 1 <= i <= len(row["operations"]):
            raise ValueError("Unknown output variable")
        correction = step["corrects_step"]
        if correction is None:
            if i != next_operation:
                raise ValueError("Trace skipped or repeated an operation without correction")
            next_operation += 1
        elif (correction != expected_index - 1 or correction < 1 or
              row["steps"][correction - 1]["output"] != step["output"] or
              row["steps"][correction - 1]["local_process_valid"] or i != next_operation - 1):
            raise ValueError("Invalid explicit correction")
        contract = row["operations"][i - 1]
        expected = apply(step["operation"], state[step["input_ref"]], step["operand"])
        arithmetic = step["claimed_result"] == expected
        dependency = step["input_ref"] == f"x{i-1}" and step["operation"] == contract["operation"] and step["operand"] == contract["operand"]
        labels = {"input_value": state[step["input_ref"]], "local_expected_result": expected,
                  "arithmetic_valid": arithmetic, "dependency_valid": dependency,
                  "local_process_valid": arithmetic and dependency,
                  "reference_value_match": step["claimed_result"] == reference[step["output"]]}
        if any(step[key] != value for key, value in labels.items()):
            raise ValueError("Inconsistent synthetic step label")
        if not arithmetic or not dependency:
            first = first or step["index"]
        state[step["output"]] = step["claimed_result"]
    if next_operation != len(row["operations"]) + 1:
        raise ValueError("Trace is incomplete")
    final = state[f"x{len(row['operations'])}"]
    answer = reference[f"x{len(row['operations'])}"]
    if (row["first_error_step"] != first or row["first_error_status"] != ("known_error" if first else "known_no_error") or row["final_answer"] != str(final) or
            row["reference_answer"] != str(answer) or row["final_answer_correct"] != (final == answer)):
        raise ValueError("Inconsistent outcome or first-error label")
