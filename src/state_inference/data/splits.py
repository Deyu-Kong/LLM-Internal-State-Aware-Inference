"""Stable group-level splits and explicit leakage auditing."""
import hashlib
import unicodedata

SPLITS = ("train", "validation", "test_id", "test_ood")


def question_id(question):
    normalized = " ".join(unicodedata.normalize("NFKC", question).split()).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def grouped_split(group_ids, seed, validation_fraction=0.1, test_fraction=0.1):
    """Rank unique roots, then split; stable under input reorder and duplication.

    Membership is fixed for a given root set, seed and protocol version. Adding
    roots can move boundaries; preserve the emitted split assignment manifest.
    """
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Invalid split fractions")
    roots = sorted(set(group_ids), key=lambda root: (hashlib.sha256(f"{seed}:{root}".encode()).hexdigest(), root))
    nv, nt = int(len(roots) * validation_fraction), int(len(roots) * test_fraction)
    return {root: "validation" if i < nv else "test_id" if i < nv + nt else "train"
            for i, root in enumerate(roots)}


def assert_no_leakage(records):
    groups, questions, ids = {}, {}, set()
    for row in records:
        if row["split"] not in SPLITS:
            raise ValueError("Unknown split")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        ids.add(row["sample_id"])
        for key, mapping in ((row["root_id"], groups), (question_id(row["question"]), questions)):
            previous = mapping.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Cross-split root or normalized-question leakage")
        if row.get("source_split") == "test" and row["split"] != "test_id":
            raise ValueError("Official test data must remain test_id")
    return {"unique_roots": len(groups), "unique_questions": len(questions), "samples": len(ids)}
