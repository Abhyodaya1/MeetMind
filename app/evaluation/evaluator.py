from app.evaluation.matcher import evaluate_items
from app.evaluation.metrics import (
    precision,
    recall,
    f1_score,
)
from app.evaluation.matcher import (
    semantic_evaluate_items,
)

def evaluate_semantic_category(
    predictions,
    ground_truth,
    matcher,
    threshold: float = 0.75,
):
    counts = semantic_evaluate_items(
        predictions,
        ground_truth,
        matcher,
        threshold,
    )

    p = precision(
        counts["true_positives"],
        counts["false_positives"],
    )

    r = recall(
        counts["true_positives"],
        counts["false_negatives"],
    )

    f1 = f1_score(
        p,
        r,
    )

    return {
        **counts,
        "precision": p,
        "recall": r,
        "f1": f1,
    }

def evaluate_category(
    predictions,
    ground_truth,
):

    counts = evaluate_items(
        predictions,
        ground_truth,
    )

    p = precision(
        counts["true_positives"],
        counts["false_positives"],
    )

    r = recall(
        counts["true_positives"],
        counts["false_negatives"],
    )

    f1 = f1_score(
        p,
        r,
    )

    return {
        **counts,
        "precision": p,
        "recall": r,
        "f1": f1,
    }