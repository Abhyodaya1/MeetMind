from app.evaluation.metrics import (
    precision,
    recall,
    f1_score,
)
from app.schemas import GroundTruthItem


def llm_evaluate_items(
    predictions: list[str],
    ground_truth: list[GroundTruthItem],
    judge,
) -> dict:

    matched_ground_truth = set()

    true_positives = 0
    false_positives = 0

    for prediction in predictions:

        best_match_index = None
        best_score = 0.0

        for index, expected in enumerate(ground_truth):

            if index in matched_ground_truth:
                continue

            result = judge.judge(
                prediction,
                expected.text,
            )

            if result.match and result.score > best_score:
                best_score = result.score
                best_match_index = index

        if best_match_index is not None:
            matched_ground_truth.add(
                best_match_index
            )
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = (
        len(ground_truth)
        - true_positives
    )

    p = precision(
        true_positives,
        false_positives,
    )

    r = recall(
        true_positives,
        false_negatives,
    )

    f1 = f1_score(p, r)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": p,
        "recall": r,
        "f1": f1,
    }