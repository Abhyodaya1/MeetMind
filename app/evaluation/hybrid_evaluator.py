from app.evaluation.metrics import (
    f1_score,
    precision,
    recall,
)
from app.schemas import GroundTruthItem


def hybrid_evaluate_items(
    predictions: list[str],
    ground_truth: list[GroundTruthItem],
    matcher,
    judge,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> dict:
    """
    Evaluate predictions using semantic candidate retrieval followed by LLM judging.

    For each prediction, retrieves up to `top_k` unmatched ground-truth items ranked by
    embedding similarity (above `min_similarity`) and passes only those candidates to
    `judge.judge(...)` to avoid N x M LLM calls.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    matched_ground_truth: set[int] = set()
    true_positives = 0
    false_positives = 0
    judge_calls = 0

    for prediction in predictions:
        candidates = []

        for index, expected in enumerate(ground_truth):
            if index in matched_ground_truth:
                continue

            similarity_score = matcher.similarity(
                prediction,
                expected.text,
            )

            if similarity_score >= min_similarity:
                candidates.append(
                    (similarity_score, index, expected)
                )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        top_candidates = candidates[:top_k]

        best_match_index = None
        best_score = 0.0

        for _, index, expected in top_candidates:
            judge_calls += 1
            result = judge.judge(
                prediction,
                expected.text,
            )

            if result.match and result.score > best_score:
                best_score = result.score
                best_match_index = index

        if best_match_index is not None:
            matched_ground_truth.add(best_match_index)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - true_positives

    p = precision(true_positives, false_positives)
    r = recall(true_positives, false_negatives)
    f1 = f1_score(p, r)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": p,
        "recall": r,
        "f1": f1,
        "judge_calls": judge_calls,
    }
