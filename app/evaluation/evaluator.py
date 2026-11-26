from app.evaluation.hybrid_evaluator import (
    hybrid_evaluate_items,
)
from app.evaluation.llm_evaluator import (
    llm_evaluate_items,
)
from app.evaluation.matcher import (
    evaluate_items,
    semantic_evaluate_items,
)
from app.evaluation.metrics import (
    f1_score,
    precision,
    recall,
)
from app.schemas import (
    EvaluationReport,
    GroundTruthMeeting,
    MeetingMinutes,
    MetricResult,
    TranscriptChunk,
)
from app.validation import (
    validate_action_evidence,
    validate_decision_evidence,
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


def evaluate_llm_category(
    predictions,
    ground_truth,
    judge,
):
    return llm_evaluate_items(
        predictions,
        ground_truth,
        judge,
    )


def evaluate_hybrid_category(
    predictions,
    ground_truth,
    matcher,
    judge,
    top_k: int = 3,
    min_similarity: float = 0.0,
):
    return hybrid_evaluate_items(
        predictions,
        ground_truth,
        matcher,
        judge,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def compute_evidence_validity(
    minutes: MeetingMinutes,
    chunks: list[TranscriptChunk],
) -> float:
    total = len(minutes.action_items) + len(minutes.decisions)
    if total == 0:
        return 0.0

    valid_count = 0
    for action in minutes.action_items:
        if validate_action_evidence(action, chunks):
            valid_count += 1
    for decision in minutes.decisions:
        if validate_decision_evidence(decision, chunks):
            valid_count += 1

    return valid_count / total


def evaluate_meeting(
    minutes: MeetingMinutes,
    ground_truth: GroundTruthMeeting,
    chunks: list[TranscriptChunk],
    mode: str = "exact",
    matcher=None,
    judge=None,
    threshold: float = 0.75,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> EvaluationReport:
    """
    Evaluate predicted MeetingMinutes against GroundTruthMeeting using the
    selected offline strategy ('exact', 'semantic', 'llm', or 'hybrid').
    """
    action_predictions = [item.task for item in minutes.action_items]
    decision_predictions = [item.description for item in minutes.decisions]

    if mode == "exact":
        action_result = evaluate_category(
            action_predictions, ground_truth.action_items
        )
        decision_result = evaluate_category(
            decision_predictions, ground_truth.decisions
        )
    elif mode == "semantic":
        if matcher is None:
            raise ValueError("matcher is required for semantic mode.")
        action_result = evaluate_semantic_category(
            action_predictions,
            ground_truth.action_items,
            matcher,
            threshold=threshold,
        )
        decision_result = evaluate_semantic_category(
            decision_predictions,
            ground_truth.decisions,
            matcher,
            threshold=threshold,
        )
    elif mode == "llm":
        if judge is None:
            raise ValueError("judge is required for llm mode.")
        action_result = evaluate_llm_category(
            action_predictions, ground_truth.action_items, judge
        )
        decision_result = evaluate_llm_category(
            decision_predictions, ground_truth.decisions, judge
        )
    elif mode == "hybrid":
        if matcher is None or judge is None:
            raise ValueError(
                "matcher and judge are required for hybrid mode."
            )
        action_result = evaluate_hybrid_category(
            action_predictions,
            ground_truth.action_items,
            matcher,
            judge,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        decision_result = evaluate_hybrid_category(
            decision_predictions,
            ground_truth.decisions,
            matcher,
            judge,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    return EvaluationReport(
        action_items=MetricResult(
            precision=action_result["precision"],
            recall=action_result["recall"],
            f1=action_result["f1"],
        ),
        decisions=MetricResult(
            precision=decision_result["precision"],
            recall=decision_result["recall"],
            f1=decision_result["f1"],
        ),
        evidence_validity=compute_evidence_validity(minutes, chunks),
    )

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


def evaluate_llm_category(
    predictions,
    ground_truth,
    judge,
):
    return llm_evaluate_items(
        predictions,
        ground_truth,
        judge,
    )


def evaluate_hybrid_category(
    predictions,
    ground_truth,
    matcher,
    judge,
    top_k: int = 3,
    min_similarity: float = 0.0,
):
    return hybrid_evaluate_items(
        predictions,
        ground_truth,
        matcher,
        judge,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def compute_evidence_validity(
    minutes: MeetingMinutes,
    chunks: list[TranscriptChunk],
) -> float:
    total = len(minutes.action_items) + len(minutes.decisions)
    if total == 0:
        return 0.0

    valid_count = 0
    for action in minutes.action_items:
        if validate_action_evidence(action, chunks):
            valid_count += 1
    for decision in minutes.decisions:
        if validate_decision_evidence(decision, chunks):
            valid_count += 1

    return valid_count / total


def evaluate_meeting(
    minutes: MeetingMinutes,
    ground_truth: GroundTruthMeeting,
    chunks: list[TranscriptChunk],
    mode: str = "exact",
    matcher=None,
    judge=None,
    threshold: float = 0.75,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> EvaluationReport:
    """
    Evaluate predicted MeetingMinutes against GroundTruthMeeting using the
    selected offline strategy ('exact', 'semantic', 'llm', or 'hybrid').
    """
    action_predictions = [item.task for item in minutes.action_items]
    decision_predictions = [item.description for item in minutes.decisions]

    if mode == "exact":
        action_result = evaluate_category(
            action_predictions, ground_truth.action_items
        )
        decision_result = evaluate_category(
            decision_predictions, ground_truth.decisions
        )
    elif mode == "semantic":
        if matcher is None:
            raise ValueError("matcher is required for semantic mode.")
        action_result = evaluate_semantic_category(
            action_predictions,
            ground_truth.action_items,
            matcher,
            threshold=threshold,
        )
        decision_result = evaluate_semantic_category(
            decision_predictions,
            ground_truth.decisions,
            matcher,
            threshold=threshold,
        )
    elif mode == "llm":
        if judge is None:
            raise ValueError("judge is required for llm mode.")
        action_result = evaluate_llm_category(
            action_predictions, ground_truth.action_items, judge
        )
        decision_result = evaluate_llm_category(
            decision_predictions, ground_truth.decisions, judge
        )
    elif mode == "hybrid":
        if matcher is None or judge is None:
            raise ValueError(
                "matcher and judge are required for hybrid mode."
            )
        action_result = evaluate_hybrid_category(
            action_predictions,
            ground_truth.action_items,
            matcher,
            judge,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        decision_result = evaluate_hybrid_category(
            decision_predictions,
            ground_truth.decisions,
            matcher,
            judge,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    return EvaluationReport(
        action_items=MetricResult(
            precision=action_result["precision"],
            recall=action_result["recall"],
            f1=action_result["f1"],
        ),
        decisions=MetricResult(
            precision=decision_result["precision"],
            recall=decision_result["recall"],
            f1=decision_result["f1"],
        ),
        evidence_validity=compute_evidence_validity(minutes, chunks),
    )