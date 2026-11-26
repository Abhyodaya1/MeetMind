import json
from pathlib import Path

from pydantic import BaseModel

from app.evaluation.evaluator import evaluate_meeting
from app.schemas import (
    EvaluationReport,
    GroundTruthMeeting,
    MeetingMinutes,
    TranscriptChunk,
)


class BenchmarkCase(BaseModel):
    case_id: str
    chunks: list[TranscriptChunk]
    minutes: MeetingMinutes
    ground_truth: GroundTruthMeeting


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(item) for item in data]


def run_benchmark(
    cases: list[BenchmarkCase],
    mode: str = "exact",
    matcher=None,
    judge=None,
    threshold: float = 0.75,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> dict:
    """
    Run offline evaluation across all benchmark cases and aggregate macro metrics.
    """
    if not cases:
        raise ValueError("Benchmark cases list cannot be empty.")

    case_reports: dict[str, EvaluationReport] = {}

    for case in cases:
        case_reports[case.case_id] = evaluate_meeting(
            minutes=case.minutes,
            ground_truth=case.ground_truth,
            chunks=case.chunks,
            mode=mode,
            matcher=matcher,
            judge=judge,
            threshold=threshold,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    n = len(case_reports)
    avg_action_f1 = sum(r.action_items.f1 for r in case_reports.values()) / n
    avg_decision_f1 = sum(r.decisions.f1 for r in case_reports.values()) / n
    avg_evidence_validity = (
        sum(r.evidence_validity for r in case_reports.values()) / n
    )

    return {
        "mode": mode,
        "num_cases": n,
        "avg_action_f1": avg_action_f1,
        "avg_decision_f1": avg_decision_f1,
        "avg_evidence_validity": avg_evidence_validity,
        "cases": {
            case_id: report.model_dump()
            for case_id, report in case_reports.items()
        },
    }


def calibrate_semantic_threshold(
    cases: list[BenchmarkCase],
    matcher,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Sweep semantic similarity thresholds over labeled benchmark cases to
    observe precision/recall/F1 trade-offs empirically.
    """
    if thresholds is None:
        thresholds = [0.55, 0.65, 0.70, 0.75, 0.80, 0.85]

    results = []
    for threshold in thresholds:
        summary = run_benchmark(
            cases,
            mode="semantic",
            matcher=matcher,
            threshold=threshold,
        )
        results.append(
            {
                "threshold": threshold,
                "avg_action_f1": summary["avg_action_f1"],
                "avg_decision_f1": summary["avg_decision_f1"],
                "avg_evidence_validity": summary["avg_evidence_validity"],
            }
        )

    return results
