import unittest

from app.evaluation.evaluator import (
    compute_evidence_validity,
    evaluate_category,
    evaluate_hybrid_category,
    evaluate_meeting,
    evaluate_semantic_category,
)
from app.evaluation.hybrid_evaluator import hybrid_evaluate_items
from app.evaluation.llm_evaluator import llm_evaluate_items
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.metrics import f1_score, precision, recall
from app.providers.llm import LLMProvider
from app.schemas import (
    ActionItem,
    Decision,
    Evidence,
    GroundTruthItem,
    GroundTruthMeeting,
    MeetingMinutes,
    TranscriptChunk,
    TranscriptSegment,
)


class FakeMatcher:
    def __init__(self, scores: dict[tuple[str, str], float]):
        self.scores = scores

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.scores.get((text_a, text_b), 0.0)


class FakeLLMProvider(LLMProvider):
    def __init__(self, decisions: dict[tuple[str, str], tuple[bool, float, str]]):
        self.decisions = decisions
        self.calls: list[tuple[str, str]] = []

    def analyze(self, prompt):
        raise NotImplementedError

    def synthesize(self, prompt):
        raise NotImplementedError

    def judge(self, prompt):
        pred_line = prompt.split("Prediction:\n", 1)[1].split("\n\nGround truth:", 1)[0]
        gt_line = prompt.split("Ground truth:\n", 1)[1].split("\n\nReturn ONLY", 1)[0]
        self.calls.append((pred_line, gt_line))
        match, score, reason = self.decisions.get(
            (pred_line, gt_line), (False, 0.0, "no match")
        )
        return {"match": match, "score": score, "reason": reason}


class TestMetrics(unittest.TestCase):
    def test_zero_division_returns_zero(self):
        self.assertEqual(precision(0, 0), 0.0)
        self.assertEqual(recall(0, 0), 0.0)
        self.assertEqual(f1_score(0.0, 0.0), 0.0)

    def test_standard_scores(self):
        p = precision(2, 1)
        r = recall(2, 2)
        self.assertAlmostEqual(p, 2 / 3)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f1_score(p, r), 2 * p * r / (p + r))


class TestEvaluators(unittest.TestCase):
    def test_exact_category(self):
        gt = [
            GroundTruthItem(text="Prepare the project presentation."),
            GroundTruthItem(text="Send weekly report."),
        ]
        preds = ["prepare the project presentation.", "Unrelated task"]
        res = evaluate_category(preds, gt)
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 1)
        self.assertEqual(res["f1"], 0.5)

    def test_semantic_category_respects_threshold_and_one_to_one(self):
        gt = [GroundTruthItem(text="gt1"), GroundTruthItem(text="gt2")]
        preds = ["p1", "p2"]
        matcher = FakeMatcher({
            ("p1", "gt1"): 0.85,
            ("p1", "gt2"): 0.40,
            ("p2", "gt1"): 0.90,
            ("p2", "gt2"): 0.60,
        })
        res = evaluate_semantic_category(preds, gt, matcher, threshold=0.75)
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 1)

    def test_llm_judge_and_evaluator(self):
        provider = FakeLLMProvider({
            ("Develop remote design", "David will create the remote control design."): (
                True,
                0.92,
                "Same task",
            ),
        })
        judge = LLMJudge(provider)
        gt = [
            GroundTruthItem(text="David will create the remote control design."),
            GroundTruthItem(text="Prepare slides."),
        ]
        res = llm_evaluate_items(["Develop remote design"], gt, judge)
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_negatives"], 1)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 0.5)

    def test_hybrid_evaluator_limits_judge_calls_to_top_k(self):
        gt = [
            GroundTruthItem(text="gt_low"),
            GroundTruthItem(text="gt_mid"),
            GroundTruthItem(text="gt_high"),
            GroundTruthItem(text="gt_other"),
        ]
        matcher = FakeMatcher({
            ("pred", "gt_low"): 0.1,
            ("pred", "gt_mid"): 0.5,
            ("pred", "gt_high"): 0.9,
            ("pred", "gt_other"): 0.2,
        })
        provider = FakeLLMProvider({
            ("pred", "gt_high"): (True, 0.95, "Best semantic match"),
        })
        judge = LLMJudge(provider)

        res = hybrid_evaluate_items(["pred"], gt, matcher, judge, top_k=2)
        self.assertEqual(res["judge_calls"], 2)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0][1], "gt_high")
        self.assertEqual(provider.calls[1][1], "gt_mid")
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["false_negatives"], 3)

    def test_hybrid_min_similarity_filters_candidates(self):
        gt = [GroundTruthItem(text="gt1"), GroundTruthItem(text="gt2")]
        matcher = FakeMatcher({("pred", "gt1"): 0.2, ("pred", "gt2"): 0.3})
        provider = FakeLLMProvider({})
        judge = LLMJudge(provider)

        res = evaluate_hybrid_category(
            ["pred"], gt, matcher, judge, top_k=2, min_similarity=0.5
        )
        self.assertEqual(res["judge_calls"], 0)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 2)

    def test_evaluate_meeting_produces_report(self):
        chunks = [
            TranscriptChunk(
                text="We will move auth to OAuth. I will handle the migration.",
                start=0.0,
                end=10.0,
                token_count=12,
                segments=[
                    TranscriptSegment(
                        text="We will move auth to OAuth.",
                        start=0.0,
                        end=5.0,
                    ),
                    TranscriptSegment(
                        text="I will handle the migration.",
                        start=5.0,
                        end=10.0,
                    ),
                ],
            )
        ]
        minutes = MeetingMinutes(
            title="Auth Sync",
            summary="Discussed auth.",
            decisions=[
                Decision(
                    description="Move auth to OAuth",
                    evidence=Evidence(
                        text="We will move auth to OAuth.",
                        start=0.0,
                        end=5.0,
                    ),
                )
            ],
            action_items=[
                ActionItem(
                    task="Handle the migration",
                    evidence=Evidence(
                        text="hallucinated quote",
                        start=5.0,
                        end=10.0,
                    ),
                )
            ],
        )
        gt = GroundTruthMeeting(
            decisions=[GroundTruthItem(text="Move auth to OAuth")],
            action_items=[GroundTruthItem(text="Handle the migration")],
        )
        report = evaluate_meeting(minutes, gt, chunks, mode="exact")
        self.assertEqual(report.action_items.f1, 1.0)
        self.assertEqual(report.decisions.f1, 1.0)
        self.assertAlmostEqual(report.evidence_validity, 0.5)
        self.assertAlmostEqual(compute_evidence_validity(minutes, chunks), 0.5)


if __name__ == "__main__":
    unittest.main()
