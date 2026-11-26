import unittest
from pathlib import Path

from app.evaluation.benchmark import (
    calibrate_semantic_threshold,
    load_benchmark,
    run_benchmark,
)
from app.evaluation.llm_judge import LLMJudge
from tests.test_evaluation import FakeLLMProvider, FakeMatcher


BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "benchmark"
    / "meetings.json"
)


class TestBenchmark(unittest.TestCase):
    def test_load_and_run_exact_benchmark(self):
        cases = load_benchmark(BENCHMARK_PATH)
        self.assertEqual(len(cases), 2)

        summary = run_benchmark(cases, mode="exact")
        self.assertEqual(summary["num_cases"], 2)
        self.assertIn("ami_remote_design_01", summary["cases"])
        self.assertIn("auth_migration_02", summary["cases"])
        # auth_migration_02 has 1 valid decision + 1 valid action + 1 invalid action => 2/3
        self.assertAlmostEqual(
            summary["cases"]["auth_migration_02"]["evidence_validity"],
            2 / 3,
        )

    def test_calibrate_semantic_threshold(self):
        cases = load_benchmark(BENCHMARK_PATH)
        matcher = FakeMatcher({
            ("Develop the working design of the remote control.", "David will create the remote control design."): 0.72,
            ("Create the project presentation.", "Prepare the project presentation."): 0.88,
            ("Use a curved ergonomic casing for the remote control.", "Adopt a curved ergonomic casing."): 0.81,
            ("Move authentication to OAuth 2.0.", "Move authentication to OAuth 2.0."): 1.0,
            ("Write the OAuth migration checklist.", "Draft the migration checklist."): 0.78,
        })
        sweep = calibrate_semantic_threshold(
            cases, matcher, thresholds=[0.70, 0.80]
        )
        self.assertEqual(len(sweep), 2)
        self.assertGreater(sweep[0]["avg_action_f1"], sweep[1]["avg_action_f1"])

    def test_run_hybrid_benchmark(self):
        cases = load_benchmark(BENCHMARK_PATH)
        matcher = FakeMatcher({
            ("Develop the working design of the remote control.", "David will create the remote control design."): 0.72,
            ("Create the project presentation.", "Prepare the project presentation."): 0.88,
            ("Use a curved ergonomic casing for the remote control.", "Adopt a curved ergonomic casing."): 0.81,
            ("Move authentication to OAuth 2.0.", "Move authentication to OAuth 2.0."): 1.0,
            ("Write the OAuth migration checklist.", "Draft the migration checklist."): 0.78,
        })
        provider = FakeLLMProvider({
            ("Develop the working design of the remote control.", "David will create the remote control design."): (True, 0.90, "Same task"),
            ("Create the project presentation.", "Prepare the project presentation."): (True, 0.95, "Same task"),
            ("Use a curved ergonomic casing for the remote control.", "Adopt a curved ergonomic casing."): (True, 0.92, "Same decision"),
            ("Move authentication to OAuth 2.0.", "Move authentication to OAuth 2.0."): (True, 1.0, "Exact"),
            ("Write the OAuth migration checklist.", "Draft the migration checklist."): (True, 0.91, "Same task"),
        })
        judge = LLMJudge(provider)
        summary = run_benchmark(
            cases, mode="hybrid", matcher=matcher, judge=judge, top_k=2
        )
        self.assertEqual(summary["avg_decision_f1"], 1.0)
        # Case 1 has F1=1.0, Case 2 has 1 TP, 1 FP, 0 FN => P=0.5, R=1.0, F1=2/3
        self.assertAlmostEqual(
            summary["avg_action_f1"], (1.0 + (2 / 3)) / 2
        )


if __name__ == "__main__":
    unittest.main()
