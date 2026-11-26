import argparse
import json
from pathlib import Path

from app.evaluation.benchmark import (
    calibrate_semantic_threshold,
    load_benchmark,
    run_benchmark,
)
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.semantic_matcher import SemanticMatcher
from app.providers.groq import GroqProvider


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).parent / "data" / "benchmark" / "meetings.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Run MeetMind offline evaluation benchmark."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to labeled benchmark JSON file.",
    )
    parser.add_argument(
        "--mode",
        choices=["exact", "semantic", "llm", "hybrid", "calibrate"],
        default="exact",
        help="Evaluation mode to run.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Semantic similarity threshold (for semantic mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K semantic candidates passed to LLMJudge (for hybrid mode).",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-oss-20b",
        help="Groq model used by LLMJudge.",
    )
    args = parser.parse_args()

    cases = load_benchmark(args.dataset)

    if args.mode == "exact":
        output = run_benchmark(cases, mode="exact")
    elif args.mode == "semantic":
        matcher = SemanticMatcher()
        output = run_benchmark(
            cases,
            mode="semantic",
            matcher=matcher,
            threshold=args.threshold,
        )
    elif args.mode == "calibrate":
        matcher = SemanticMatcher()
        output = calibrate_semantic_threshold(cases, matcher)
    elif args.mode == "llm":
        judge = LLMJudge(GroqProvider(model=args.judge_model))
        output = run_benchmark(cases, mode="llm", judge=judge)
    else:
        matcher = SemanticMatcher()
        judge = LLMJudge(GroqProvider(model=args.judge_model))
        output = run_benchmark(
            cases,
            mode="hybrid",
            matcher=matcher,
            judge=judge,
            top_k=args.top_k,
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
