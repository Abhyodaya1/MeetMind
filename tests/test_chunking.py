import unittest

from app.chunking import chunk_transcript
from app.prompting import (
    create_chunk_prompt,
    estimate_chunk_prompt_overhead,
)
from app.schemas import TranscriptSegment


class WordTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class TestPromptAwareChunking(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()
        self.segments = [
            TranscriptSegment(
                text=f"Segment number {i} discusses project timeline.",
                start=float(i * 5),
                end=float((i + 1) * 5),
            )
            for i in range(6)
        ]

    def test_legacy_chunking_preserves_behavior(self):
        chunks = chunk_transcript(
            self.segments,
            tokenizer=self.tokenizer,
            max_tokens=15,
            overlap_segments=1,
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 15)

    def test_prompt_aware_chunking_stays_within_total_budget(self):
        overhead = estimate_chunk_prompt_overhead(self.tokenizer)
        max_total = overhead + 35

        chunks = chunk_transcript(
            self.segments,
            tokenizer=self.tokenizer,
            max_tokens=max_total,
            overlap_segments=1,
            prompt_aware=True,
            prompt_overhead_tokens=overhead,
        )
        self.assertGreater(len(chunks), 1)
        for idx, chunk in enumerate(chunks, start=1):
            full_prompt = create_chunk_prompt(chunk, idx, len(chunks))
            actual_tokens = self.tokenizer.count_tokens(full_prompt)
            self.assertLessEqual(actual_tokens, max_total)


if __name__ == "__main__":
    unittest.main()
