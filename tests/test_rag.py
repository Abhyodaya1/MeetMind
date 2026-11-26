import unittest

from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.rag import MeetingIndex, answer_question
from app.schemas import TranscriptChunk, TranscriptSegment


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic keyword-axis embedder for offline RAG unit tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([
                1.0 if "oauth" in lower or "auth" in lower else 0.0,
                1.0 if "remote" in lower or "design" in lower else 0.0,
                1.0 if "budget" in lower else 0.0,
            ])
        return vectors


class FakeQALLMProvider(LLMProvider):
    def __init__(self):
        self.last_prompt = None

    def analyze(self, prompt):
        raise NotImplementedError

    def synthesize(self, prompt):
        self.last_prompt = prompt
        return {"answer": "Authentication is moving to OAuth 2.0."}

    def judge(self, prompt):
        raise NotImplementedError


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.index = MeetingIndex(FakeEmbeddingProvider())
        self.index.add_meeting(
            "m_auth",
            [
                TranscriptChunk(
                    text="We decided to move auth to OAuth 2.0.",
                    start=0.0,
                    end=10.0,
                    token_count=8,
                    segments=[
                        TranscriptSegment(
                            text="We decided to move auth to OAuth 2.0.",
                            start=0.0,
                            end=10.0,
                            speaker="Speaker 1",
                        )
                    ],
                )
            ],
        )
        self.index.add_meeting(
            "m_remote",
            [
                TranscriptChunk(
                    text="David will create the remote control design.",
                    start=5.0,
                    end=15.0,
                    token_count=7,
                    segments=[
                        TranscriptSegment(
                            text="David will create the remote control design.",
                            start=5.0,
                            end=15.0,
                            speaker="Speaker 2",
                        )
                    ],
                )
            ],
        )

    def test_cross_meeting_search_and_filters(self):
        results = self.index.search("What happened with auth?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.meeting_id, "m_auth")
        self.assertEqual(results[0].chunk.speakers, ["Speaker 1"])

        filtered = self.index.search("anything", top_k=3, speaker="Speaker 2")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk.meeting_id, "m_remote")

    def test_answer_question_returns_grounded_answer(self):
        llm = FakeQALLMProvider()
        res = answer_question("How are we handling auth?", self.index, llm, top_k=1)
        self.assertEqual(res.answer, "Authentication is moving to OAuth 2.0.")
        self.assertEqual(len(res.citations), 1)
        self.assertEqual(res.citations[0].chunk.meeting_id, "m_auth")
        self.assertIn("m_auth", llm.last_prompt)

    def test_empty_index_skips_llm(self):
        empty_index = MeetingIndex(FakeEmbeddingProvider())
        llm = FakeQALLMProvider()
        res = answer_question("What about OAuth?", empty_index, llm)
        self.assertEqual(res.citations, [])
        self.assertIsNone(llm.last_prompt)


if __name__ == "__main__":
    unittest.main()
