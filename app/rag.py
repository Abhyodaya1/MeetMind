import math

from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.schemas import (
    GroundedAnswer,
    IndexedChunk,
    RetrievedChunk,
    TranscriptChunk,
)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MeetingIndex:
    """In-memory vector index for cross-meeting TranscriptChunk retrieval."""

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self._entries: list[tuple[IndexedChunk, list[float]]] = []

    def add_meeting(
        self,
        meeting_id: str,
        chunks: list[TranscriptChunk],
    ) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed(texts)

        added = 0
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            speakers = sorted(
                {
                    seg.speaker
                    for seg in chunk.segments
                    if seg.speaker is not None
                }
            )
            indexed = IndexedChunk(
                meeting_id=meeting_id,
                chunk_index=idx,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                speakers=speakers,
            )
            self._entries.append((indexed, vector))
            added += 1

        return added

    def search(
        self,
        query: str,
        top_k: int = 3,
        meeting_id: str | None = None,
        speaker: str | None = None,
    ) -> list[RetrievedChunk]:
        if top_k < 1 or not self._entries:
            return []

        query_vec = self.embedder.embed([query])[0]
        scored: list[RetrievedChunk] = []

        for indexed, vector in self._entries:
            if meeting_id is not None and indexed.meeting_id != meeting_id:
                continue
            if speaker is not None and speaker not in indexed.speakers:
                continue

            score = _cosine_similarity(query_vec, vector)
            scored.append(RetrievedChunk(chunk=indexed, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def answer_question(
    query: str,
    index: MeetingIndex,
    llm: LLMProvider,
    top_k: int = 3,
    meeting_id: str | None = None,
    speaker: str | None = None,
) -> GroundedAnswer:
    """Answer a cross-meeting question grounded in retrieved transcript chunks."""
    retrieved = index.search(
        query=query,
        top_k=top_k,
        meeting_id=meeting_id,
        speaker=speaker,
    )
    if not retrieved:
        return GroundedAnswer(
            answer="No relevant meeting records found.",
            citations=[],
        )

    context_blocks = []
    for item in retrieved:
        c = item.chunk
        end_str = f"{c.end:.2f}s" if c.end is not None else "unknown"
        speakers_str = ", ".join(c.speakers) if c.speakers else "unknown"
        context_blocks.append(
            f"[Meeting: {c.meeting_id} | {c.start:.2f}s → {end_str} | Speakers: {speakers_str}]\n"
            f"{c.text}"
        )

    prompt = (
        "You are MeetMind, an evidence-grounded meeting intelligence assistant.\n"
        "Answer the user question using ONLY the meeting excerpts below.\n"
        'Return ONLY valid JSON: {"answer": "string"}\n\n'
        f"Question:\n{query}\n\n"
        "Meeting excerpts:\n----------------\n"
        + "\n\n".join(context_blocks)
        + "\n----------------\n"
    )

    response = llm.synthesize(prompt)
    return GroundedAnswer(
        answer=response.get("answer", ""),
        citations=retrieved,
    )
