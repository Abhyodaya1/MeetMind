from math import isclose
import re

from app.schemas import ActionItem, Decision, Evidence, TranscriptChunk


TIMESTAMP_TOLERANCE = 0.05

def normalize_text(text: str) -> str:
    return " ".join(
        text.lower().split()
    )

def get_transcript_text_for_range(
    start: float,
    end: float | None,
    chunks: list[TranscriptChunk],
) -> str:

    texts = []

    for chunk in chunks:
        for segment in chunk.segments:

            segment_end = segment.end

            if segment_end is None:
                segment_end = float("inf")

            evidence_end = (
                end
                if end is not None
                else start
            )

            overlaps = (
                segment.start <= evidence_end
                and segment_end >= start
            )

            if overlaps:
                texts.append(segment.text)

    return " ".join(texts)



def validate_evidence_text(
        evidence,
        chunks: list[TranscriptChunk],
    ) -> bool:
    transcript_text = get_transcript_text_for_range(
        evidence.start,
        evidence.end,
        chunks,
    )

    if not transcript_text:
        return False

    evidence_text = normalize_text(
        evidence.text
    )

    transcript_text = normalize_text(
        transcript_text
    )

    return evidence_text in transcript_text

def timestamp_exists(
    timestamp: float,
    chunks: list[TranscriptChunk],
) -> bool:

    for chunk in chunks:

        for segment in chunk.segments:

            if isclose(
                timestamp,
                segment.start,
                abs_tol=TIMESTAMP_TOLERANCE,
            ):
                return True

            if segment.end is not None:

                if isclose(
                    timestamp,
                    segment.end,
                    abs_tol=TIMESTAMP_TOLERANCE,
                ):
                    return True

    return False

def validate_evidence(
    evidence,
    chunks: list[TranscriptChunk],
) -> tuple[bool, str]:

    if evidence is None:
        return False, "missing_evidence"

    if evidence.start < 0:
        return False, "negative_start"

    if (
        evidence.end is not None
        and evidence.end < evidence.start
    ):
        return False, "invalid_range"

    if not timestamp_exists(
        evidence.start,
        chunks,
    ):
        return False, "start_timestamp_not_found"

    if (
        evidence.end is not None
        and not timestamp_exists(
            evidence.end,
            chunks,
        )
    ):
        return False, "end_timestamp_not_found"

    transcript_text = get_transcript_text_for_range(
        evidence.start,
        evidence.end,
        chunks,
    )

    if not transcript_text:
        return False, "no_transcript_in_range"

    normalized_evidence = normalize_text(
        evidence.text
    )

    normalized_transcript = normalize_text(
        transcript_text
    )

    if normalized_evidence not in normalized_transcript:
        return False, "evidence_text_not_found"

    return True, "valid"

def validate_action_evidence(
    action: ActionItem,
    chunks: list[TranscriptChunk],
) -> bool:

    valid, _ = validate_evidence(
        action.evidence,
        chunks,
    )

    return valid
    
def validate_decision_evidence(
    decision: Decision,
    chunks: list[TranscriptChunk],
) -> bool:

    valid, _ = validate_evidence(
        decision.evidence,
        chunks,
    )

    return valid


def _normalize_for_repair(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _token_containment(evidence_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not evidence_tokens or not candidate_tokens:
        return 0.0
    ev_set = set(evidence_tokens)
    cand_set = set(candidate_tokens)
    return len(ev_set & cand_set) / len(ev_set)


def repair_evidence(
    evidence: Evidence | None,
    chunks: list[TranscriptChunk],
    min_overlap_ratio: float = 0.75,
    max_window_segments: int = 3,
) -> tuple[Evidence | None, bool, str]:
    """
    Attempt deterministic evidence repair when strict validation fails.

    Preserves validate_evidence() as the strict baseline. Never invents
    timestamps: only snaps to real TranscriptSegment boundaries in `chunks`.
    """
    valid, reason = validate_evidence(evidence, chunks)
    if valid:
        return evidence, True, "valid"

    if evidence is None or not evidence.text.strip():
        return evidence, False, "missing_evidence"

    norm_ev = _normalize_for_repair(evidence.text)
    ev_tokens = norm_ev.split()
    if not ev_tokens:
        return evidence, False, "empty_evidence"

    seen: set[tuple[float, float | None, str]] = set()
    segments = []
    for chunk in chunks:
        for seg in chunk.segments:
            key = (seg.start, seg.end, seg.text)
            if key not in seen:
                seen.add(key)
                segments.append(seg)

    best_score = 0.0
    best_span = None

    for start_idx in range(len(segments)):
        for span_len in range(1, max_window_segments + 1):
            end_idx = start_idx + span_len
            if end_idx > len(segments):
                break
            window = segments[start_idx:end_idx]
            window_text = " ".join(s.text for s in window)
            norm_window = _normalize_for_repair(window_text)

            if norm_ev in norm_window:
                repaired = Evidence(
                    text=window_text,
                    start=window[0].start,
                    end=window[-1].end,
                )
                return repaired, True, "repaired_exact_span"

            score = _token_containment(ev_tokens, norm_window.split())
            if score > best_score:
                best_score = score
                best_span = (window, window_text)

    if best_span is not None and best_score >= min_overlap_ratio:
        window, window_text = best_span
        repaired = Evidence(
            text=window_text,
            start=window[0].start,
            end=window[-1].end,
        )
        return repaired, True, "repaired_token_overlap"

    return evidence, False, reason