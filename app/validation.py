from math import isclose

from app.schemas import ActionItem, TranscriptChunk , Decision


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