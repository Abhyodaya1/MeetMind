from math import isclose

from app.schemas import ActionItem, TranscriptChunk


TIMESTAMP_TOLERANCE = 0.05


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


def validate_action_evidence(
    action: ActionItem,
    chunks: list[TranscriptChunk],
) -> bool:

    if action.evidence is None:
        return False

    evidence = action.evidence

    if evidence.start < 0:
        return False

    if (
        evidence.end is not None
        and evidence.end < evidence.start
    ):
        return False

    if not timestamp_exists(
        evidence.start,
        chunks,
    ):
        return False

    if evidence.end is not None:

        if not timestamp_exists(
            evidence.end,
            chunks,
        ):
            return False

    return True