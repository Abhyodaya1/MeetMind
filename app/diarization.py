from app.schemas import SpeakerTurn, TranscriptSegment


def _overlap_duration(
    seg_start: float,
    seg_end: float | None,
    turn: SpeakerTurn,
) -> float:
    if seg_end is None:
        return 1.0 if turn.start <= seg_start <= turn.end else 0.0

    overlap_start = max(seg_start, turn.start)
    overlap_end = min(seg_end, turn.end)
    return max(0.0, overlap_end - overlap_start)


def align_speakers(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """
    Assign the best-overlapping SpeakerTurn label to each TranscriptSegment.

    Returns new TranscriptSegment instances; segments with no overlapping turn
    keep `speaker=None`.
    """
    if not turns:
        return list(segments)

    aligned: list[TranscriptSegment] = []
    for segment in segments:
        best_speaker = None
        best_overlap = 0.0

        for turn in turns:
            overlap = _overlap_duration(
                segment.start,
                segment.end,
                turn,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        aligned.append(
            segment.model_copy(update={"speaker": best_speaker})
        )

    return aligned
