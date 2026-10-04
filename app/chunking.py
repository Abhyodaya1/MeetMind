from app.schemas import TranscriptSegment, TranscriptChunk


def chunk_transcript(
    segments: list[TranscriptSegment],
    max_chars: int = 12000,
    overlap_segments: int = 2,
) -> list[TranscriptChunk]:

    if not segments:
        return []

    chunks = []

    start_index = 0
    total_segments = len(segments)

    while start_index < total_segments:

        current_segments = []
        current_length = 0

        index = start_index

        while index < total_segments:

            segment = segments[index]

            segment_length = len(segment.text)

            if (
                current_segments
                and current_length + segment_length > max_chars
            ):
                break

            current_segments.append(segment)
            current_length += segment_length

            index += 1

        if not current_segments:
            break

        chunk_text = " ".join(
            segment.text
            for segment in current_segments
        )

        chunk_start = current_segments[0].start
        chunk_end = current_segments[-1].end

        chunks.append(
            TranscriptChunk(
                text=chunk_text,
                start=chunk_start,
                end=chunk_end,
            )
        )

        if index >= total_segments:
            break

        next_start = index - overlap_segments

        start_index = max(
            start_index + 1,
            next_start,
        )

    return chunks