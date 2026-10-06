from app.schemas import TranscriptSegment, TranscriptChunk


def chunk_transcript(
    segments: list[TranscriptSegment],
    tokenizer,
    max_tokens: int = 8000,
    overlap_segments: int = 2,
) -> list[TranscriptChunk]:

    if not segments:
        return []

    chunks = []

    start_index = 0
    total_segments = len(segments)

    while start_index < total_segments:

        current_segments = []

        index = start_index

        while index < total_segments:

            segment = segments[index]

            candidate_segments = (
                current_segments + [segment]
            )

            candidate_text = " ".join(
                item.text
                for item in candidate_segments
            )

            token_count = tokenizer.count_tokens(
                candidate_text
            )

            if (
                current_segments
                and token_count > max_tokens
            ):
                break

            current_segments.append(segment)

            index += 1

        if not current_segments:
            break

        chunk_text = " ".join(
            segment.text
            for segment in current_segments
        )
        chunk_token_count = tokenizer.count_tokens(
        chunk_text
         )
        chunk_start = current_segments[0].start
        chunk_end = current_segments[-1].end

        chunks.append(
            TranscriptChunk(
                text=chunk_text,
                start=chunk_start,
                end=chunk_end,
                token_count=chunk_token_count,
                segments=current_segments,
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