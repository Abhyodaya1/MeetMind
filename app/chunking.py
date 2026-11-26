from app.prompting import format_timestamped_segment
from app.schemas import TranscriptSegment, TranscriptChunk


def chunk_transcript(
    segments: list[TranscriptSegment],
    tokenizer,
    max_tokens: int = 8000,
    overlap_segments: int = 2,
    prompt_aware: bool = False,
    prompt_overhead_tokens: int = 0,
) -> list[TranscriptChunk]:

    if not segments:
        return []

    effective_max_tokens = max(1, max_tokens - prompt_overhead_tokens)

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

            if prompt_aware:
                candidate_text = "\n".join(
                    format_timestamped_segment(item)
                    for item in candidate_segments
                )
            else:
                candidate_text = " ".join(
                    item.text
                    for item in candidate_segments
                )

            token_count = tokenizer.count_tokens(
                candidate_text
            )

            if (
                current_segments
                and token_count > effective_max_tokens
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
        if prompt_aware:
            formatted_text = "\n".join(
                format_timestamped_segment(segment)
                for segment in current_segments
            )
            chunk_token_count = (
                tokenizer.count_tokens(formatted_text)
                + prompt_overhead_tokens
            )
        else:
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