from app.transcription import transcribe_audio
from app.transcript import normalize_transcript
from app.chunking import chunk_transcript
from app.validation import validate_action_evidence

from app.prompting import (
    create_chunk_prompt,
    create_synthesis_prompt,
)

from app.llm import (
    analyze_chunk,
    synthesize_meeting,
)

from app.schemas import (
    ChunkAnalysis,
    MeetingMinutes,
)


def process_meeting(audio_file):

    print("Transcribing meeting...")

    transcript_result = transcribe_audio(audio_file)

    print("Normalizing transcript...")
    segments = normalize_transcript(
        transcript_result
    )

    if not segments:
        raise ValueError(
            "No valid transcript segments were produced."
        )

    print(
        f"Valid transcript segments: {len(segments)}"
    )

    print(
        "Creating timestamp-aware transcript chunks..."
    )

    chunks = chunk_transcript(segments)

    if not chunks:
        raise ValueError(
            "Transcript produced no chunks."
        )

    print(
        f"Created {len(chunks)} transcript chunks."
    )

    analyses = []
    failed_chunks = []

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"\nAnalyzing chunk "
            f"{index}/{len(chunks)}..."
        )

        print(
            f"Time range: "
            f"{chunk.start:.2f}s → "
            f"{chunk.end}"
        )

        prompt = create_chunk_prompt(
            chunk,
            index,
            len(chunks),
        )

        try:

            raw_analysis = analyze_chunk(prompt)

            analysis = ChunkAnalysis.model_validate(
                raw_analysis
            )

            analyses.append(analysis)

        except Exception as error:

            print(
                f"Chunk {index} failed: {error}"
            )

            failed_chunks.append(index)

    if not analyses:

        raise RuntimeError(
            "All transcript chunks failed analysis."
        )

    print(
        f"\nSuccessfully analyzed "
        f"{len(analyses)}/{len(chunks)} chunks."
    )

    if failed_chunks:
      raise RuntimeError(
        f"Chunk analysis failed for chunks: {failed_chunks}"
    )

    print(
        "\nSynthesizing final meeting analysis..."
    )

    synthesis_input = [
        analysis.model_dump()
        for analysis in analyses
    ]

    synthesis_prompt = create_synthesis_prompt(
        synthesis_input
    )

    raw_minutes = synthesize_meeting(
        synthesis_prompt
    )

    minutes = MeetingMinutes.model_validate(
        raw_minutes
    )

    print("\nValidating evidence...")

    for index, action in enumerate(
        minutes.action_items,
        start=1,
    ):

        valid = validate_action_evidence(
            action,
            chunks,
        )

        print(
            f"Action {index} evidence: "
            f"{'VALID' if valid else 'INVALID'}"
        )

    return minutes