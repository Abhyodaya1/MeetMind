from app.transcript import normalize_transcript
from app.chunking import chunk_transcript
from app.validation import validate_action_evidence
from app.providers.whisper import WhisperProvider
from app.providers.groq import GroqProvider
from app.providers.hf_tokenizer import HFTokenizerProvider
from app.config import MAX_TRANSCRIPT_TOKENS



from app.prompting import (
    create_chunk_prompt,
    create_synthesis_prompt,
)

from app.schemas import (
    ChunkAnalysis,
    MeetingMinutes,
)


def process_meeting(audio_file):

    asr = WhisperProvider()
    llm = GroqProvider()
    tokenizer = HFTokenizerProvider()

    print("Transcribing meeting...")

    transcript_result = asr.transcribe(audio_file)

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

    chunks = chunk_transcript(
    segments,
    tokenizer=tokenizer,
    max_tokens=MAX_TRANSCRIPT_TOKENS,
)

    
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
            f"Chunk {index}: "
        f"{chunk.token_count} tokens | "
        f"{chunk.start:.2f}s → "
        f"{chunk.end if chunk.end is not None else 'unknown'}s"
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

            raw_analysis = llm.analyze(prompt)

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

    raw_minutes = llm.synthesize(
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