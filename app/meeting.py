from app.transcript import normalize_transcript
from app.chunking import chunk_transcript
from app.diarization import align_speakers
from app.multimodal import extract_visual_materials, format_visual_context
from app.validation import validate_action_evidence, validate_decision_evidence, validate_evidence
from app.providers.asr import ASRProvider
from app.providers.diarization import DiarizationProvider
from app.providers.llm import LLMProvider
from app.providers.tokenizer import TokenizerProvider
from app.providers.vision import VisionProvider
from app.providers.whisper import WhisperProvider
from app.providers.groq import GroqProvider
from app.providers.hf_tokenizer import HFTokenizerProvider
from app.config import MAX_CONTEXT_TOKENS, RESERVED_PROMPT_TOKENS
from app.prompting import (
    create_chunk_prompt,
    create_synthesis_prompt,
    estimate_chunk_prompt_overhead,
)

from app.schemas import (
    ChunkAnalysis,
    EvidenceValidationSummary,
    MeetingMinutes,
    MeetingProcessingResult,
)


def process_meeting_full(
    audio_file,
    diarizer: DiarizationProvider | None = None,
    visual_inputs: list | None = None,
    vision_provider: VisionProvider | None = None,
    asr: ASRProvider | None = None,
    llm: LLMProvider | None = None,
    tokenizer: TokenizerProvider | None = None,
) -> MeetingProcessingResult:
    """
    Execute the end-to-end meeting intelligence pipeline and return a typed
    MeetingProcessingResult contract suitable for downstream APIs and RAG indexing.
    """
    asr = asr or WhisperProvider()
    llm = llm or GroqProvider()
    tokenizer = tokenizer or HFTokenizerProvider()

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

    if diarizer is not None:
        print("Running speaker diarization...")
        turns = diarizer.diarize(audio_file)
        segments = align_speakers(segments, turns)

    print(
        f"Valid transcript segments: {len(segments)}"
    )

    print(
        "Creating prompt-aware, timestamp-aware transcript chunks..."
    )

    prompt_overhead = max(
        RESERVED_PROMPT_TOKENS,
        estimate_chunk_prompt_overhead(tokenizer),
    )

    chunks = chunk_transcript(
        segments,
        tokenizer=tokenizer,
        max_tokens=MAX_CONTEXT_TOKENS,
        prompt_aware=True,
        prompt_overhead_tokens=prompt_overhead,
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

    visual_materials = extract_visual_materials(
        visual_inputs or [],
        vision_provider=vision_provider,
    )
    visual_context = format_visual_context(visual_materials)

    print(
        "\nSynthesizing final meeting analysis..."
    )

    synthesis_input = [
        analysis.model_dump()
        for analysis in analyses
    ]

    synthesis_prompt = create_synthesis_prompt(
        synthesis_input,
        visual_context=visual_context,
    )

    raw_minutes = llm.synthesize(
        synthesis_prompt
    )

    minutes = MeetingMinutes.model_validate(
        raw_minutes
    )

    print("\nValidating evidence...")

    actions_valid = 0
    for index, action in enumerate(
        minutes.action_items,
        start=1,
    ):

        valid, reason = validate_evidence(
            action.evidence,
            chunks,
        )
        if valid:
            actions_valid += 1

        print(
            f"Action {index} evidence: "
            f"{'VALID' if valid else 'INVALID'} "
            f"({reason})"
        )

    decisions_valid = 0
    for i, decision in enumerate(
        minutes.decisions,
        start=1,
    ):
        valid = validate_decision_evidence(
            decision,
            chunks,
        )
        if valid:
            decisions_valid += 1

        print(
            f"Decision {i} evidence: "
            f"{'VALID' if valid else 'INVALID'}"
        )

    total_items = len(minutes.action_items) + len(minutes.decisions)
    validity_ratio = (
        (actions_valid + decisions_valid) / total_items
        if total_items > 0
        else 0.0
    )

    return MeetingProcessingResult(
        minutes=minutes,
        segments=segments,
        chunks=chunks,
        visual_materials=visual_materials,
        validation_summary=EvidenceValidationSummary(
            action_items_valid=actions_valid,
            action_items_total=len(minutes.action_items),
            decisions_valid=decisions_valid,
            decisions_total=len(minutes.decisions),
            validity_ratio=validity_ratio,
        ),
    )


def process_meeting(
    audio_file,
    diarizer: DiarizationProvider | None = None,
) -> MeetingMinutes:
    return process_meeting_full(
        audio_file,
        diarizer=diarizer,
    ).minutes