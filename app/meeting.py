from app.transcription import transcribe_audio
from app.prompting import (
    create_chunk_prompt,
    create_synthesis_prompt,
)
from app.llm import analyze_chunk, synthesize_meeting
from app.schemas import ChunkAnalysis, MeetingMinutes
from app.chunking import chunk_text


def process_meeting(audio_file):

    print("Transcribing meeting...")
    transcript = transcribe_audio(audio_file)

    print("Creating transcript chunks...")
    chunks = chunk_text(transcript)
    print(f"Created {len(chunks)} transcript chunks.")

    analyses = []
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Analyzing chunk {index}/{len(chunks)}..."
        )
        prompt = create_chunk_prompt(
            chunk,
            index,
            len(chunks),
        )

        raw_analysis = analyze_chunk(prompt)
        analysis = ChunkAnalysis.model_validate(
            raw_analysis
        )
        analyses.append(analysis)
        
    print("Synthesizing final meeting analysis...")

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

    return minutes