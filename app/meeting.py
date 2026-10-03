from app.transcription import transcribe_audio
from app.prompting import create_meeting_prompt
from app.llm import generate_minutes
from app.schemas import MeetingMinutes


def process_meeting(audio_file):
    """
    Process a meeting audio file from start to finish.

    Audio
        ↓
    Transcription
        ↓
    Prompt creation
        ↓
    LLM analysis
        ↓
    Pydantic validation

    Returns:
        MeetingMinutes
    """

    print("Transcribing meeting...")

    transcript = transcribe_audio(audio_file)

    print("Generating meeting analysis...")

    prompt = create_meeting_prompt(transcript)

    raw_minutes = generate_minutes(prompt)

    print("Validating meeting analysis...")

    minutes = MeetingMinutes.model_validate(raw_minutes)

    return minutes