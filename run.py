from app.transcription import transcribe_audio
from pathlib import Path



audio_file = Path(__file__).parent / "data" / "audio" / "harvard.wav"

transcription = transcribe_audio(audio_file)

print("Transcription result:", transcription)