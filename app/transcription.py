from transformers import pipeline
import torch

def transcribe_audio(audio_file):
  pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",
    dtype=torch.float16,
    device="cuda" if torch.cuda.is_available() else "cpu"
  )

  result = pipe(str(audio_file))

  return result["text"]