from transformers import pipeline
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",  
    dtype=DTYPE,
    device=DEVICE
)

def transcribe_audio(audio_file):
  pipe = whisper_pipeline
  result = pipe(str(audio_file))

  return result["text"]