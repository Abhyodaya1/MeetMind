from transformers import pipeline
import torch

from app.providers.asr import ASRProvider


class WhisperProvider(ASRProvider):

    def __init__(
        self,
        model_name="openai/whisper-small.en",
    ):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        dtype = (
            torch.float16
            if device == "cuda"
            else torch.float32
        )

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=dtype,
            device=device,
        )

    def transcribe(self, audio_file):

        return self.pipeline(
            str(audio_file),
            return_timestamps=True,
        )