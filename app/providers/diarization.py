from abc import ABC, abstractmethod

from app.schemas import SpeakerTurn


class DiarizationProvider(ABC):
    @abstractmethod
    def diarize(self, audio_file) -> list[SpeakerTurn]:
        pass
