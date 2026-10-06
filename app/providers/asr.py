from abc import ABC, abstractmethod

class ASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_file):
        """
        Transcribe the given audio file and return the transcribed text.

        :param audio_file_path: Path to the audio file to be transcribed.
        :return: Transcribed text.
        """
        pass