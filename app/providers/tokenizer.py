from abc import ABC, abstractmethod

class TokenizerProvider(ABC):

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass