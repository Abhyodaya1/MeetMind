from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def analyze(self, prompt):
        pass

    @abstractmethod
    def synthesize(self, prompt):
        pass