import unittest

from app.meeting import process_meeting_full
from app.providers.asr import ASRProvider
from app.providers.diarization import DiarizationProvider
from app.providers.llm import LLMProvider
from app.providers.tokenizer import TokenizerProvider
from app.providers.vision import VisionProvider
from app.schemas import SpeakerTurn, VisualMaterial


class FakeASR(ASRProvider):
    def transcribe(self, audio_file):
        return {
            "chunks": [
                {
                    "text": "We are moving authentication to OAuth 2.0.",
                    "timestamp": [0.0, 5.0],
                },
                {
                    "text": "I will handle the migration.",
                    "timestamp": [5.0, 10.0],
                },
            ]
        }


class FakeDiarizer(DiarizationProvider):
    def diarize(self, audio_file):
        return [
            SpeakerTurn(speaker="Speaker 1", start=0.0, end=5.0),
            SpeakerTurn(speaker="Speaker 2", start=5.0, end=10.0),
        ]


class FakeTokenizer(TokenizerProvider):
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeVision(VisionProvider):
    def analyze_material(self, file_path, material_type="slide", timestamp=None):
        return VisualMaterial(
            source_path=file_path,
            material_type=material_type,
            timestamp=timestamp,
            extracted_text="Current -> JWT | Proposed -> OAuth 2.0",
            summary="Architecture slide comparing JWT and OAuth 2.0.",
        )


class FakePipelineLLM(LLMProvider):
    def __init__(self):
        self.synthesis_prompt = None

    def analyze(self, prompt):
        return {
            "key_points": ["Authentication migration planned"],
            "decisions": [
                {
                    "description": "Move authentication to OAuth 2.0.",
                    "evidence": {
                        "text": "We are moving authentication to OAuth 2.0.",
                        "start": 0.0,
                        "end": 5.0,
                    },
                }
            ],
            "action_items": [
                {
                    "task": "Handle the migration",
                    "owner": "Speaker 2",
                    "deadline": None,
                    "evidence": {
                        "text": "I will handle the migration.",
                        "start": 5.0,
                        "end": 10.0,
                    },
                }
            ],
        }

    def synthesize(self, prompt):
        self.synthesis_prompt = prompt
        return {
            "title": "Auth Migration Sync",
            "summary": "Agreed to move from JWT to OAuth 2.0.",
            "key_points": ["Authentication migration planned"],
            "decisions": [
                {
                    "description": "Move authentication to OAuth 2.0.",
                    "evidence": {
                        "text": "We are moving authentication to OAuth 2.0.",
                        "start": 0.0,
                        "end": 5.0,
                    },
                }
            ],
            "action_items": [
                {
                    "task": "Handle the migration",
                    "owner": "Speaker 2",
                    "deadline": None,
                    "evidence": {
                        "text": "I will handle the migration.",
                        "start": 5.0,
                        "end": 10.0,
                    },
                }
            ],
        }

    def judge(self, prompt):
        raise NotImplementedError


class TestMultimodalAndContract(unittest.TestCase):
    def test_process_meeting_full_contract_with_multimodal(self):
        llm = FakePipelineLLM()
        result = process_meeting_full(
            "dummy.wav",
            diarizer=FakeDiarizer(),
            visual_inputs=[{"source_path": "slide_1.png", "material_type": "slide", "timestamp": 2.0}],
            vision_provider=FakeVision(),
            asr=FakeASR(),
            llm=llm,
            tokenizer=FakeTokenizer(),
        )

        self.assertEqual(result.minutes.title, "Auth Migration Sync")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].speaker, "Speaker 1")
        self.assertEqual(result.segments[1].speaker, "Speaker 2")
        self.assertEqual(len(result.visual_materials), 1)
        self.assertIn("Current -> JWT | Proposed -> OAuth 2.0", llm.synthesis_prompt)
        self.assertEqual(result.validation_summary.action_items_valid, 1)
        self.assertEqual(result.validation_summary.decisions_valid, 1)
        self.assertEqual(result.validation_summary.validity_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
