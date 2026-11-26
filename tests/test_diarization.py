import unittest

from app.diarization import align_speakers
from app.prompting import format_timestamped_segment
from app.providers.diarization import DiarizationProvider
from app.schemas import SpeakerTurn, TranscriptSegment


class FakeDiarizationProvider(DiarizationProvider):
    def diarize(self, audio_file) -> list[SpeakerTurn]:
        return [
            SpeakerTurn(speaker="Speaker 1", start=0.0, end=10.0),
            SpeakerTurn(speaker="Speaker 2", start=10.0, end=20.0),
        ]


class TestDiarization(unittest.TestCase):
    def test_align_speakers_picks_max_overlap(self):
        turns = FakeDiarizationProvider().diarize("dummy.wav")
        segments = [
            TranscriptSegment(
                text="We should move authentication to OAuth.",
                start=2.0,
                end=9.0,
            ),
            TranscriptSegment(
                text="Agreed, I will handle the migration.",
                start=9.5,
                end=15.0,
            ),
            TranscriptSegment(
                text="Final remark.",
                start=18.0,
                end=None,
            ),
            TranscriptSegment(
                text="Outside turns.",
                start=25.0,
                end=30.0,
            ),
        ]

        aligned = align_speakers(segments, turns)
        self.assertEqual(aligned[0].speaker, "Speaker 1")
        self.assertEqual(aligned[1].speaker, "Speaker 2")
        self.assertEqual(aligned[2].speaker, "Speaker 2")
        self.assertIsNone(aligned[3].speaker)

    def test_format_timestamped_segment_with_and_without_speaker(self):
        seg_with = TranscriptSegment(
            text="I'll handle it.",
            start=10.32,
            end=12.5,
            speaker="Speaker 2",
        )
        seg_without = TranscriptSegment(
            text="Hello.",
            start=0.0,
            end=2.0,
        )
        self.assertEqual(
            format_timestamped_segment(seg_with),
            "[Speaker 2 | 10.32s → 12.5s]\nI'll handle it.",
        )
        self.assertEqual(
            format_timestamped_segment(seg_without),
            "[0.00s → 2.0s]\nHello.",
        )


if __name__ == "__main__":
    unittest.main()
