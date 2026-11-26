import unittest

from app.schemas import Evidence, TranscriptChunk, TranscriptSegment
from app.validation import validate_evidence


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            TranscriptChunk(
                text="We should move authentication to OAuth 2.0.",
                start=10.0,
                end=15.0,
                token_count=9,
                segments=[
                    TranscriptSegment(
                        text="We should move authentication to OAuth 2.0.",
                        start=10.0,
                        end=15.0,
                    )
                ],
            )
        ]

    def test_valid_evidence(self):
        ev = Evidence(
            text="move authentication to oauth 2.0",
            start=10.0,
            end=15.0,
        )
        valid, reason = validate_evidence(ev, self.chunks)
        self.assertTrue(valid)
        self.assertEqual(reason, "valid")

    def test_missing_evidence(self):
        valid, reason = validate_evidence(None, self.chunks)
        self.assertFalse(valid)
        self.assertEqual(reason, "missing_evidence")

    def test_timestamp_not_found(self):
        ev = Evidence(
            text="We should move authentication to OAuth 2.0.",
            start=99.0,
            end=100.0,
        )
        valid, reason = validate_evidence(ev, self.chunks)
        self.assertFalse(valid)
        self.assertEqual(reason, "start_timestamp_not_found")

    def test_evidence_text_not_found(self):
        ev = Evidence(
            text="something never said",
            start=10.0,
            end=15.0,
        )
        valid, reason = validate_evidence(ev, self.chunks)
        self.assertFalse(valid)
        self.assertEqual(reason, "evidence_text_not_found")


if __name__ == "__main__":
    unittest.main()
