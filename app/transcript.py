from app.schemas import TranscriptSegment

def normalize_transcript(result):
    segments = []

    for chunk in result.get("chunks", []):
        text = chunk.get("text", "").strip()
        timestamp = chunk.get("timestamp")

        if not text or not timestamp:
            continue

        if len(timestamp) != 2:
            continue

        start, end = timestamp
        segment = TranscriptSegment(text=text, start=start, end=end)
        segments.append(segment)

    return segments