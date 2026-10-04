def create_chunk_prompt(chunk, chunk_number, total_chunks):
    return f"""
You are MeetMind, an AI meeting intelligence system.

You are analyzing part {chunk_number} of {total_chunks}
of a larger meeting transcript.

This transcript section covers approximately:

START TIME: {chunk.start:.2f} seconds
END TIME: {chunk.end if chunk.end is not None else "unknown"} seconds

Extract ONLY information explicitly supported by this
transcript section.

Return EXACTLY this JSON structure:

{{
    "key_points": [],
    "decisions": [],
    "action_items": [
        {{
            "task": "",
            "owner": null,
            "deadline": null,
            "evidence": ""
        }}
    ]
}}

Rules:

1. Do not invent information.
2. Only include decisions explicitly made.
3. Only include action items explicitly stated or assigned.
4. Owner must be null if not explicitly identified.
5. Deadline must be null if not explicitly identified.
6. Evidence must be directly supported by this transcript section.
7. If there are no items, return empty arrays.
8. Keep key points concise.
9. Return ONLY valid JSON.

Transcript section:
----------------
{chunk.text}
----------------
"""


def create_synthesis_prompt(chunk_analyses):
    return f"""
You are MeetMind, an AI meeting intelligence system.

You are given structured analyses extracted from different
sections of the same meeting.

Combine them into one final meeting analysis.

Return EXACTLY this JSON structure:

{{
    "title": "string",
    "summary": "string",
    "key_points": [],
    "decisions": [],
    "action_items": [
        {{
            "task": "string",
            "owner": null,
            "deadline": null,
            "evidence": ""
        }}
    ]
}}

Rules:

1. The title MUST be concise and descriptive.
2. Never return an empty title.
3. Do not invent information.
4. Merge duplicate key points.
5. Preserve distinct decisions.
6. Preserve distinct action items.
7. Do not create an owner unless explicitly present.
8. Do not create a deadline unless explicitly present.
9. Evidence must come from the provided analyses.
10. Keep the final summary concise.
11. If there are no decisions, return [].
12. If there are no action items, return [].
13. Return ONLY valid JSON.

Chunk analyses:
----------------
{chunk_analyses}
----------------
"""