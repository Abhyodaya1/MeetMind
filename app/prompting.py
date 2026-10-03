def create_chunk_prompt(chunk, chunk_number, total_chunks):
    return f"""
You are MeetMind, an AI meeting intelligence system.

You are analyzing part {chunk_number} of {total_chunks}
of a larger meeting transcript.

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
{chunk}
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

1. Do not invent information.
2. Merge duplicate key points.
3. Preserve distinct decisions.
4. Preserve distinct action items.
5. Do not create an owner unless explicitly present.
6. Do not create a deadline unless explicitly present.
7. Evidence must come from the provided analyses.
8. Keep the final summary concise.
9. If there are no decisions, return [].
10. If there are no action items, return [].
11. Return ONLY valid JSON.

Chunk analyses:
----------------
{chunk_analyses}
----------------
"""