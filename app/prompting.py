def create_chunk_prompt(chunk, chunk_number, total_chunks):

    timestamped_transcript = "\n".join(
        (
            f"[{segment.start:.2f}s → "
            f"{segment.end if segment.end is not None else 'unknown'}s]\n"
            f"{segment.text}"
        )
        for segment in chunk.segments
    )

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

    "decisions": [
        {{
            "description": "",
            "evidence": {{
                "text": "",
                "start": 0.0,
                "end": null
            }}
        }}
    ],

    "action_items": [
        {{
            "task": "",
            "owner": null,
            "deadline": null,
            "evidence": {{
                "text": "",
                "start": 0.0,
                "end": null
            }}
        }}
    ]
}}

Rules:

1. Do not invent information.
2. Only include decisions explicitly made in the transcript.
3. Only include action items explicitly stated or assigned.
4. Owner must be null if not explicitly identified.
5. Deadline must be null if not explicitly identified.
6. Evidence must be directly supported by the transcript.
7. If there are no key points, return [].
8. If there are no decisions, return [].
9. If there are no action items, return [].
10. Keep key points concise.
11. Return ONLY valid JSON.

Decision rules:

12. A decision must represent something the participants
    explicitly decided, agreed upon, selected, approved,
    or committed to.
13. Do not convert a suggestion, question, opinion, or
    possibility into a decision.
14. Every decision should include supporting evidence when
    the transcript provides it.

Action item rules:

15. An action item must represent work that someone is
    explicitly expected to perform.
16. Do not create an action item from a general discussion.
17. Do not infer an owner.
18. Do not infer a deadline.

Evidence rules:

19. Evidence.text must come directly from the transcript.
20. Evidence.start must correspond to a real transcript
    segment start timestamp.
21. Evidence.end must correspond to a real transcript
    segment end timestamp.
22. Never invent timestamps.
23. If evidence spans multiple adjacent transcript
    segments, use the earliest start and latest end.
24. Use the smallest timestamp range that fully supports
    the decision or action item.
25. Do not use the entire transcript section when a
    smaller range is sufficient.
26. Different decisions or action items should use
    different evidence ranges when separate evidence
    exists.
27. Evidence may be null only when there is genuinely
    no reliable supporting evidence.

The transcript below contains timestamped segments.
These timestamps are the source of truth.

Transcript section:
----------------

{timestamped_transcript}

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

    "decisions": [
        {{
            "description": "",
            "evidence": {{
                "text": "",
                "start": 0.0,
                "end": null
            }}
        }}
    ],

    "action_items": [
        {{
            "task": "string",
            "owner": null,
            "deadline": null,
            "evidence": {{
                "text": "",
                "start": 0.0,
                "end": null
            }}
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
9. Evidence must come from the provided chunk analyses.
10. Keep the final summary concise.
11. If there are no key points, return [].
12. If there are no decisions, return [].
13. If there are no action items, return [].
14. Return ONLY valid JSON.

Decision rules:

15. Preserve the description of each supported decision.
16. Do not create new decisions during synthesis.
17. Merge duplicate decisions when multiple chunks describe
    the same decision.
18. Preserve evidence from the chunk analysis.
19. Never invent or modify evidence timestamps.

Action item rules:

20. Preserve distinct action items.
21. Merge duplicate action items when multiple chunks refer
    to the same task.
22. Preserve the most directly supporting evidence when
    merging duplicate action items.
23. Do not create owners or deadlines that are not explicitly
    supported by the chunk analyses.

Evidence rules:

24. Evidence.text must come from the provided analyses.
25. Evidence timestamps must be preserved exactly.
26. Never invent, modify, or approximate evidence timestamps.
27. Evidence may be null when no supporting evidence exists.

Chunk analyses:
----------------

{chunk_analyses}

----------------
"""