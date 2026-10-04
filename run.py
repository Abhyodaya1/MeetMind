from pathlib import Path

from app.meeting import process_meeting


audio_file = (
    Path(__file__).parent/ "data"
    / "audio"
    / "ES2002a.Mix-Headset.wav"
)

minutes = process_meeting(audio_file)


print("\n--- MEETING MINUTES ---")

print("\nTitle:")
print(minutes.title)

print("\nSummary:")
print(minutes.summary)

print("\nKey Points:")

for point in minutes.key_points:
    print("-", point)

print("\nDecisions:")

for decision in minutes.decisions:
    print("-", decision)

print("\nAction Items:")

for action in minutes.action_items:

    print("-", action.task)

    print(
        "  Owner:",
        action.owner
    )

    print(
        "  Deadline:",
        action.deadline
    )

    print(
        "  Evidence:",
        action.evidence
    )