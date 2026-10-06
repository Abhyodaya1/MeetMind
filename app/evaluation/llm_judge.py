from app.schemas import JudgeResult


class LLMJudge:

    def __init__(self, provider):
        self.provider = provider

    def judge(
        self,
        prediction: str,
        ground_truth: str,
    ) -> JudgeResult:

        prompt = f"""
You are evaluating an AI meeting-minutes system.

Determine whether the predicted item and the ground-truth
item represent the same underlying information.

Prediction:
{prediction}

Ground truth:
{ground_truth}

Return ONLY valid JSON:

{{
    "match": true,
    "score": 0.0,
    "reason": "brief explanation"
}}

Rules:

1. match must be true only when the two items are
   semantically equivalent.

2. Do not require identical wording.

3. Do not treat merely related information as equivalent.

4. Do not infer information that is missing.

5. score must be between 0 and 1.

6. Return only valid JSON.
"""

        result = self.provider.judge(prompt)

        return JudgeResult(**result)