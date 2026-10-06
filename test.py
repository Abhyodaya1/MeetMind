from app.evaluation.evaluator import (
    evaluate_semantic_category,
)
from app.evaluation.semantic_matcher import (
    SemanticMatcher,
)
from app.schemas import GroundTruthItem


ground_truth = [
    GroundTruthItem(
        text="David will create the remote control design."
    ),
    GroundTruthItem(
        text="Prepare the project presentation."
    ),
]

predictions = [
    "Develop the working design of the remote control.",
    "Create the project presentation.",
]

matcher = SemanticMatcher()

result = evaluate_semantic_category(
    predictions,
    ground_truth,
    matcher,
    threshold=0.75,
)

print(result)   