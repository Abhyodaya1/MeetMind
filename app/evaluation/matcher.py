from app.schemas import GroundTruthItem


def normalize_text(text: str) -> str:
    return " ".join(
        text.lower().split()
    )


def exact_match(
    prediction: str,
    ground_truth: str,
) -> bool:

    return (
        normalize_text(prediction)
        == normalize_text(ground_truth)
    )


def evaluate_items(
    predictions: list[str],
    ground_truth: list[GroundTruthItem],
) -> dict:

    matched_ground_truth = set()

    true_positives = 0
    false_positives = 0

    for prediction in predictions:

        matched = False

        for index, expected in enumerate(
            ground_truth
        ):

            if index in matched_ground_truth:
                continue

            if exact_match(
                prediction,
                expected.text,
            ):
                matched_ground_truth.add(index)
                true_positives += 1
                matched = True
                break

        if not matched:
            false_positives += 1

    false_negatives = (
        len(ground_truth)
        - true_positives
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

def semantic_evaluate_items(
    predictions: list[str],
    ground_truth: list[GroundTruthItem],
    matcher,
    threshold: float = 0.75,
) -> dict:

    matched_ground_truth = set()

    true_positives = 0
    false_positives = 0

    for prediction in predictions:

        best_match_index = None
        best_score = 0.0

        for index, expected in enumerate(
            ground_truth
        ):

            if index in matched_ground_truth:
                continue

            score = matcher.similarity(
                prediction,
                expected.text,
            )

            if score > best_score:
                best_score = score
                best_match_index = index

        if (
            best_match_index is not None
            and best_score >= threshold
        ):
            matched_ground_truth.add(
                best_match_index
            )
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = (
        len(ground_truth)
        - true_positives
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }