def precision(
    true_positives: int,
    false_positives: int,
) -> float:

    total = true_positives + false_positives

    if total == 0:
        return 0.0

    return true_positives / total


def recall(
    true_positives: int,
    false_negatives: int,
) -> float:

    total = true_positives + false_negatives

    if total == 0:
        return 0.0

    return true_positives / total


def f1_score(
    precision_value: float,
    recall_value: float,
) -> float:

    total = precision_value + recall_value

    if total == 0:
        return 0.0

    return (
        2
        * precision_value
        * recall_value
        / total
    )