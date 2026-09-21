"""Measuring how close two embeddings are."""

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine of the angle between two vectors, from -1.0 to 1.0.

    Cosine similarity compares *direction* and ignores *length*. That is what we want:
    a 900-character chunk and a 200-character chunk about the same topic point the same
    way but have different magnitudes, and we care about the topic. A consequence worth
    knowing is that `cosine_similarity(v, k * v) == 1.0` for any positive k.

    1.0 means the same direction, 0.0 means unrelated (perpendicular), -1.0 means
    opposite.

    Production vector databases scale every vector to length 1 on the way in, which makes
    the two divisions below unnecessary and turns this into a plain dot product. We keep
    the explicit form here because it is the definition; Qdrant will do the fast version.
    """
    if len(left) != len(right):
        raise ValueError(
            f"cannot compare vectors of different lengths: {len(left)} and {len(right)}"
        )
    if not left:
        raise ValueError("cannot compare empty vectors")

    dot_product = sum(x * y for x, y in zip(left, right, strict=True))
    left_length = math.sqrt(sum(x * x for x in left))
    right_length = math.sqrt(sum(y * y for y in right))

    if left_length == 0.0 or right_length == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    return dot_product / (left_length * right_length)
