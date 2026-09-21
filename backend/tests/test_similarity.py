import pytest

from voicelm.retrieval.similarity import cosine_similarity


def test_identical_vectors_score_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_opposite_vectors_score_minus_one() -> None:
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_perpendicular_vectors_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_length_is_ignored() -> None:
    # The defining property: only direction matters, so scaling a vector changes nothing.
    # This is why a long chunk and a short chunk on the same topic still match.
    assert cosine_similarity([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_known_value() -> None:
    # 45 degrees between (1,0) and (1,1): cos(45 degrees) = 1/sqrt(2).
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(0.7071067, abs=1e-6)


def test_partial_similarity_falls_between() -> None:
    score = cosine_similarity([1.0, 1.0, 0.0], [1.0, 0.0, 0.0])

    assert 0.0 < score < 1.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        cosine_similarity([], [])


def test_zero_vector_is_rejected() -> None:
    # The angle to a zero-length vector is undefined, so there is no honest answer.
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])
