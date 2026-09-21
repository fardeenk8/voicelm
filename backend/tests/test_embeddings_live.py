"""Tests that talk to a real Ollama instance.

Skipped automatically when Ollama is not running, so the suite still passes on a machine
without it. The offline tests in test_ollama_embedder.py cover the client's logic; these
check the thing no mock can: that the embeddings actually carry meaning.
"""

import httpx2
import pytest

from voicelm.embeddings.ollama import DEFAULT_BASE_URL, OllamaEmbedder
from voicelm.retrieval.similarity import cosine_similarity

NOMIC_EMBED_DIMENSIONS = 768


def ollama_is_running() -> bool:
    try:
        return httpx2.get(f"{DEFAULT_BASE_URL}/api/version", timeout=2.0).status_code == 200
    except httpx2.RequestError:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_is_running(), reason="Ollama is not running on 127.0.0.1:11434"
)


@pytest.fixture(scope="module")
def embedder() -> OllamaEmbedder:
    return OllamaEmbedder()


def test_real_embeddings_have_the_expected_shape(embedder: OllamaEmbedder) -> None:
    [vector] = embedder.embed_texts(["A sentence about databases."])

    assert len(vector) == NOMIC_EMBED_DIMENSIONS
    assert all(isinstance(value, float) for value in vector)


def test_the_same_text_embeds_identically(embedder: OllamaEmbedder) -> None:
    first, second = embedder.embed_texts(["stable input", "stable input"])

    assert first == second


def test_related_text_scores_higher_than_unrelated_text(embedder: OllamaEmbedder) -> None:
    """The whole premise of semantic search, checked end to end.

    Note that the related pair shares no words with the query, and the unrelated pair
    shares one ("the"). Keyword matching would get this backwards; embeddings should not.
    """
    query, related, unrelated = embedder.embed_texts(
        [
            "How do I care for a young cat?",
            "Kittens need feeding several times a day.",
            "The mortgage interest rate rose by half a point.",
        ]
    )

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_paraphrases_score_close_to_each_other(embedder: OllamaEmbedder) -> None:
    original, paraphrase, unrelated = embedder.embed_texts(
        [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "Compile the kernel with optimisations enabled.",
        ]
    )

    assert cosine_similarity(original, paraphrase) > cosine_similarity(original, unrelated)
