"""Tests for the PDF embedding and environment setup."""

# pylint: disable=import-error

import os
import pytest
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


@pytest.fixture
def model() -> SentenceTransformer:
    """Fixture that returns the embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def test_env_key_loaded() -> None:
    """Test that the OpenAI API key is loaded."""
    assert "OPENAI_API_KEY" in os.environ
    assert os.environ["OPENAI_API_KEY"].startswith("sk-")


def test_embedding_output_shape(model: SentenceTransformer) -> None:
    """Test embedding output shape."""
    vec = model.encode(["test string"])
    assert vec.shape[1] == 384
