import os
import pytest
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


@pytest.fixture
def embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def test_env_key_loaded() -> None:
    assert "OPENAI_API_KEY" in os.environ
    assert os.environ["OPENAI_API_KEY"].startswith("sk-")


def test_embedding_output_shape(
    embedding_model: SentenceTransformer,
) -> None:
    vec = embedding_model.encode(["test string"])
    assert vec.shape[1] == 384
