import os
from dotenv import load_dotenv
import pytest
from sentence_transformers import SentenceTransformer


# Load environment variables for testing
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


@pytest.fixture
def embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def test_env_key_loaded():
    assert "OPENAI_API_KEY" in os.environ
    assert os.environ["OPENAI_API_KEY"].startswith("sk-")


def test_embedding_output_shape(embedding_model):
    vec = embedding_model.encode(["test string"])
    assert vec.shape[1] == 384
