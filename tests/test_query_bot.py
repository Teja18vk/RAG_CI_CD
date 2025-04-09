import os
from dotenv import load_dotenv
import pytest
from sentence_transformers import SentenceTransformer


# Explicitly set the path to .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

@pytest.fixture
def embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def test_env_key_loaded():
    assert "OPENAI_API_KEY" in os.environ
    assert os.environ["OPENAI_API_KEY"].startswith("sk-")
