"""Interactive chatbot that answers questions."""

# pylint: disable=invalid-name

import pickle
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI()

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load metadata and chunks
with open("metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

with open("all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    # User input
    query = input("Ask me something based on the PDF content: ")

    # Embed the query
    query_vec = model.encode([query]).astype("float32")

    # Perform FAISS search
    top_k = 5
    distances, indices = index.search(query_vec, top_k)

    # Collect context from top results
    context = ""
    for i in indices[0]:
        chunk = all_chunks[i]
        meta = metadatas[i]
        source_info = f"\n(Source: {meta['source']}, Page {meta['page']})\n"
        chunk_text = f"{chunk['content']}\n"
        context += source_info + chunk_text

    # Define prompt
    prompt = (
        "You are a helpful assistant. "
        "Use the following context to answer the question:\n\n"
        f"{context}\n"
        f"Question: {query}\nAnswer:"
    )

    # Get answer from OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
        top_p=0.9,
    )

    # Show answer
    print("\n🧠 Answer:\n", response.choices[0].message.content)
