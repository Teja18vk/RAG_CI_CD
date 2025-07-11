from __future__ import annotations

"""Simple Flask web interface for uploading PDFs and chatting."""

import os
import pickle

import faiss
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

from main import extract_text_from_pdf, generate_embeddings

load_dotenv()

app = Flask(__name__)

INDEX_FILE = os.environ.get("INDEX_FILE", "faiss_index.index")
METADATA_FILE = os.environ.get("METADATA_FILE", "metadata.pkl")
CHUNKS_FILE = os.environ.get("CHUNKS_FILE", "all_chunks.pkl")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploaded_pdfs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIM = 384
model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()


    """Load index, metadata and chunks from disk."""
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(MODEL_DIM)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            metadatas = pickle.load(f)
    else:
        metadatas = []
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
    else:
        chunks = []
    return index, metadatas, chunks


def save_store(index: faiss.IndexFlatL2, metadatas: list, chunks: list) -> None:
    """Persist index and data."""
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadatas, f)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)


index, metadatas, all_chunks = load_store()


def chunk_pdf(file_path: str) -> list[dict[str, object]]:
    """Chunk a single PDF."""
    pages = extract_text_from_pdf(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks: list[dict[str, object]] = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for idx, text in enumerate(texts):
            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": page["page_num"],
                        "chunk_index": idx,
                    },
                }
            )
    return chunks


@app.get("/")
def index_page() -> str:

    """Handle PDF upload and update FAISS store."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    chunks = chunk_pdf(file_path)
    embeds, metas = generate_embeddings(chunks)
    global index, metadatas, all_chunks
    index.add(embeds)
    metadatas.extend(metas)
    all_chunks.extend(chunks)
    save_store(index, metadatas, all_chunks)

    return jsonify({"status": "uploaded", "chunks": len(chunks)})


@app.post("/chat")
def chat() -> tuple[dict[str, object], int] | dict[str, str]:
    """Answer questions using uploaded PDFs."""

    if not question:
        return jsonify({"error": "no question"}), 400

    query_vec = model.encode([question]).astype("float32")
    top_k = 5
    distances, indices = index.search(query_vec, top_k)
    context = ""
    for i in indices[0]:
        chunk = all_chunks[i]
        meta = metadatas[i]

    prompt = (
        "You are a helpful assistant. "
        "Use the following context to answer the question:\n\n"
        f"{context}\nQuestion: {question}\nAnswer:"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
        top_p=0.9,
    )
    answer = response.choices[0].message.content
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
