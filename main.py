"""Process PDFs, split text into chunks, embed,store them in a FAISS index."""

import os
import pickle
from typing import List, Dict, Any

import pdfplumber
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract cleaned text from each page of a PDF file."""
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg_num, pg in enumerate(pdf.pages):
            text = pg.extract_text()
            if text:
                cleaned_text = " ".join(text.split())
                extracted_text.append(
                    {
                        "page_num": pg_num + 1,
                        "text": cleaned_text,
                    }
                )
    return extracted_text


def load_and_chunk_pdfs(pdf_folder_path: str) -> List[Dict[str, Any]]:
    """Load PDF files and chunk them into text blocks."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder_path, filename)
            pages = extract_text_from_pdf(path)

            for page_data in pages:
                chunks = splitter.split_text(page_data["text"])
                for idx, chunk in enumerate(chunks):
                    all_chunks.append(
                        {
                            "content": chunk,
                            "metadata": {
                                "source": filename,
                                "page": page_data["page_num"],
                                "chunk_index": idx,
                            },
                        }
                    )
    return all_chunks


def preview_chunks(all_chunks: List[Dict[str, Any]]) -> None:
    """Print a preview of the first few chunks."""
    for chunk_dict in all_chunks[:3]:
        meta = chunk_dict["metadata"]
        content = str(chunk_dict["content"])
        print("\n--- Chunk ---")
        print(
            f"Source: {meta['source']}, Page: {meta['page']}, "
            f"Chunk #: {meta['chunk_index']}"
        )
        print(f"Content: {content[:300]}...")


def generate_embeddings(
    all_chunks: List[Dict[str, Any]],
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate embeddings and return vectors + metadata."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [str(chunk["content"]) for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    print("🔄 Generating embeddings...")
    embeddings = model.encode(texts)
    embeddings_np = np.array(embeddings).astype("float32")
    return embeddings_np, metadatas


def main() -> None:
    """Main function."""
    pdf_folder_path = "pdfs"
    all_chunks = load_and_chunk_pdfs(pdf_folder_path)
    preview_chunks(all_chunks)

    embeddings_np, metadatas = generate_embeddings(all_chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)  # pylint: disable=no-value-for-parameter
    faiss.write_index(index, "faiss_index.index")

    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadatas, f)

    with open("all_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Embeddings stored in FAISS index!")


if __name__ == "__main__":
    main()
