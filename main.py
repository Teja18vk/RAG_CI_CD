import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, cast


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    extracted_text: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                cleaned_text = " ".join(text.split())
                extracted_text.append(
                    {
                        "page_num": page_num + 1,
                        "text": cleaned_text,
                    }
                )
    return extracted_text


# PDF folder
pdf_folder = "pdfs"
all_chunks: List[Dict[str, Any]] = []

# Text splitter config
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Loop and chunk
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        pages = extract_text_from_pdf(path)

        for page in pages:
            chunks = splitter.split_text(page["text"])
            for idx, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "source": filename,
                            "page": page["page_num"],
                            "chunk_index": idx,
                        },
                    }
                )

# Print preview
for chunk_dict in all_chunks[:3]:
    meta_dict = cast(Dict[str, Any], chunk_dict.get("metadata", {}))
    content = str(chunk_dict.get("content", ""))

    print("\n--- Chunk ---")
    print(
        f"Source: {meta_dict.get('source', 'N/A')}, "
        f"Page: {meta_dict.get('page', 'N/A')}, "
        f"Chunk #: {meta_dict.get('chunk_index', 'N/A')}"
    )
    print(f"Content: {content[:300]}...")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract just the content and metadata
texts: List[str] = []
metadatas: List[Dict[str, Any]] = []

for entry in all_chunks:
    if isinstance(entry, dict):
        content = str(entry.get("content", ""))
        meta_info = cast(Dict[str, Any], entry.get("metadata", {}))

        texts.append(content)
        metadatas.append(meta_info)


# Create embeddings
print("🔄 Generating embeddings...")
embeddings = model.encode(texts)
embeddings_np = np.array(embeddings).astype("float32")

# Create FAISS index
dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_np)

# Save index and metadata
faiss.write_index(index, "faiss_index.index")

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

with open("all_chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Embeddings stored in FAISS index!")
