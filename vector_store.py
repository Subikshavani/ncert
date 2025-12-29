import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
CHUNKS_ROOT = "extracted_text"
DB_PATH = "vector_db"

# Load multilingual embedding model
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Create / load ChromaDB (auto-persistent)
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=DB_PATH,
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(
    name="ncert_chunks"
)


def load_chunks_into_db():
    doc_id = 0

    for root, dirs, files in os.walk(CHUNKS_ROOT):
        for file in files:
            if not file.endswith("_chunks.json"):
                continue

            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for chunk in data["chunks"]:
                embedding = model.encode(chunk["text"]).tolist()

                metadata = {
                    "class": data["class"],
                    "subject": data["subject"],
                    "language": data["language"],
                    "source_pdf": data["source_pdf"],
                    "page": chunk["page"]
                }

                collection.add(
                    documents=[chunk["text"]],
                    metadatas=[metadata],
                    ids=[f"chunk_{doc_id}"],
                    embeddings=[embedding]
                )

                doc_id += 1

    print(f"✅ Stored {doc_id} chunks in Vector DB")


if __name__ == "__main__":
    load_chunks_into_db()
