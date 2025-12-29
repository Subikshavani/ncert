'''import os
import json
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---------- Paths ----------
CHUNKS_ROOT = "extracted_text"
DB_PATH = "vector_db"

# ---------- Load embedding model ----------
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------- Load / Initialize ChromaDB ----------
client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    anonymized_telemetry=False
))

# Create collection if it doesn't exist
if "ncert_chunks" in [c.name for c in client.list_collections()]:
    collection = client.get_collection("ncert_chunks")
else:
    collection = client.create_collection("ncert_chunks")

# ---------- Recursive function to find all _chunks.json files ----------
def find_chunk_files(root_dir):
    chunk_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith("_chunks.json"):
                chunk_files.append(os.path.join(dirpath, file))
    return chunk_files

# ---------- Insert Chunks ----------
def insert_chunks():
    chunk_files = find_chunk_files(CHUNKS_ROOT)
    if not chunk_files:
        print("No _chunks.json files found. Check your extracted_text folder!")
        return

    count = 0
    for file_path in chunk_files:
        print(f"Processing file: {file_path}")  # Debug print
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        # Determine if data is a list or a dict with "chunks"
        if isinstance(data, list):
            chunks_list = data
        elif isinstance(data, dict) and "chunks" in data:
            chunks_list = data["chunks"]
        else:
            print(f"Skipping {file_path}, unexpected format: {type(data)}")
            continue

        for chunk in chunks_list:
            text = chunk.get("text")
            if not text:
                continue

            embedding = embedding_model.encode(text).tolist()

            # Metadata: fallback for None values
            metadata = {
                "class": chunk.get("class") or "unknown_class",
                "subject": chunk.get("subject") or "unknown_subject",
                "language": chunk.get("language") or "unknown_language",
                "page": chunk.get("page") or 0,
                "source_pdf": chunk.get("source_pdf") or "unknown_pdf"
            }

            # Unique ID for ChromaDB
            chunk_id = str(uuid.uuid4())

            # Insert into ChromaDB
            collection.add(
                ids=[chunk_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            count += 1

    print(f"Inserted {count} chunks into ChromaDB.")

# ---------- Run ----------
if __name__ == "__main__":
    insert_chunks()




'''
import os
import json
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
#from chromadb.config import Settings

# ---------------- CONFIG ----------------
CHUNKS_ROOT = "extracted_text"
DB_PATH = os.path.abspath("vector_db")
COLLECTION_NAME = "ncert_chunks"

# ---------------- MODEL ----------------
embedding_model = SentenceTransformer(
    "./models/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------------- CHROMA ----------------
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ---------------- LOAD CHUNKS ----------------
texts = []
metadatas = []
ids = []

for root, _, files in os.walk(CHUNKS_ROOT):
    for file in files:
        if not file.endswith("_chunks.json"):
            continue

        path = os.path.join(root, file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # YOUR FORMAT: { ..., "chunks": [ { "text": ... } ] }
            for chunk in data.get("chunks", []):
                text = chunk.get("text")
                if not text:
                    continue

                texts.append(text)
                metadatas.append({
                    "source": path,
                    "page": chunk.get("page", -1)
                })
                ids.append(str(uuid.uuid4()))

        except Exception as e:
            print(f"❌ Error reading {path}: {e}")

# ---------------- SAFETY CHECK ----------------
if not texts:
    print("❌ No valid chunks found. Check chunking.py output.")
    exit()

# ---------------- EMBEDDINGS ----------------
print(f"🔢 Creating embeddings for {len(texts)} chunks...")
embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

# ---------------- INSERT ----------------
collection.add(
    documents=texts,
    metadatas=metadatas,
    ids=ids,
    embeddings=embeddings.tolist()
)

print(f"✅ Inserted {len(texts)} chunks into ChromaDB successfully.")
print("🔍 VERIFY COUNT:", collection.count())
