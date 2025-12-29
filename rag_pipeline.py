'''import os
from langdetect import detect
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings

# ---------- Paths ----------
DB_PATH = "vector_db"

# ---------- Load models ----------
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# ---------- Load / Initialize Vector DB ----------
client = chromadb.Client(Settings(
    persist_directory=DB_PATH,
    anonymized_telemetry=False
))

# Create collection if it doesn't exist
if "ncert_chunks" in [c.name for c in client.list_collections()]:
    collection = client.get_collection("ncert_chunks")
else:
    collection = client.create_collection("ncert_chunks")

# ---------- Helper Functions ----------

def detect_language(text):
    """Detect language of the question."""
    try:
        return detect(text)
    except:
        return "en"

def retrieve_chunks(question, class_level=None, subject=None, language=None, top_k=5, similarity_threshold=0.8):
    """Retrieve top-k relevant chunks from vector DB using embeddings and metadata filters."""
    q_embedding = embedding_model.encode(question).tolist()
    
    # Build proper metadata filter using $and
    filters = []
    if class_level:
        filters.append({"class": {"$eq": class_level}})
    if subject:
        filters.append({"subject": {"$eq": subject}})
    if language:
        filters.append({"language": {"$eq": language}})
    
    metadata_filter = {"$and": filters} if filters else None
    
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        where=metadata_filter
    )
    
    chunks = []
    for doc, meta, score in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [1]*len(results['documents'][0]))):
        if score <= similarity_threshold:  # lower distance = more similar
            chunks.append((doc, meta))
    
    return chunks

def generate_answer(question, chunks):
    """Generate answer using retrieved chunks, including citations."""
    if not chunks:
        return "Sorry, this is not covered in the NCERT textbooks."
    
    context = "\n\n".join([f"Page {c[1]['page']} from {c[1]['source_pdf']}: {c[0]}" for c in chunks])
    prompt = f"Answer the question based on the context below. Cite the page number and source PDF.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    answer = qa_pipeline(prompt, max_length=300, do_sample=False)[0]['generated_text']
    return answer

def rag_qa(question, class_level=None, subject=None):
    """Main RAG function."""
    language = detect_language(question)
    
    chunks = retrieve_chunks(question, class_level=class_level, subject=subject, language=language)
    answer = generate_answer(question, chunks)
    
    return answer

# ---------- Example Usage ----------
if __name__ == "__main__":
    question = "What is Mathematics?"
    class_level = "6"
    subject = "science"
    
    answer = rag_qa(question, class_level, subject)
    print("\nQuestion:", question)
    print("\nAnswer:", answer)


'''
'''import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import chromadb
#from chromadb.config import Settings

# ---------------- CONFIG ----------------
DB_PATH = os.path.abspath("vector_db")
COLLECTION_NAME = "ncert_chunks"
TOP_K = 3

# ---------------- MODELS ----------------
embedding_model = SentenceTransformer(
    "./models/paraphrase-multilingual-MiniLM-L12-v2"
)

tokenizer = AutoTokenizer.from_pretrained("./models/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/flan-t5-small")

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

# ---------------- CHROMA ----------------
client = chromadb.PersistentClient(path=DB_PATH)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

count = collection.count()
print(f"⚡ Debug: Number of chunks in collection = {count}")

if count == 0:
    print("❌ Vector DB is empty. Run populate_vector_db.py first.")
    exit()

# ---------------- RETRIEVAL ----------------
def retrieve_chunks(query):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    if results["documents"]:
        return results["documents"][0]
    return []

# ---------------- MAIN LOOP ----------------
print("\n📘 Multilingual NCERT Doubt-Solver RAG Pipeline")
print("Type 'exit' to quit")

while True:
    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit"]:
        break

    docs = retrieve_chunks(question)

    if not docs:
        print("⚠️ No relevant content found.")
        continue

    context = " ".join(docs)
    prompt = f"question: {question} context: {context}"

    answer = qa_pipeline(prompt, max_length=200)[0]["generated_text"]
    print("\n✅ Answer:", answer)
'''
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import chromadb

# ---------------- CONFIG ----------------
DB_PATH = os.path.abspath("vector_db")
COLLECTION_NAME = "ncert_chunks"
TOP_K = 3
MAX_CONTEXT_CHARS = 1200   # 🔑 context cut to avoid 512 token error

# ---------------- EMBEDDING MODEL ----------------
embedding_model = SentenceTransformer(
    "./models/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------------- QA MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained("./models/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/flan-t5-small")

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1   # CPU
)

# ---------------- CHROMA DB ----------------
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

count = collection.count()
print(f"⚡ Debug: Number of chunks in collection = {count}")

if count == 0:
    print("❌ Vector DB is empty. Run populate_vector_db.py first.")
    exit()

# ---------------- RETRIEVAL ----------------
def retrieve_chunks(query):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    if results and results.get("documents"):
        return results["documents"][0]
    return []

# ---------------- MAIN LOOP ----------------
print("\n📘 Multilingual NCERT Doubt-Solver RAG Pipeline")
print("Type 'exit' to quit")

while True:
    question = input("\nAsk a question: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    docs = retrieve_chunks(question)

    if not docs:
        print("⚠️ No relevant content found.")
        continue

    # 🔑 join + trim context safely
    context = " ".join(docs)
    context = context[:MAX_CONTEXT_CHARS]

    prompt = (
        f"Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )

    result = qa_pipeline(
        prompt,
        max_new_tokens=200,   # ✅ only this (no conflict)
        do_sample=False
    )

    answer = result[0]["generated_text"]
    print("\n✅ Answer:", answer)
