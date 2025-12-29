'''import streamlit as st
import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
DB_PATH = os.path.abspath("vector_db")
COLLECTION_NAME = "ncert_chunks"
TOP_K = 3

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="NCERT AI Tutor",
    page_icon="📘",
    layout="wide"
)

st.markdown(
    """
    <style>
    .title { font-size:40px; font-weight:700; }
    .subtitle { font-size:18px; color:gray; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">📘 NCERT AI Tutor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Offline RAG-based Doubt Solver (Class 6 Maths)</div>', unsafe_allow_html=True)
st.divider()

# ---------------- LOAD MODELS (CACHE) ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
    tokenizer = AutoTokenizer.from_pretrained("./models/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("./models/flan-t5-small")
    qa = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return embedder, qa

embedding_model, qa_pipeline = load_models()

# ---------------- LOAD CHROMA ----------------
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

count = collection.count()
st.info(f"📦 Total chunks loaded: {count}")

if count == 0:
    st.error("Vector DB is empty. Run populate_vector_db.py first.")
    st.stop()

# ---------------- QUERY FUNCTION ----------------
def retrieve_chunks(query):
    emb = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=TOP_K
    )
    return results["documents"][0] if results["documents"] else []

# ---------------- UI INPUT ----------------
question = st.text_input(
    "Ask your NCERT question 👇",
    placeholder="Eg: What is Mathematics?"
)

if st.button("🔍 Get Answer"):

    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking... 🤔"):
            docs = retrieve_chunks(question)

            if not docs:
                st.error("No relevant content found.")
            else:
                context = " ".join(docs[:3])[:1500]
                prompt = f"question: {question} context: {context}"

                answer = qa_pipeline(
                    prompt,
                    max_new_tokens=256
                )[0]["generated_text"]

                st.success("✅ Answer")
                st.write(answer)

                with st.expander("📚 Source Context"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:** {d[:300]}...")


'''

import streamlit as st
import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
DB_PATH = os.path.abspath("vector_db")
COLLECTION_NAME = "ncert_chunks"
DEFAULT_TOP_K = 3

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="NCERT AI Tutor",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
<style>
.title { font-size:40px; font-weight:700; }
.subtitle { font-size:18px; color:gray; }
.card { border:1px solid #ccc; border-radius:10px; padding:10px; margin:5px 0; background:#f9f9f9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📘 NCERT AI Tutor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Offline RAG-based Doubt Solver (Class 6 Maths)</div>', unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")
lang = st.sidebar.selectbox("Language", ["English", "Hindi", "Tamil"])
top_k = st.sidebar.slider("Top K Results", 1, 10, DEFAULT_TOP_K)

# ---------------- CHAT HISTORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("./models/paraphrase-multilingual-MiniLM-L12-v2")
    tokenizer = AutoTokenizer.from_pretrained("./models/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("./models/flan-t5-small")
    qa_pipeline_obj = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return embedder, qa_pipeline_obj

embedding_model, qa_pipeline = load_models()

# ---------------- LOAD TRANSLATION MODELS (optional) ----------------
translation_models = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Tamil": "Helsinki-NLP/opus-mt-en-tam"
}

@st.cache_resource
def load_translation_models_safe():
    models = {}
    for l, model_name in translation_models.items():
        try:
            from transformers import MarianMTModel, MarianTokenizer
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            models[l] = (tokenizer, model)
        except Exception as e:
            st.warning(f"Translation model for {l} could not be loaded. Will use English instead.")
    return models

translation_models_loaded = load_translation_models_safe()

# ---------------- LOAD CHROMA ----------------
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

count = collection.count()
st.info(f"📦 Total chunks loaded: {count}")
if count == 0:
    st.error("Vector DB is empty. Run populate_vector_db.py first.")
    st.stop()

# ---------------- QUERY FUNCTION ----------------
def retrieve_chunks(query):
    emb = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

# ---------------- UI INPUT ----------------
question = st.text_input("Ask your NCERT question 👇", placeholder="Eg: What is Mathematics?")

if st.button("🔍 Get Answer") and question.strip():
    with st.spinner("Thinking... 🤔"):
        docs = retrieve_chunks(question)
        if not docs:
            st.error("No relevant content found.")
        else:
            context = " ".join([d["text"] if isinstance(d, dict) else d for d in docs[:3]])[:1500]
            prompt = f"question: {question} context: {context}"
            answer = qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"]

            # Translate if needed
            if lang != "English" and lang in translation_models_loaded:
                try:
                    tokenizer_t, model_t = translation_models_loaded[lang]
                    translated = model_t.generate(**tokenizer_t(answer, return_tensors="pt", padding=True))
                    answer = tokenizer_t.decode(translated[0], skip_special_tokens=True)
                except Exception as e:
                    st.warning(f"Translation failed. Showing answer in English.")

            # Append to chat history
            st.session_state.chat_history.append({"user": question, "bot": answer, "docs": docs, "lang": lang})

# ---------------- DISPLAY CHAT HISTORY ----------------
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f'<div class="card"><b>Tutor ({chat["lang"]}):</b> {chat["bot"]}</div>', unsafe_allow_html=True)
    
    with st.expander("📚 Source Context"):
        for i, d in enumerate(chat["docs"], 1):
            text = d.get("text", d) if isinstance(d, dict) else d
            page = d.get("page", "N/A") if isinstance(d, dict) else "N/A"
            st.markdown(f"**Chunk {i} (Page {page}):** {text[:300]}...")
    st.divider()

