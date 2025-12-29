from sentence_transformers import SentenceTransformer

# Download the multilingual embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Save it locally
embedding_model.save("./models/paraphrase-multilingual-MiniLM-L12-v2")
print("Embedding model saved locally!")
