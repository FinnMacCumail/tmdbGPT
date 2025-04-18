import json
from sentence_transformers import SentenceTransformer
import chromadb

# Load parameter data
with open("data/tmdb_parameters.json") as f:
    parameters = json.load(f)

# Initialize embedding model and Chroma collection
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("tmdb_parameters")

# Build embeddings
ids = []
documents = []
metadatas = []

for param in parameters:
    ids.append(param["name"])
    text = f"Parameter: {param['name']}\n{param['description']}"
    documents.append(text)
    metadatas.append({
        "name": param["name"],
        "in": param["in"],
        "used_in": json.dumps(param["used_in"])  # 🔧 Fix: serialize list to string
    })

print(f"📚 Embedding {len(documents)} parameter descriptions...")
embeddings = model.encode(documents, show_progress_bar=True).tolist()

collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

print("✅ Finished embedding parameter collection into ChromaDB.")
