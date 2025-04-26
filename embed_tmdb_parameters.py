import json
from sentence_transformers import SentenceTransformer
import chromadb

# Load parameter data
with open("data/tmdb_parameters.json") as f:
    parameters = json.load(f)

# Initialize embedding model and ChromaDB collection
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("tmdb_parameters")

# Build embeddings for parameter search
ids = []
documents = []
metadatas = []
param_to_entity = {}  # New dictionary to generate param_to_entity_map.json

for param in parameters:
    ids.append(param["name"])
    text = f"Parameter: {param['name']}\n{param['description']}"
    documents.append(text)
    metadatas.append({
        "name": param["name"],
        "in": param["in"],
        "used_in": json.dumps(param.get("used_in", []))
    })

    # Build param to entity map if entity_type is available
    if "entity_type" in param:
        param_to_entity[param["name"]] = param["entity_type"]

# 🛠 Patch in special cases manually
param_to_entity["credit_id"] = "credit"
param_to_entity["review_id"] = "review"

# Save param_to_entity_map.json
with open("data/param_to_entity_map.json", "w") as f:
    json.dump(param_to_entity, f, indent=2)

# Embed into ChromaDB
print(f"📚 Embedding {len(documents)} parameter descriptions...")
embeddings = model.encode(documents, show_progress_bar=True).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("✅ Finished embedding tmdb_parameters collection and generating param_to_entity_map.json.")

