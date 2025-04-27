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
param_to_entity = {}  # New dictionary to generate param_to_entity_map_generated.json safely

for param in parameters:
    ids.append(param["name"])
    text = f"Parameter: {param['name']}\nType: {param.get('entity_type', 'unknown')}\nImportance: {param.get('importance', 'optional')}\nDescription: {param.get('description', '')}"
    documents.append(text)
    metadatas.append({
        "name": param["name"],
        "in": param["in"],
        "used_in": json.dumps(param.get("used_in", [])),
        "description": param.get("description", ""),
        "type": param.get("entity_type", "unknown"),
        "importance": param.get("importance", "optional")
    })

    # Build param-to-entity map if entity_type exists
    if "entity_type" in param:
        param_to_entity[param["name"]] = param["entity_type"]

# Embed into ChromaDB
print(f"📚 Embedding {len(documents)} parameter descriptions...")
embeddings = model.encode(documents, show_progress_bar=True).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("✅ Finished embedding tmdb_parameters collection.")

# --- 🛡️ Safe Optional Step: Save param_to_entity_map_generated.json if mappings exist ---
if param_to_entity:
    output_path = "data/param_to_entity_map_generated.json"
    with open(output_path, "w") as f:
        json.dump(param_to_entity, f, indent=2)

    print(f"✅ Saved generated param-to-entity mapping to {output_path}")
    print("⚡ WARNING: This does NOT overwrite your live param_to_entity_map.json!")
else:
    print("⚠️ No param-to-entity mappings detected. Skipped writing param_to_entity_map_generated.json.")
