import json
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
PARAMETERS_PATH = PROJECT_ROOT / "data" / "tmdb_parameters.json"

# Load parameter data
with PARAMETERS_PATH.open("r", encoding="utf-8") as f:
    parameters = json.load(f)

# Initialize embedding model and ChromaDB collection
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection("tmdb_parameters")

# Build embeddings for parameter search
ids = []
documents = []
metadatas = []
# New dictionary to generate param_to_entity_map_generated.json safely
param_to_entity = {}

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
embeddings = model.encode(documents, show_progress_bar=True).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


# --- 🛡️ Safe Optional Step: Save param_to_entity_map_generated.json if mappings exist ---
if param_to_entity:
    output_path = "data/param_to_entity_map_generated.json"
    with open(output_path, "w") as f:
        json.dump(param_to_entity, f, indent=2)

else:
