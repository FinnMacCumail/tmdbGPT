import os
import json
import chromadb

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Connect to ChromaDB and retrieve endpoint metadata
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("tmdb_endpoints")
entries = collection.get(include=["metadatas"])

# Prepare log content
log_lines = ["# TMDB Endpoint Metadata Log\n"]

for meta in entries["metadatas"]:
    path = meta.get("path", "UNKNOWN")
    intents = json.loads(meta.get("intents", "[]"))
    consumes = json.loads(meta.get("consumes_entities", "[]"))
    produces = json.loads(meta.get("produces_entities", "[]"))
    params = json.loads(meta.get("param_names", "[]"))
    entities = meta.get("entities", "")

    log_lines.append(f"## {path}")
    log_lines.append(f"- Intents: {[i['intent'] for i in intents]}")
    log_lines.append(f"- Entities: {entities}")
    log_lines.append(f"- Param Names: {params}")
    log_lines.append(f"- Consumes: {consumes}")
    log_lines.append(f"- Produces: {produces}")
    log_lines.append("")

# Write to file
log_path = "logs/tmdb_endpoints_metadata_log.md"
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))

print(f"✅ Log written to {log_path}")
