import json
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
import re

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")


def load_tmdb_schema(filepath="./data/tmdb.json"):
    """Load TMDB API schema from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_api_metadata(openapi_spec):
    """Extract API metadata, including endpoints, descriptions, and parameters."""
    metadata = []
    for path, methods in openapi_spec.get("paths", {}).items():
        for method, details in methods.items():
            parameters = details.get("parameters", [])

            # Identify path parameters automatically
            path_params = re.findall(r"\{(\w+)\}", path)
            existing_param_names = {param["name"] for param in parameters}

            # Add path parameters explicitly if missing
            for path_param in path_params:
                if path_param not in existing_param_names:
                    parameters.append({
                        "name": path_param,
                        "in": "path",
                        "schema": {"type": "integer"},
                        "description": f"The ID of the {path_param.replace('_', ' ')}.",
                        "required": True
                    })

            metadata.append({
                "endpoint": path,
                "method": method.upper(),
                "description": details.get("summary", ""),
                "parameters": parameters
            })
    return metadata


def create_embeddings(api_metadata):
    """Generate embeddings for API descriptions using Hugging Face model."""
    descriptions = [entry["description"] for entry in api_metadata]
    return model.encode(descriptions).tolist()


def perform_clustering(embeddings, num_clusters):
    """Apply K-means clustering on embeddings."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.array(embeddings))
    return labels, kmeans.cluster_centers_


def store_clusters(api_metadata, cluster_labels):
    """Store clustered API functions in ChromaDB with valid metadata types."""
    for i, entry in enumerate(api_metadata):
        entry["cluster"] = cluster_labels[i]

        # Ensure all metadata values are valid types for ChromaDB
        valid_metadata = {
            "endpoint": entry["endpoint"],
            "method": entry["method"],
            "description": entry["description"],
            "parameters": json.dumps(entry["parameters"]) if entry["parameters"] else "[]"  # Ensure valid JSON array
        }

        collection.add(
            ids=[str(i)],
            embeddings=[model.encode(entry["description"]).tolist()],
            metadatas=[valid_metadata],  # Use validated metadata
            documents=[entry["description"]]
        )


def main():
    print("🔄 Loading TMDB API Specification...")
    openapi_spec = load_tmdb_schema()

    print("🔍 Extracting API Metadata...")
    api_metadata = extract_api_metadata(openapi_spec)

    print("🔢 Creating Embeddings...")
    embeddings = create_embeddings(api_metadata)

    num_clusters = max(1, len(api_metadata) // 5)
    print(f"📊 Performing Clustering into {num_clusters} clusters...")
    cluster_labels, _ = perform_clustering(embeddings, num_clusters)

    print("🗄️ Storing Clusters in ChromaDB...")
    store_clusters(api_metadata, cluster_labels)

    print("✅ Encoding and Clustering Completed!")


if __name__ == "__main__":
    main()
