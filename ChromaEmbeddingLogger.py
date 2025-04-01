import chromadb
import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict


class ChromaEmbeddingLogger:
    def __init__(self):
        self.log_dir = "logs"
        self.collection_name = "tmdb_endpoints"
        self.client = chromadb.PersistentClient(path="./sec_intent_chroma_db")

    def _ensure_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_collection(self):
        return self.client.get_collection(name=self.collection_name)

    def _generate_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"chroma_embeddings_{timestamp}.log"

    def _format_entry(self, entry: Dict) -> str:
        metadata = entry["metadata"]
        embedding_text = metadata.get("embedding_text", "[Embedding text not available]")

        return (
            f"ID: {entry['id']}\n"
            f"Metadata: {json.dumps(metadata, indent=2)}\n"
            f"Embedding: Vector of length {len(entry['embedding'])} "
            f"(First 5 dims: {entry['embedding'][:5]})\n"
            f"Embedding Text:\n{embedding_text}\n"
            + "=" * 80 + "\n"
        )

    def log_embeddings(self):
        try:
            self._ensure_log_dir()
            collection = self._get_collection()
            embeddings = collection.get(include=["embeddings", "metadatas"])

            ids = embeddings.get("ids", [])
            if not ids:
                print("No embeddings found in collection")
                return

            # Count occurrences of each ID
            id_counts = Counter(ids)
            duplicates = {k: v for k, v in id_counts.items() if v > 1}

            if duplicates:
                print("🚨 Duplicate IDs detected:")
                for dup_id, count in duplicates.items():
                    print(f" - {dup_id} → {count} times")
            else:
                print("✅ No duplicate IDs found.")

            # Proceed with logging
            log_content = []
            for idx in range(len(ids)):
                entry = {
                    "id": ids[idx],
                    "metadata": embeddings["metadatas"][idx],
                    "embedding": embeddings["embeddings"][idx]
                }
                log_content.append(self._format_entry(entry))

            filename = os.path.join(self.log_dir, self._generate_filename())
            with open(filename, "w", encoding="utf-8") as f:
                f.writelines(log_content)

            print(f"Successfully logged {len(log_content)} embeddings to {filename}")

        except Exception as e:
            print(f"Error logging embeddings: {str(e)}")


if __name__ == "__main__":
    logger = ChromaEmbeddingLogger()
    logger.log_embeddings()
