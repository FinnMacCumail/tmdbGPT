import json
import chromadb
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import re
from typing import List, Dict

# Load environment variables
load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./semantic_chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load API schema
with open("./data/tmdb.json", "r", encoding="utf-8") as f:
    tmdb_schema = json.load(f)

api_endpoints = tmdb_schema.get("paths", {})

def analyze_parameters(parameters: List) -> Dict:
    """Dynamically classifies parameters with type safety"""
    analysis = {
        "search_params": [],
        "filter_params": [],
        "pagination_params": [],
        "id_params": [],
        "required_params": []
    }
    
    for param in parameters:
        # Skip non-dictionary parameters
        if not isinstance(param, dict):
            continue
            
        param_name = param.get("name", "")
        param_in = param.get("in", "")
        required = param.get("required", False)
        
        if param_in == "path":
            analysis["id_params"].append(param_name)
        elif param_in == "query":
            if param_name == "query":
                analysis["search_params"].append(param_name)
            elif "page" in param_name:
                analysis["pagination_params"].append(param_name)
            elif any(k in param_name for k in [".gte", ".lte", "sort_by", "with_"]):
                analysis["filter_params"].append(param_name)
        
        if required:
            analysis["required_params"].append(param_name)
    
    return analysis

# In semantic_embed.py
def infer_api_context(method_data: dict) -> str:  # Remove "path" parameter
    """Dynamically infer API purpose from parameters and summary"""
    parameters = method_data.get("parameters", [])
    summary = method_data.get("summary", "").lower()
    
    # 1. Detect search endpoints by parameter presence
    if any(p.get("name") == "query" for p in parameters):
        return "Text search endpoint"
    
    # 2. Identify discovery endpoints
    discovery_flags = {"sort_by", "with_", "without_", ".gte", ".lte"}
    if any(any(flag in p.get("name", "") for flag in discovery_flags) for p in parameters):
        return "Filtered discovery endpoint"
    
    # 3. Check summary semantics as fallback
    if "search" in summary:
        return "Search-oriented endpoint"
    if "discover" in summary:
        return "Content discovery endpoint"
    
    # 4. Default context
    return "General API endpoint"
def generate_embedding_components(path, method, method_data):
    """Constructs embedding metadata through dynamic analysis"""
    parameters = method_data.get("parameters", [])
    param_analysis = analyze_parameters(parameters)
    context = infer_api_context(method_data)
    
    return {
        "path": path,
        "method": method.upper(),
        "summary": method_data.get("summary", "No summary"),
        "context": context,
        "parameters": {
            "search": param_analysis["search_params"],
            "filters": param_analysis["filter_params"],
            "pagination": param_analysis["pagination_params"],
            "ids": param_analysis["id_params"],
            "required": param_analysis["required_params"]
        }
    }

def create_embedding_text(components):
    """Generates semantic text for embedding from analyzed components"""
    parts = [
        f"API Purpose: {components['summary']}",
        f"Context Type: {components['context']}",
        f"HTTP Method: {components['method']}",
        f"Path Pattern: {components['path']}",
        "Parameters:",
        f"- Search: {components['parameters']['search'] or 'None'}",
        f"- Filters: {components['parameters']['filters'] or 'None'}",
        f"- Pagination: {components['parameters']['pagination'] or 'None'}",
        f"- IDs: {components['parameters']['ids'] or 'None'}",
        f"- Required: {components['parameters']['required'] or 'None'}"
    ]
    return "\n".join(parts)

def store_api_embeddings():
    """Store embeddings with ChromaDB-compatible metadata"""
    for path, methods in api_endpoints.items():
        for method, method_data in methods.items():
            parameters = method_data.get("parameters", [])
            
            # Convert parameters to JSON string
            metadata = {
                "path": path,
                "method": method.upper(),
                "parameters": json.dumps([p for p in parameters if isinstance(p, dict)]),
                "context": infer_api_context(method_data),  # Remove "path" argument
                "requires_resolution": bool(re.search(r"{\w+}", path))
            }
            
            collection.add(
                ids=[f"{path}-{method}"],
                embeddings=[model.encode(metadata["context"]).tolist()],
                metadatas=[metadata]
            )

if __name__ == "__main__":
    store_api_embeddings()