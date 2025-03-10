import spacy
from spacy.pipeline import EntityRuler
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
import requests
from dotenv import load_dotenv# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env") 
load_dotenv(dotenv_path, override=True)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")



# Load the Transformer-based spaCy model
nlp = spacy.load("en_core_web_trf")

# ✅ Ensure EntityRuler is added **before** the Named Entity Recognizer (NER)
if "ner" in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.add_pipe("entity_ruler")

# ✅ Define dynamic entity mapping (All 49+ Entities)
ENTITY_MAPPING = {
    "DATE": "year",
    "PERSON": "with_people",
    "GPE": "region",
    "ORG": "with_companies",
    "LANGUAGE": "with_original_language",
    "EVENT": "with_keywords",
    "NORP": "with_keywords",
    "WORK_OF_ART": "query",
    "FAC": "query",
    "LOC": "query",
    "PRODUCT": "query",
    "CARDINAL": "page",
    "ORDINAL": "page",
}

# ✅ List of known movie genres from TMDB (case-insensitive matching)
GENRE_MAPPING = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80, "Documentary": 99,
    "Drama": 18, "Family": 10751, "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878, "Thriller": 53, "War": 10752, "Western": 37
}

# ✅ Add genre patterns (ensure case-insensitivity)
patterns = [{"label": "GENRE", "pattern": [{"LOWER": genre.lower()}]} for genre in GENRE_MAPPING.keys()]

ruler.add_patterns(patterns)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="tmdb_queries")


def extract_entities(query):
    """Extract relevant entities dynamically using spaCy."""
    doc = nlp(query)
    
    extracted = {ent.label_: ent.text for ent in doc.ents}

    # ✅ Convert extracted entities to TMDB-compatible parameters
    mapped_entities = {ENTITY_MAPPING.get(key, key): value for key, value in extracted.items()}

    # ✅ Convert recognized genres into TMDB Genre IDs
    if "with_genres" in mapped_entities:
        genre_name = mapped_entities["with_genres"].title()
        if genre_name in GENRE_MAPPING:
            mapped_entities["with_genres"] = GENRE_MAPPING[genre_name]

    return mapped_entities

def match_query_to_cluster(query, extracted_entities, n_results=5):
    """Find and filter the closest API clusters using cosine similarity."""
    query_vector = model.encode([query]).tolist()
    search_results = collection.query(query_texts=[query], n_results=n_results)

    if not search_results or "metadatas" not in search_results or not search_results["metadatas"]:
        print("❌ No matching clusters found in ChromaDB.")
        return []

    matched_apis = search_results["metadatas"]
    final_apis = []

    for api_list in matched_apis:
        for api in api_list:
            if isinstance(api, dict):
                raw_params = api.get("parameters", "[]")

                try:
                    parsed_params = json.loads(raw_params)
                    api["parameters"] = {
                        param["name"]: extracted_entities.get(param["name"], None)
                        for param in parsed_params if "name" in param
                    }
                except (json.JSONDecodeError, TypeError):
                    print(f"⚠️ Warning: Could not parse parameters for {api['endpoint']}")
                    api["parameters"] = {}

                # ✅ Exclude APIs that don't have any matching parameters
                if any(param in extracted_entities for param in api["parameters"]):
                    final_apis.append(api)

    print("✅ Matched APIs After Parameter Injection:", final_apis)
    return final_apis

def execute_api_calls(matched_apis):
    """Executes API calls dynamically based on matched API endpoints."""
    headers = {
        "Authorization": f"Bearer {TMDB_API_KEY}",
        "Content-Type": "application/json"
    }
    responses = []
    person_id = None  # Store person ID if found

    for api in matched_apis:
        url = f"https://api.themoviedb.org/3{api['endpoint']}"
        params = {key: value for key, value in api["parameters"].items() if value is not None}

        print(f"🌐 Making API Call: {url}")
        print(f"🔍 Parameters: {params}")

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            responses.append(data)

            # ✅ Extract person_id dynamically from `/search/person`
            if api["endpoint"] == "/search/person" and data["results"]:
                person_id = data["results"][0]["id"]
                print(f"✅ Found Person ID: {person_id}")

                # ✅ Dynamically fetch additional details
                responses.append(requests.get(f"https://api.themoviedb.org/3/person/{person_id}", headers=headers).json())
                responses.append(requests.get(f"https://api.themoviedb.org/3/person/{person_id}/movie_credits", headers=headers).json())

        else:
            print(f"❌ API request failed for {api['endpoint']} with status {response.status_code}")

    return responses

# Extracted Entities
#query = "What horror movies were released in 2022?"
query = "Who is Sofia Coppola?"
entities = extract_entities(query)
matched_apis = match_query_to_cluster(query, entities)
api_responses = execute_api_calls(matched_apis)

# ✅ Print the API responses
for i, response in enumerate(api_responses, start=1):
    print(f"✅ API Response {i}:")
    print(response)
