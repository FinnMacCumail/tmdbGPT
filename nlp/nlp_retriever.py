import chromadb
import os
import spacy
import huggingface_hub
from importlib import import_module

# sentence-transformers versions prior to v2.3 expect cached_download from
# huggingface_hub. Recent huggingface_hub releases removed this helper. Insert a
# compatibility shim so the import succeeds regardless of the installed hub
# version.
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download as cached_download
    huggingface_hub.cached_download = cached_download

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re
from typing import Dict, List, Optional, Set, Any

from core.embeddings.hybrid_retrieval import retrieve_semantic_matches, convert_matches_to_execution_steps
from core.planner.plan_validator import PlanValidator, SymbolicConstraintFilter
from pathlib import Path

# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

# Initialize ChromaDB client

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)


# Load NLP and embedding models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# TMDB API configuration
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_API_KEY}"}


def expand_plan_with_dependencies(state, newly_resolved: dict) -> list:
    """
    Use newly resolved entities to find and append follow-up steps to the plan.

    Args:
        state (AppState): current app state
        newly_resolved (dict): keys like {"person_id": 1234}

    Returns:
        List[dict]: list of new execution steps (if any)
    """
    if not newly_resolved:
        return []

    current_intents = state.extraction_result.get("intents", [])
    followup_matches = DependencyEndpointSuggester.suggest_followups(
        newly_resolved, current_intents)

    existing_endpoints = {step["endpoint"] for step in state.plan_steps}
    new_matches = [m for m in followup_matches if m["endpoint"]
                   not in existing_endpoints]

    return convert_matches_to_execution_steps(new_matches, state.extraction_result, state.resolved_entities)


class DependencyEndpointSuggester:
    @staticmethod
    def suggest_followups(new_entities: dict, current_intents: list, limit: int = 10) -> list:
        """
        Given new resolved entities (e.g., person_id), suggest follow-up endpoints that are now actionable.

        Args:
            new_entities (dict): newly resolved keys like {'person_id': 6193}
            current_intents (list): list of active user intents from LLM extraction
            limit (int): max number of results to return

        Returns:
            list of dicts with suggested endpoint metadata
        """
        queries = []
        for entity_key in new_entities.keys():
            for intent in current_intents:
                prompt = f"Fetch endpoints requiring {entity_key} for intent {intent}"
                queries.append(prompt)

        all_matches = []
        for q in queries:
            results = retrieve_semantic_matches(q, top_k=limit)
            all_matches.extend(results)

        # De-duplicate by endpoint
        seen = set()
        unique = []
        for m in all_matches:
            eid = m.get("endpoint")
            if eid and eid not in seen:
                seen.add(eid)
                unique.append(m)

        return unique


class PathRewriter:
    @staticmethod
    def rewrite(path: str, resolved_entities: dict) -> str:
        def replacer(match):
            key = match.group(1)
            value = resolved_entities.get(key)
            if isinstance(value, list) and len(value) == 1:
                return str(value[0])
            return str(value) if value is not None else match.group(0)

        rewritten = re.sub(r"{(\w+)}", replacer, path)
        return rewritten or path  # ✅ fallback to original if empty


class PostStepUpdater:
    # 🔎 Extracts resolved entity IDs from /search/* API responses (e.g., person, movie).
    # Updates state.resolved_entities and logs extracted results for downstream use.
    # may want to extend this logic to handle /credits or /discover responses in the future.
    @staticmethod
    def update(state, step, json_data):
        path = step.get("endpoint")
        step_id = step.get("step_id", "unknown_step")

        extracted = {}

        if path.startswith("/search/"):
            for item in json_data.get("results", []):
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("id")
                entity_name = item.get("name") or item.get("title")
                if entity_id:
                    entity_type = PostStepUpdater._infer_entity_type(path)
                    if entity_type:
                        key = f"{entity_type}_id"
                        extracted.setdefault(key, []).append(entity_id)

        if extracted:
            state.resolved_entities.update(extracted)
            state.responses.append({"step": step_id, "extracted": extracted})

        return state

    @staticmethod
    def _infer_entity_type(path):
        if "person" in path:
            return "person"
        elif "movie" in path:
            return "movie"
        elif "tv" in path:
            return "tv"
        elif "company" in path:
            return "company"
        elif "collection" in path:
            return "collection"
        elif "keyword" in path:
            return "keyword"
        elif "network" in path:
            return "network"
        elif "genre" in path:
            return "genre"
        return None


class RerankPlanning:
    @staticmethod
    def rerank_matches(matches, resolved_entities):
        """
        Reorder and annotate matches based on parameter feasibility.
        Promote steps with resolved entities; demote those with missing params.
        Boost endpoints that support optional semantically inferred parameters.
        """
        reranked = []

        # ✅ Load optional semantic parameters if query available
        validator = PlanValidator()
        # ⚡ expect __query injected upstream
        query_text = resolved_entities.get("__query", "")
        optional_params = validator.infer_semantic_parameters(
            query_text) if query_text else []
        SAFE_OPTIONAL_PARAMS = {
            "vote_average.gte", "vote_count.gte", "primary_release_year",
            "release_date.gte", "with_runtime.gte", "with_runtime.lte",
            "with_original_language", "region"
        }

        for match in matches:
            endpoint = match.get("endpoint", "")
            needs = []
            penalty = 0.0
            boost = 0.0

            # --- Symbolic Entity Missing Penalty ---
            for key in ["person_id", "movie_id", "tv_id", "collection_id", "company_id"]:
                if f"{{{key}}}" in endpoint and not resolved_entities.get(key):
                    needs.append(key)
                    penalty += 0.4

            # --- Boost if Optional Parameters Supported ---
            supported = match.get("supported_parameters", [])
            optional_match_count = 0

            for param in optional_params:
                if param in SAFE_OPTIONAL_PARAMS and param in supported:
                    boost += 0.02
                    optional_match_count += 1

            # --- Bonus Boost for Strong Param Coverage ---
            if optional_match_count >= 3:
                boost += 0.05

            # --- Apply Pre-Recorded Penalty from SymbolicConstraintFilter ---
            additional_penalty = match.get("penalty", 0.0)
            if additional_penalty > 0:
                penalty += additional_penalty

            # --- NEW: Penalty if Missing Supported Intents ---
            supported_intents = SymbolicConstraintFilter._extract_supported_intents(
                match.get("metadata", match))
            if not supported_intents:
                penalty += 0.1

            # --- Final Scoring ---
            base_score = match.get("final_score", 0.0)
            final_score = round(base_score + boost - penalty, 3)

            match.update({
                "final_score": final_score,
                "missing_entities": needs,
                "is_entrypoint": bool("/search" in endpoint)
            })
            reranked.append(match)

            # --- Clean Debug Output ---

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    @staticmethod
    def validate_parameters(endpoint, resolved_entities):
        """
        Check if endpoint has all the required resolved parameters.
        Return a flag indicating if the step is executable.
        """
        for key in ["person_id", "movie_id", "tv_id", "collection_id"]:
            if f"{{{key}}}" in endpoint and not resolved_entities.get(key):
                return False
        return True

    @staticmethod
    def filter_feasible_steps(ranked_matches, resolved_entities, extraction_result=None):
        """
        Return only steps that can be executed now, plus role-aware entrypoints if applicable.
        """
        feasible = []
        deferred = []

        # Extract question type and roles from extraction result
        question_type = extraction_result.get(
            "question_type", "summary") if extraction_result else "summary"
        query_entities = extraction_result.get(
            "query_entities", []) if extraction_result else []

        for match in ranked_matches:
            endpoint_path = match.get("endpoint") or match.get("path", "")
            # 🔐 Standard validation: do we have required parameters?
            is_valid = RerankPlanning.validate_parameters(
                endpoint_path, resolved_entities)
            # phase 19.7 ✅ Strategic override: allow /person/{id}/movie_credits or tv_credits if role + count query
            if not is_valid:
                if endpoint_path in {"/person/{person_id}/movie_credits", "/person/{person_id}/tv_credits"}:
                    if question_type == "count":
                        if any(
                            ent.get("type") == "person" and ent.get("role") in {
                                "director", "cast", "writer", "producer", "composer"}
                            for ent in query_entities
                        ):
                            is_valid = True

            if is_valid:
                feasible.append(match)
            elif match.get("is_entrypoint"):
                feasible.append(match)
            else:
                deferred.append(match)

        return feasible, deferred


class ResultExtractor:
    @staticmethod
    def extract(json_data: dict, endpoint: str, resolved_entities: dict = None) -> list:
        resolved_entities = resolved_entities or {}

        if not json_data:
            return []

        # ✅ Movie detail endpoint
        if endpoint.startswith("/movie/") and endpoint.count("/") == 2:
            return ResultExtractor._extract_movie_details(json_data, endpoint)

        # ✅ TV detail endpoint (e.g. Breaking Bad)
        if endpoint.startswith("/tv/") and endpoint.count("/") == 2:
            return ResultExtractor._extract_tv_details(json_data, endpoint)

        summaries = []
        if "/credits" in endpoint:
            summaries += ResultExtractor.extract_cast_and_crew_credits(
                json_data, endpoint)
            # f"✅ Total summaries extracted from credits: {len(summaries)}")
            return summaries  # ← THIS is critical
        # ✅ Credits endpoints: tv or movie
        if "tv_credits" in endpoint or "movie_credits" in endpoint:
            return ResultExtractor._extract_credits(json_data, endpoint, resolved_entities=resolved_entities)

        # ✅ Discovery endpoint
        if "/discover/" in endpoint:
            return ResultExtractor._extract_discovery(json_data, endpoint)

        # ✅ Search: person
        if "/search/person" in endpoint:
            return ResultExtractor._extract_search_person(json_data)

        # ✅ Person profile (not a sub-resource like /credits, /tv, /movie)
        if "/person/" in endpoint and not any(k in endpoint for k in ["/credits", "/images", "/tv", "/movie"]):
            return ResultExtractor._extract_person_profile(json_data, endpoint)

        # ✅ Company / Network lookups
        if "/company/" in endpoint or "/network/" in endpoint:
            return ResultExtractor._extract_company_or_network(json_data)

        # ✅ Generic fallback
        return ResultExtractor._extract_generic(json_data, endpoint)

    @staticmethod
    def _extract_discovery(json_data, endpoint):
        summaries = []
        results = json_data.get("results", [])

        for item in results:
            title = item.get("title") or item.get("name", "Untitled")
            overview = item.get("overview") or "No synopsis available."
            score = item.get("vote_average", 0) / 10.0
            release_date = item.get(
                "release_date") or item.get("first_air_date")
            entity_id = item.get("id")

            summaries.append({
                "id": entity_id,
                "type": "tv_summary" if "tv" in endpoint else "movie_summary",
                "title": title,
                "overview": overview.strip(),
                "source": endpoint,
                "final_score": round(score, 2),
                "release_date": release_date,
                "media_type": "tv" if "tv" in endpoint else "movie"
            })

        return summaries

    @staticmethod
    def _extract_search_person(json_data):
        summaries = []
        seen = set()
        results = json_data.get("results", [])

        for result in results:
            name = result.get("name", "").strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())

            known_for = result.get("known_for", [])
            known_titles = [
                k.get("title") or k.get("name")
                for k in known_for if isinstance(k, dict) and (k.get("title") or k.get("name"))
            ]

            overview = f"Known for: {', '.join(known_titles)}" if known_titles else "No major known works."

            summaries.append({
                "type": "person_profile",
                "title": name,
                "overview": overview,
                "source": "/search/person"
            })

        return summaries

    @staticmethod
    def _extract_credits(json_data: dict, endpoint: str, resolved_entities=None) -> list:
        """
        Dispatcher for credit extraction.
        Routes to TV or Movie logic based on endpoint path.
        """
        person_id = None
        if resolved_entities and "person_id" in resolved_entities:
            person_id = resolved_entities["person_id"]

        if "tv_credits" in endpoint:
            return ResultExtractor._extract_tv_credits(json_data, person_id=person_id)
        elif "movie_credits" in endpoint:
            return ResultExtractor._extract_movie_credits(json_data, endpoint)
        return []

    @staticmethod
    def _extract_tv_credits(json_data: dict, person_id: int = None) -> list:
        """
        Extracts TV credits (cast and selected crew roles) from the TMDB credits response.

        Args:
            json_data (dict): TMDB /tv/{id}/credits response
            person_id (int, optional): Used to tag cast summaries for symbolic role enrichment

        Returns:
            list of dict: Summarized TV credit entries with optional role tagging
        """
        cast = json_data.get("cast", [])
        crew = json_data.get("crew", [])
        # f"🟢 _extract_tv_credits called — cast: {len(cast)}, crew: {len(crew)}")

        summaries = []

        # ➤ Extract cast members
        for entry in cast:
            entity_id = entry.get("id")
            title = entry.get("name") or entry.get(
                "original_name") or "Untitled"
            overview = entry.get("overview") or entry.get(
                "character") or "Cast"
            release_date = entry.get("first_air_date") or "Unknown"

            summary = {
                "id": entity_id,
                "type": "tv_summary",
                "title": title,
                "overview": overview,
                "release_date": release_date,
                "final_score": 1.0,
                "source": "/tv_credits",
                "job": "cast",
                "media_type": "tv"
            }

            # 🎯 Inject actor ID for role-based indexing (symbolic constraint satisfaction)
            if person_id:
                summary["_actor_id"] = person_id

            summaries.append(summary)

        # ➤ Extract selected crew members (director, writer, producer)
        allowed_jobs = {"director", "writer", "producer"}
        for entry in crew:
            job = (entry.get("job") or "").lower()
            if job in allowed_jobs:
                entity_id = entry.get("id")
                title = entry.get("name") or entry.get(
                    "original_name") or "Untitled"
                overview = entry.get("overview") or job.title()
                release_date = entry.get("first_air_date") or "Unknown"

                summaries.append({
                    "id": entity_id,
                    "type": "tv_summary",
                    "title": title,
                    "overview": overview,
                    "release_date": release_date,
                    "final_score": 1.0,
                    "source": "/tv_credits",
                    "job": job,
                    "media_type": "tv",
                    "genre_ids": entry.get("genre_ids", []),
                })

        return summaries

    @staticmethod
    def _extract_movie_credits(json_data, endpoint, resolved_entities=None):
        movie_id = extract_id_from_endpoint(endpoint)
        results = []

        for cast in json_data.get("cast", []):
            results.append({
                "id": cast.get("id", movie_id),
                "title": cast.get("title") or cast.get("original_title"),
                "release_date": cast.get("release_date"),
                "poster_path": cast.get("poster_path"),
                "overview": cast.get("overview"),
                "type": "movie_summary",
                "job": "cast",
                "character": cast.get("character"),
                "source": endpoint
            })

        for crew in json_data.get("crew", []):
            results.append({
                "id": crew.get("id", movie_id),
                "title": crew.get("title") or crew.get("original_title"),
                "release_date": crew.get("release_date"),
                "poster_path": crew.get("poster_path"),
                "overview": crew.get("overview"),
                "type": "movie_summary",
                "job": crew.get("job"),  # e.g., Director
                "department": crew.get("department"),
                "source": endpoint
            })

        return results

    @staticmethod
    def _extract_person_profile(json_data, endpoint):
        name = json_data.get("name", "Unknown")
        bio = json_data.get("biography", "No biography available.")
        department = json_data.get("known_for_department", "N/A")
        birthday = json_data.get("birthday")
        place = json_data.get("place_of_birth")

        return [{
            "id": json_data.get("id"),
            "type": "person_profile",
            "name": name,
            "biography": bio.strip(),
            "known_for": department,
            "birthplace": place,
            "birthday": birthday,
            "final_score": 1.0,
            "media_type": "person",
            "source": endpoint
        }]

    @staticmethod
    def _extract_company_or_network(json_data):
        name = json_data.get("name", "Unknown")
        description = json_data.get("description", "No description available.")

        return [{
            "type": "company_profile",
            "id": json_data.get("id"),
            "title": name,
            "overview": description.strip(),
            "source": "/company_or_network/profile",
            "final_score": 1.0
        }]

    @staticmethod
    def _extract_generic(json_data, endpoint):
        summaries = []
        seen = set()

        # --- General List-based fallback ---
        candidate_lists = [
            v for v in json_data.values() if isinstance(v, list)
        ]
        if not candidate_lists and "results" in json_data:
            candidate_lists = [json_data["results"]]

        for item_list in candidate_lists:
            for item in item_list:
                if not isinstance(item, dict):
                    continue

                title = item.get("title") or item.get("name", "Untitled")
                overview = item.get("overview") or item.get("job") or item.get(
                    "character") or item.get("description") or "No synopsis available."
                score = item.get("vote_average", 0) / 10.0
                release_date = item.get(
                    "release_date") or item.get("first_air_date")

                summaries.append({
                    "id": item.get("id"),
                    "type": "movie_summary",
                    "title": title,
                    "overview": overview.strip(),
                    "source": endpoint,
                    "final_score": round(score, 2),
                    "release_date": release_date,
                    "media_type": "tv" if "tv" in endpoint else "movie"
                })

        # --- Flat object fallback (for /person/{id} etc.) ---
        flat_title = json_data.get("title") or json_data.get("name")
        flat_overview = json_data.get(
            "overview") or json_data.get("biography") or ""

        if flat_title or flat_overview:
            is_person_profile = (
                "/person/" in endpoint
                and not any(k in endpoint for k in ["/credits", "/images", "/tv", "/movie"])
            )
            profile_type = "person_profile" if is_person_profile else "movie_summary"

            summaries.append({
                "id": item.get("id"),
                "type": profile_type,
                "title": flat_title or "Untitled",
                "overview": flat_overview.strip() or "No bio available.",
                "source": endpoint,
                "final_score": 1.0,
                "media_type": "tv" if "tv" in endpoint else "movie"
            })

        return summaries

    @staticmethod
    def should_post_filter(endpoint: str, applied_params: dict = None) -> bool:
        """
        Determines whether post-filtering is needed for a given TMDB endpoint.

        Post-filtering is only applied if:
        - The endpoint is a broad retrieval type (e.g. /discover, /search)
        - No strong filtering (e.g. with_people, with_genres) was applied in the query
        """
        if not endpoint:
            return False

        applied_params = applied_params or {}

        # Strong filters make post-filtering unnecessary
        if any(p in applied_params for p in ("with_people", "with_genres", "with_companies", "with_networks")):
            return False

        return endpoint.startswith("/discover") or endpoint.startswith("/search")

    @staticmethod
    def _extract_movie_details(json_data, endpoint):
        title = json_data.get("title", "Untitled")
        overview = json_data.get("overview") or "No synopsis available."
        release_date = json_data.get("release_date")
        score = json_data.get("vote_average", 0) / 10.0

        # ✅ Extract all crew roles from credits, if available
        directors = []
        writers = []
        composers = []
        producers = []
        cast = []
        
        if "credits" in json_data:
            # Extract crew roles
            for crew_member in json_data["credits"].get("crew", []):
                job = crew_member.get("job", "").lower()
                name = crew_member.get("name")
                if name:
                    if job == "director":
                        directors.append(name)
                    elif job in ["writer", "screenplay", "story", "characters"]:
                        if name not in writers:  # Avoid duplicates
                            writers.append(name)
                    elif job in ["original music composer", "music", "composer", "music by"]:
                        if name not in composers:
                            composers.append(name)
                    elif job in ["producer", "executive producer"]:
                        if name not in producers:
                            producers.append(name)
            
            # Extract main cast (limit to top 5 to avoid overly long responses)
            for cast_member in json_data["credits"].get("cast", [])[:5]:
                name = cast_member.get("name")
                if name:
                    cast.append(name)

        # ✅ Extract additional fields for fact queries
        runtime = json_data.get("runtime")
        genres = json_data.get("genres", [])
        revenue = json_data.get("revenue")

        return [{
            "id": json_data.get("id"),
            "type": "movie_summary",
            "title": title,
            "overview": overview.strip(),
            "release_date": release_date,
            "directors": directors,
            "writers": writers,  # ✅ Add writers field
            "composers": composers,  # ✅ Add composers field
            "producers": producers,  # ✅ Add producers field
            "cast": cast,  # ✅ Add main cast field
            "runtime": runtime,  # ✅ Add runtime field
            "genres": genres,    # ✅ Add genres field
            "revenue": revenue,  # ✅ Add revenue field
            "final_score": round(score, 2),
            "source": endpoint,
            "media_type": "movie"
        }]

    @staticmethod
    def _extract_tv_details(json_data, endpoint):
        title = json_data.get("name", "Untitled")
        overview = json_data.get(
            "overview", "").strip() or "No synopsis available."
        release_date = json_data.get("first_air_date")
        score = round(json_data.get("vote_average", 0) / 10.0, 2)

        # ✅ Extract creators from created_by field and all crew roles from credits
        creators = []
        writers = []
        producers = []
        cast = []
        
        # Extract from created_by field
        created_by_data = json_data.get("created_by", [])
        if created_by_data:
            for creator in created_by_data:
                if isinstance(creator, dict) and creator.get("name"):
                    creators.append(creator["name"])
        
        # ✅ Extract all crew and cast roles from credits, if available
        if "credits" in json_data:
            # Extract crew roles
            for crew_member in json_data["credits"].get("crew", []):
                job = crew_member.get("job", "").lower()
                name = crew_member.get("name")
                if name:
                    if job in ["creator", "executive producer", "director"]:
                        if name not in creators:
                            creators.append(name)
                    elif job in ["writer", "screenplay", "story", "teleplay", "television writer"]:
                        if name not in writers:
                            writers.append(name)
                    elif job in ["producer", "executive producer", "co-executive producer"]:
                        if name not in producers:
                            producers.append(name)
            
            # Extract main cast (limit to top 5 for TV shows)
            for cast_member in json_data["credits"].get("cast", [])[:5]:
                name = cast_member.get("name")
                if name:
                    cast.append(name)

        # ✅ Extract additional fields for TV fact queries
        number_of_seasons = json_data.get("number_of_seasons")
        number_of_episodes = json_data.get("number_of_episodes") 
        first_air_date = json_data.get("first_air_date")

        return [{
            "id": json_data.get("id"),
            "type": "tv_summary",
            "title": title,
            "overview": overview,
            "release_date": release_date,
            "first_air_date": first_air_date,  # ✅ Add first air date field
            "created_by": json_data.get("created_by", []),
            "creators": creators,  # ✅ Add extracted creators list
            "writers": writers,  # ✅ Add writers field
            "producers": producers,  # ✅ Add producers field
            "cast": cast,  # ✅ Add main cast field
            "number_of_seasons": number_of_seasons,    # ✅ Add seasons field
            "number_of_episodes": number_of_episodes,  # ✅ Add episodes field
            "final_score": score,
            "source": endpoint,
            "media_type": "tv"
        }]

    @staticmethod
    def extract_cast_and_crew_credits(json_data, endpoint):
        summaries = []

        cast_list = json_data.get("cast", [])
        crew_list = json_data.get("crew", [])

        for member in cast_list:
            name = member.get("name") or "Unknown"
            character = member.get("character") or "Unknown role"
            summaries.append({
                "type": "movie_summary",
                "title": name,
                "overview": character,
                "source": endpoint,
                "final_score": 0.0,
                "release_date": None
            })

        for member in crew_list:
            name = member.get("name") or "Unknown"
            job = member.get("job") or "Crew"
            summaries.append({
                "type": "movie_summary",
                "title": name,
                "overview": job,
                "source": endpoint,
                "final_score": 0.0,
                "release_date": None
            })

        return summaries


def extract_id_from_endpoint(endpoint):
    """
    Extracts numeric ID from an endpoint like /movie/594/credits.
    """
    match = re.search(r"/movie/(\d+)", endpoint)
    if match:
        return int(match.group(1))
    return None
