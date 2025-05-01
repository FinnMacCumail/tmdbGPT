import chromadb
import os
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re
from typing import Dict, List, Optional, Set, Any


from fallback_handler import FallbackHandler
from llm_client import OpenAILLMClient

from json import JSONDecodeError
from hybrid_retrieval_test import hybrid_search, convert_matches_to_execution_steps

from plan_validator import PlanValidator, SymbolicConstraintFilter

# Load API keys
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="tmdb_endpoints",
    metadata={"hnsw:space": "cosine"}
)

# Load NLP and embedding models
nlp = spacy.load("en_core_web_trf")
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
    followup_matches = DependencyEndpointSuggester.suggest_followups(newly_resolved, current_intents)

    existing_endpoints = {step["endpoint"] for step in state.plan_steps}
    new_matches = [m for m in followup_matches if m["endpoint"] not in existing_endpoints]

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
            results = hybrid_search(q, top_k=limit)
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
        """
        Replaces unresolved placeholders in endpoint paths with resolved entity values.
        Example: /person/{person_id} → /person/123 if person_id is in resolved_entities.

        Args:
            path (str): the endpoint path with placeholders
            resolved_entities (dict): dictionary of resolved entity keys and values

        Returns:
            str: updated path with substitutions applied
        """

        def replacer(match):
            key = match.group(1)
            return str(resolved_entities.get(key, match.group(0)))

        return re.sub(r"{(\w+)}", replacer, path)

class PostStepUpdater:
    @staticmethod
    def update(state, step, json_data):
        path = step.get("endpoint")
        step_id = step.get("step_id", "unknown_step")

        extracted = {}

        if path.startswith("/search/"):
            print("🔎 Raw /search/person results:")
            for item in json_data.get("results", []):
                print(f"  → {item.get('name')} (id={item.get('id')})")
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("id")
                entity_name = item.get("name") or item.get("title")
                if entity_id:
                    entity_type = PostStepUpdater._infer_entity_type(path)
                    if entity_type:
                        key = f"{entity_type}_id"
                        extracted.setdefault(key, []).append(entity_id)
                        print(f"🔍 Resolved {entity_type}: '{entity_name}' → {entity_id}")

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
        query_text = resolved_entities.get("__query", "")  # ⚡ expect __query injected upstream
        optional_params = validator.infer_semantic_parameters(query_text) if query_text else []
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
                print(f"⭐ Strong optional param coverage bonus applied (+0.05) for {endpoint}")

            # --- Apply Pre-Recorded Penalty from SymbolicConstraintFilter ---
            additional_penalty = match.get("penalty", 0.0)
            if additional_penalty > 0:
                print(f"🔻 Applying additional penalty of {additional_penalty} to {endpoint}")
                penalty += additional_penalty

            # --- NEW: Penalty if Missing Supported Intents ---
            supported_intents = SymbolicConstraintFilter._extract_supported_intents(match.get("metadata", match))
            if not supported_intents:
                penalty += 0.1
                print(f"⚠️ Missing supported_intents penalty applied to {endpoint} (-0.1)")

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
            print("\n📊 Rerank Debug Info")
            print(f"Endpoint: {endpoint}")
            print(f"🔹 Base Score: {base_score}")
            print(f"➕ Boost Applied: {round(boost, 3)}")
            print(f"➖ Penalty Applied: {round(penalty, 3)}")
            print(f"🎯 Final Score: {final_score}")
            print(f"🔎 Missing Entities (path params): {needs}")
            print("-" * 40)


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
        question_type = extraction_result.get("question_type", "summary") if extraction_result else "summary"
        query_entities = extraction_result.get("query_entities", []) if extraction_result else []

        for match in ranked_matches:
            endpoint_path = match.get("endpoint") or match.get("path", "")
            print(f"🔍 Evaluating: {endpoint_path}")
            # 🔐 Standard validation: do we have required parameters?
            is_valid = RerankPlanning.validate_parameters(endpoint_path, resolved_entities)
            print(f"🔎 validate_parameters = {is_valid} for {endpoint_path}")
            # phase 19.7 ✅ Strategic override: allow /person/{id}/movie_credits or tv_credits if role + count query
            if not is_valid:
                print(f"⚠️ Rejected in feasibility: {endpoint_path} due to missing resolved params")
                if endpoint_path in {"/person/{person_id}/movie_credits", "/person/{person_id}/tv_credits"}:
                    if question_type == "count":
                        if any(
                            ent.get("type") == "person" and ent.get("role") in {"director", "cast", "writer", "producer", "composer"}
                            for ent in query_entities
                        ):
                            print(f"✅ Allowing role-aware count query: {endpoint_path}")
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
    def extract(endpoint: str, json_data: dict, resolved_entities: dict = None) -> list:
        resolved_entities = resolved_entities or {}

        if not json_data:
            return []

        if "movie_credits" in endpoint or "tv_credits" in endpoint:
            return ResultExtractor._extract_credits(json_data, endpoint)

        if "/discover/" in endpoint:
            return ResultExtractor._extract_discovery(json_data, endpoint)

        if "/search/person" in endpoint:
            return ResultExtractor._extract_search_person(json_data)

        if "/person/" in endpoint and not any(k in endpoint for k in ["/credits", "/images", "/tv", "/movie"]):
            return ResultExtractor._extract_person_profile(json_data)

        if "/company/" in endpoint or "/network/" in endpoint:
            return ResultExtractor._extract_company_or_network(json_data)

        return ResultExtractor._extract_generic(json_data, endpoint)

    @staticmethod
    def _extract_discovery(json_data, endpoint):
        summaries = []
        results = json_data.get("results", [])

        for item in results:
            title = item.get("title") or item.get("name", "Untitled")
            overview = item.get("overview") or "No synopsis available."
            score = item.get("vote_average", 0) / 10.0
            release_date = item.get("release_date") or item.get("first_air_date")

            summaries.append({
                "type": "movie_summary",
                "title": title,
                "overview": overview.strip(),
                "source": endpoint,
                "final_score": round(score, 2),
                "release_date": release_date
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
    def _extract_credits(json_data, endpoint):
        summaries = []

        # ✅ Crew roles: director, writer, producer, composer
        for crew in json_data.get("crew", []):
            job = crew.get("job", "").lower()
            if job in {"director", "writer", "producer", "composer"}:
                title = crew.get("title") or crew.get("name") or "Untitled"
                
                summaries.append({
                    "type": "movie_summary",
                    "title": crew.get("title") or crew.get("original_title") or crew.get("name", "Untitled"),
                    "overview": job.capitalize(),
                    "source": endpoint,
                    "release_date": crew.get("release_date") or crew.get("first_air_date"),
                    "final_score": 1.0,
                    "job": job  # ✅ Needed for count_summary formatter
                })

        # ✅ Cast
        for cast in json_data.get("cast", []):
            title = cast.get("title") or cast.get("name") or "Untitled"
            character = cast.get("character", "")
            
            summaries.append({
                "type": "movie_summary",
                "title": cast.get("title") or cast.get("original_title") or cast.get("name", "Untitled"),
                "overview": cast.get("character") or "Cast",
                "source": endpoint,
                "release_date": cast.get("release_date") or cast.get("first_air_date"),
                "final_score": 1.0,
                "job": "cast"  # ✅ Also helps count actor roles
            })
        
        return summaries


    @staticmethod
    def _extract_person_profile(json_data):
        name = json_data.get("name", "Unknown")
        bio = json_data.get("biography", "No biography available.")

        return [{
            "type": "person_profile",
            "title": name,
            "overview": bio.strip(),
            "source": "/person/profile",
            "final_score": 1.0
        }]

    @staticmethod
    def _extract_company_or_network(json_data):
        name = json_data.get("name", "Unknown")
        description = json_data.get("description", "No description available.")

        return [{
            "type": "company_profile",
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
                overview = item.get("overview") or item.get("job") or item.get("character") or item.get("description") or "No synopsis available."
                score = item.get("vote_average", 0) / 10.0
                release_date = item.get("release_date") or item.get("first_air_date")

                summaries.append({
                    "type": "movie_summary",
                    "title": title,
                    "overview": overview.strip(),
                    "source": endpoint,
                    "final_score": round(score, 2),
                    "release_date": release_date
                })

        # --- Flat object fallback (for /person/{id} etc.) ---
        flat_title = json_data.get("title") or json_data.get("name")
        flat_overview = json_data.get("overview") or json_data.get("biography") or ""

        if flat_title or flat_overview:
            is_person_profile = (
                "/person/" in endpoint
                and not any(k in endpoint for k in ["/credits", "/images", "/tv", "/movie"])
            )
            profile_type = "person_profile" if is_person_profile else "movie_summary"

            summaries.append({
                "type": profile_type,
                "title": flat_title or "Untitled",
                "overview": flat_overview.strip() or "No bio available.",
                "source": endpoint,
                "final_score": 1.0
            })

        return summaries
   

    @staticmethod

    def post_filter_responses(responses, query_entities, extraction_result):
        """
        Post-filter responses only if necessary.
        If the API step already filtered by genre/person/company, skip aggressive filtering.
        """
        if not responses:
            return []

        # ⚡ Check if original planning already injected strong filters (e.g., with_genres, with_people)
        strong_filter_applied = False
        applied_params = extraction_result.get("applied_parameters", {})
        if applied_params:
            strong_filter_applied = any(
                key in applied_params
                for key in ("with_genres", "with_people", "with_companies", "with_networks")
            )

        if strong_filter_applied:
            print("⚡ Strong filters already applied in plan — skipping aggressive post-filtering.")
            return responses  # ✅ Skip aggressive filtering if strong param filtering was applied.

        # ✅ Otherwise fallback to textual entity matching
        genre_names = [e["name"].lower() for e in query_entities if e.get("type") == "genre"]
        person_names = [e["name"].lower() for e in query_entities if e.get("type") == "person"]

        filtered = []
        for r in responses:
            text = (r.get("title", "") + " " + r.get("overview", "")).lower()
            keep = True

            for genre in genre_names:
                if genre not in text:
                    keep = False

            for person in person_names:
                if person not in text:
                    keep = False

            if keep:
                filtered.append(r)

        return filtered

     