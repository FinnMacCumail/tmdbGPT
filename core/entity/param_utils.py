import json

import os
import requests


def normalize_parameters(value):
    """
    Normalize TMDB parameters field to ensure it is a dict.

    Handles:
    - JSON strings → parsed dict
    - Lists (e.g. OpenAPI parameter specs) → empty dict
    - Anything else → return only if already dict
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    elif isinstance(value, list):
        return {}
    return value if isinstance(value, dict) else {}


class GenreNormalizer:
    GENRE_ALIASES = {
        "sci-fi": "science fiction",
        "scifi": "science fiction",
        "romcom": "romance",
        "dramedy": "comedy",
        "action adventure": "action",
        "doc": "documentary",
        "biopic": "history",
        "kids": "family",
        "animation": "animation",
        "suspense": "thriller",
        "feel good": "comedy",
        "fantasy epic": "fantasy",
    }

    TV_GENRE_ALIASES = {
        "sci-fi": "sci-fi & fantasy",
        "scifi": "sci-fi & fantasy",
        "science fiction": "sci-fi & fantasy",
        "romcom": "comedy",
        "dramedy": "comedy",
        "action adventure": "action & adventure",
        "doc": "documentary",
        "biopic": "history",
        "kids": "family",
        "animation": "animation",
        "suspense": "mystery",
        "feel good": "comedy",
        "fantasy epic": "sci-fi & fantasy",
    }

    @staticmethod
    def normalize(name: str, media_type: str = "movie") -> str:
        name = name.strip().lower()
        if media_type == "tv":
            normalized = GenreNormalizer.TV_GENRE_ALIASES.get(name, name)
        else:
            normalized = GenreNormalizer.GENRE_ALIASES.get(name, name)
        if name != normalized:
            print(
                f"🎭 Normalized genre alias for {media_type}: '{name}' → '{normalized}'")
        return normalized


class ParameterMapper:
    VALUE_PARAM_MAP = {
        # Numeric filters
        "rating": ("vote_average.gte", float),
        "votes": ("vote_count.gte", int),
        "runtime": ("with_runtime.gte", int),
        "runtime_max": ("with_runtime.lte", int),
        "year": ("primary_release_year", int),
        "date": ("primary_release_year", int),

        # Language and region filters
        "language": ("with_original_language", str),
        "country": ("region", str),
        "language_name": ("with_original_language", str),

        # Optional advanced date filters
        "release_after": ("primary_release_date.gte", str),
        "release_before": ("primary_release_date.lte", str),
    }

    @staticmethod
    def inject_parameters_from_entities(query_entities: list, step_parameters: dict) -> None:
        for ent in query_entities:
            ent_type = ent.get("type")
            ent_value = ent.get("name", "").strip()

            mapping = ParameterMapper.VALUE_PARAM_MAP.get(ent_type)
            if not mapping:
                continue

            param_name, cast_fn = mapping
            try:
                if ent_value:
                    step_parameters[param_name] = cast_fn(ent_value)
                    print(
                        f"✅ Injected {param_name} = {step_parameters[param_name]}")
            except ValueError:
                print(
                    f"⚠️ Failed to parse value '{ent_value}' for param '{param_name}'")


# --- Dynamic Entity → Parameter Map for Phase 2 ---

# --- Load param_to_entity_map.json ---
param_to_entity_map_path = os.path.join("data", "param_to_entity_map.json")

with open(param_to_entity_map_path, "r", encoding="utf-8") as f:
    PARAM_TO_ENTITY_MAP = json.load(f)

# --- Reverse mapping: ENTITY → list of PARAMS ---
ENTITY_TO_PARAM_MAP = {}

for param, entity in PARAM_TO_ENTITY_MAP.items():
    if entity not in ENTITY_TO_PARAM_MAP:
        ENTITY_TO_PARAM_MAP[entity] = []
    ENTITY_TO_PARAM_MAP[entity].append(param)

# --- Function to resolve best param for a given entity ---


def resolve_parameter_for_entity(entity_type: str) -> str:
    """
    Resolve a TMDB parameter for a given entity type.
    Prefer 'with_*' symbolic query parameters first.
    Then fallback to '*_id' path slot parameters.
    """
    candidates = ENTITY_TO_PARAM_MAP.get(entity_type)

    if not candidates:
        return None

    # ✅ First priority: parameters starting with 'with_'
    for param in candidates:
        if param.startswith("with_"):
            return param

    # ✅ Second priority: parameters ending with '_id'
    for param in candidates:
        if param.endswith("_id"):
            return param

    # ✅ Fallback: first available candidate
    return candidates[0]


def normalize_entity_type(entity_type: str) -> str:
    """
    Normalize entity type strings.
    Currently a no-op but ready for future extensions.
    """
    return entity_type.lower().strip()


# Cached map loaded from file
_param_to_entity_map = None


def load_param_to_entity_map(path="data/param_to_entity_map.json"):
    global _param_to_entity_map
    if _param_to_entity_map is None:
        with open(path, "r") as f:
            _param_to_entity_map = json.load(f)
    return _param_to_entity_map

# phase 21.5


def get_param_key_for_type(type_, prefer="default"):
    param_map = load_param_to_entity_map()
    for param, ent_type in param_map.items():
        if ent_type == type_:
            if prefer != "default" and param.startswith(prefer):
                return param
    for param, ent_type in param_map.items():
        if ent_type == type_:
            return param  # fallback
    return type_  # final fallback


def update_symbolic_registry(entity: dict, registry: dict, *, credits=None, keywords=None, release_info=None, watch_providers=None):
    """
    Symbolically enrich a TMDB entity (movie or TV show) into the constraint registry for filtering.

    - Handles cast/crew/person roles
    - Handles genres, companies, networks
    - Handles keywords, certification, providers
    - Handles language, region, runtime, year
    - Supports both full enrichment input and direct fallback via *_ids
    """
    if not isinstance(entity, dict):
        return

    # 🔒 Ensure movie_id and tv_id are always dict[str → set[int]]
    for key in ["movie_id", "tv_id"]:
        current = registry.get(key)
        if not isinstance(current, dict):
            print(f"❌ CORRUPTED {key} registry — resetting to dict.")
            registry[key] = {}
        elif isinstance(current, set):
            print(f"⚠️ Found malformed set in {key}. Wrapping into dict.")
            registry[key] = {str(v): {v} for v in current}

    print(
        f"🔍 Registry types: movie_id={type(registry.get('movie_id'))}, tv_id={type(registry.get('tv_id'))}")

    media_type = entity.get("media_type") or (
        "tv" if "first_air_date" in entity else "movie")
    entity_id = entity.get("id")

    if not entity_id:
        return

    # ✅ NEW: Fallback person indexing from _actor_id
    if "_actor_id" in entity:
        actor_id = entity["_actor_id"]
        entity_id = entity.get("id")

        if isinstance(actor_id, list):  # 🛠 fix nested list
            actor_id = actor_id[0]
        actor_id = str(actor_id)

        if entity_id:
            registry.setdefault("with_people", {}).setdefault(
                actor_id, set()).add(entity_id)
            registry.setdefault("cast", {}).setdefault(
                actor_id, set()).add(entity_id)
            print(f"✅ Explicit cast[{actor_id}] → {entity_id}")
        else:
            print(
                f"⚠️ Skipped cast indexing: entity missing ID for actor {actor_id}")

    # ✅ Ensure self-indexing of ID (for constraint tree matching)
    enrich_media_id(entity, registry)

    # 🔹 Preferred enrichment paths (external API results passed in)
    enrich_person_roles(entity, credits, registry, media_type)
    enrich_genres(entity, registry, media_type)
    enrich_networks(entity, registry)
    enrich_companies(entity, registry)
    enrich_keywords(entity, keywords, registry)
    enrich_certification(entity, release_info, registry)
    enrich_language(entity, registry)
    enrich_region(entity, registry)
    enrich_runtime(entity, registry)
    enrich_year(entity, registry)
    enrich_watch_providers(entity, watch_providers, registry)

    if "4495" not in registry.get("cast", {}):
        print("❌ cast_4495 not indexed in registry!")

    # 🔁 Direct fallback indexing from *_ids if enrich_* failed or wasn't used
    def fallback_index(entity, field_key, registry_key):
        """
        Generic fallback: if entity[field_key] is present (either list or scalar), index it under registry_key
        """
        values = entity.get(field_key)

        if not values:
            print(f"🔕 No values for {field_key} in {entity.get('id')}")
            return

        if not isinstance(values, list):
            values = [values]  # ✅ wrap single int/str in list

        for v in values:
            if v is None:
                continue
            print(f"✅ Indexing fallback: {registry_key}[{v}] → {entity['id']}")
            registry.setdefault(registry_key, {}).setdefault(
                str(v), set()).add(entity["id"])

    fallback_index(entity, "genre_ids", "with_genres")
    fallback_index(entity, "network_ids", "with_networks")
    fallback_index(entity, "production_company_ids", "with_companies")
    fallback_index(entity, "keyword_ids", "with_keywords")

    # 🔁 Language fallback
    lang = entity.get("original_language")
    if lang:
        registry.setdefault("with_original_language", {}).setdefault(
            lang, set()).add(entity_id)

    # 🔁 Region fallback (if present)
    if "origin_country" in entity:
        origin_val = entity["origin_country"]
        regions = origin_val if isinstance(origin_val, list) else [origin_val]
        for region in regions:
            registry.setdefault("with_origin_country", {}).setdefault(
                region, set()).add(entity_id)

    # 🔁 Year fallback (extract from release_date or first_air_date)
    date_field = entity.get("release_date") or entity.get("first_air_date")
    if date_field and len(date_field) >= 4:
        year = date_field[:4]
        registry.setdefault("primary_release_year", {}).setdefault(
            year, set()).add(entity_id)

    debug_keys = list(registry.keys())
    entity_title = entity.get("title") or entity.get("name") or "Unknown"
    entity_id = entity.get("id")
    print(
        f"🧩 Enriched {entity_title} (ID: {entity_id}) with keys: {sorted(debug_keys)}")
    for key in sorted(registry):
        subkeys = registry[key]
        for subk, ids in subkeys.items():
            # ✅ Defensive: only check `in` if ids is a set
            if isinstance(ids, set) and str(entity_id) in ids:
                print(f"   ↳ {key}[{subk}]: matched ID {entity_id}")


def enrich_person_roles(entity, credits, registry, media_type):
    if not credits:
        return

    eid = entity.get("id")
    if not eid:
        return

    # ✅ Cast roles
    for person in credits.get("cast", []):
        _add_role_index(registry, eid, person, "cast")

    # ✅ Crew roles
    for person in credits.get("crew", []):
        job = person.get("job", "").lower()
        role = _map_job_to_symbolic_role(job)
        _add_role_index(registry, eid, person, role)


def _add_role_index(registry, eid, person, role=None):
    pid = person.get("id")
    if not pid:
        return
    pid = str(pid)
    if role:
        registry.setdefault(role, {}).setdefault(pid, set()).add(eid)
    registry.setdefault("with_people", {}).setdefault(pid, set()).add(eid)


def _map_job_to_symbolic_role(job):
    if job == "director":
        return "director"
    elif job in {"writer", "screenplay"}:
        return "writer"
    elif "producer" in job:
        return "producer"
    elif any(k in job for k in {"composer", "music", "score"}):
        return "composer"
    return None  # other crew jobs are still indexed in with_people


def enrich_genres(entity, registry, media_type):
    genre_ids = entity.get("genre_ids")
    if not genre_ids:
        genre_ids = [g["id"] for g in entity.get("genres", [])]
    elif not isinstance(genre_ids, list):
        genre_ids = [genre_ids]

    for gid in genre_ids:
        registry.setdefault("with_genres", {}).setdefault(
            str(gid), set()).add(entity["id"])
        print(f"✅ Indexing genre_id[{gid}] → {entity['id']}")


def enrich_networks(entity, registry):
    networks = entity.get("networks", [])
    if isinstance(networks, dict):
        networks = [networks]

    for n in networks:
        nid = n.get("id")
        if nid:
            registry.setdefault("with_networks", {}).setdefault(
                str(nid), set()).add(entity["id"])


def enrich_companies(entity, registry):
    companies = entity.get("production_companies", [])
    if isinstance(companies, dict):
        companies = [companies]

    for c in companies:
        cid = c.get("id")
        if cid:
            registry.setdefault("with_companies", {}).setdefault(
                str(cid), set()).add(entity["id"])


def enrich_keywords(entity, keywords, registry):
    if not keywords:
        return

    keyword_list = []
    if isinstance(keywords, dict) and isinstance(keywords.get("keywords"), list):
        keyword_list = keywords["keywords"]

    for kw in keyword_list:
        kid = kw.get("id")
        if kid:
            registry.setdefault("with_keywords", {}).setdefault(
                str(kid), set()).add(entity["id"])


def enrich_certification(entity, release_info, registry):
    if not release_info:
        return
    for entry in release_info.get("results", []):
        for release in entry.get("release_dates", []):
            cert = release.get("certification")
            if cert:
                registry.setdefault("certification", {}).setdefault(
                    cert, set()).add(entity["id"])


def enrich_language(entity, registry):
    lang = entity.get("original_language")
    if lang:
        registry.setdefault("with_original_language", {}).setdefault(
            lang, set()).add(entity["id"])


def enrich_region(entity, registry):
    country = entity.get("origin_country") or []
    for r in country:
        registry.setdefault("region", {}).setdefault(
            r, set()).add(entity["id"])


def enrich_runtime(entity, registry):
    runtime = entity.get("runtime")
    if runtime:
        registry.setdefault("runtime", {}).setdefault(
            str(runtime), set()).add(entity["id"])


def enrich_year(entity, registry):
    date = entity.get("release_date") or entity.get("first_air_date")
    if date and len(date) >= 4:
        year = date[:4]
        registry.setdefault("primary_release_year", {}).setdefault(
            year, set()).add(entity["id"])


def enrich_watch_providers(entity, watch_providers, registry):
    if not watch_providers:
        return
    for key in ["flatrate", "ads", "buy", "rent"]:
        for p in watch_providers.get("results", {}).get("US", {}).get(key, []):
            pid = p.get("provider_id")
            if pid:
                registry.setdefault("with_watch_providers", {}).setdefault(
                    str(pid), set()).add(entity["id"])


def enrich_media_id(entity: dict, registry: dict):
    """
    Ensure the entity is indexed under its own media ID:
    - movie_id["27205"] = {27205}
    - tv_id["12345"] = {12345}
    """
    entity_id = entity.get("id")
    if not entity_id:
        return

    media_type = entity.get("media_type") or (
        "tv" if "first_air_date" in entity else "movie")

    key = "movie_id" if media_type == "movie" else "tv_id"
    registry.setdefault(key, {}).setdefault(
        str(entity_id), set()).add(entity_id)
    print(f"✅ Indexed {key}[{entity_id}] → {entity_id}")


def enrich_symbolic_registry(movie, registry, *, credits=None, keywords=None, release_info=None, watch_providers=None):
    """
    Robust wrapper to symbolically index a movie into the constraint registry.
    Supports full enrichment from credits, keywords, release dates, and providers.

    Args:
        movie (dict): The movie or TV entity dictionary.
        registry (dict): The symbolic constraint registry (e.g., state.data_registry).
        credits (dict, optional): Result of /movie/{id}/credits or /tv/{id}/credits.
        keywords (list, optional): List of keyword objects from /keywords endpoint.
        release_info (dict, optional): Certification data from /release_dates.
        watch_providers (dict, optional): Data from /watch/providers.
    """

    # 🧩 Index this result into the symbolic registry using its credits metadata.
    # This enables symbolic filtering and constraint matching across roles, genres, companies, etc.
    # e.g., updates: with_people[6193] → movie_ids, director[1032] → movie_ids
    try:
        update_symbolic_registry(
            entity=movie,
            registry=registry,
            credits=credits,
            keywords=keywords,
            release_info=release_info,
            watch_providers=watch_providers
        )
    except Exception as e:
        print(
            f"⚠️ Failed symbolic indexing for movie ID {movie.get('id')}: {e}")


def enrich_symbolic_fields(summary: dict, state) -> dict:
    """Enrich a fallback summary with missing symbolic metadata via /tv/{id} or /movie/{id}"""
    media_type = summary.get("media_type")
    if media_type not in ("tv", "movie"):
        return summary

    endpoint = f"/{media_type}/{summary['id']}"
    try:
        res = requests.get(f"{state.base_url}{endpoint}",
                           headers=state.headers)
        if res.status_code != 200:
            return summary
        details = res.json()
    except Exception as e:
        print(f"⚠️ Failed to enrich {media_type}/{summary['id']}: {e}")
        return summary

    # Patch genre_ids
    if not summary.get("genre_ids") and "genres" in details:
        summary["genre_ids"] = [g["id"] for g in details.get("genres", [])]

    # Patch network_ids
    if media_type == "tv" and not summary.get("network_ids") and "networks" in details:
        summary["network_ids"] = [n["id"] for n in details["networks"]]

    # Patch production_company_ids
    if media_type == "movie" and not summary.get("production_company_ids") and "production_companies" in details:
        summary["production_company_ids"] = [c["id"]
                                             for c in details["production_companies"]]

    # Patch original_language
    if not summary.get("original_language") and "original_language" in details:
        summary["original_language"] = details["original_language"]

    return summary
