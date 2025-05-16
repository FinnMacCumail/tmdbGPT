import json

import os


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
    """
    if not isinstance(entity, dict):
        return

    media_type = entity.get("media_type") or (
        "tv" if "first_air_date" in entity else "movie")
    entity_id = entity.get("id")

    if not entity_id:
        return

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

    debug_keys = list(registry.keys())
    entity_title = entity.get("title") or entity.get("name") or "Unknown"
    entity_id = entity.get("id")
    print(
        f"🧩 Enriched {entity_title} (ID: {entity_id}) with keys: {sorted(debug_keys)}")
    for key in sorted(registry):
        subkeys = registry[key]
        for subk, ids in subkeys.items():
            if entity_id and str(entity_id) in ids:
                print(f"   ↳ {key}[{subk}]: matched ID {entity_id}")


def enrich_person_roles(entity, credits, registry, media_type):
    if not credits:
        return
    cast = credits.get("cast", [])
    crew = credits.get("crew", [])
    for person in cast:
        pid = person.get("id")
        if pid:
            registry.setdefault("with_people", {}).setdefault(
                str(pid), set()).add(entity["id"])
    for person in crew:
        pid = person.get("id")
        job = person.get("job", "").lower()
        if pid:
            registry.setdefault("with_people", {}).setdefault(
                str(pid), set()).add(entity["id"])
            if job in {"director", "writer", "screenplay", "producer", "composer"}:
                registry.setdefault(job, {}).setdefault(
                    str(pid), set()).add(entity["id"])


def enrich_genres(entity, registry, media_type):
    genre_ids = entity.get("genre_ids") or [
        g["id"] for g in entity.get("genres", [])]
    for gid in genre_ids:
        registry.setdefault("with_genres", {}).setdefault(
            str(gid), set()).add(entity["id"])


def enrich_networks(entity, registry):
    for n in entity.get("networks", []):
        nid = n.get("id")
        if nid:
            registry.setdefault("with_networks", {}).setdefault(
                str(nid), set()).add(entity["id"])


def enrich_companies(entity, registry):
    for c in entity.get("production_companies", []):
        cid = c.get("id")
        if cid:
            registry.setdefault("with_companies", {}).setdefault(
                str(cid), set()).add(entity["id"])


def enrich_keywords(entity, keywords, registry):
    if not keywords:
        return
    for kw in keywords.get("keywords", []):
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
