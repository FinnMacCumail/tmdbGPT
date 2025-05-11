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


def update_symbolic_registry(entity: dict, registry: dict, credits: dict = None, keywords: list = None, release_info: dict = None, watch_providers: dict = None):
    """
    Updates symbolic registry with known symbolic constraints.
    - credits: from /movie/{id}/credits
    - keywords: from /movie/{id}/keywords
    - release_info: from /movie/{id}/release_dates
    - watch_providers: from /movie/{id}/watch/providers
    """
    entity_id = entity.get("id")
    if not entity_id:
        return

    # genre
    for gid in entity.get("genre_ids", []):
        registry.setdefault("with_genres", {}).setdefault(
            str(gid), set()).add(entity_id)

    # company
    for comp in entity.get("production_companies", []):
        cid = comp.get("id")
        if cid:
            registry.setdefault("with_companies", {}).setdefault(
                str(cid), set()).add(entity_id)

    # network
    for net in entity.get("networks", []):
        nid = net.get("id")
        if nid:
            registry.setdefault("with_networks", {}).setdefault(
                str(nid), set()).add(entity_id)

    # language
    lang = entity.get("original_language")
    if lang:
        registry.setdefault("with_original_language", {}).setdefault(
            str(lang), set()).add(entity_id)

    # country
    for country in entity.get("origin_country", []):
        registry.setdefault("watch_region", {}).setdefault(
            str(country), set()).add(entity_id)

    # person indexing from credits
    if credits:
        for cast_member in credits.get("cast", []):
            pid = cast_member.get("id")
            if pid:
                registry.setdefault("with_people", {}).setdefault(
                    str(pid), set()).add(entity_id)
                registry.setdefault("with_cast", {}).setdefault(
                    str(pid), set()).add(entity_id)
        for crew_member in credits.get("crew", []):
            pid = crew_member.get("id")
            job = crew_member.get("job", "").lower()
            if pid:
                registry.setdefault("with_people", {}).setdefault(
                    str(pid), set()).add(entity_id)
                registry.setdefault("with_crew", {}).setdefault(
                    str(pid), set()).add(entity_id)
                if job == "director":
                    registry.setdefault("with_crew_director", {}).setdefault(
                        str(pid), set()).add(entity_id)

    # keywords
    if keywords:
        for kw in keywords:
            kid = kw.get("id")
            if kid:
                registry.setdefault("with_keywords", {}).setdefault(
                    str(kid), set()).add(entity_id)

    # certifications
    if release_info:
        results = release_info.get("results", [])
        for region_block in results:
            country_code = region_block.get("iso_3166_1")
            for release in region_block.get("release_dates", []):
                cert = release.get("certification")
                if cert:
                    registry.setdefault("certification", {}).setdefault(
                        cert, set()).add(entity_id)
                if country_code:
                    registry.setdefault("certification_country", {}).setdefault(
                        country_code, set()).add(entity_id)

    # watch providers and monetization types
    if watch_providers:
        results = watch_providers.get("results", {})
        for country_code, info in results.items():
            registry.setdefault("watch_region", {}).setdefault(
                country_code, set()).add(entity_id)
            for m_type in ["flatrate", "buy", "rent", "ads"]:
                if m_type in info:
                    registry.setdefault("with_watch_monetization_types", {}).setdefault(
                        m_type, set()).add(entity_id)
                    for provider in info[m_type]:
                        pid = provider.get("provider_id")
                        if pid:
                            registry.setdefault("with_watch_providers", {}).setdefault(
                                str(pid), set()).add(entity_id)

    # Numeric/date value fields for .gte/.lte filtering
    if "vote_average" in entity:
        registry.setdefault("vote_average", {})[
            entity_id] = entity["vote_average"]
    if "vote_count" in entity:
        registry.setdefault("vote_count", {})[entity_id] = entity["vote_count"]
    if "release_date" in entity:
        registry.setdefault("release_date", {})[
            entity_id] = entity["release_date"]
    if "first_air_date" in entity:
        registry.setdefault("first_air_date", {})[
            entity_id] = entity["first_air_date"]

# core/entity/symbolic_indexer.py


def enrich_symbolic_registry(
    movie: dict,
    registry: dict,
    *,
    credits: dict = None,
    keywords: list = None,
    release_info: dict = None,
    watch_providers: dict = None
) -> None:
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
