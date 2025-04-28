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
            print(f"🎭 Normalized genre alias for {media_type}: '{name}' → '{normalized}'")
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
                    print(f"✅ Injected {param_name} = {step_parameters[param_name]}")
            except ValueError:
                print(f"⚠️ Failed to parse value '{ent_value}' for param '{param_name}'")

                
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
