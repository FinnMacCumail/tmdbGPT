# generate_param_to_entity_map.py (final refactor version)

import json
import os

def generate_param_to_entity_map():
    tmdb_parameters_path = os.path.join("data", "tmdb_parameters.json")
    output_path = os.path.join("data", "param_to_entity_map.json")

    with open(tmdb_parameters_path, "r", encoding="utf-8") as f:
        tmdb_parameters = json.load(f)

    param_to_entity = {}

    # ✅ Define fields that should be EXCLUDED — control/technical params, not filters
    exclusion_list = {
        "page", "sort_by", "include_adult", "query", "timezone",
        "append_to_response", "include_image_language", "include_video",
    }
    # ✅ (Important): we now **include** fields like 'include_null_first_air_dates' and 'screened_theatrically'
    # because they are valid TV show filters — no longer excluded!

    for param in tmdb_parameters:
        name = param.get("name", "").strip()
        entity_type = param.get("entity_type", "").strip()

        if not name:
            continue

        if name in exclusion_list:
            continue  # ❌ Skip irrelevant control parameters

        if entity_type:
            # ✅ If entity_type explicitly defined, use it
            param_to_entity[name] = entity_type
        else:
            # ✅ Otherwise, mark as "general" (value filters, toggles, etc.)
            param_to_entity[name] = "general"

    # ✅ Manual patches for critical fields that TMDB schema might miss
    manual_patches = {
        # Path placeholder fields
        "review_id": "review",
        "credit_id": "credit",
        "season_number": "season",
        "episode_number": "episode",
        
        # Important symbolic entity fields (correct classification)
        "with_people": "person",
        "with_cast": "person",
        "with_crew": "person",
        "with_genres": "genre",
        "without_genres": "genre",
        "with_keywords": "keyword",
        "without_keywords": "keyword",
        "with_companies": "company",
        "without_companies": "company",
        "with_networks": "network",
    }
    param_to_entity.update(manual_patches)

    # Save the final param_to_entity_map.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(param_to_entity, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully wrote {len(param_to_entity)} parameter mappings to {output_path}")

if __name__ == "__main__":
    generate_param_to_entity_map()
