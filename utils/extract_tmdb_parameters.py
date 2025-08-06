import json
import re
from collections import defaultdict

# Load the TMDB schema
with open("data/tmdb.json") as f:
    schema = json.load(f)

param_index = {}

# Extract declared parameters
for path, methods in schema.get("paths", {}).items():
    for method, config in methods.items():
        if not isinstance(config, dict):
            continue

        for param in config.get("parameters", []):
            pname = param.get("name")
            if not pname:
                continue

            if pname not in param_index:
                param_index[pname] = {
                    "name": pname,
                    "description": param.get("description", ""),
                    "in": param.get("in", "query"),
                    "used_in": set()
                }

            param_index[pname]["used_in"].add(path)

# Detect path-style parameters from the endpoint path strings
for path in schema.get("paths", {}):
    path_params = re.findall(r"{(.*?)}", path)
    for pname in path_params:
        if pname not in param_index:
            param_index[pname] = {
                "name": pname,
                "description": "(extracted from path)",
                "in": "path",
                "used_in": set()
            }
        param_index[pname]["used_in"].add(path)

# Map known path parameters to entity types
ENTITY_PARAM_MAP = {
    "person_id": "person",
    "movie_id": "movie",
    "tv_id": "tv",
    "collection_id": "collection",
    "company_id": "company",
    "review_id": "review",
    "network_id": "network",
    "season_number": "season",
    "episode_number": "episode"
}

# Convert sets to lists and apply entity type annotations
parameter_data = []
for p in param_index.values():
    p["used_in"] = sorted(list(p["used_in"]))
    if p["name"] in ENTITY_PARAM_MAP:
        p["entity_type"] = ENTITY_PARAM_MAP[p["name"]]
    parameter_data.append(p)

for p in parameter_data[:10]:
    entity = f" (entity: {p['entity_type']})" if "entity_type" in p else ""

# Save output
with open("data/tmdb_parameters.json", "w") as f:
    json.dump(parameter_data, f, indent=2)

