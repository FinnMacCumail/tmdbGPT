from core.formatting.registry import register_renderer
from typing import List
import re


def generate_explanation(extraction_result) -> str:
    query_entities = extraction_result.get("query_entities", [])
    entity_descriptions = [e.get("name")
                           for e in query_entities if e.get("name")]
    if entity_descriptions:
        return f"No results found for: {', '.join(entity_descriptions)}"
    return "No relevant information found."


@register_renderer("fallback")
def format_fallback(state) -> dict:
    explanation = generate_explanation(state.extraction_result)
    return {
        "response_format": "summary",
        "question_type": state.question_type or "summary",
        "text": f"⚠️ {explanation}",
        "entries": [f"⚠️ {explanation}"]
    }


@register_renderer("count_summary")
def format_count_summary(state) -> dict:
    movie_count = 0
    tv_count = 0

    entity = state.extraction_result.get("query_entities", [{}])[0]
    name = entity.get("name", "This person")
    role = entity.get("role", "director").lower()
    role_label = {
        "cast": "actor", "actor": "actor",
        "director": "director",
        "writer": "screenwriter",
        "producer": "producer",
        "composer": "composer"
    }.get(role, role)

    for r in state.responses:
        if not isinstance(r, dict):
            continue
        if r.get("type") != "movie_summary":
            continue
        src = r.get("source", "")
        job = r.get("job", "").lower()
        if "movie_credits" in src and job == "director":
            movie_count += 1
        elif "tv_credits" in src and job == "director":
            tv_count += 1

    text = f"🎬 {name} worked as a {role_label} in {movie_count} movie(s) and {tv_count} TV show(s)."
    return {
        "response_format": "count_summary",
        "question_type": "count",
        "entity": name,
        "text": text,
        "entries": [text]
    }


@register_renderer("ranked_list")
def format_ranked_list(state, include_debug=False):
    formatted = []
    for idx, item in enumerate(state.responses):
        if not isinstance(item, dict):
            continue
        line = f"{idx+1}. {item.get('title') or item.get('name')}"
        if include_debug and "_provenance" in item:
            prov = item["_provenance"]
            matched = ', '.join(prov.get("matched_constraints", []))
            relaxed = ', '.join(prov.get("relaxed_constraints", []))
            validated = ', '.join(prov.get("post_validations", []))
            line += f"  [matched: {matched} | relaxed: {relaxed} | validated: {validated}]"
        formatted.append(line)
    return formatted


@register_renderer("summary")
def format_summary(state) -> dict:
    responses = state.responses
    question_type = state.extraction_result.get("question_type", "summary")
    entries = []
    
    # For fact questions, try to detect the specific fact type being asked
    query_text = getattr(state, 'input', '').lower() if question_type == "fact" else ""

    SUMMARY_TYPES = {
        "movie_summary": "🎬",
        "tv_summary": "📺",
        "person_profile": "👤",
        "company_profile": "🏢",
        "network_profile": "📡",
        "collection_profile": "🎞️",
    }

    for r in responses:
        if not isinstance(r, dict):
            continue

        r_type = r.get("type")
        emoji = SUMMARY_TYPES.get(r_type, "🔹")
        title = r.get("title") or r.get("name", "Untitled")
        overview = r.get("overview") or r.get(
            "biography") or "No summary available."

        # 👤 Person profile rendering
        if r_type == "person_profile":
            name = r.get("name", "Unknown")
            bio = r.get("biography", "No biography available.")
            known_for = r.get("known_for")
            birthday = r.get("birthday")

            person_entry = f"{emoji} {name}: {bio.strip()}"
            if known_for:
                person_entry += f"\n🧠 Known For: {known_for}"
            if birthday:
                person_entry += f"\n🎂 Birthday: {birthday}"

            entries.append(person_entry)
            continue

        # 🧠 Fact-style rendering for known roles
        if question_type == "fact":
            # Enhanced fact extraction based on query intent
            
            # Detect specific fact type being requested
            is_year_question = any(keyword in query_text for keyword in ["year", "when", "released", "came out", "aired"])
            is_runtime_question = any(keyword in query_text for keyword in ["long", "runtime", "duration", "minutes", "hours"])
            is_genre_question = any(keyword in query_text for keyword in ["genre", "type of", "kind of", "category"])
            is_director_question = any(keyword in query_text for keyword in ["direct", "director"])
            is_budget_question = any(keyword in query_text for keyword in ["budget", "cost", "money", "expensive"])
            is_creator_question = any(keyword in query_text for keyword in ["create", "creator", "made by"])
            
            # Try to extract specific facts based on available data and query intent
            fact_extracted = False
            
            # 🎬 Movie facts
            if r_type == "movie_summary":
                # Prioritize based on query intent
                if is_runtime_question:
                    runtime = r.get("runtime")
                    if runtime and isinstance(runtime, (int, float)) and runtime > 0:
                        entries.append(f"{emoji} {title} has a runtime of {runtime} minutes.")
                        fact_extracted = True
                        continue
                
                if is_genre_question:
                    genres = r.get("genres")
                    if genres and isinstance(genres, list) and len(genres) > 0:
                        genre_names = [g.get("name") if isinstance(g, dict) else str(g) for g in genres]
                        genre_names = [g for g in genre_names if g]  # Filter out empty values
                        if genre_names:
                            genre_text = ", ".join(genre_names)
                            entries.append(f"{emoji} {title} is a {genre_text} film.")
                            fact_extracted = True
                            continue
                
                if is_director_question:
                    if r.get("directors"):
                        directors = ", ".join(r["directors"])
                        entries.append(f"{emoji} {title} was directed by {directors}.")
                        fact_extracted = True
                        continue
                
                if is_budget_question:
                    budget = r.get("budget")
                    if budget and isinstance(budget, (int, float)) and budget > 0:
                        entries.append(f"{emoji} {title} had a budget of ${budget:,}.")
                        fact_extracted = True
                        continue
                
                # Year/release date questions (including default for fact questions)
                if is_year_question or not fact_extracted:
                    release_date = r.get("release_date")
                    if release_date and len(release_date) >= 4:
                        year = release_date[:4]
                        entries.append(f"{emoji} {title} was released in {year}.")
                        fact_extracted = True
                        continue

            # 📺 TV facts
            if r_type == "tv_summary":
                # Prioritize based on query intent for TV shows
                if is_creator_question:
                    if r.get("created_by"):
                        creators = [
                            c.get("name") for c in r["created_by"] if c.get("name")
                        ]
                        if creators:
                            entries.append(
                                f"{emoji} {title} was created by {', '.join(creators)}.")
                            fact_extracted = True
                            continue
                
                # Episode/season count facts
                is_episodes_question = any(keyword in query_text for keyword in ["episodes", "seasons", "how many", "number of"])
                if is_episodes_question:
                    number_of_seasons = r.get("number_of_seasons")
                    number_of_episodes = r.get("number_of_episodes")
                    if number_of_seasons and isinstance(number_of_seasons, (int, float)):
                        if number_of_episodes and isinstance(number_of_episodes, (int, float)):
                            entries.append(f"{emoji} {title} has {number_of_seasons} seasons and {number_of_episodes} episodes.")
                        else:
                            entries.append(f"{emoji} {title} has {number_of_seasons} seasons.")
                        fact_extracted = True
                        continue
                
                # Year/first air date questions (including default for TV fact questions)
                if is_year_question or not fact_extracted:
                    first_air_date = r.get("first_air_date")
                    if first_air_date and len(first_air_date) >= 4:
                        year = first_air_date[:4]
                        entries.append(f"{emoji} {title} first aired in {year}.")
                        fact_extracted = True
                        continue

            # Continue to next response if we extracted a specific fact
            
            # If we extracted a specific fact, don't fall through to default rendering
            if fact_extracted:
                continue

        # 🧾 Default rendering
        entries.append(f"{emoji} {title}: {overview.strip()}")

    return {
        "response_format": "summary",
        "question_type": question_type,
        "entries": entries or ["⚠️ No summary available."]
    }


@register_renderer("timeline")
def format_timeline(state) -> dict:
    entries = []
    for item in state.responses:
        if not isinstance(item, dict) or item.get("type") != "movie_summary":
            continue
        title = item.get("title", "Untitled")
        overview = item.get("overview", "No synopsis available.")
        source = item.get("source", "")
        score = item.get("final_score", 1.0)
        year = item.get("release_date", "")[
            :4] if "release_date" in item else None
        if not year:
            match = re.search(r"(19|20)\\d{2}", overview)
            year = match.group(0) if match else None
        entries.append({
            "title": title,
            "overview": overview,
            "release_year": int(year) if year and year.isdigit() else None,
            "source": source,
            "score": score
        })

    entries.sort(key=lambda x: x.get("release_year") or 3000)
    name = state.extraction_result.get("query_entities", [{}])[
        0].get("name", "")
    return {
        "response_format": "timeline",
        "question_type": "timeline",
        "entity": name,
        "entries": entries
    }


@register_renderer("comparison")
def format_comparison(state) -> dict:
    query_entities = state.extraction_result.get("query_entities", [])
    if len(query_entities) != 2:
        return {
            "response_format": "comparison",
            "question_type": "comparison",
            "entries": ["⚠️ Comparison requires exactly two entities."]
        }

    left_id = str(query_entities[0].get("resolved_id"))
    right_id = str(query_entities[1].get("resolved_id"))
    left_name = query_entities[0].get("name", "Entity A")
    right_name = query_entities[1].get("name", "Entity B")

    left_entries = []
    right_entries = []

    for r in state.responses:
        if not isinstance(r, dict) or r.get("type") != "movie_summary":
            continue
        src = r.get("source", "")
        entry = {
            "title": r.get("title", ""),
            "overview": r.get("overview", ""),
            "score": r.get("final_score", 1.0),
            "source": src
        }
        if left_id in src:
            left_entries.append(entry)
        elif right_id in src:
            right_entries.append(entry)

    left_entries.sort(key=lambda x: x["score"], reverse=True)
    right_entries.sort(key=lambda x: x["score"], reverse=True)

    return {
        "response_format": "comparison",
        "question_type": "comparison",
        "left": {"name": left_name, "entries": left_entries[:3]},
        "right": {"name": right_name, "entries": right_entries[:3]}
    }


def generate_relaxation_explanation(dropped_constraints: List[str]) -> str:
    """
    Generate a human-readable explanation of which constraints were relaxed.
    """
    if not dropped_constraints:
        return ""

    pieces = []
    mapping = {
        "with_people": "specific actors",
        "with_networks": "specific networks",
        "with_companies": "specific studios",
        "director_id": "specific directors",
        "with_genres": "specific genres",
        "primary_release_year": "specific release year",
        "vote_average.gte": "minimum rating",
        "with_runtime.gte": "minimum runtime",
        "with_runtime.lte": "maximum runtime"
    }

    for param in dropped_constraints:
        label = mapping.get(param, param)
        pieces.append(f"relaxed {label}")

    explanation = ", ".join(pieces)
    return f"⚠️ Note: Some filters were relaxed to find results ({explanation})."


class QueryExplanationBuilder:
    @staticmethod
    def build_final_explanation(extraction_result, relaxed_parameters: list = None, fallback_used: bool = False) -> str:
        query_entities = extraction_result.get("query_entities", []) or []
        explanation_parts = []

        def describe_entity(ent):
            name = ent.get("name", "")
            ent_type = ent.get("type", "")
            role = ent.get("role", "actor").capitalize()

            if ent_type == "genre":
                return f"{name} genre"
            elif ent_type == "company":
                return f"produced by {name}"
            elif ent_type == "network":
                return f"aired on {name}"
            elif ent_type == "person":
                return f"{role} {name}"
            return name

        if query_entities:
            entity_descriptions = [describe_entity(
                ent) for ent in query_entities if describe_entity(ent)]
            if entity_descriptions:
                applied_summary = " and ".join(entity_descriptions)
                explanation_parts.append(f"Planned for {applied_summary}.")

        if relaxed_parameters:
            relaxed_parameters = sorted(set(relaxed_parameters))
            if relaxed_parameters:
                if len(relaxed_parameters) == 1:
                    relaxed_text = relaxed_parameters[0]
                else:
                    relaxed_text = ", ".join(
                        relaxed_parameters[:-1]) + f" and {relaxed_parameters[-1]}"
                explanation_parts.append(
                    f"Relaxed constraints on {relaxed_text} to find matches.")

        if fallback_used:
            explanation_parts.append(
                "Fallback discovery was used to broaden the search results.")

        if not explanation_parts:
            return "Performed a general search based on available information."

        return " ".join(explanation_parts)
