class EntityAwareReranker:
    @staticmethod
    def boost_by_entity_mentions(results: list, query_entities: list) -> list:
        if not results:
            return []

        if not query_entities:
            return results

        boosted = []
        genre_names = [e["name"].lower()
                       for e in query_entities if e.get("type") == "genre"]
        person_names = [e["name"].lower()
                        for e in query_entities if e.get("type") == "person"]
        role_map = {e["name"].lower(): e.get("role")
                    for e in query_entities if e.get("type") == "person"}

        for movie in results:
            score = movie.get("final_score", 0)

            text = (movie.get("title", "") + " " +
                    movie.get("overview", "")).lower()
            matched = False

            # 🎯 Boost for genres
            for genre in genre_names:
                if genre in text:
                    score += 0.3
                    matched = True

            # 🎯 Boost for people (cast/director)
            for person in person_names:
                if person in text:
                    role = role_map.get(person, "")
                    if role == "cast":
                        score += 0.5
                    elif role == "director":
                        score += 0.5
                    else:
                        score += 0.3
                    matched = True

            if matched:
                movie["final_score"] = score

            boosted.append(movie)

        boosted.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return boosted


class RoleAwareReranker:
    @staticmethod
    def boost_matches_by_role(ranked_matches, extraction_result, intended_media_type):
        """
        Boost or demote endpoint matches based on entity role, media type,
        and endpoint alignment. Updates ranked_matches in place.
        """
        query_entities = extraction_result.get("query_entities", [])
        if len(query_entities) != 1:
            return  # only apply for single-person queries

        entity = query_entities[0]
        if entity.get("type") != "person":
            return

        role = entity.get("role", "cast")  # default to cast
        media = intended_media_type or "movie"

        for match in ranked_matches:
            endpoint = match.get("endpoint", "")
            params = match.get("parameters", {})

            # Prefer direct credit lookups
            if f"/person/{{person_id}}/{media}_credits" in endpoint:
                match["_boost_score"] = 0.4

            # Prefer discover endpoints with proper media and with_people
            if role == "cast" and "with_people" in params:
                if media == "tv" and endpoint == "/discover/tv":
                    match["_boost_score"] = 0.3
                elif media == "movie" and endpoint == "/discover/movie":
                    match["_boost_score"] = 0.3

            # Demote cross-media discover endpoints
            if role == "cast" and media == "tv" and endpoint == "/discover/movie":
                match["_demote_score"] = 0.1
            if role == "cast" and media == "movie" and endpoint == "/discover/tv":
                match["_demote_score"] = 0.1

        return ranked_matches
