class EntityAwareReranker:
    @staticmethod
    def boost_by_entity_mentions(results: list, query_entities: list) -> list:
        if not results:
            return []

        if not query_entities:
            return results

        boosted = []
        genre_names = [e["name"].lower() for e in query_entities if e.get("type") == "genre"]
        person_names = [e["name"].lower() for e in query_entities if e.get("type") == "person"]
        role_map = {e["name"].lower(): e.get("role") for e in query_entities if e.get("type") == "person"}

        for movie in results:
            score = movie.get("final_score", 0)

            text = (movie.get("title", "") + " " + movie.get("overview", "")).lower()
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
