class ParamResolver:
    GENRE_MAP = {
        "action": 28, "adventure": 12, "animation": 16,
        "comedy": 35, "crime": 80, "documentary": 99,
        "drama": 18, "family": 10751, "fantasy": 14,
        "history": 36, "horror": 27, "music": 10402,
        "mystery": 9648, "romance": 10749, "sci-fi": 878,
        "tv movie": 10770, "thriller": 53, "war": 10752,
        "western": 37
    }

    TIME_WINDOWS = {"day": "day", "week": "week"}

    def resolve(self, entity_type: str, value: str) -> int:
        """Resolve parameters to TMDB IDs"""
        if entity_type == "genre":
            return self.GENRE_MAP.get(value.lower())
        return None