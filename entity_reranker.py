class EntityAwareReranker:
    @staticmethod
    def boost_by_entity_mentions(matches: list, query_entities: list, boost_weight: float = 0.1) -> list:
        """
        Boost final_score if endpoint metadata includes one or more query_entities.

        Args:
            matches (list): list of semantic_retrieval match results
            query_entities (list): entities from extraction_result["query_entities"]
            boost_weight (float): bonus to apply per matching entity mention

        Returns:
            list: boosted and re-sorted match list
        """
        for m in matches:
            mentions = 0
            for q in query_entities:
                name = q.get("name") if isinstance(q, dict) else q
                if isinstance(name, str) and name.lower() in str(m.get("entities", [])).lower():
                    mentions += 1
            m["final_score"] += mentions * boost_weight
            m["final_score"] = round(m["final_score"], 3)

        return sorted(matches, key=lambda x: x["final_score"], reverse=True)