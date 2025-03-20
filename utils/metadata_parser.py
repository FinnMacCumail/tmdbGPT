from typing import List, Dict

class MetadataParser:
    @staticmethod
    def parse_list(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()] if value else []

    @staticmethod
    def parse_boolean(value: str) -> bool:
        return value.lower() == "true" if value else False

    @staticmethod
    def parse_filters(metadata: Dict) -> Dict[str, bool]:
        return {
            "temporal": MetadataParser.parse_boolean(metadata.get("filter_temporal", "False")),
            "genre": MetadataParser.parse_boolean(metadata.get("filter_genre", "False")),
            "relationships": MetadataParser.parse_boolean(metadata.get("filter_relationships", "False"))
        }