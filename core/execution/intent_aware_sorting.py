# core/execution/intent_aware_sorting.py

from typing import Optional, Dict, Any


class IntentAwareSorting:
    """
    Implements intent-aware sorting logic based on user query patterns.
    
    This class analyzes user queries for temporal, quality, and other sorting intents
    and returns appropriate TMDB API sort parameters.
    """
    
    # Intent keyword mapping
    TEMPORAL_RECENT_KEYWORDS = ["latest", "recent", "newest", "new", "current", "most recent"]
    TEMPORAL_CHRONOLOGICAL_KEYWORDS = ["first", "earliest", "debut", "initial", "original", "chronological"]
    
    QUALITY_HIGH_KEYWORDS = ["highest rated", "best rated", "top rated", "highly rated", "best reviewed", "greatest", "highest-rated"]
    QUALITY_LOW_KEYWORDS = ["worst", "bad", "terrible", "lowest rated", "poorly rated", "awful", "horrible"]
    
    POPULARITY_KEYWORDS = ["popular", "famous", "well-known", "mainstream", "blockbuster"]
    
    @classmethod
    def determine_sort_strategy(cls, query: str, question_type: str = None) -> Optional[Dict[str, str]]:
        """
        Analyze query text and determine appropriate sorting strategy.
        
        Args:
            query (str): User's query text
            question_type (str): Type of question (list, fact, count, etc.)
            
        Returns:
            Dict with sort parameters or None if no special sorting needed
        """
        query_lower = query.lower()
        
        # 🔧 DEBUG: Intent analysis
        print(f"🧠 DEBUG: Analyzing intent for query: {query}")
        
        # Check for temporal intents first (highest specificity)
        temporal_sort = cls._check_temporal_intent(query_lower)
        if temporal_sort:
            print(f"🕒 DEBUG: Temporal intent detected - {temporal_sort}")
            return temporal_sort
        
        # Check for quality/rating intents  
        quality_sort = cls._check_quality_intent(query_lower)
        if quality_sort:
            print(f"⭐ DEBUG: Quality intent detected - {quality_sort}")
            return quality_sort
        
        # Default to popularity for list queries if no specific intent
        if question_type == "list":
            print(f"📈 DEBUG: Default popularity sort for list query")
            return {"sort_by": "popularity.desc"}
        
        print(f"🎯 DEBUG: No special sorting intent detected")
        return None
    
    @classmethod
    def _check_temporal_intent(cls, query_lower: str) -> Optional[Dict[str, str]]:
        """Check for temporal sorting intents"""
        
        # Recent/latest intent - newest first
        if any(keyword in query_lower for keyword in cls.TEMPORAL_RECENT_KEYWORDS):
            return {"sort_by": "release_date.desc"}
        
        # Chronological/first intent - oldest first  
        if any(keyword in query_lower for keyword in cls.TEMPORAL_CHRONOLOGICAL_KEYWORDS):
            return {"sort_by": "release_date.asc"}
        
        return None
    
    @classmethod  
    def _check_quality_intent(cls, query_lower: str) -> Optional[Dict[str, str]]:
        """Check for quality/rating sorting intents"""
        
        # High quality intent - best rated first
        if any(keyword in query_lower for keyword in cls.QUALITY_HIGH_KEYWORDS):
            return {
                "sort_by": "vote_average.desc",
                "vote_count.gte": "50"  # Minimum votes for meaningful ratings
            }
        
        # Low quality intent - worst rated first
        if any(keyword in query_lower for keyword in cls.QUALITY_LOW_KEYWORDS):
            return {
                "sort_by": "vote_average.asc", 
                "vote_count.gte": "10"  # Lower threshold for bad movies (they get fewer votes)
            }
        
        return None
    
    @classmethod
    def apply_sort_to_step_parameters(cls, step: Dict[str, Any], query: str, question_type: str = None) -> Dict[str, Any]:
        """
        Apply intent-aware sorting to a step's parameters.
        
        Args:
            step: Execution step dictionary
            query: Original user query
            question_type: Type of question
            
        Returns:
            Modified step with appropriate sort parameters
        """
        # Get sort strategy
        sort_strategy = cls.determine_sort_strategy(query, question_type)
        
        if sort_strategy:
            # Ensure parameters dict exists
            step.setdefault("parameters", {})
            
            # Apply sort strategy - override existing sort_by for intent-aware sorting
            for param, value in sort_strategy.items():
                if param == "sort_by":
                    # Always override sort_by when we have intent-aware sorting
                    existing_value = step["parameters"].get(param, "NOT_SET")
                    step["parameters"][param] = value
                    print(f"📊 DEBUG: Override {param}: {existing_value} → {value} for step {step.get('step_id', 'unknown')}")
                elif param not in step["parameters"]:
                    # For other parameters (like vote_count.gte), only add if not present
                    step["parameters"][param] = value
                    print(f"📊 DEBUG: Applied {param}={value} to step {step.get('step_id', 'unknown')}")
        
        return step
    
    @classmethod
    def get_debug_info(cls, query: str) -> Dict[str, Any]:
        """Get debug information about intent detection for a query"""
        query_lower = query.lower()
        
        return {
            "query": query,
            "temporal_recent_detected": any(kw in query_lower for kw in cls.TEMPORAL_RECENT_KEYWORDS),
            "temporal_chronological_detected": any(kw in query_lower for kw in cls.TEMPORAL_CHRONOLOGICAL_KEYWORDS), 
            "quality_high_detected": any(kw in query_lower for kw in cls.QUALITY_HIGH_KEYWORDS),
            "quality_low_detected": any(kw in query_lower for kw in cls.QUALITY_LOW_KEYWORDS),
            "recommended_sort": cls.determine_sort_strategy(query)
        }