# core/execution/financial_filter.py

class FinancialFilter:
    """Handler for revenue constraint filtering"""
    
    @staticmethod
    def apply_financial_filters(results: list, constraint_tree, max_results: int = 100) -> list:
        """
        Apply revenue threshold filtering to API results.
        
        Args:
            results: List of movie/TV results from TMDB API
            constraint_tree: ConstraintGroup containing revenue constraints
            max_results: Maximum number of results to return
            
        Returns:
            Filtered list of results that meet revenue constraints
        """
        if not results or not constraint_tree:
            return results
            
        # Extract revenue constraints
        revenue_constraints = []
        for constraint in constraint_tree.flatten():
            if hasattr(constraint, 'type') and constraint.type == 'revenue':
                revenue_constraints.append(constraint)
        
        if not revenue_constraints:
            return results
            
        filtered_results = []
        
        for result in results:
            passes_all_constraints = True
            
            for constraint in revenue_constraints:
                if not FinancialFilter._check_financial_constraint(result, constraint):
                    passes_all_constraints = False
                    break
            
            if passes_all_constraints:
                filtered_results.append(result)
                
                # Early termination for efficiency - especially important for "less than" queries
                # where we're processing results in ascending order
                if len(filtered_results) >= max_results:
                    break
                    
        return filtered_results
    
    @staticmethod
    def _check_financial_constraint(result: dict, constraint) -> bool:
        """Check if a single result meets a financial constraint"""
        constraint_type = constraint.type
        threshold = constraint.metadata.get('threshold', 0)
        operator = constraint.metadata.get('threshold_operator', 'less_than')
        
        if constraint_type == 'revenue':
            actual_value = result.get('revenue', 0)
        else:
            return True  # Unknown constraint type, pass through
            
        # Handle cases where financial data is missing or unreliable
        if actual_value is None or actual_value == 0:
            # Exclude movies with missing/zero revenue data as unreliable
            # These are typically unreleased films, data entry errors, or missing information
            return False
        
        # Apply the threshold comparison
        if operator == 'less_than':
            return actual_value < threshold
        elif operator == 'less_than_equal':
            return actual_value <= threshold
        elif operator == 'greater_than':
            return actual_value > threshold
        elif operator == 'greater_than_equal':
            return actual_value >= threshold
        elif operator == 'between':
            # For "between" operator, threshold should be a tuple/list
            if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                return threshold[0] <= actual_value <= threshold[1]
                
        return False
    
    @staticmethod
    def should_apply_financial_filtering(constraint_tree) -> bool:
        """Check if revenue filtering should be applied to results"""
        if not constraint_tree:
            return False
            
        for constraint in constraint_tree.flatten():
            if hasattr(constraint, 'type') and constraint.type == 'revenue':
                return True
                
        return False
    
    @staticmethod
    def estimate_result_limit(constraint_tree) -> int:
        """
        Estimate appropriate result limit based on financial constraints.
        For "less than" queries, we want to stop early when we hit the threshold.
        """
        if not constraint_tree:
            return 100  # Default limit
            
        for constraint in constraint_tree.flatten():
            if hasattr(constraint, 'type') and constraint.type == 'revenue':
                operator = constraint.metadata.get('threshold_operator', 'less_than')
                
                if operator in ('less_than', 'less_than_equal'):
                    # For "less than" queries with popularity.desc sorting,
                    # process fewer results for efficiency
                    return 50  # Lower limit for efficiency
                else:
                    # For "greater than" queries with revenue.desc sorting
                    return 200
                    
        return 100  # Default