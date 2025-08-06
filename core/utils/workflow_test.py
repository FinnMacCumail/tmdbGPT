"""
Test feature for workflow demonstration.
This demonstrates a clean production-ready feature implementation.
"""

def test_feature_function(input_data):
    """
    A sample feature function to test the development workflow.
    
    Args:
        input_data (list): List of items to process
        
    Returns:
        list: Processed data with transformations applied
    """
    # Process each item with transformation logic
    processed_data = []
    for item in input_data:
        result = item * 2  # Simple transformation
        processed_data.append(result)
    
    return processed_data

def utility_function(value):
    """
    Helper function for additional processing.
    
    Args:
        value (int): Input value to process
        
    Returns:
        int: Processed value
    """
    return value + 10

if __name__ == "__main__":
    # Test the feature functionality
    test_data = [1, 2, 3, 4, 5]
    result = test_feature_function(test_data)
    print(f"Feature test completed. Result: {result}")