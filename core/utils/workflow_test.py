"""
Test feature for workflow demonstration.
This file demonstrates the feature development process with debug statements.
"""

def test_feature_function(input_data):
    """
    A sample feature function to test the development workflow.
    """
    print(f"🔧 Debug: Starting test feature with input: {input_data}")
    
    # Simulate some processing
    processed_data = []
    for item in input_data:
        print(f"📊 Debug: Processing item: {item}")
        result = item * 2  # Simple transformation
        processed_data.append(result)
        print(f"⚙️ Debug: Item {item} transformed to {result}")
    
    print(f"✅ Debug: Completed processing. Total items: {len(processed_data)}")
    return processed_data

def utility_function(value):
    """Helper function with debug output."""
    print(f"🛠️ Debug: Utility function called with value: {value}")
    return value + 10

if __name__ == "__main__":
    # Test the feature
    test_data = [1, 2, 3, 4, 5]
    result = test_feature_function(test_data)
    print(f"🎯 Debug: Final result: {result}")