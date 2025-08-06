import json
import os

# Define input and output file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "tmdb.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "tmdb_cleaned.json")

def clean_tmdb_schema(input_path, output_path):
    """Cleans the TMDB schema JSON by removing unnecessary example responses."""
    
    # Load the original TMDB JSON file
    with open(input_path, "r", encoding="utf-8") as file:
        tmdb_data = json.load(file)

    cleaned_data = {"paths": {}}

    # Iterate through all API paths
    for path, methods in tmdb_data.get("paths", {}).items():
        cleaned_data["paths"][path] = {}

        for method, details in methods.items():
            if not isinstance(details, dict):
                continue  # Skip invalid/malformed entries
            
            # Retain only necessary fields
            cleaned_details = {
                "summary": details.get("summary", "No description available"),
                "description":details.get("description", "No description available"),
                "parameters": details.get("parameters", []),
                "method": method.upper()
            }

            cleaned_data["paths"][path][method] = cleaned_details

    # Save the cleaned JSON file
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(cleaned_data, file, indent=4)


# Run the cleaning process
if __name__ == "__main__":
    clean_tmdb_schema(INPUT_FILE, OUTPUT_FILE)
