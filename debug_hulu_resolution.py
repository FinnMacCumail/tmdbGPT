#!/usr/bin/env python3
"""
Debug Hulu Entity Resolution
Check which Hulu network ID is being resolved and why
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from dotenv import load_dotenv
from core.entity.entity_resolution import TMDBEntityResolver

# Load environment
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

headers = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "Content-Type": "application/json;charset=utf-8"
}

def search_all_hulu_networks():
    """Search for all Hulu networks in TMDB"""
    print("🔍 Searching TMDB for all Hulu networks...")
    print("=" * 50)
    
    search_url = f"{TMDB_BASE_URL}/search/network"
    search_params = {"query": "Hulu"}
    
    try:
        response = requests.get(search_url, headers=headers, params=search_params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            print(f"Found {len(results)} Hulu networks:")
            for i, network in enumerate(results):
                name = network.get("name", "Unknown")
                network_id = network.get("id")
                origin = network.get("origin_country", "Unknown")
                logo = "✅" if network.get("logo_path") else "❌"
                
                print(f"   {i+1}. {name} (ID: {network_id})")
                print(f"      Origin: {origin}")
                print(f"      Logo: {logo}")
                print()
                
                # Test each network to see if it has content
                test_network_content(network_id, name)
            
            return results
        else:
            print(f"❌ Search failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_network_content(network_id, network_name):
    """Test if a network has actual TV content"""
    discover_url = f"{TMDB_BASE_URL}/discover/tv"
    discover_params = {
        "with_networks": str(network_id),
        "page": 1
    }
    
    try:
        response = requests.get(discover_url, headers=headers, params=discover_params)
        if response.status_code == 200:
            data = response.json()
            total_results = data.get("total_results", 0)
            results = data.get("results", [])
            
            print(f"      Content: {total_results} total shows")
            if results:
                # Show first few shows
                for j, show in enumerate(results[:3]):
                    show_name = show.get("name", "Unknown")
                    year = show.get("first_air_date", "")[:4] if show.get("first_air_date") else "Unknown"
                    print(f"         - {show_name} ({year})")
            print()
        
    except Exception as e:
        print(f"      Content check failed: {e}")

def test_entity_resolver():
    """Test our entity resolver with Hulu"""
    print("🧪 Testing Entity Resolver with 'Hulu'...")
    print("=" * 50)
    
    resolver = TMDBEntityResolver(TMDB_API_KEY, headers)
    
    # Test direct resolution
    resolved_id = resolver.resolve_entity("Hulu", "network")
    print(f"resolve_entity('Hulu', 'network') → {resolved_id}")
    
    if resolved_id:
        # Get details about the resolved network
        network_url = f"{TMDB_BASE_URL}/network/{resolved_id}"
        response = requests.get(network_url, headers=headers)
        
        if response.status_code == 200:
            network_data = response.json()
            name = network_data.get("name", "Unknown")
            origin = network_data.get("origin_country", "Unknown") 
            
            print(f"Resolved to: {name} (Origin: {origin})")
            
            # Test content
            print("Testing content from resolved network:")
            test_network_content(resolved_id, name)

def test_full_query_flow():
    """Test the complete query flow for Hulu Originals"""
    print("🔄 Testing Full Query Flow...")
    print("=" * 50)
    
    try:
        from app import build_app_graph
        
        graph = build_app_graph()
        result = graph.invoke({"input": "Hulu Originals"})
        
        # Check entity resolution
        extraction = result.get("extraction_result", {})
        query_entities = extraction.get("query_entities", [])
        
        print("Query entities extracted:")
        for entity in query_entities:
            name = entity.get("name")
            entity_type = entity.get("type")
            resolved_id = entity.get("resolved_id")
            print(f"   {name} → {entity_type} (ID: {resolved_id})")
            
            if resolved_id and entity_type == "network":
                # Check what this network actually is
                network_url = f"{TMDB_BASE_URL}/network/{resolved_id}"
                response = requests.get(network_url, headers=headers)
                if response.status_code == 200:
                    network_data = response.json()
                    actual_name = network_data.get("name", "Unknown")
                    origin = network_data.get("origin_country", "Unknown")
                    print(f"      Actually: {actual_name} (Origin: {origin})")
        
    except Exception as e:
        print(f"❌ Query flow test failed: {e}")

def main():
    """Main debug function"""
    print("🔍 Debugging Hulu Entity Resolution Issue")
    print("=" * 60)
    
    # Step 1: Find all Hulu networks in TMDB
    all_hulu_networks = search_all_hulu_networks()
    
    # Step 2: Test our entity resolver
    test_entity_resolver()
    
    # Step 3: Test full query flow
    test_full_query_flow()
    
    print("🎯 Analysis:")
    print("- Check which Hulu networks have actual content")
    print("- Verify our entity resolver picks the right one")
    print("- Confirm US preference logic is working")

if __name__ == "__main__":
    if not TMDB_API_KEY:
        print("❌ TMDB_API_KEY not found in environment")
        exit(1)
    
    main()