#!/usr/bin/env python3
"""
Debug Company/Studio Constraint Validation
Investigates why resolved company/network entities fail to return results
"""

import requests
import os
from dotenv import load_dotenv
import json

# Load environment
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

headers = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "Content-Type": "application/json;charset=utf-8"
}

def debug_company_discover_endpoint(company_id, company_name):
    """Test TMDB discover/movie endpoint with company constraint"""
    
    print(f"\n=== Debugging Company: {company_name} (ID: {company_id}) ===")
    
    # Test direct company info
    company_url = f"{TMDB_BASE_URL}/company/{company_id}"
    company_response = requests.get(company_url, headers=headers)
    
    if company_response.status_code == 200:
        company_data = company_response.json()
        print(f"✅ Company exists: {company_data.get('name', 'Unknown')}")
        print(f"   Origin: {company_data.get('origin_country', 'Unknown')}")
        print(f"   HQ: {company_data.get('headquarters', 'Unknown')}")
    else:
        print(f"❌ Company not found: {company_response.status_code}")
        return
    
    # Test discover/movie with company constraint
    discover_url = f"{TMDB_BASE_URL}/discover/movie"
    discover_params = {
        "with_companies": str(company_id),
        "sort_by": "popularity.desc",
        "page": 1
    }
    
    discover_response = requests.get(discover_url, headers=headers, params=discover_params)
    
    if discover_response.status_code == 200:
        discover_data = discover_response.json()
        total_results = discover_data.get("total_results", 0)
        results = discover_data.get("results", [])
        
        print(f"✅ Discover API success: {total_results} total movies")
        
        if results:
            print("   Top 3 movies:")
            for movie in results[:3]:
                title = movie.get("title", "Unknown")
                year = movie.get("release_date", "Unknown")[:4] if movie.get("release_date") else "Unknown"
                print(f"   - {title} ({year})")
        else:
            print("   ⚠️ No movies returned despite successful API call")
    else:
        print(f"❌ Discover API failed: {discover_response.status_code}")
        print(f"   Response: {discover_response.text}")

def debug_network_discover_endpoint(network_id, network_name):
    """Test TMDB discover/tv endpoint with network constraint"""
    
    print(f"\n=== Debugging Network: {network_name} (ID: {network_id}) ===")
    
    # Test direct network info
    network_url = f"{TMDB_BASE_URL}/network/{network_id}"
    network_response = requests.get(network_url, headers=headers)
    
    if network_response.status_code == 200:
        network_data = network_response.json()
        print(f"✅ Network exists: {network_data.get('name', 'Unknown')}")
        print(f"   Origin: {network_data.get('origin_country', 'Unknown')}")
        print(f"   HQ: {network_data.get('headquarters', 'Unknown')}")
    else:
        print(f"❌ Network not found: {network_response.status_code}")
        return
    
    # Test discover/tv with network constraint
    discover_url = f"{TMDB_BASE_URL}/discover/tv"
    discover_params = {
        "with_networks": str(network_id),
        "sort_by": "popularity.desc", 
        "page": 1
    }
    
    discover_response = requests.get(discover_url, headers=headers, params=discover_params)
    
    if discover_response.status_code == 200:
        discover_data = discover_response.json()
        total_results = discover_data.get("total_results", 0)
        results = discover_data.get("results", [])
        
        print(f"✅ Discover API success: {total_results} total TV shows")
        
        if results:
            print("   Top 3 shows:")
            for show in results[:3]:
                name = show.get("name", "Unknown")
                year = show.get("first_air_date", "Unknown")[:4] if show.get("first_air_date") else "Unknown"
                print(f"   - {name} ({year})")
        else:
            print("   ⚠️ No shows returned despite successful API call")
    else:
        print(f"❌ Discover API failed: {discover_response.status_code}")
        print(f"   Response: {discover_response.text}")

def search_hbo_networks():
    """Find the correct HBO network ID"""
    search_url = f"{TMDB_BASE_URL}/search/network"
    search_params = {"query": "HBO"}
    
    response = requests.get(search_url, headers=headers, params=search_params)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        
        print("🔍 HBO Network Search Results:")
        for network in results:
            name = network.get("name", "Unknown")
            network_id = network.get("id")
            origin = network.get("origin_country", "Unknown")
            print(f"   - {name} (ID: {network_id}, Origin: {origin})")
        
        # Find US HBO
        us_hbo = [n for n in results if n.get("origin_country") == "US" and "HBO" in n.get("name", "")]
        if us_hbo:
            return us_hbo[0].get("id")
        
        # Fallback to first HBO result
        if results:
            return results[0].get("id")
    
    return None

def test_constraint_validation():
    """Test constraint validation for problematic companies/networks"""
    
    print("🔍 TMDB Company/Network Constraint Validation Debug")
    print("=" * 60)
    
    # Find correct HBO network ID
    correct_hbo_id = search_hbo_networks()
    
    # Test Marvel Studios (from our failed test)
    debug_company_discover_endpoint(420, "Marvel Studios")
    
    # Test HBO with correct ID
    if correct_hbo_id:
        debug_network_discover_endpoint(correct_hbo_id, f"HBO (Correct ID: {correct_hbo_id})")
    
    # Test HBO that our system found (wrong one)
    debug_network_discover_endpoint(8102, "HBO (System Found - Wrong)")
    
    # Test a few other major studios for comparison
    debug_company_discover_endpoint(2, "Walt Disney Pictures")  # Known large studio
    debug_network_discover_endpoint(49, "HBO")  # This should actually be Netflix based on previous results
    
    print("\n" + "=" * 60)
    print("🧐 Analysis Summary:")
    print("- Marvel Studios ID 420: Should work but tmdbGPT constraint building fails")
    print("- HBO ID 8102: Wrong HBO (Poland) - entity resolution issue")
    print("- Need to fix US origin preference in entity resolution")
    print("- Need to debug why Marvel constraint validation fails in tmdbGPT")

if __name__ == "__main__":
    if not TMDB_API_KEY:
        print("❌ TMDB_API_KEY not found in environment")
        exit(1)
    
    test_constraint_validation()