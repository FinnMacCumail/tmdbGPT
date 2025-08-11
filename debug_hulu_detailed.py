#!/usr/bin/env python3
"""
Detailed Hulu Investigation
Check both company and network IDs, find US Hulu if it exists
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

headers = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "Content-Type": "application/json;charset=utf-8"
}

def search_companies_and_networks():
    """Search both companies and networks for Hulu"""
    print("🔍 Searching for Hulu in both companies and networks...")
    print("=" * 60)
    
    # Search companies
    print("📦 COMPANIES:")
    company_url = f"{TMDB_BASE_URL}/search/company"
    response = requests.get(company_url, headers=headers, params={"query": "Hulu"})
    
    if response.status_code == 200:
        companies = response.json().get("results", [])
        for company in companies:
            name = company.get("name", "Unknown")
            company_id = company.get("id")
            origin = company.get("origin_country", "Unknown")
            print(f"   {name} (ID: {company_id}, Origin: {origin})")
    else:
        print(f"   Company search failed: {response.status_code}")
    
    print()
    
    # Search networks using search/multi (since search/network failed)
    print("📺 NETWORKS (via search/multi):")
    multi_url = f"{TMDB_BASE_URL}/search/multi"
    response = requests.get(multi_url, headers=headers, params={"query": "Hulu"})
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        networks = [r for r in results if r.get("media_type") == "network" or "network" in str(r)]
        
        if not networks:
            print("   No networks found via multi search")
            # Try to find networks by checking all results
            print("   All multi search results:")
            for result in results:
                media_type = result.get("media_type", "unknown")
                name = result.get("name") or result.get("title", "Unknown")
                result_id = result.get("id")
                print(f"      {name} (ID: {result_id}, Type: {media_type})")
        else:
            for network in networks:
                name = network.get("name", "Unknown") 
                network_id = network.get("id")
                origin = network.get("origin_country", "Unknown")
                print(f"   {name} (ID: {network_id}, Origin: {origin})")
    else:
        print(f"   Multi search failed: {response.status_code}")

def check_specific_ids():
    """Check the specific IDs we found"""
    print("\n🔍 Checking Specific IDs Found...")
    print("=" * 60)
    
    # Check Japanese Hulu network (1772)
    print("📺 Network ID 1772 (Japanese Hulu from direct resolver):")
    network_url = f"{TMDB_BASE_URL}/network/1772"
    response = requests.get(network_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"   Name: {data.get('name', 'Unknown')}")
        print(f"   Origin: {data.get('origin_country', 'Unknown')}")
        print(f"   HQ: {data.get('headquarters', 'Unknown')}")
    
    # Test its content
    discover_url = f"{TMDB_BASE_URL}/discover/tv"
    response = requests.get(discover_url, headers=headers, params={"with_networks": "1772"})
    if response.status_code == 200:
        data = response.json()
        total = data.get("total_results", 0)
        shows = data.get("results", [])
        print(f"   Content: {total} shows")
        for show in shows[:3]:
            name = show.get("name", "Unknown")
            print(f"      - {name}")
    
    print()
    
    # Check Hulu company (140361) 
    print("📦 Company ID 140361 (from query flow):")
    company_url = f"{TMDB_BASE_URL}/company/140361"
    response = requests.get(company_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"   Name: {data.get('name', 'Unknown')}")
        print(f"   Origin: {data.get('origin_country', 'Unknown')}")
        print(f"   HQ: {data.get('headquarters', 'Unknown')}")
    
    # Test its content
    discover_url = f"{TMDB_BASE_URL}/discover/tv"
    response = requests.get(discover_url, headers=headers, params={"with_companies": "140361"})
    if response.status_code == 200:
        data = response.json()
        total = data.get("total_results", 0)
        shows = data.get("results", [])
        print(f"   Content (TV): {total} shows")
        for show in shows[:3]:
            name = show.get("name", "Unknown")
            print(f"      - {name}")
    
    # Also test movies for the company
    discover_url = f"{TMDB_BASE_URL}/discover/movie"
    response = requests.get(discover_url, headers=headers, params={"with_companies": "140361"})
    if response.status_code == 200:
        data = response.json()
        total = data.get("total_results", 0)
        movies = data.get("results", [])
        print(f"   Content (Movies): {total} movies")
        for movie in movies[:3]:
            name = movie.get("title", "Unknown")
            print(f"      - {name}")

def find_us_hulu():
    """Try to find US Hulu by searching for variations"""
    print("\n🇺🇸 Searching for US Hulu variations...")
    print("=" * 60)
    
    search_terms = ["Hulu US", "Hulu United States", "Hulu America", "Hulu LLC"]
    
    for term in search_terms:
        print(f"Searching for: '{term}'")
        
        # Try company search
        response = requests.get(f"{TMDB_BASE_URL}/search/company", 
                              headers=headers, params={"query": term})
        if response.status_code == 200:
            results = response.json().get("results", [])
            for result in results:
                name = result.get("name", "Unknown")
                company_id = result.get("id")
                origin = result.get("origin_country", "Unknown")
                print(f"   Company: {name} (ID: {company_id}, Origin: {origin})")
        
        # Try multi search  
        response = requests.get(f"{TMDB_BASE_URL}/search/multi",
                              headers=headers, params={"query": term})
        if response.status_code == 200:
            results = response.json().get("results", [])
            for result in results:
                if "hulu" in result.get("name", "").lower():
                    name = result.get("name", "Unknown")
                    result_id = result.get("id")
                    media_type = result.get("media_type", "unknown")
                    origin = result.get("origin_country", "Unknown")
                    print(f"   {media_type.title()}: {name} (ID: {result_id}, Origin: {origin})")
        
        print()

def main():
    """Main investigation"""
    search_companies_and_networks()
    check_specific_ids() 
    find_us_hulu()
    
    print("🎯 ANALYSIS:")
    print("=" * 60)
    print("1. Check if US Hulu exists as separate entity")
    print("2. Determine if company ID 140361 is better than network ID 1772")
    print("3. Fix entity resolution to prefer US entities")
    print("4. Update dynamic service classification if needed")

if __name__ == "__main__":
    main()