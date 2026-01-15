#!/usr/bin/env python3
"""
============================================================================
SCRIPT 07: TRUSTPILOT SCRAPER (TOP 25 BY SIZE)
============================================================================

PURPOSE: Search Trustpilot for the top 25 largest factors by property count.

INPUT:  data/csv/factors_register.csv
OUTPUT: data/csv/trustpilot_reviews_top25.csv

COLUMNS OUTPUT:
    - factor_registration_number
    - factor_name
    - property_count
    - trustpilot_id
    - trustpilot_name
    - trustpilot_url
    - rating (1-5 TrustScore)
    - review_count
    - categories

USAGE:
    python 07_trustpilot_scrape_top25.py

DEPENDENCIES:
    pip install requests beautifulsoup4

TIME: ~1-2 minutes for 25 factors
============================================================================
"""

import csv
import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import requests
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = Path("data/csv/factors_register.csv")
OUTPUT_CSV = Path("data/csv/trustpilot_reviews_top25.csv")

BASE_URL = "https://www.trustpilot.com"
SEARCH_URL = "https://www.trustpilot.com/search"

# Top N factors by size
TOP_N = 25

# Rate limiting
REQUEST_DELAY = 1.5  # Seconds between requests
REQUEST_TIMEOUT = 15

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# ============================================================================
# SCRAPING FUNCTIONS
# ============================================================================

def search_trustpilot(session: requests.Session, query: str) -> Optional[Dict]:
    """Search Trustpilot for a company."""
    
    search_params = {'query': query}
    
    try:
        response = session.get(
            SEARCH_URL,
            params=search_params,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Method 1: Look for JSON data in script tags
        for script in soup.select('script[type="application/json"]'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    businesses = find_businesses_in_json(data)
                    if businesses:
                        return parse_business_data(businesses[0])
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Method 2: Look for data attributes
        result = soup.select_one('[data-business-unit-json]')
        if result:
            try:
                data = json.loads(result['data-business-unit-json'])
                return parse_business_data(data)
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Method 3: Parse HTML result cards
        result_cards = soup.select('.styles_businessCard__')
        if not result_cards:
            result_cards = soup.select('.business-unit-card')
        if not result_cards:
            result_cards = soup.select('[data-business-unit-id]')
        
        if result_cards:
            return parse_result_card(result_cards[0])
        
        return None
        
    except requests.RequestException as e:
        print(f"    ⚠️ Request error: {e}")
        return None


def find_businesses_in_json(data: Dict, depth: int = 0) -> List[Dict]:
    """Recursively find business data in JSON structure."""
    if depth > 10:
        return []
    
    businesses = []
    
    if isinstance(data, dict):
        if 'businessUnitId' in data or 'identifyingName' in data:
            businesses.append(data)
        
        for value in data.values():
            businesses.extend(find_businesses_in_json(value, depth + 1))
    
    elif isinstance(data, list):
        for item in data:
            businesses.extend(find_businesses_in_json(item, depth + 1))
    
    return businesses


def parse_business_data(data: Dict) -> Dict:
    """Parse business data from JSON."""
    return {
        'trustpilot_id': data.get('businessUnitId', data.get('id', '')),
        'trustpilot_name': data.get('displayName', data.get('name', '')),
        'trustpilot_url': f"{BASE_URL}/review/{data.get('identifyingName', '')}",
        'rating': data.get('trustScore', data.get('score', 0)),
        'review_count': data.get('numberOfReviews', data.get('reviewCount', 0)),
        'categories': ','.join(
            c.get('name', '') for c in data.get('categories', [])
        ),
    }


def parse_result_card(card) -> Dict:
    """Parse data from an HTML result card."""
    data = {
        'trustpilot_id': '',
        'trustpilot_name': '',
        'trustpilot_url': '',
        'rating': 0,
        'review_count': 0,
        'categories': '',
    }
    
    data['trustpilot_id'] = card.get('data-business-unit-id', '')
    
    name_el = card.select_one('.typography_heading-xs__jSwUz, .business-name, h3')
    if name_el:
        data['trustpilot_name'] = name_el.get_text(strip=True)
    
    link = card.select_one('a[href*="/review/"]')
    if link:
        href = link.get('href', '')
        if href.startswith('/'):
            href = BASE_URL + href
        data['trustpilot_url'] = href
    
    rating_el = card.select_one('[data-rating]')
    if rating_el:
        data['rating'] = float(rating_el.get('data-rating', 0))
    else:
        rating_text = card.select_one('.typography_body-m__xgxZ_, .star-rating')
        if rating_text:
            match = re.search(r'(\d+\.?\d*)', rating_text.get_text())
            if match:
                data['rating'] = float(match.group(1))
    
    review_el = card.select_one('.typography_body-m__xgxZ_, .review-count')
    if review_el:
        text = review_el.get_text()
        match = re.search(r'([\d,]+)\s*review', text, re.IGNORECASE)
        if match:
            data['review_count'] = int(match.group(1).replace(',', ''))
    
    return data


def name_match_score(factor_name: str, trustpilot_name: str) -> float:
    """Calculate how well the names match."""
    
    def normalize(s):
        s = s.lower()
        s = re.sub(r'(ltd|limited|llp|plc|inc)\.?$', '', s)
        s = re.sub(r'[^\w\s]', ' ', s)
        return ' '.join(s.split())
    
    n1 = normalize(factor_name)
    n2 = normalize(trustpilot_name)
    
    if n1 == n2:
        return 1.0
    
    if n1 in n2 or n2 in n1:
        return 0.8
    
    words1 = set(n1.split())
    words2 = set(n2.split())
    
    overlap = len(words1 & words2)
    total = len(words1 | words2)
    
    return overlap / total if total > 0 else 0.0


def get_top_factors_by_size(factors: List[Dict], n: int = 25) -> List[Dict]:
    """Filter to top N factors by property_count, excluding expired registrations."""
    
    # Filter to registered (active) factors only
    active = [
        f for f in factors 
        if f.get('status', '').strip().lower() == 'registered'
    ]
    
    # Sort by property_count descending
    sorted_factors = sorted(
        active,
        key=lambda x: int(x.get('property_count', 0) or 0),
        reverse=True
    )
    
    return sorted_factors[:n]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scrape Trustpilot for top 25 property factors by size")
    parser.add_argument('--top', type=int, default=TOP_N,
                        help=f'Number of top factors to process (default: {TOP_N})')
    parser.add_argument('--min-reviews', type=int, default=1,
                        help='Minimum reviews to include (default: 1)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRUSTPILOT SCRAPER - TOP 25 BY SIZE")
    print("=" * 60)
    
    # Check input file
    if not INPUT_CSV.exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        return
    
    # Load factors
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        factors = list(csv.DictReader(f))
    
    print(f"📋 Loaded {len(factors)} total factors")
    
    # Get top N by size
    top_factors = get_top_factors_by_size(factors, args.top)
    
    print(f"🎯 Targeting top {len(top_factors)} factors by property count")
    print()
    
    # Show the factors we'll be searching
    print("Factors to search:")
    print("-" * 60)
    for i, f in enumerate(top_factors, 1):
        name = f.get('name', '')[:40]
        count = int(f.get('property_count', 0) or 0)
        print(f"  {i:2}. {name:<42} ({count:,} properties)")
    print("-" * 60)
    print()
    
    results = []
    
    # Create session
    session = requests.Session()
    session.headers.update(HEADERS)
    
    print("🔍 Searching Trustpilot...")
    print()
    
    found_count = 0
    
    for i, factor in enumerate(top_factors):
        pf = factor.get('registration_number', '')
        name = factor.get('name', '')
        prop_count = int(factor.get('property_count', 0) or 0)
        
        print(f"[{i+1}/{len(top_factors)}] {name[:45]}...", end=" ", flush=True)
        
        # Search
        data = search_trustpilot(session, name)
        
        if data and data.get('review_count', 0) >= args.min_reviews:
            match_score = name_match_score(name, data.get('trustpilot_name', ''))
            
            if match_score >= 0.3:
                result = {
                    'factor_registration_number': pf,
                    'factor_name': name,
                    'property_count': prop_count,
                    **data,
                    'match_score': match_score,
                }
                results.append(result)
                found_count += 1
                
                print(f"✅ {data['rating']}★ ({data['review_count']} reviews)")
            else:
                print(f"⚠️ Poor match ({match_score:.0%})")
        else:
            print("❌ Not found")
        
        # Rate limiting
        if i < len(top_factors) - 1:
            time.sleep(REQUEST_DELAY)
    
    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'factor_registration_number', 'factor_name', 'property_count',
        'trustpilot_id', 'trustpilot_name', 'trustpilot_url', 
        'rating', 'review_count', 'categories', 'match_score'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    print()
    print("=" * 60)
    print("TRUSTPILOT SCRAPING COMPLETE")
    print("=" * 60)
    
    coverage = found_count / len(top_factors) * 100 if top_factors else 0
    
    print(f"✅ Found: {len(results)}/{len(top_factors)} factors on Trustpilot")
    print(f"📊 Coverage: {coverage:.1f}%")
    
    if results:
        avg_rating = sum(r['rating'] for r in results if r.get('rating')) / len([r for r in results if r.get('rating')])
        total_reviews = sum(r.get('review_count', 0) for r in results)
        total_properties = sum(r.get('property_count', 0) for r in results)
        print(f"📊 Average rating: {avg_rating:.2f}")
        print(f"📊 Total reviews: {total_reviews:,}")
        print(f"📊 Properties covered: {total_properties:,}")
    
    print(f"\n📄 Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
