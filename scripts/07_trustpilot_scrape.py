#!/usr/bin/env python3
"""
============================================================================
SCRIPT 07: TRUSTPILOT SCRAPER
============================================================================

PURPOSE: Search Trustpilot for factor profiles and extract ratings/reviews.

INPUT:  data/csv/factors_register.csv
OUTPUT: data/csv/trustpilot_reviews.csv

COLUMNS OUTPUT:
    - factor_registration_number
    - factor_name
    - trustpilot_id
    - trustpilot_name
    - trustpilot_url
    - rating (1-5 TrustScore)
    - review_count
    - categories

USAGE:
    python 07_trustpilot_scrape.py
    python 07_trustpilot_scrape.py --resume
    python 07_trustpilot_scrape.py --limit 10

DEPENDENCIES:
    pip install requests beautifulsoup4

TIME: ~15-20 minutes for 340 factors

NOTE: Trustpilot coverage for property factors is typically ~20-25%
============================================================================
"""

import csv
import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = Path("data/csv/factors_register.csv")
OUTPUT_CSV = Path("data/csv/trustpilot_reviews.csv")
CHECKPOINT_FILE = Path("data/trustpilot_checkpoint.json")

BASE_URL = "https://www.trustpilot.com"
SEARCH_URL = "https://www.trustpilot.com/search"

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
                # Navigate through possible structures
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
        # Check if this looks like a business object
        if 'businessUnitId' in data or 'identifyingName' in data:
            businesses.append(data)
        
        # Recurse into values
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
    
    # ID from data attribute
    data['trustpilot_id'] = card.get('data-business-unit-id', '')
    
    # Name
    name_el = card.select_one('.typography_heading-xs__jSwUz, .business-name, h3')
    if name_el:
        data['trustpilot_name'] = name_el.get_text(strip=True)
    
    # URL
    link = card.select_one('a[href*="/review/"]')
    if link:
        href = link.get('href', '')
        if href.startswith('/'):
            href = BASE_URL + href
        data['trustpilot_url'] = href
    
    # Rating
    rating_el = card.select_one('[data-rating]')
    if rating_el:
        data['rating'] = float(rating_el.get('data-rating', 0))
    else:
        # Try to find rating text
        rating_text = card.select_one('.typography_body-m__xgxZ_, .star-rating')
        if rating_text:
            match = re.search(r'(\d+\.?\d*)', rating_text.get_text())
            if match:
                data['rating'] = float(match.group(1))
    
    # Review count
    review_el = card.select_one('.typography_body-m__xgxZ_, .review-count')
    if review_el:
        text = review_el.get_text()
        match = re.search(r'([\d,]+)\s*review', text, re.IGNORECASE)
        if match:
            data['review_count'] = int(match.group(1).replace(',', ''))
    
    return data


def fetch_company_page(session: requests.Session, url: str) -> Optional[Dict]:
    """Fetch additional details from company's Trustpilot page."""
    
    if not url:
        return None
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data = {}
        
        # Get more accurate rating from page
        rating_el = soup.select_one('[data-rating-typography]')
        if rating_el:
            try:
                data['rating'] = float(rating_el.get_text(strip=True))
            except ValueError:
                pass
        
        # Get review count
        review_el = soup.select_one('[itemprop="reviewCount"]')
        if review_el:
            data['review_count'] = int(review_el.get('content', 0))
        
        return data
        
    except Exception:
        return None


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

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint() -> Dict:
    """Load checkpoint data."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'results': []}


def save_checkpoint(data: Dict):
    """Save checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scrape Trustpilot for property factors")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--limit', type=int, help='Limit number of factors')
    parser.add_argument('--min-reviews', type=int, default=1,
                        help='Minimum reviews to include (default: 1)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRUSTPILOT SCRAPER")
    print("=" * 60)
    
    # Check input file
    if not INPUT_CSV.exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        print("   Run 02_registry_enrich.py first.")
        return
    
    # Load factors
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        factors = list(csv.DictReader(f))
    
    print(f"📋 Loaded {len(factors)} factors")
    
    # Load checkpoint
    checkpoint = {'processed': [], 'results': []}
    if args.resume:
        checkpoint = load_checkpoint()
        print(f"   Resuming: {len(checkpoint['processed'])} already processed")
    
    processed_set = set(checkpoint['processed'])
    
    # Filter to unprocessed
    pending = [f for f in factors if f['registration_number'] not in processed_set]
    
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} to process")
    
    if not pending:
        print("\n✅ All factors already processed!")
        return
    
    results = checkpoint['results']
    
    # Create session
    session = requests.Session()
    session.headers.update(HEADERS)
    
    print(f"\n🔍 Searching Trustpilot...")
    
    found_count = 0
    
    for i, factor in enumerate(pending):
        pf = factor.get('registration_number', '')
        name = factor.get('name', '')
        
        print(f"[{i+1}/{len(pending)}] {name[:45]}...", end=" ", flush=True)
        
        # Search
        data = search_trustpilot(session, name)
        
        if data and data.get('review_count', 0) >= args.min_reviews:
            # Verify it's a reasonable match
            match_score = name_match_score(name, data.get('trustpilot_name', ''))
            
            if match_score >= 0.3:  # Lenient threshold
                result = {
                    'factor_registration_number': pf,
                    'factor_name': name,
                    **data,
                    'match_score': match_score,
                }
                results.append(result)
                found_count += 1
                
                print(f"✅ {data['rating']}★ ({data['review_count']} reviews)")
            else:
                print(f"⚠️ Poor match ({match_score:.0%})")
        else:
            print("❌")
        
        # Update checkpoint
        checkpoint['processed'].append(pf)
        checkpoint['results'] = results
        
        if (i + 1) % 20 == 0:
            save_checkpoint(checkpoint)
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    # Save final checkpoint
    save_checkpoint(checkpoint)
    
    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'factor_registration_number', 'factor_name', 'trustpilot_id',
        'trustpilot_name', 'trustpilot_url', 'rating', 'review_count',
        'categories', 'match_score'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRUSTPILOT SCRAPING COMPLETE")
    print("=" * 60)
    
    total_factors = len(checkpoint['processed'])
    coverage = found_count / total_factors * 100 if total_factors > 0 else 0
    
    print(f"✅ Found: {len(results)} factors on Trustpilot")
    print(f"📊 Coverage: {coverage:.1f}%")
    
    if results:
        avg_rating = sum(r['rating'] for r in results if r.get('rating')) / len([r for r in results if r.get('rating')])
        total_reviews = sum(r.get('review_count', 0) for r in results)
        print(f"📊 Average rating: {avg_rating:.2f}")
        print(f"📊 Total reviews: {total_reviews:,}")
    
    print(f"\n📄 Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
