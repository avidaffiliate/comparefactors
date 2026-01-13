#!/usr/bin/env python3
"""
============================================================================
SCRIPT 02: REGISTRY ENRICHMENT (Direct Detail Pages)
============================================================================

PURPOSE: Fetch full details for each factor from their individual pages at
         https://www.propertyfactorregister.gov.scot/property-factor/PF000XXX

INPUT:  data/csv/factors_registry_raw.csv (from browser extraction)
OUTPUT: data/csv/factors_register.csv

EXTRACTS:
    - registration_number (PF000XXX)
    - name
    - trading_name  
    - trading_type
    - address
    - city
    - postcode
    - website
    - property_count
    - status (Registered/Expired)

USAGE:
    python scripts/02_registry_enrich.py
    python scripts/02_registry_enrich.py --resume          # Resume from checkpoint
    python scripts/02_registry_enrich.py --limit 10        # Test with 10 factors
    python scripts/02_registry_enrich.py --pf PF000257     # Test single factor

DEPENDENCIES:
    pip install requests beautifulsoup4

TIME: ~20-30 minutes for 675 factors (with rate limiting)
============================================================================
"""

import csv
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = Path("data/csv/factors_registry_raw.csv")
OUTPUT_CSV = Path("data/csv/factors_register.csv")
CHECKPOINT_FILE = Path("data/registry_enrich_checkpoint.json")

BASE_URL = "https://www.propertyfactorregister.gov.scot/property-factor"

# Rate limiting - be polite to the government server
REQUEST_DELAY = 1.0  # seconds between requests
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.9',
}

# ============================================================================
# DATA STRUCTURE
# ============================================================================

@dataclass
class FactorDetails:
    registration_number: str
    name: str
    trading_name: str = ""
    trading_type: str = ""
    address: str = ""
    city: str = ""
    postcode: str = ""
    website: str = ""
    property_count: int = 0
    status: str = ""
    scrape_success: bool = False
    scrape_error: str = ""

# ============================================================================
# SCRAPING FUNCTIONS
# ============================================================================

def extract_postcode(text: str) -> str:
    """Extract Scottish postcode from text."""
    if not text:
        return ""
    # Scottish postcodes pattern
    pattern = r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b'
    match = re.search(pattern, text.upper())
    return match.group(1).strip() if match else ""


def extract_city(address_lines: list) -> str:
    """Extract city from address lines."""
    # Common Scottish cities
    cities = [
        'Edinburgh', 'Glasgow', 'Aberdeen', 'Dundee', 'Inverness',
        'Stirling', 'Perth', 'Paisley', 'Livingston', 'Hamilton',
        'Kirkcaldy', 'Ayr', 'Kilmarnock', 'Dunfermline', 'Greenock',
        'Falkirk', 'Cumbernauld', 'East Kilbride', 'Motherwell', 'Coatbridge',
        'Airdrie', 'Clydebank', 'Irvine', 'Dumfries', 'Glenrothes',
        'Musselburgh', 'Arbroath', 'Bathgate', 'Wishaw', 'Bellshill'
    ]
    
    for line in address_lines:
        line_clean = line.strip()
        for city in cities:
            if city.lower() in line_clean.lower():
                return city
        # If line looks like a city (no numbers, not a postcode)
        if line_clean and not re.search(r'\d', line_clean) and len(line_clean) < 30:
            if not re.match(r'^[A-Z]{1,2}\d', line_clean.upper()):
                return line_clean
    
    return ""


def fetch_factor_details(session: requests.Session, pf_number: str) -> FactorDetails:
    """Fetch and parse a factor's detail page."""
    
    details = FactorDetails(registration_number=pf_number, name="")
    
    url = f"{BASE_URL}/{pf_number}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 404:
                details.scrape_error = "Page not found (404)"
                details.status = "Expired"
                return details
            
            if response.status_code != 200:
                details.scrape_error = f"HTTP {response.status_code}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return details
            
            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get the title (factor name)
            title = soup.select_one('h1.ds_page-header__title')
            if title:
                details.name = title.get_text(strip=True)
            
            # Parse the summary list items
            # Structure: <li class="ds_summary-list__item">
            #              <span class="ds_summary-list__key">Label</span>
            #              <span class="ds_summary-list__value">Value</span>
            #            </li>
            
            for item in soup.select('li.ds_summary-list__item'):
                key_el = item.select_one('.ds_summary-list__key')
                value_el = item.select_one('.ds_summary-list__value')
                
                if not key_el or not value_el:
                    continue
                
                key = key_el.get_text(strip=True).lower()
                
                # Handle different value types
                if key == 'business address':
                    # Address has multiple lines with <br> tags
                    # Get text with newlines preserved
                    address_lines = []
                    for child in value_el.children:
                        if hasattr(child, 'get_text'):
                            text = child.get_text(strip=True)
                        else:
                            text = str(child).strip()
                        if text and text != '\n':
                            address_lines.append(text)
                    
                    details.address = ', '.join(address_lines)
                    details.postcode = extract_postcode(details.address)
                    details.city = extract_city(address_lines)
                    
                elif key == 'business website':
                    link = value_el.select_one('a')
                    if link:
                        details.website = link.get('href', '')
                    else:
                        details.website = value_el.get_text(strip=True)
                        
                elif key == 'number of properties factored':
                    value = value_el.get_text(strip=True)
                    # Extract number
                    match = re.search(r'(\d[\d,]*)', value.replace(',', ''))
                    details.property_count = int(match.group(1)) if match else 0
                    
                elif key == 'registration status':
                    details.status = value_el.get_text(strip=True)
                    
                elif key == 'company name':
                    # Use this if we didn't get name from title
                    if not details.name:
                        details.name = value_el.get_text(strip=True)
                        
                elif key == 'trading name':
                    details.trading_name = value_el.get_text(strip=True)
                    
                elif key == 'trading type':
                    details.trading_type = value_el.get_text(strip=True)
            
            details.scrape_success = True
            return details
            
        except requests.RequestException as e:
            details.scrape_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return details
        except Exception as e:
            details.scrape_error = f"Parse error: {e}"
            return details
    
    return details


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
    parser = argparse.ArgumentParser(description="Enrich factor data from detail pages")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--limit', type=int, help='Limit number of factors to process')
    parser.add_argument('--pf', type=str, help='Process single PF number (for testing)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROPERTY FACTOR REGISTRY ENRICHMENT")
    print("=" * 60)
    print(f"Source: {BASE_URL}/PF000XXX")
    print()
    
    # Single factor test mode
    if args.pf:
        session = requests.Session()
        session.headers.update(HEADERS)
        
        pf = args.pf.upper()
        if not pf.startswith('PF'):
            pf = f'PF{pf}'
        
        print(f"Testing single factor: {pf}")
        details = fetch_factor_details(session, pf)
        
        print(f"\nResults:")
        for key, value in asdict(details).items():
            print(f"  {key}: {value}")
        return
    
    # Check input file
    if not INPUT_CSV.exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        print("   Run the browser extraction script first.")
        return
    
    # Load factors from CSV
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        factors = list(reader)
    
    print(f"📋 Loaded {len(factors)} factors from {INPUT_CSV}")
    
    # Load checkpoint if resuming
    checkpoint = {'processed': [], 'results': []}
    if args.resume:
        checkpoint = load_checkpoint()
        print(f"   Resuming: {len(checkpoint['processed'])} already processed")
    
    processed_set = set(checkpoint['processed'])
    
    # Filter to unprocessed
    pending = []
    for f in factors:
        pf = f.get('registration_number', '').strip().upper()
        if pf and pf not in processed_set:
            pending.append(pf)
    
    # Apply limit
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} factors to process")
    
    if not pending:
        print("\n✅ All factors already processed!")
        # Still write output from checkpoint
        if checkpoint['results']:
            write_output(checkpoint['results'])
        return
    
    # Estimate time
    est_minutes = len(pending) * REQUEST_DELAY / 60
    print(f"   Estimated time: ~{est_minutes:.0f} minutes")
    print()
    
    # Create session
    session = requests.Session()
    session.headers.update(HEADERS)
    
    results = checkpoint['results']
    
    print("🔍 Fetching factor details...")
    print()
    
    success_count = 0
    error_count = 0
    
    for i, pf_number in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] {pf_number}...", end=" ", flush=True)
        
        details = fetch_factor_details(session, pf_number)
        
        if details.scrape_success:
            props = details.property_count
            print(f"✅ {details.name[:35]} ({props} properties)")
            success_count += 1
        else:
            print(f"⚠️ {details.scrape_error}")
            error_count += 1
        
        results.append(asdict(details))
        checkpoint['processed'].append(pf_number)
        checkpoint['results'] = results
        
        # Save checkpoint every 20 factors
        if (i + 1) % 20 == 0:
            save_checkpoint(checkpoint)
            print(f"   💾 Checkpoint saved ({i+1}/{len(pending)})")
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    # Final checkpoint save
    save_checkpoint(checkpoint)
    
    # Write output
    write_output(results)
    
    # Summary
    print()
    print("=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"⚠️ Errors: {error_count}")
    
    # Stats
    total_props = sum(r.get('property_count', 0) for r in results)
    with_props = sum(1 for r in results if r.get('property_count', 0) > 0)
    with_website = sum(1 for r in results if r.get('website'))
    registered = sum(1 for r in results if r.get('status') == 'Registered')
    
    print()
    print("📊 Statistics:")
    print(f"   Total factors: {len(results)}")
    print(f"   Registered: {registered}")
    print(f"   With property counts: {with_props}")
    print(f"   Total properties: {total_props:,}")
    print(f"   With websites: {with_website}")
    
    print()
    print(f"📄 Output saved to: {OUTPUT_CSV}")


def write_output(results: list):
    """Write results to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'registration_number', 'name', 'trading_name', 'trading_type',
        'address', 'city', 'postcode', 'website', 'property_count', 'status'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
