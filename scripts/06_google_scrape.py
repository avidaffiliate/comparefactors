#!/usr/bin/env python3
"""
Google Places API Scraper for Scottish Property Factors
========================================================
Uses the official Google Places API for reliable data extraction.

Setup:
    1. Go to https://console.cloud.google.com/
    2. Create a new project (or select existing)
    3. Enable "Places API" and "Places API (New)"
    4. Create an API key: APIs & Services > Credentials > Create Credentials > API Key
    5. Add billing (you get $200 free credit/month - this script costs ~$6 for 340 factors)

Usage:
    python google_places_api.py --api-key YOUR_API_KEY --search "James Gibb Property Management"
    python google_places_api.py --api-key YOUR_API_KEY --input factors.csv --output results.json

Pricing (as of 2024):
    - Find Place: $17 per 1,000 requests
    - Place Details: $17 per 1,000 requests
    - Total for 340 factors: ~$6-12 (depending on if multiple branches)
"""

import argparse
import csv
import json
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = Path("data/csv/factors_register.csv")
OUTPUT_CSV = Path("data/csv/google_reviews.csv")

# Scottish postcodes
SCOTTISH_POSTCODES = {'AB', 'DD', 'DG', 'EH', 'FK', 'G', 'HS', 'IV', 'KA', 'KW', 'KY', 'ML', 'PA', 'PH', 'TD', 'ZE'}

# Scotland bounds
SCOTLAND_LAT_MIN = 54.5
SCOTLAND_LAT_MAX = 61.0
SCOTLAND_LON_MIN = -8.0
SCOTLAND_LON_MAX = -0.5


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def load_factors_register(path: Path) -> Dict[str, dict]:
    """Load factors register for cross-referencing validation."""
    factors = {}
    if not path.exists():
        return factors
    
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg_num = row.get('registration_number', '').strip().upper()
            if reg_num:
                factors[reg_num] = {
                    'name': row.get('name', ''),
                    'trading_name': row.get('trading_name', ''),
                    'address': row.get('address', ''),
                    'city': row.get('city', ''),
                    'postcode': row.get('postcode', '').strip().upper(),
                    'website': row.get('website', '').strip(),
                    'status': row.get('status', ''),
                }
    return factors


def normalize_website(url: str) -> str:
    """Normalize website URL for comparison."""
    if not url:
        return ''
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    url = url.split('/')[0]
    return url


def websites_match(url1: str, url2: str) -> bool:
    """Check if two websites match."""
    if not url1 or not url2:
        return False
    norm1 = normalize_website(url1)
    norm2 = normalize_website(url2)
    if not norm1 or not norm2:
        return False
    return norm1 == norm2 or norm1 in norm2 or norm2 in norm1


def extract_postcode(address: str) -> str:
    """Extract postcode from address."""
    if not address:
        return ''
    match = re.search(r'([A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2})', address.upper())
    return match.group(1).replace(' ', '') if match else ''


def get_postcode_area(postcode: str) -> str:
    """Get area prefix from postcode (e.g., 'EH8 9DH' -> 'EH')."""
    if not postcode:
        return ''
    match = re.match(r'^([A-Z]{1,2})\d', postcode.upper())
    return match.group(1) if match else ''


def is_scottish_postcode(postcode: str) -> bool:
    """Check if postcode is Scottish."""
    area = get_postcode_area(postcode)
    return area in SCOTTISH_POSTCODES


def is_in_england(address: str) -> bool:
    """Check if address is clearly in England."""
    if not address:
        return False
    address_lower = address.lower()
    
    england_keywords = [
        'london', 'manchester', 'birmingham', 'liverpool', 'leeds', 'bristol',
        'newcastle upon tyne', 'sheffield', 'nottingham', 'leicester',
        'upon tyne', 'upon-tyne'
    ]
    
    for kw in england_keywords:
        if kw in address_lower:
            # Skip if it's just a street name
            if kw == 'london' and ('london rd' in address_lower or 'london road' in address_lower):
                continue
            return True
    
    # Check for English postcodes
    postcode = extract_postcode(address)
    if postcode and not is_scottish_postcode(postcode):
        area = get_postcode_area(postcode)
        if area:  # Has a valid postcode that's not Scottish
            return True
    
    return False


def name_similarity(name1: str, name2: str) -> float:
    """Calculate name similarity (0-1)."""
    if not name1 or not name2:
        return 0.0
    
    # Normalize
    def normalize(s):
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        return set(s.split())
    
    # Remove common words
    stopwords = {'ltd', 'limited', 'property', 'management', 'services', 
                 'factors', 'factoring', 'the', 'and', 'of', 'uk', 'scotland'}
    
    words1 = normalize(name1) - stopwords
    words2 = normalize(name2) - stopwords
    
    if not words1 or not words2:
        return 0.3
    
    # Jaccard similarity
    overlap = len(words1 & words2)
    total = len(words1 | words2)
    
    return overlap / total if total > 0 else 0.0


def validate_result(factor_data: dict, google_data: dict, factors_register: Dict) -> Tuple[bool, str]:
    """
    Validate a Google result against the factors register.
    Returns (is_valid, rejection_reason).
    """
    if not google_data:
        return False, "No Google data"
    
    pf = factor_data.get('registration_number', '').upper()
    factor_name = factor_data.get('name', '')
    
    google_name = google_data.get('google_name', '')
    google_address = google_data.get('google_address', '')
    google_website = google_data.get('google_website', '')
    lat = google_data.get('latitude')
    lon = google_data.get('longitude')
    business_types = google_data.get('types', [])
    business_status = google_data.get('business_status', '')
    
    # Get registered data
    registered = factors_register.get(pf, {})
    registered_postcode = registered.get('postcode', '')
    registered_website = registered.get('website', '')
    registered_city = registered.get('city', '')
    trading_name = registered.get('trading_name', '')
    
    google_postcode = extract_postcode(google_address)
    
    # ===================
    # HARD REJECTIONS
    # ===================
    
    # Check business status
    if business_status in ['CLOSED_PERMANENTLY']:
        return False, f"Business permanently closed"
    
    # Check business type - reject clearly wrong types
    wrong_business_types = {
        'restaurant', 'cafe', 'bar', 'hotel', 'lodging', 'food',
        'meal_takeaway', 'meal_delivery', 'bakery', 'grocery_store',
        'supermarket', 'clothing_store', 'shoe_store', 'hair_care',
        'beauty_salon', 'gym', 'hospital', 'doctor', 'dentist',
        'pharmacy', 'bank', 'atm', 'gas_station', 'car_dealer',
        'car_repair', 'car_wash', 'parking', 'transit_station',
        'airport', 'church', 'mosque', 'synagogue', 'cemetery',
        'funeral_home', 'pet_store', 'veterinary_care', 'zoo',
        'aquarium', 'museum', 'art_gallery', 'movie_theater',
        'night_club', 'casino', 'amusement_park', 'bowling_alley',
        'school', 'university', 'library', 'post_office', 'police',
        'fire_station', 'embassy', 'city_hall', 'courthouse',
    }
    
    if business_types:
        type_set = set(t.lower() for t in business_types)
        wrong_matches = type_set & wrong_business_types
        if wrong_matches and 'real_estate' not in ' '.join(business_types).lower():
            return False, f"Wrong business type: {', '.join(wrong_matches)}"
    
    # Check if outside UK by coordinates
    if lat is not None:
        try:
            lat_f = float(lat)
            if lat_f < 0:
                return False, f"Southern hemisphere (lat={lat_f:.1f})"
            if lat_f < 50:
                return False, f"Outside UK (lat={lat_f:.1f})"
            if lat_f < 54.5 and is_in_england(google_address):
                return False, f"England (lat={lat_f:.1f})"
        except (ValueError, TypeError):
            pass
    
    # Check if clearly in England
    if is_in_england(google_address):
        return False, f"English location: {google_address[:50]}"
    
    # Check for non-UK addresses
    if google_address:
        non_uk = ['australia', 'usa', 'canada', 'new zealand', ' nsw ', ' qld ', ' vic ']
        if any(kw in google_address.lower() for kw in non_uk):
            return False, f"Outside UK: {google_address[:50]}"
    
    # ===================
    # STRONG ACCEPTS
    # ===================
    
    # Website match = confirmed
    if registered_website and google_website:
        if websites_match(registered_website, google_website):
            return True, ""
    
    # Exact postcode match = confirmed
    if registered_postcode and google_postcode:
        reg_pc = registered_postcode.replace(' ', '').upper()
        goog_pc = google_postcode.replace(' ', '').upper()
        if reg_pc == goog_pc:
            return True, ""
    
    # ===================
    # VALIDATION CHECKS
    # ===================
    
    # Name similarity
    name_sim = name_similarity(factor_name, google_name)
    trading_sim = name_similarity(trading_name, google_name) if trading_name else 0
    best_sim = max(name_sim, trading_sim)
    
    # Postcode area match
    postcode_area_ok = False
    if registered_postcode and google_postcode:
        postcode_area_ok = get_postcode_area(registered_postcode) == get_postcode_area(google_postcode)
    
    # City match
    city_ok = False
    if registered_city and google_address:
        city_ok = registered_city.lower() in google_address.lower()
    
    # Is it in Scotland?
    in_scotland = is_scottish_postcode(google_postcode) if google_postcode else False
    
    # Decision logic
    if best_sim >= 0.5:
        if in_scotland or postcode_area_ok or city_ok:
            return True, ""
    
    if best_sim >= 0.3:
        if postcode_area_ok or city_ok:
            return True, ""
        if in_scotland:
            return True, ""
    
    # Low name match - need strong location confirmation
    if postcode_area_ok and in_scotland:
        return True, ""
    if city_ok and in_scotland:
        return True, ""
    
    # Rejection
    if best_sim < 0.3:
        return False, f"Low name match ({best_sim:.0%}): {google_name[:30]}"
    if not in_scotland and not postcode_area_ok:
        return False, f"Location not confirmed: {google_address[:40]}"
    
    # Marginal - accept
    return True, ""


@dataclass
class PlaceResult:
    """Structured result from Google Places API."""
    factor_name: str
    place_id: Optional[str] = None
    google_name: Optional[str] = None
    google_rating: Optional[float] = None
    google_review_count: Optional[int] = None
    google_address: Optional[str] = None
    google_phone: Optional[str] = None
    google_website: Optional[str] = None
    google_maps_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    business_status: Optional[str] = None
    types: list = field(default_factory=list)
    opening_hours: Optional[dict] = None
    price_level: Optional[int] = None
    reviews: list = field(default_factory=list)  # Up to 5 recent reviews
    scraped_at: str = ""
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()


class GooglePlacesAPI:
    """
    Google Places API client for extracting business data.
    
    Uses the official API - reliable, fast, and no blocking issues.
    """
    
    BASE_URL = "https://maps.googleapis.com/maps/api/place"
    
    def __init__(self, api_key: str, delay: float = 0.1):
        """
        Initialize the API client.
        
        Args:
            api_key: Your Google Places API key
            delay: Seconds between requests (to avoid rate limits)
        """
        self.api_key = api_key
        self.delay = delay
        self.session = requests.Session()
        self.request_count = 0
    
    def find_place(self, query: str, location_bias: str = "Scotland, UK") -> Optional[str]:
        """
        Find a place and return its place_id.
        
        Args:
            query: Business name to search for
            location_bias: Location context for better results
            
        Returns:
            place_id if found, None otherwise
        """
        url = f"{self.BASE_URL}/findplacefromtext/json"
        
        params = {
            "input": f"{query} {location_bias}",
            "inputtype": "textquery",
            "fields": "place_id,name,formatted_address",
            "key": self.api_key
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            self.request_count += 1
            
            if data.get("status") == "OK" and data.get("candidates"):
                return data["candidates"][0].get("place_id")
            elif data.get("status") == "ZERO_RESULTS":
                return None
            else:
                logger.warning(f"Find place failed: {data.get('status')} - {data.get('error_message', '')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def text_search(self, query: str, location: str = "Scotland") -> list[dict]:
        """
        Search for places matching the query (returns multiple results).
        
        Args:
            query: Search query
            location: Location context
            
        Returns:
            List of place dictionaries with basic info
        """
        url = f"{self.BASE_URL}/textsearch/json"
        
        params = {
            "query": f"{query} {location}",
            "key": self.api_key
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            self.request_count += 1
            
            if data.get("status") == "OK":
                return data.get("results", [])
            else:
                logger.warning(f"Text search failed: {data.get('status')}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
    
    def get_place_details(self, place_id: str) -> Optional[dict]:
        """
        Get detailed information about a place.
        
        Args:
            place_id: Google Place ID
            
        Returns:
            Dictionary with place details
        """
        url = f"{self.BASE_URL}/details/json"
        
        # Request all useful fields
        fields = [
            "place_id",
            "name",
            "formatted_address",
            "formatted_phone_number",
            "international_phone_number",
            "website",
            "url",  # Google Maps URL
            "rating",
            "user_ratings_total",
            "geometry",
            "business_status",
            "types",
            "opening_hours",
            "price_level",
            "reviews",  # Up to 5 most recent reviews
        ]
        
        params = {
            "place_id": place_id,
            "fields": ",".join(fields),
            "key": self.api_key
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            self.request_count += 1
            
            if data.get("status") == "OK":
                return data.get("result")
            else:
                logger.warning(f"Place details failed: {data.get('status')} - {data.get('error_message', '')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def search_factor(self, factor_name: str, get_all_branches: bool = False) -> list[PlaceResult]:
        """
        Search for a property factor and get their Google data.
        
        Args:
            factor_name: Name of the property factor
            get_all_branches: If True, return all matching locations
            
        Returns:
            List of PlaceResult objects (usually 1, more if get_all_branches=True)
        """
        results = []
        
        # Build search query
        search_name = factor_name
        if 'factor' not in factor_name.lower() and 'management' not in factor_name.lower():
            search_name = f"{factor_name} property management"
        
        time.sleep(self.delay)
        
        if get_all_branches:
            # Use text search to get multiple results
            places = self.text_search(search_name, "Scotland")
            
            for place in places[:5]:  # Limit to top 5 matches
                place_id = place.get("place_id")
                if place_id:
                    time.sleep(self.delay)
                    details = self.get_place_details(place_id)
                    if details:
                        result = self._parse_details(details, factor_name)
                        results.append(result)
        else:
            # Find single best match
            place_id = self.find_place(search_name, "Scotland, UK")
            
            if place_id:
                time.sleep(self.delay)
                details = self.get_place_details(place_id)
                if details:
                    result = self._parse_details(details, factor_name)
                    results.append(result)
            else:
                # No result found
                results.append(PlaceResult(
                    factor_name=factor_name,
                    error="No matching place found"
                ))
        
        return results
    
    def _parse_details(self, details: dict, factor_name: str) -> PlaceResult:
        """Parse API response into PlaceResult."""
        geometry = details.get("geometry", {}).get("location", {})
        
        # Parse opening hours if available
        opening_hours = None
        if details.get("opening_hours"):
            opening_hours = {
                "open_now": details["opening_hours"].get("open_now"),
                "weekday_text": details["opening_hours"].get("weekday_text", [])
            }
        
        # Parse reviews if available
        reviews = []
        for review in details.get("reviews", []):
            reviews.append({
                "author_name": review.get("author_name"),
                "rating": review.get("rating"),
                "text": review.get("text"),
                "time": review.get("time"),  # Unix timestamp
                "relative_time": review.get("relative_time_description"),
            })
        
        return PlaceResult(
            factor_name=factor_name,
            place_id=details.get("place_id"),
            google_name=details.get("name"),
            google_rating=details.get("rating"),
            google_review_count=details.get("user_ratings_total"),
            google_address=details.get("formatted_address"),
            google_phone=details.get("formatted_phone_number") or details.get("international_phone_number"),
            google_website=details.get("website"),
            google_maps_url=details.get("url"),
            latitude=geometry.get("lat"),
            longitude=geometry.get("lng"),
            business_status=details.get("business_status"),
            types=details.get("types", []),
            opening_hours=opening_hours,
            price_level=details.get("price_level"),
            reviews=reviews,
        )
    
    def search_multiple_factors(self, factor_names: list[str], 
                                 get_all_branches: bool = False,
                                 output_file: str = None) -> list[PlaceResult]:
        """
        Search for multiple factors with progress tracking.
        
        Args:
            factor_names: List of factor names
            get_all_branches: Get all locations for each factor
            output_file: Optional JSON file to save results progressively
            
        Returns:
            List of all PlaceResult objects
        """
        all_results = []
        total = len(factor_names)
        
        logger.info(f"Searching {total} factors...")
        
        for i, name in enumerate(factor_names, 1):
            print(f"[{i}/{total}] {name}")
            
            try:
                results = self.search_factor(name, get_all_branches)
                
                for result in results:
                    all_results.append(result)
                    
                    if result.google_rating:
                        print(f"    ✓ {result.google_name}: {result.google_rating}★ ({result.google_review_count} reviews)")
                    elif result.error:
                        print(f"    ✗ {result.error}")
                    else:
                        print(f"    ~ {result.google_name} (no rating)")
                
                # Save progress periodically
                if output_file and i % 10 == 0:
                    self._save_results(all_results, output_file)
                    
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                all_results.append(PlaceResult(factor_name=name, error=str(e)))
        
        # Final save
        if output_file:
            self._save_results(all_results, output_file)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: list[PlaceResult], filepath: str):
        """Save results to JSON file."""
        output = []
        for r in results:
            d = asdict(r)
            # Convert types list for JSON
            output.append(d)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self, results: list[PlaceResult]):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        total = len(results)
        with_rating = [r for r in results if r.google_rating]
        errors = [r for r in results if r.error]
        
        print(f"Total results:     {total}")
        print(f"With rating:       {len(with_rating)} ({len(with_rating)/total*100:.1f}%)")
        print(f"Errors/Not found:  {len(errors)}")
        print(f"API requests made: {self.request_count}")
        
        if with_rating:
            ratings = [r.google_rating for r in with_rating]
            reviews = [r.google_review_count or 0 for r in with_rating]
            
            print(f"\nAverage rating:    {sum(ratings)/len(ratings):.2f}★")
            print(f"Rating range:      {min(ratings):.1f}★ - {max(ratings):.1f}★")
            print(f"Total reviews:     {sum(reviews):,}")
        
        # Estimate cost
        cost_per_1000 = 17  # $17 per 1000 requests
        estimated_cost = (self.request_count / 1000) * cost_per_1000
        print(f"\nEstimated cost:    ${estimated_cost:.2f}")


def export_to_csv(results: list[PlaceResult], filepath: str):
    """Export results to CSV."""
    if not results:
        return
    
    # Flatten for CSV (remove nested dicts/lists)
    rows = []
    for r in results:
        row = {
            'factor_name': r.factor_name,
            'place_id': r.place_id,
            'google_name': r.google_name,
            'google_rating': r.google_rating,
            'google_review_count': r.google_review_count,
            'google_address': r.google_address,
            'google_phone': r.google_phone,
            'google_website': r.google_website,
            'google_maps_url': r.google_maps_url,
            'latitude': r.latitude,
            'longitude': r.longitude,
            'business_status': r.business_status,
            'types': ', '.join(r.types) if r.types else '',
            'scraped_at': r.scraped_at,
            'error': r.error
        }
        rows.append(row)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Exported {len(rows)} results to {filepath}")


def load_factors_from_csv(filepath: str, column: str = 'name') -> list[str]:
    """Load factor names from CSV file."""
    names = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try common column names
            for col in [column, 'name', 'registered_name', 'trading_name', 'factor_name']:
                if col in row and row[col]:
                    names.append(row[col].strip())
                    break
    return names


def main():
    parser = argparse.ArgumentParser(
        description='Google Places API scraper for Scottish property factors (with validation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Scrape remaining factors:
    python google_places_api.py --api-key YOUR_KEY
    
  Test with a few factors:
    python google_places_api.py --api-key YOUR_KEY --limit 5
    
  Single search:
    python google_places_api.py --api-key YOUR_KEY --search "James Gibb"

Setup:
  1. Go to https://console.cloud.google.com/
  2. Enable "Places API"
  3. Create API key: APIs & Services > Credentials > Create Credentials
  4. Set GOOGLE_API_KEY env var or pass --api-key
        """
    )
    
    parser.add_argument('--api-key', '-k', 
                        default=os.environ.get('GOOGLE_API_KEY'),
                        help='Google Places API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--search', '-s', help='Single factor to search')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of factors to process')
    parser.add_argument('--delay', '-d', type=float, default=0.1, help='Delay between API calls (seconds)')
    parser.add_argument('--include-expired', action='store_true', help='Include expired registrations')
    parser.add_argument('--rescrape', action='store_true', help='Re-scrape all factors (ignore existing data)')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ API key required. Set GOOGLE_API_KEY or use --api-key")
        return
    
    api = GooglePlacesAPI(api_key=args.api_key, delay=args.delay)
    
    if args.search:
        # Single search mode
        print(f"Searching for: {args.search}")
        results = api.search_factor(args.search, get_all_branches=True)
        
        print(f"\nFound {len(results)} location(s):\n")
        for i, result in enumerate(results, 1):
            print(f"--- Location {i} ---")
            print(f"Name:     {result.google_name}")
            print(f"Rating:   {result.google_rating}★ ({result.google_review_count} reviews)")
            print(f"Address:  {result.google_address}")
            print(f"Phone:    {result.google_phone}")
            print(f"Website:  {result.google_website}")
            print(f"Maps URL: {result.google_maps_url}")
            print()
        return
    
    # ===================
    # BATCH MODE
    # ===================
    
    print("=" * 70)
    print("GOOGLE PLACES API SCRAPER (with validation)")
    print("=" * 70)
    
    # Load factors register
    if not INPUT_CSV.exists():
        print(f"❌ Factors register not found: {INPUT_CSV}")
        return
    
    factors_register = load_factors_register(INPUT_CSV)
    print(f"📋 Loaded {len(factors_register)} factors from register")
    
    # Load all factors with their data
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        all_factors = list(csv.DictReader(f))
    
    # Load existing results to avoid re-scraping (unless --rescrape)
    existing_reg_nums = set()
    existing_names = set()
    existing_results = []
    
    if not args.rescrape and OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, 'r', encoding='utf-8-sig') as f:
            existing_results = list(csv.DictReader(f))
            for row in existing_results:
                reg = row.get('factor_registration_number', '').strip()
                name = row.get('factor_name', '').strip()
                if reg:
                    existing_reg_nums.add(reg)
                if name:
                    existing_names.add(name)
        print(f"   Found {len(existing_reg_nums)} already scraped in {OUTPUT_CSV.name}")
    
    # Filter factors to process
    pending = []
    skipped_existing = 0
    skipped_expired = 0
    
    for f in all_factors:
        reg = f.get('registration_number', '').strip()
        name = f.get('name', '').strip()
        status = f.get('status', '').lower()
        
        # Skip if already scraped
        if reg in existing_reg_nums or name in existing_names:
            skipped_existing += 1
            continue
        
        # Skip expired unless requested
        if 'expired' in status and not args.include_expired:
            skipped_expired += 1
            continue
        
        pending.append(f)
    
    if args.rescrape:
        print(f"   🔄 Re-scraping all factors (--rescrape)")
    else:
        print(f"   Skipped {skipped_existing} already scraped")
    print(f"   Skipped {skipped_expired} expired registrations")
    
    # Apply limit
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} factors to process")
    
    if not pending:
        print("\n✅ All factors already scraped!")
        return
    
    # Estimate cost (text search + details with reviews per branch)
    # Reviews field adds ~$0.005 per place
    est_cost = len(pending) * 0.06  # ~$0.032 text search + ~$0.025 per branch (details + reviews)
    print(f"\n💰 Estimated cost: ~${est_cost:.2f} (multi-branch + reviews)")
    print(f"⏱️  Estimated time: {len(pending) * 0.5:.0f} seconds\n")
    
    # Process factors
    new_results = []
    all_reviews = []  # Store individual reviews separately
    verified_count = 0
    rejected_count = 0
    not_found_count = 0
    branch_count = 0
    
    for i, factor in enumerate(pending, 1):
        reg = factor.get('registration_number', '')
        name = factor.get('name', '')
        city = factor.get('city', 'Scotland')
        
        print(f"[{i}/{len(pending)}] {name[:50]}...", end=" ", flush=True)
        
        # Search with multi-branch enabled
        results = api.search_factor(name, get_all_branches=True)
        
        if not results or not results[0].google_name:
            not_found_count += 1
            print("❌ Not found")
            continue
        
        # Process each branch/location
        factor_verified = 0
        factor_rejected = 0
        
        for loc_num, result in enumerate(results, 1):
            # Convert to dict for validation
            google_data = {
                'google_name': result.google_name,
                'google_address': result.google_address,
                'google_website': result.google_website,
                'latitude': result.latitude,
                'longitude': result.longitude,
                'types': result.types,
                'business_status': result.business_status,
            }
            
            # Validate each location
            is_valid, rejection_reason = validate_result(factor, google_data, factors_register)
            
            if is_valid:
                # Save verified result as individual row
                row = {
                    'factor_registration_number': reg,
                    'factor_name': name,
                    'google_place_id': result.place_id,
                    'google_name': result.google_name,
                    'rating': result.google_rating,
                    'review_count': result.google_review_count or 0,
                    'address': result.google_address,
                    'phone': result.google_phone,
                    'website': result.google_website,
                    'is_verified': True,
                    'google_maps_url': result.google_maps_url,
                    'latitude': result.latitude,
                    'longitude': result.longitude,
                    'location_number': loc_num,
                    'business_status': result.business_status,
                    'business_types': ', '.join(result.types) if result.types else '',
                }
                new_results.append(row)
                
                # Store reviews separately
                for review in result.reviews:
                    all_reviews.append({
                        'factor_registration_number': reg,
                        'factor_name': name,
                        'google_place_id': result.place_id,
                        'location_number': loc_num,
                        'author_name': review.get('author_name', ''),
                        'review_rating': review.get('rating'),
                        'review_text': review.get('text', ''),
                        'review_time': review.get('time'),
                        'review_relative_time': review.get('relative_time', ''),
                    })
                
                factor_verified += 1
                branch_count += 1
            else:
                factor_rejected += 1
        
        # Report for this factor
        if factor_verified > 0:
            verified_count += 1
            if factor_verified == 1:
                r = results[0]
                if r.google_rating:
                    print(f"✅ {r.google_rating}★ ({r.google_review_count} reviews)")
                else:
                    print(f"✅ (no rating)")
            else:
                # Multiple locations
                total_reviews = sum(r.google_review_count or 0 for r in results[:factor_verified])
                print(f"✅ {factor_verified} locations ({total_reviews} total reviews)")
        else:
            rejected_count += 1
            # Show first rejection reason
            google_data = {
                'google_name': results[0].google_name,
                'google_address': results[0].google_address,
                'google_website': results[0].google_website,
                'latitude': results[0].latitude,
                'longitude': results[0].longitude,
                'types': results[0].types,
                'business_status': results[0].business_status,
            }
            _, reason = validate_result(factor, google_data, factors_register)
            print(f"❌ Rejected: {reason}")
        
        # Checkpoint every 20
        if i % 20 == 0:
            _save_results(existing_results + new_results, OUTPUT_CSV)
            if all_reviews:
                reviews_csv = OUTPUT_CSV.parent / "google_reviews_text.csv"
                _save_reviews(all_reviews, reviews_csv)
            print(f"   [Checkpoint: {verified_count} factors, {branch_count} locations, {len(all_reviews)} reviews]")
    
    # Final save
    all_results = existing_results + new_results
    _save_results(all_results, OUTPUT_CSV)
    
    # Save reviews to separate file
    if all_reviews:
        reviews_csv = OUTPUT_CSV.parent / "google_reviews_text.csv"
        _save_reviews(all_reviews, reviews_csv)
    
    # Summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"✅ Factors verified: {verified_count}")
    print(f"✅ Total locations:  {branch_count}")
    print(f"✅ Reviews captured: {len(all_reviews)}")
    print(f"❌ Factors rejected: {rejected_count}")
    print(f"❌ Not found:        {not_found_count}")
    print(f"📊 Total rows in CSV: {len(all_results)}")
    print(f"💰 API requests: {api.request_count}")
    print(f"💰 Est. cost:    ${api.request_count * 0.017:.2f}")
    print(f"\n✅ Results saved to {OUTPUT_CSV}")
    if all_reviews:
        print(f"✅ Reviews saved to {reviews_csv}")


def _save_results(results: list, filepath: Path):
    """Save results to CSV."""
    if not results:
        return
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'factor_registration_number', 'factor_name', 'google_place_id',
        'google_name', 'rating', 'review_count', 'address', 'phone',
        'website', 'is_verified', 'google_maps_url', 'latitude', 'longitude',
        'location_number', 'business_status', 'business_types'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def _save_reviews(reviews: list, filepath: Path):
    """Save reviews to separate CSV."""
    if not reviews:
        return
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'factor_registration_number', 'factor_name', 'google_place_id',
        'location_number', 'author_name', 'review_rating', 'review_text',
        'review_time', 'review_relative_time'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reviews)


if __name__ == "__main__":
    main()