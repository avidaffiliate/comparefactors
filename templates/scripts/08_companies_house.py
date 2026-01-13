#!/usr/bin/env python3
"""
============================================================================
SCRIPT 08: COMPANIES HOUSE SCRAPER (FULL DATA)
============================================================================

Comprehensive Companies House data extraction with proper fuzzy matching.

CAPTURES:
    - Company profile (status, type, incorporation, SIC codes)
    - Red flags (insolvency history, charges, disputed address)
    - Previous company names (rebrand tracking)
    - Directors (count, tenure, names)
    - PSCs (persons with significant control / beneficial owners)
    - Filing history (late filings, overdue accounts)
    - Accounts status

USAGE:
    CH_API_KEY=your_key python 08_companies_house_full.py
    python 08_companies_house_full.py --resume
    python 08_companies_house_full.py --limit 10
    python 08_companies_house_full.py --min-score 75

DEPENDENCIES:
    pip install requests rapidfuzz

TIME: ~25-30 minutes for 675 factors (3 API calls per factor)
============================================================================
"""

import os
import csv
import json
import time
import re
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple

import requests

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("⚠️  rapidfuzz not installed. Run: pip install rapidfuzz")
    print("   Falling back to basic matching (less accurate)")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = Path("data/csv/factors_register.csv")
OUTPUT_CSV = Path("data/csv/companies_house.csv")
CHECKPOINT_FILE = Path("data/companies_house_checkpoint.json")
LOG_FILE = Path("data/ch_matching.log")

API_KEY = os.getenv("CH_API_KEY", "YOUR_API_KEY_HERE")
API_BASE = "https://api.company-information.service.gov.uk"

# Matching
MIN_MATCH_SCORE = 70
SCOTTISH_BONUS = 5
ACTIVE_BONUS = 3

# Rate limiting (600 requests per 5 min = 2/sec, but be conservative)
REQUEST_DELAY = 0.6
REQUEST_TIMEOUT = 15

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# NAME NORMALIZATION & MATCHING
# ============================================================================

# Words that appear in many factor names and add noise to matching
# These will be removed before comparison to focus on distinctive terms
STOPWORDS = {
    # Legal suffixes (39%+ of names)
    'ltd', 'limited', 'llp', 'plc', 'inc', 'company', 'co', 'lp',
    # Industry terms (common in property factors)
    'property', 'properties', 'housing', 'management', 'services',
    'factors', 'factoring', 'factor', 'lettings', 'letting',
    'estates', 'estate', 'homes', 'residential', 'maintenance',
    'association', 'solutions', 'group', 'trust', 'developments',
    # Geographic (will match wrong companies otherwise)
    'scotland', 'scottish', 'uk', 'british', 'national',
    'edinburgh', 'glasgow', 'aberdeen', 'dundee', 'inverness',
    'highland', 'highlands', 'lowland', 'central', 'north', 'south', 'east', 'west',
    # Generic words
    'the', 'and', 'of', 'for', 'a', 'an',
}


def normalize_company_name(name: str, remove_stopwords: bool = False) -> str:
    """Normalize company name for comparison.
    
    Args:
        name: Company name to normalize
        remove_stopwords: If True, remove common industry words for matching
    """
    if not name:
        return ""
    
    name = name.lower().strip()
    
    # Remove bracketed suffixes
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)  # (scotland), (uk), (expired), etc.
    
    # Normalize & -> and
    name = name.replace('&', ' and ')
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Tokenize
    words = name.split()
    
    # Remove stopwords if requested
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    
    # Normalize whitespace
    return ' '.join(words)


def normalize_for_matching(name: str) -> str:
    """Normalize name specifically for matching (removes stopwords)."""
    return normalize_company_name(name, remove_stopwords=True)


def calculate_match_score(query: str, candidate: str) -> Tuple[float, str, str]:
    """Calculate similarity score between two names.
    
    Returns:
        Tuple of (score, query_normalized, candidate_normalized)
    """
    
    # First normalize with stopwords for matching
    q_match = normalize_for_matching(query)
    c_match = normalize_for_matching(candidate)
    
    # Also get full normalized versions for logging
    q_full = normalize_company_name(query, remove_stopwords=False)
    c_full = normalize_company_name(candidate, remove_stopwords=False)
    
    # If stopword removal leaves nothing, fall back to full normalization
    if not q_match or not c_match:
        q_match = q_full
        c_match = c_full
    
    if not q_match or not c_match:
        return 0.0, q_match, c_match
    
    # Exact match after normalization (with stopwords removed)
    if q_match == c_match:
        return 100.0, q_match, c_match
    
    if HAS_RAPIDFUZZ:
        # Use token_sort_ratio for word order independence
        score = fuzz.token_sort_ratio(q_match, c_match)
    else:
        # Basic Jaccard similarity fallback
        q_words = set(q_match.split())
        c_words = set(c_match.split())
        
        if not q_words or not c_words:
            return 0.0, q_match, c_match
        
        overlap = len(q_words & c_words)
        total = len(q_words | c_words)
        score = (overlap / total) * 100 if total > 0 else 0
    
    return score, q_match, c_match


def find_best_match(
    query: str, 
    results: List[Dict],
    min_score: float = MIN_MATCH_SCORE
) -> Tuple[Optional[Dict], float, str]:
    """Find the best matching company from search results.
    
    Returns:
        Tuple of (best_match, score, match_debug_info)
    """
    
    if not results:
        return None, 0.0, "no results"
    
    scored_results = []
    
    for r in results:
        company_name = r.get('title', '')
        company_number = r.get('company_number', '')
        status = r.get('company_status', '')
        
        score, q_norm, c_norm = calculate_match_score(query, company_name)
        
        # Bonuses
        bonus = 0
        if company_number.startswith(('SC', 'SO', 'SL', 'SP')):  # Scottish prefixes
            bonus += SCOTTISH_BONUS
        if status == 'active':
            bonus += ACTIVE_BONUS
        
        total_score = score + bonus
        
        scored_results.append({
            'result': r,
            'base_score': score,
            'total_score': total_score,
            'company_name': company_name,
            'query_normalized': q_norm,
            'candidate_normalized': c_norm,
        })
    
    scored_results.sort(key=lambda x: x['total_score'], reverse=True)
    best = scored_results[0]
    
    # Create debug info
    debug_info = f"'{best['query_normalized']}' vs '{best['candidate_normalized']}'"
    
    # Log for debugging
    logger.debug(f"Query: {query}")
    logger.debug(f"  Normalized: '{best['query_normalized']}'")
    for i, sr in enumerate(scored_results[:3]):
        logger.debug(f"  #{i+1}: {sr['company_name'][:40]} = {sr['base_score']:.0f}% ('{sr['candidate_normalized']}')")
    
    if best['base_score'] >= min_score:
        return best['result'], best['base_score'], debug_info
    else:
        logger.info(f"No match: '{query}' -> best: '{best['company_name'][:40]}' ({best['base_score']:.0f}%)")
        logger.info(f"  Compared: {debug_info}")
        return None, best['base_score'], debug_info


# ============================================================================
# API CLIENT
# ============================================================================

class CompaniesHouseClient:
    """Companies House API client with all endpoints."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.request_count = 0
        
        auth = base64.b64encode(f"{api_key}:".encode()).decode()
        self.session.headers.update({
            'Authorization': f'Basic {auth}',
            'Accept': 'application/json',
        })
    
    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limit handling."""
        url = f"{API_BASE}{endpoint}"
        self.request_count += 1
        
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                print("\n    ⏳ Rate limited, waiting 60s...", end="", flush=True)
                time.sleep(60)
                print(" resuming")
                return self._get(endpoint, params)
            else:
                logger.warning(f"API error {response.status_code}: {endpoint}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def search_companies(self, query: str, items_per_page: int = 20) -> List[Dict]:
        """Search for companies by name."""
        data = self._get('/search/companies', {
            'q': query,
            'items_per_page': items_per_page,
        })
        return data.get('items', []) if data else []
    
    def get_company(self, company_number: str) -> Optional[Dict]:
        """Get company profile."""
        return self._get(f'/company/{company_number}')
    
    def get_officers(self, company_number: str) -> List[Dict]:
        """Get company officers (directors, secretaries)."""
        data = self._get(f'/company/{company_number}/officers')
        return data.get('items', []) if data else []
    
    def get_pscs(self, company_number: str) -> List[Dict]:
        """Get persons with significant control (beneficial owners)."""
        data = self._get(f'/company/{company_number}/persons-with-significant-control')
        return data.get('items', []) if data else []
    
    def get_filing_history(self, company_number: str, items_per_page: int = 50) -> List[Dict]:
        """Get recent filing history."""
        data = self._get(f'/company/{company_number}/filing-history', {
            'items_per_page': items_per_page,
        })
        return data.get('items', []) if data else []
    
    def get_charges(self, company_number: str) -> List[Dict]:
        """Get charges/mortgages (only if has_charges is True)."""
        data = self._get(f'/company/{company_number}/charges')
        return data.get('items', []) if data else []


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_director_info(officers: List[Dict]) -> Dict:
    """Extract director information."""
    
    directors = [
        o for o in officers
        if o.get('officer_role') == 'director' and not o.get('resigned_on')
    ]
    
    secretaries = [
        o for o in officers
        if o.get('officer_role') == 'secretary' and not o.get('resigned_on')
    ]
    
    tenures = []
    today = date.today()
    
    director_details = []
    for d in directors:
        appointed = d.get('appointed_on')
        years = None
        if appointed:
            try:
                start = datetime.strptime(appointed, '%Y-%m-%d').date()
                years = round((today - start).days / 365.25, 1)
                tenures.append(years)
            except ValueError:
                pass
        
        director_details.append({
            'name': d.get('name', ''),
            'appointed': appointed or '',
            'tenure_years': years,
            'occupation': d.get('occupation', ''),
            'nationality': d.get('nationality', ''),
            'country_of_residence': d.get('country_of_residence', ''),
        })
    
    return {
        'director_count': len(directors),
        'secretary_count': len(secretaries),
        'directors': director_details,
        'avg_director_tenure_years': round(sum(tenures) / len(tenures), 1) if tenures else None,
        'min_director_tenure_years': round(min(tenures), 1) if tenures else None,
        'max_director_tenure_years': round(max(tenures), 1) if tenures else None,
    }


def extract_psc_info(pscs: List[Dict]) -> Dict:
    """Extract persons with significant control."""
    
    active_pscs = [p for p in pscs if not p.get('ceased_on')]
    
    psc_details = []
    for p in active_pscs:
        psc_details.append({
            'name': p.get('name', p.get('name_elements', {}).get('surname', '')),
            'kind': p.get('kind', ''),
            'natures_of_control': p.get('natures_of_control', []),
            'notified_on': p.get('notified_on', ''),
            'nationality': p.get('nationality', ''),
            'country_of_residence': p.get('country_of_residence', ''),
        })
    
    # Check for corporate PSCs (could indicate complex ownership)
    corporate_pscs = [p for p in active_pscs if 'corporate' in p.get('kind', '').lower()]
    
    return {
        'psc_count': len(active_pscs),
        'corporate_psc_count': len(corporate_pscs),
        'pscs': psc_details,
    }


def extract_filing_info(filings: List[Dict]) -> Dict:
    """Extract filing history analysis."""
    
    if not filings:
        return {
            'total_filings': 0,
            'filings_last_year': 0,
            'late_filings_count': 0,
            'last_filing_date': None,
            'last_filing_type': None,
        }
    
    today = date.today()
    one_year_ago = date(today.year - 1, today.month, today.day)
    
    filings_last_year = 0
    late_count = 0
    
    for f in filings:
        filing_date_str = f.get('date')
        if filing_date_str:
            try:
                filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').date()
                if filing_date >= one_year_ago:
                    filings_last_year += 1
            except ValueError:
                pass
        
        # Check for late filing indicators in description
        desc = f.get('description', '').lower()
        if 'late' in desc or 'penalty' in desc:
            late_count += 1
    
    latest = filings[0] if filings else {}
    
    return {
        'total_filings': len(filings),
        'filings_last_year': filings_last_year,
        'late_filings_count': late_count,
        'last_filing_date': latest.get('date'),
        'last_filing_type': latest.get('type'),
        'last_filing_description': latest.get('description', '')[:100],
    }


def extract_company_data(client: CompaniesHouseClient, company_number: str) -> Optional[Dict]:
    """Extract comprehensive company data."""
    
    # 1. Get company profile
    profile = client.get_company(company_number)
    if not profile:
        return None
    
    time.sleep(REQUEST_DELAY)
    
    # 2. Get officers
    officers = client.get_officers(company_number)
    director_info = extract_director_info(officers)
    
    time.sleep(REQUEST_DELAY)
    
    # 3. Get PSCs
    pscs = client.get_pscs(company_number)
    psc_info = extract_psc_info(pscs)
    
    time.sleep(REQUEST_DELAY)
    
    # 4. Get filing history
    filings = client.get_filing_history(company_number)
    filing_info = extract_filing_info(filings)
    
    # === Extract from profile ===
    
    # Company URL for manual verification
    company_url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
    
    # Previous names (rebrand tracking)
    previous_names = profile.get('previous_company_names', [])
    previous_names_list = [
        {
            'name': pn.get('name', ''),
            'effective_from': pn.get('effective_from', ''),
            'ceased_on': pn.get('ceased_on', ''),
        }
        for pn in previous_names
    ]
    
    # Accounts info
    accounts = profile.get('accounts', {})
    last_accounts = accounts.get('last_accounts', {})
    
    # Check if accounts overdue
    next_accounts_due = accounts.get('next_due')
    accounts_overdue = False
    days_overdue = None
    if next_accounts_due:
        try:
            due_date = datetime.strptime(next_accounts_due, '%Y-%m-%d').date()
            if due_date < date.today():
                accounts_overdue = True
                days_overdue = (date.today() - due_date).days
        except ValueError:
            pass
    
    # Confirmation statement
    conf_stmt = profile.get('confirmation_statement', {})
    next_conf_due = conf_stmt.get('next_due')
    conf_overdue = False
    if next_conf_due:
        try:
            due_date = datetime.strptime(next_conf_due, '%Y-%m-%d').date()
            conf_overdue = due_date < date.today()
        except ValueError:
            pass
    
    # Registered address
    addr = profile.get('registered_office_address', {})
    address_parts = []
    for field in ['premises', 'address_line_1', 'address_line_2', 'locality', 'region', 'postal_code', 'country']:
        if addr.get(field):
            address_parts.append(addr[field])
    registered_address = ', '.join(address_parts)
    
    # SIC codes
    sic_codes = profile.get('sic_codes', [])
    
    return {
        # Basic info
        'company_number': company_number,
        'company_name': profile.get('company_name', ''),
        'company_url': company_url,
        'incorporated_date': profile.get('date_of_creation', ''),
        'company_status': profile.get('company_status', ''),
        'company_type': profile.get('type', ''),
        'jurisdiction': profile.get('jurisdiction', ''),
        'sic_codes': ','.join(sic_codes),
        'registered_address': registered_address,
        
        # 🚨 RED FLAGS
        'has_insolvency_history': profile.get('has_insolvency_history', False),
        'has_charges': profile.get('has_charges', False),
        'registered_office_in_dispute': profile.get('registered_office_is_in_dispute', False),
        'undeliverable_address': profile.get('undeliverable_registered_office_address', False),
        
        # Previous names (rebrand tracking)
        'previous_names_count': len(previous_names),
        'previous_names': json.dumps(previous_names_list) if previous_names_list else None,
        
        # Accounts
        'last_accounts_date': last_accounts.get('made_up_to', ''),
        'last_accounts_type': last_accounts.get('type', ''),
        'next_accounts_due': next_accounts_due,
        'accounts_overdue': accounts_overdue,
        'accounts_days_overdue': days_overdue,
        
        # Confirmation statement
        'last_confirmation_date': conf_stmt.get('last_made_up_to', ''),
        'next_confirmation_due': next_conf_due,
        'confirmation_overdue': conf_overdue,
        
        # Directors
        'director_count': director_info['director_count'],
        'secretary_count': director_info['secretary_count'],
        'avg_director_tenure_years': director_info['avg_director_tenure_years'],
        'min_director_tenure_years': director_info['min_director_tenure_years'],
        'max_director_tenure_years': director_info['max_director_tenure_years'],
        'directors': json.dumps(director_info['directors']),
        
        # PSCs (beneficial owners)
        'psc_count': psc_info['psc_count'],
        'corporate_psc_count': psc_info['corporate_psc_count'],
        'pscs': json.dumps(psc_info['pscs']) if psc_info['pscs'] else None,
        
        # Filing history
        'total_filings': filing_info['total_filings'],
        'filings_last_year': filing_info['filings_last_year'],
        'late_filings_count': filing_info['late_filings_count'],
        'last_filing_date': filing_info['last_filing_date'],
        'last_filing_type': filing_info['last_filing_type'],
    }


# ============================================================================
# CHECKPOINT
# ============================================================================

def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'results': [], 'no_match': []}


def save_checkpoint(data: Dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Companies House Full Data Scraper")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--limit', type=int, help='Limit number of factors')
    parser.add_argument('--api-key', type=str, help='Companies House API key')
    parser.add_argument('--min-score', type=int, default=MIN_MATCH_SCORE,
                        help=f'Minimum match score (default: {MIN_MATCH_SCORE})')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("COMPANIES HOUSE SCRAPER (FULL DATA)")
    print("=" * 60)
    print(f"📊 Captures: profile, directors, PSCs, filings, red flags")
    print(f"🎯 Min match score: {args.min_score}%")
    
    api_key = args.api_key or API_KEY
    if api_key == "YOUR_API_KEY_HERE":
        print("\n❌ No API key provided!")
        print("   Set CH_API_KEY environment variable or use --api-key")
        print("   Get a key at: https://developer.company-information.service.gov.uk/")
        return
    
    if not INPUT_CSV.exists():
        print(f"\n❌ Input file not found: {INPUT_CSV}")
        return
    
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        factors = list(csv.DictReader(f))
    
    print(f"\n📋 Loaded {len(factors)} factors")
    
    checkpoint = {'processed': [], 'results': [], 'no_match': []}
    if args.resume:
        checkpoint = load_checkpoint()
        if 'no_match' not in checkpoint:
            checkpoint['no_match'] = []
        print(f"   Resuming: {len(checkpoint['processed'])} already processed")
    
    processed_set = set(checkpoint['processed'])
    pending = [f for f in factors if f.get('registration_number') not in processed_set]
    
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} to process")
    print(f"   ~{len(pending) * 4} API calls needed")
    print(f"   Estimated time: {len(pending) * 4 * REQUEST_DELAY / 60:.0f} minutes")
    
    if not pending:
        print("\n✅ All factors already processed!")
        return
    
    results = checkpoint['results']
    no_match = checkpoint.get('no_match', [])
    
    client = CompaniesHouseClient(api_key)
    
    print(f"\n🔍 Searching Companies House...\n")
    
    found_count = 0
    red_flag_count = 0
    start_time = time.time()
    
    for i, factor in enumerate(pending):
        pf = factor.get('registration_number', '')
        name = factor.get('name', '')
        
        print(f"[{i+1}/{len(pending)}] {name[:42]}...", end=" ", flush=True)
        
        # Search
        search_results = client.search_companies(name)
        match, score, debug_info = find_best_match(name, search_results, args.min_score)
        
        if match:
            company_number = match.get('company_number', '')
            data = extract_company_data(client, company_number)
            
            if data:
                result = {
                    'factor_registration_number': pf,
                    'factor_name': name,
                    'match_score': score,
                    'match_debug': debug_info,
                    **data,
                }
                results.append(result)
                found_count += 1
                
                # Check for red flags
                flags = []
                if data.get('has_insolvency_history'):
                    flags.append('INSOLV')
                if data.get('accounts_overdue'):
                    flags.append('ACC_DUE')
                if data.get('confirmation_overdue'):
                    flags.append('CONF_DUE')
                if data.get('registered_office_in_dispute'):
                    flags.append('ADDR_DISPUTE')
                if data.get('company_status') != 'active':
                    flags.append(data.get('company_status', '').upper()[:8])
                
                if flags:
                    red_flag_count += 1
                    flag_str = ','.join(flags)
                    print(f"🚩 {company_number} [{flag_str}]")
                else:
                    print(f"✅ {company_number} ({score:.0f}%)")
            else:
                print(f"⚠️  Could not fetch {company_number}")
                no_match.append({'pf': pf, 'name': name, 'reason': 'fetch_failed'})
        else:
            print(f"❌ No match ({score:.0f}%) {debug_info[:40]}")
            no_match.append({'pf': pf, 'name': name, 'best_score': score, 'debug': debug_info})
        
        # Update checkpoint
        checkpoint['processed'].append(pf)
        checkpoint['results'] = results
        checkpoint['no_match'] = no_match
        
        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            remaining = (len(pending) - i - 1) / rate if rate > 0 else 0
            print(f"    💾 Saved checkpoint | {rate:.1f}/min | ~{remaining:.0f}min left")
        
        time.sleep(REQUEST_DELAY)
    
    # Final save
    save_checkpoint(checkpoint)
    
    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'factor_registration_number', 'factor_name', 'match_score', 'match_debug',
        'company_number', 'company_name', 'company_url', 'incorporated_date', 
        'company_status', 'company_type', 'jurisdiction', 'sic_codes',
        'registered_address',
        # Red flags
        'has_insolvency_history', 'has_charges', 
        'registered_office_in_dispute', 'undeliverable_address',
        # Previous names
        'previous_names_count', 'previous_names',
        # Accounts
        'last_accounts_date', 'last_accounts_type', 'next_accounts_due',
        'accounts_overdue', 'accounts_days_overdue',
        # Confirmation
        'last_confirmation_date', 'next_confirmation_due', 'confirmation_overdue',
        # Directors
        'director_count', 'secretary_count', 
        'avg_director_tenure_years', 'min_director_tenure_years', 'max_director_tenure_years',
        'directors',
        # PSCs
        'psc_count', 'corporate_psc_count', 'pscs',
        # Filings
        'total_filings', 'filings_last_year', 'late_filings_count',
        'last_filing_date', 'last_filing_type',
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    total = len(checkpoint['processed'])
    print(f"✅ Matched: {len(results)} ({len(results)/total*100:.1f}%)")
    print(f"❌ No match: {len(no_match)} ({len(no_match)/total*100:.1f}%)")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"📡 API calls: {client.request_count}")
    
    if results:
        active = sum(1 for r in results if r.get('company_status') == 'active')
        insolvency = sum(1 for r in results if r.get('has_insolvency_history'))
        charges = sum(1 for r in results if r.get('has_charges'))
        acc_overdue = sum(1 for r in results if r.get('accounts_overdue'))
        rebranded = sum(1 for r in results if r.get('previous_names_count', 0) > 0)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Active: {active}")
        print(f"   🚩 Insolvency history: {insolvency}")
        print(f"   🚩 Has charges/loans: {charges}")
        print(f"   🚩 Accounts overdue: {acc_overdue}")
        print(f"   🔄 Rebranded (prev names): {rebranded}")
        
        if results:
            avg_score = sum(r.get('match_score', 0) for r in results) / len(results)
            tenures = [r['avg_director_tenure_years'] for r in results if r.get('avg_director_tenure_years')]
            print(f"   📈 Avg match score: {avg_score:.1f}%")
            if tenures:
                print(f"   📈 Avg director tenure: {sum(tenures)/len(tenures):.1f} years")
    
    print(f"\n📄 Output: {OUTPUT_CSV}")
    print(f"📄 Log: {LOG_FILE}")


if __name__ == "__main__":
    main()