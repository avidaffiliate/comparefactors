#!/usr/bin/env python3
"""
============================================================================
SCRIPT 03: FOI ENRICHMENT (Postcode Coverage from FOI Release)
============================================================================

PURPOSE: Extract postcode coverage areas from FOI spreadsheets and match
         to registered factors. This enables geographic classification
         (national/regional/local) and "factors in your area" lookups.

SOURCE:  https://www.gov.scot/publications/foi-202500448749/
         FOI 202500448749 - Property Factor Register data (2024 Annual Update)

INPUT:   data/csv/factors_register.csv (from script 02)
         data/foi/FOI_202500448749_-_Information_Released_-_Spreadsheet_04.xlsx

OUTPUT:  data/csv/factors_postcodes.csv
         data/foi/foi_historical_stats.csv (optional - year-by-year stats)

EXTRACTS:
    - postcode_areas: Comma-separated 2-char postcodes (e.g., "EH,G,DD")
    - postcode_count: Number of unique postcode areas
    - cities: Comma-separated cities from land parcels
    - geographic_reach: national/regional/local classification

USAGE:
    python scripts/03_enrich_foi.py
    python scripts/03_enrich_foi.py --test              # Show matching preview
    python scripts/03_enrich_foi.py --threshold 80      # Adjust match threshold

DEPENDENCIES:
    pip install pandas openpyxl rapidfuzz

TIME: ~10 seconds (fuzzy matching only, no network requests)
============================================================================
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
REGISTRY_CSV = Path("data/csv/factors_register.csv")
FOI_SPREADSHEET_04 = Path("data/foi/FOI_202500448749_-_Information_Released_-_Spreadsheet_04.xlsx")
FOI_SPREADSHEET_01 = Path("data/foi/FOI_202500448749_-_Information_Released_-_Spreadsheet_01.xlsx")
FOI_SPREADSHEET_03 = Path("data/foi/FOI_202500448749_-_Information_Released_-_Spreadsheet_03.xlsx")

# Output files
OUTPUT_CSV = Path("data/csv/factors_postcodes.csv")
HISTORICAL_CSV = Path("data/foi/foi_historical_stats.csv")
UNMATCHED_CSV = Path("data/foi/foi_unmatched.csv")

# Manual mappings for fuzzy matching failures
MANUAL_MAPPINGS_CSV = Path("data/manual/foi_name_mappings.csv")

# Matching configuration
MATCH_THRESHOLD = 85  # Minimum fuzzy match score (0-100)

# Geographic classification thresholds
NATIONAL_THRESHOLD = 15   # 15+ postcode areas = national
REGIONAL_THRESHOLD = 5    # 5-14 postcode areas = regional
                          # <5 = local

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FOIMatch:
    registration_number: str
    registry_name: str
    foi_name: str
    match_score: int
    postcode_areas: str      # Comma-separated: "EH,G,DD"
    postcode_count: int
    cities: str              # Comma-separated from Land sheet
    geographic_reach: str    # national/regional/local


# ============================================================================
# NAME NORMALIZATION
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize factor name for matching."""
    if not name:
        return ""
    
    name = name.strip().lower()
    
    # Common abbreviations and variations
    replacements = [
        ('limited', 'ltd'),
        ('property management', 'pm'),
        ('property factor', 'pf'),
        ('property factors', 'pf'),
        ('property services', 'ps'),
        ('housing association', 'ha'),
        ('  ', ' '),
    ]
    
    for old, new in replacements:
        name = name.replace(old, new)
    
    # Remove common suffixes for matching
    suffixes = [' ltd', ' llp', ' plc', ' cic']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    return name.strip()


# ============================================================================
# FOI DATA LOADING
# ============================================================================

def load_foi_postcodes(filepath: Path) -> Dict[str, Dict]:
    """
    Load FOI Spreadsheet 04 and aggregate postcodes per factor.
    
    Returns dict: foi_name -> {postcodes: set, cities: set}
    """
    print(f"📂 Loading FOI data from {filepath}")
    
    factors = defaultdict(lambda: {'postcodes': set(), 'cities': set()})
    
    # Load Properties sheet (factor -> postcode)
    df_props = pd.read_excel(filepath, sheet_name='Properties')
    print(f"   Properties sheet: {len(df_props)} rows")
    
    for _, row in df_props.iterrows():
        factor = str(row['Property Factor']).strip()
        postcode = str(row['postcode']).strip().upper()
        if factor and postcode and postcode != 'NAN':
            factors[factor]['postcodes'].add(postcode)
    
    # Load Land sheet (factor -> city)
    df_land = pd.read_excel(filepath, sheet_name='Land')
    print(f"   Land sheet: {len(df_land)} rows")
    
    for _, row in df_land.iterrows():
        factor = str(row['Property Factor']).strip()
        city = str(row['City/Town']).strip()
        if factor and city and city != 'nan':
            factors[factor]['cities'].add(city)
    
    print(f"   Unique FOI factors: {len(factors)}")
    
    return dict(factors)


def load_registry_factors(filepath: Path) -> Dict[str, str]:
    """
    Load registry factors for matching.
    
    Returns dict: registration_number -> name
    """
    print(f"📂 Loading registry from {filepath}")
    
    factors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pf = row.get('registration_number', '').strip()
            name = row.get('name', '').strip()
            if pf and name:
                factors[pf] = name
    
    print(f"   Registry factors: {len(factors)}")
    return factors


# ============================================================================
# MANUAL MAPPINGS
# ============================================================================

def load_manual_mappings(filepath: Path) -> Dict[str, str]:
    """
    Load manual FOI name -> registration number mappings.
    
    CSV format: foi_name,registration_number
    """
    mappings = {}
    
    if not filepath.exists():
        return mappings
    
    print(f"📂 Loading manual mappings from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            foi_name = row.get('foi_name', '').strip()
            pf = row.get('registration_number', '').strip().upper()
            if foi_name and pf:
                mappings[foi_name] = pf
    
    print(f"   Manual mappings: {len(mappings)}")
    return mappings


# ============================================================================
# FUZZY MATCHING
# ============================================================================

def match_foi_to_registry(
    foi_factors: Dict[str, Dict],
    registry_factors: Dict[str, str],
    manual_mappings: Dict[str, str] = None,
    threshold: int = MATCH_THRESHOLD
) -> Tuple[List[FOIMatch], List[str]]:
    """
    Match FOI factor names to registry using fuzzy matching.
    Manual mappings take priority over fuzzy matching.
    
    Returns:
        - List of successful matches
        - List of unmatched FOI names
    """
    print(f"\n🔗 Matching FOI names to registry (threshold: {threshold})")
    
    manual_mappings = manual_mappings or {}
    
    # Build lookup: normalized_name -> (pf_number, original_name)
    registry_lookup = {}
    registry_names = []
    for pf, name in registry_factors.items():
        norm = normalize_name(name)
        registry_lookup[norm] = (pf, name)
        registry_names.append(norm)
    
    # Also build pf -> name lookup for manual mappings
    pf_to_name = {pf: name for pf, name in registry_factors.items()}
    
    matches = []
    unmatched = []
    manual_count = 0
    
    for foi_name, data in foi_factors.items():
        foi_norm = normalize_name(foi_name)
        
        # Check manual mappings first
        if foi_name in manual_mappings:
            pf_number = manual_mappings[foi_name]
            registry_name = pf_to_name.get(pf_number, "")
            
            if registry_name:
                pc_count = len(data['postcodes'])
                if pc_count >= NATIONAL_THRESHOLD:
                    reach = 'national'
                elif pc_count >= REGIONAL_THRESHOLD:
                    reach = 'regional'
                else:
                    reach = 'local'
                
                match = FOIMatch(
                    registration_number=pf_number,
                    registry_name=registry_name,
                    foi_name=foi_name,
                    match_score=100,  # Manual = 100%
                    postcode_areas=','.join(sorted(data['postcodes'])),
                    postcode_count=pc_count,
                    cities=','.join(sorted(data['cities'])),
                    geographic_reach=reach
                )
                matches.append(match)
                manual_count += 1
                continue
        
        # Fall back to fuzzy matching
        result = process.extractOne(
            foi_norm,
            registry_names,
            scorer=fuzz.token_sort_ratio
        )
        
        if result and result[1] >= threshold:
            matched_norm, score, _ = result
            pf_number, registry_name = registry_lookup[matched_norm]
            
            # Calculate geographic reach
            pc_count = len(data['postcodes'])
            if pc_count >= NATIONAL_THRESHOLD:
                reach = 'national'
            elif pc_count >= REGIONAL_THRESHOLD:
                reach = 'regional'
            else:
                reach = 'local'
            
            match = FOIMatch(
                registration_number=pf_number,
                registry_name=registry_name,
                foi_name=foi_name,
                match_score=score,
                postcode_areas=','.join(sorted(data['postcodes'])),
                postcode_count=pc_count,
                cities=','.join(sorted(data['cities'])),
                geographic_reach=reach
            )
            matches.append(match)
        else:
            unmatched.append(foi_name)
    
    print(f"   ✅ Matched: {len(matches)} ({manual_count} manual, {len(matches) - manual_count} fuzzy)")
    print(f"   ❌ Unmatched: {len(unmatched)}")
    
    return matches, unmatched


# ============================================================================
# HISTORICAL STATS (Optional)
# ============================================================================

def extract_historical_stats(
    spreadsheet_01: Path,
    spreadsheet_03: Path,
    output_path: Path
):
    """Extract year-by-year statistics for content/analysis."""
    
    if not spreadsheet_01.exists() or not spreadsheet_03.exists():
        print("   ⚠️ Historical spreadsheets not found, skipping")
        return
    
    print(f"\n📊 Extracting historical statistics")
    
    # Load factor counts
    df_counts = pd.read_excel(spreadsheet_01, sheet_name='Number of PFs per year ')
    df_counts = df_counts[['Year', 'Number of Registered Property Factors']].dropna()
    df_counts.columns = ['year', 'registered_count']
    
    # Load strike-off data
    df_strikes = pd.read_excel(spreadsheet_03, sheet_name='Sheet1')
    df_strikes = df_strikes[[
        'Year',
        'Number of Property Factors removed for failing to comply with a PFEO',
        'Number of Property Factors removed for failing to comply with the Code of Conduct',
        'Number of Property Factors removed due to the property factor no longer being considered a fit and proper person'
    ]].dropna(subset=['Year'])
    df_strikes.columns = ['year', 'removed_pfeo', 'removed_code', 'removed_unfit']
    
    # Merge
    df = pd.merge(df_counts, df_strikes, on='year', how='outer')
    df = df.sort_values('year')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Saved to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enrich factors with FOI postcode data")
    parser.add_argument('--test', action='store_true', help='Show matching preview only')
    parser.add_argument('--threshold', type=int, default=MATCH_THRESHOLD,
                        help=f'Fuzzy match threshold (default: {MATCH_THRESHOLD})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FOI ENRICHMENT - Postcode Coverage")
    print("=" * 60)
    print(f"Source: gov.scot/publications/foi-202500448749")
    print()
    
    # Check input files
    if not REGISTRY_CSV.exists():
        print(f"❌ Registry file not found: {REGISTRY_CSV}")
        print("   Run script 02_registry_enrich.py first")
        return
    
    if not FOI_SPREADSHEET_04.exists():
        print(f"❌ FOI spreadsheet not found: {FOI_SPREADSHEET_04}")
        print("   Download from: https://www.gov.scot/publications/foi-202500448749/")
        print(f"   Save to: {FOI_SPREADSHEET_04}")
        return
    
    # Load data
    foi_factors = load_foi_postcodes(FOI_SPREADSHEET_04)
    registry_factors = load_registry_factors(REGISTRY_CSV)
    manual_mappings = load_manual_mappings(MANUAL_MAPPINGS_CSV)
    
    # Match
    matches, unmatched = match_foi_to_registry(
        foi_factors, 
        registry_factors,
        manual_mappings=manual_mappings,
        threshold=args.threshold
    )
    
    # Test mode - just show preview
    if args.test:
        print("\n" + "=" * 60)
        print("MATCH PREVIEW (test mode)")
        print("=" * 60)
        
        print("\n📍 Sample matches:")
        for m in sorted(matches, key=lambda x: -x.postcode_count)[:10]:
            print(f"   [{m.match_score}%] {m.foi_name}")
            print(f"         → {m.registration_number}: {m.registry_name}")
            print(f"         → {m.postcode_count} areas ({m.geographic_reach}): {m.postcode_areas[:50]}...")
        
        if unmatched:
            print(f"\n❌ Unmatched FOI names ({len(unmatched)}):")
            for name in unmatched[:10]:
                print(f"   - {name}")
            if len(unmatched) > 10:
                print(f"   ... and {len(unmatched) - 10} more")
        
        return
    
    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'registration_number', 'registry_name', 'foi_name', 'match_score',
        'postcode_areas', 'postcode_count', 'cities', 'geographic_reach'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            writer.writerow(asdict(match))
    
    print(f"\n📄 Output saved to: {OUTPUT_CSV}")
    
    # Save unmatched for manual review
    if unmatched:
        UNMATCHED_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(UNMATCHED_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['foi_name', 'postcodes', 'cities'])
            for name in unmatched:
                data = foi_factors[name]
                writer.writerow([
                    name,
                    ','.join(sorted(data['postcodes'])),
                    ','.join(sorted(data['cities']))
                ])
        print(f"📄 Unmatched saved to: {UNMATCHED_CSV}")
    
    # Extract historical stats
    extract_historical_stats(FOI_SPREADSHEET_01, FOI_SPREADSHEET_03, HISTORICAL_CSV)
    
    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    
    # Geographic distribution
    national = sum(1 for m in matches if m.geographic_reach == 'national')
    regional = sum(1 for m in matches if m.geographic_reach == 'regional')
    local = sum(1 for m in matches if m.geographic_reach == 'local')
    
    print(f"\n📊 Geographic Distribution:")
    print(f"   National (15+ areas): {national}")
    print(f"   Regional (5-14 areas): {regional}")
    print(f"   Local (<5 areas): {local}")
    
    # Top coverage
    top_coverage = sorted(matches, key=lambda x: -x.postcode_count)[:5]
    print(f"\n🏆 Widest Coverage:")
    for m in top_coverage:
        print(f"   {m.postcode_count} areas: {m.registry_name}")
    
    match_rate = len(matches) / len(foi_factors) * 100 if foi_factors else 0
    print(f"\n✅ Match rate: {match_rate:.1f}% ({len(matches)}/{len(foi_factors)})")


if __name__ == "__main__":
    main()
