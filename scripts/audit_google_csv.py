#!/usr/bin/env python3
"""
Audit Google Reviews CSV for Bad Matches

Compares factor_name to google_name using fuzzy matching to identify
reviews that were incorrectly associated with the wrong Google Business listing.

Usage:
    python audit_google_csv.py --input google_reviews.csv
    python audit_google_csv.py --input google_reviews.csv --threshold 40
    python audit_google_csv.py --input google_reviews.csv --factors factors_register.csv
"""

import csv
import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Try to import rapidfuzz, fall back to difflib
try:
    from rapidfuzz import fuzz
    USE_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    USE_RAPIDFUZZ = False
    print("Note: Install rapidfuzz for better matching: pip install rapidfuzz")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_THRESHOLD = 40  # Minimum similarity score (0-100)
SEVERE_THRESHOLD = 25   # Below this is almost certainly wrong

# Words to remove when comparing names
NOISE_WORDS = {
    'limited', 'ltd', 'llp', 'plc', 'inc', 'corp', 'corporation',
    'property', 'properties', 'factor', 'factors', 'factoring',
    'management', 'services', 'service', 'residential', 'commercial',
    'scotland', 'scottish', 'glasgow', 'edinburgh', 'aberdeen',
    'the', 'and', '&', 'of', 'uk', 'group'
}


# =============================================================================
# NAME SIMILARITY
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize a company name for comparison."""
    if not name:
        return ""
    
    name = name.lower()
    name = re.sub(r'[^\w\s]', ' ', name)
    words = [w for w in name.split() if w not in NOISE_WORDS]
    return ' '.join(words).strip()


def calculate_similarity(name1: str, name2: str) -> Tuple[int, str, str]:
    """Calculate similarity between two names."""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if not norm1 or not norm2:
        return 0, norm1, norm2
    
    if USE_RAPIDFUZZ:
        score = fuzz.token_set_ratio(norm1, norm2)
    else:
        score = int(SequenceMatcher(None, norm1, norm2).ratio() * 100)
    
    return score, norm1, norm2


# =============================================================================
# AUDIT
# =============================================================================

@dataclass
class AuditResult:
    row_num: int
    reg_number: str
    factor_name: str
    google_name: str
    google_address: str
    similarity_score: int
    rating: str
    review_count: str
    place_id: str
    severity: str  # 'severe', 'suspicious', 'ok'
    original_row: dict


def audit_csv(input_path: Path, threshold: int = DEFAULT_THRESHOLD) -> List[AuditResult]:
    """Audit CSV file for bad matches."""
    
    results = []
    
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # Detect column names (handle variations)
        fieldnames = reader.fieldnames
        
        # Map possible column names
        def get_col(row, *names):
            for name in names:
                if name in row and row[name]:
                    return row[name].strip()
            return ''
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            factor_name = get_col(row, 'factor_name', 'name', 'factor')
            google_name = get_col(row, 'google_name', 'business_name', 'place_name')
            google_address = get_col(row, 'google_address', 'address', 'formatted_address')
            reg_number = get_col(row, 'registration_number', 'reg_number', 'pf_number', 'factor_registration_number')
            rating = get_col(row, 'rating', 'google_rating')
            review_count = get_col(row, 'review_count', 'google_review_count')
            place_id = get_col(row, 'place_id', 'google_place_id')
            
            if not factor_name or not google_name:
                continue
            
            score, _, _ = calculate_similarity(factor_name, google_name)
            
            if score < SEVERE_THRESHOLD:
                severity = 'severe'
            elif score < threshold:
                severity = 'suspicious'
            else:
                severity = 'ok'
            
            results.append(AuditResult(
                row_num=row_num,
                reg_number=reg_number,
                factor_name=factor_name,
                google_name=google_name,
                google_address=google_address,
                similarity_score=score,
                rating=rating,
                review_count=review_count,
                place_id=place_id,
                severity=severity,
                original_row=row
            ))
    
    return results


def print_report(results: List[AuditResult], threshold: int):
    """Print audit report."""
    
    severe = [r for r in results if r.severity == 'severe']
    suspicious = [r for r in results if r.severity == 'suspicious']
    ok = [r for r in results if r.severity == 'ok']
    
    print("=" * 80)
    print("GOOGLE REVIEWS CSV AUDIT REPORT")
    print("=" * 80)
    print(f"Total rows:        {len(results)}")
    print(f"Severe (<{SEVERE_THRESHOLD}%):     {len(severe)} ← DELETE THESE")
    print(f"Suspicious (<{threshold}%): {len(suspicious)} ← REVIEW THESE")
    print(f"OK (>={threshold}%):        {len(ok)}")
    print("=" * 80)
    
    if severe:
        print(f"\n{'='*80}")
        print("SEVERE MISMATCHES - Almost certainly wrong")
        print(f"{'='*80}\n")
        
        for r in sorted(severe, key=lambda x: x.similarity_score):
            print(f"  Row {r.row_num}: {r.reg_number}")
            print(f"    Factor: {r.factor_name[:50]}")
            print(f"    Google: {r.google_name[:50]}")
            print(f"    Score:  {r.similarity_score}% | Rating: {r.rating}★ ({r.review_count} reviews)")
            if r.google_address:
                print(f"    Addr:   {r.google_address[:60]}")
            print()
    
    if suspicious:
        print(f"\n{'='*80}")
        print("SUSPICIOUS MATCHES - Manual review recommended")
        print(f"{'='*80}\n")
        
        for r in sorted(suspicious, key=lambda x: x.similarity_score)[:20]:  # Show first 20
            print(f"  Row {r.row_num}: {r.factor_name[:40]} → {r.google_name[:40]} ({r.similarity_score}%)")
        
        if len(suspicious) > 20:
            print(f"\n  ... and {len(suspicious) - 20} more (see CSV output)")


def export_flagged_csv(results: List[AuditResult], output_path: Path):
    """Export flagged results to CSV."""
    
    flagged = [r for r in results if r.severity in ('severe', 'suspicious')]
    
    if not flagged:
        print("No flagged entries to export.")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'severity',
            'row_num',
            'registration_number',
            'factor_name',
            'google_name',
            'similarity_score',
            'google_address',
            'rating',
            'review_count',
            'place_id',
            'action'
        ])
        
        for r in sorted(flagged, key=lambda x: (x.severity != 'severe', x.similarity_score)):
            writer.writerow([
                r.severity,
                r.row_num,
                r.reg_number,
                r.factor_name,
                r.google_name,
                r.similarity_score,
                r.google_address,
                r.rating,
                r.review_count,
                r.place_id,
                'DELETE' if r.severity == 'severe' else 'REVIEW'
            ])
    
    print(f"\nExported {len(flagged)} flagged entries to: {output_path}")


def export_clean_csv(results: List[AuditResult], original_path: Path, output_path: Path):
    """Export cleaned CSV with bad matches removed."""
    
    # Get row numbers to remove
    remove_rows = {r.row_num for r in results if r.severity == 'severe'}
    
    if not remove_rows:
        print("No rows to remove.")
        return
    
    kept = 0
    removed = 0
    
    with open(original_path, 'r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row_num, row in enumerate(reader, start=2):
                if row_num in remove_rows:
                    removed += 1
                else:
                    writer.writerow(row)
                    kept += 1
    
    print(f"\nExported clean CSV: {output_path}")
    print(f"  Kept: {kept} rows")
    print(f"  Removed: {removed} rows")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Audit Google reviews CSV for bad matches')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file (google_reviews.csv)')
    parser.add_argument('--threshold', '-t', type=int, default=DEFAULT_THRESHOLD,
                        help=f'Similarity threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--output', '-o', default='flagged_google_matches.csv',
                        help='Output CSV for flagged entries')
    parser.add_argument('--clean', '-c', default='google_reviews_clean.csv',
                        help='Output CSV with bad matches removed')
    parser.add_argument('--no-clean', action='store_true',
                        help="Don't generate clean CSV")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1
    
    # Run audit
    print(f"Auditing: {input_path}")
    results = audit_csv(input_path, args.threshold)
    
    if not results:
        print("No data found in CSV.")
        return 1
    
    # Print report
    print_report(results, args.threshold)
    
    # Export flagged
    export_flagged_csv(results, Path(args.output))
    
    # Export clean CSV
    if not args.no_clean:
        severe_count = len([r for r in results if r.severity == 'severe'])
        if severe_count > 0:
            export_clean_csv(results, input_path, Path(args.clean))
    
    # Summary
    severe = [r for r in results if r.severity == 'severe']
    if severe:
        print(f"\n⚠️  Found {len(severe)} severe mismatches.")
        print(f"    Use google_reviews_clean.csv for your pipeline.")
    
    return 0


if __name__ == '__main__':
    exit(main())
