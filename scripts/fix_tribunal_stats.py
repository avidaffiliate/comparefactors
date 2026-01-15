#!/usr/bin/env python3
"""
Fix tribunal stats by importing authoritative values from factors_enriched.csv.

This script:
1. Imports tribunal_negative_outcomes as tribunal_cases_upheld (case-based adverse count)
2. Imports tribunal_positive_outcomes as tribunal_cases_dismissed
3. Imports tribunal_total_cases as tribunal_case_count
4. Recalculates adverse outcomes from tribunal_cases table for validation

Usage:
    python fix_tribunal_stats.py [db_path] [csv_path]
    python fix_tribunal_stats.py  # defaults to data/comparefactors.db and data/csv/factors_enriched.csv
"""

import sqlite3
import csv
import sys
from pathlib import Path


def normalize_pf(pf):
    """Normalize PF number to PF000XXX format."""
    if not pf:
        return None
    pf = str(pf).strip().upper()
    if pf.startswith('PF'):
        num = pf[2:].lstrip('0')
        return f"PF{int(num):06d}" if num.isdigit() else None
    return None


def fix_database(db_path: Path, csv_path: Path):
    """Import enriched values and fix tribunal stats."""
    
    print(f"Database: {db_path}")
    print(f"CSV: {csv_path}")
    print("-" * 60)
    
    # Load enriched CSV
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        enriched = {r['registration_number']: r for r in csv.DictReader(f)}
    
    print(f"Loaded {len(enriched)} factors from enriched CSV")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    updated = 0
    for pf, data in enriched.items():
        total = int(data.get('tribunal_total_cases') or 0)
        if total == 0:
            continue
        
        negative = int(data.get('tribunal_negative_outcomes') or 0)
        positive = int(data.get('tribunal_positive_outcomes') or 0)
        pfeo_issued = int(data.get('tribunal_pfeo_issued') or 0)
        pfeo_proposed = int(data.get('tribunal_pfeo_proposed') or 0)
        
        conn.execute("""
            UPDATE factors SET
                tribunal_case_count = ?,
                tribunal_cases_upheld = ?,
                tribunal_cases_dismissed = ?,
                tribunal_pfeo_count = COALESCE(?, tribunal_pfeo_count)
            WHERE registration_number = ?
        """, [
            total,
            negative,  # Case-based adverse outcomes
            positive,  # Dismissed + Rejected
            pfeo_issued + pfeo_proposed,
            pf,
        ])
        updated += 1
    
    conn.commit()
    
    # Show results
    print(f"\nUpdated {updated} factors with enriched values")
    print("\nSample results:")
    print("-" * 90)
    print(f"{'PF':<10} {'Name':<35} {'Cases':>6} {'Adverse':>8} {'Dismissed':>9} {'Rate':>6}")
    print("-" * 90)
    
    sample = conn.execute("""
        SELECT registration_number, name, tribunal_case_count, 
               tribunal_cases_upheld, tribunal_cases_dismissed
        FROM factors
        WHERE tribunal_case_count > 10
        ORDER BY tribunal_case_count DESC
        LIMIT 15
    """).fetchall()
    
    for r in sample:
        total = r['tribunal_case_count'] or 0
        upheld = r['tribunal_cases_upheld'] or 0
        rate = (upheld / total * 100) if total > 0 else 0
        print(f"{r['registration_number']:<10} {r['name'][:34]:<35} {total:>6} {upheld:>8} {r['tribunal_cases_dismissed'] or 0:>9} {rate:>5.1f}%")
    
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        db_path = Path(sys.argv[1])
        csv_path = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        db_path = Path(sys.argv[1])
        csv_path = Path("data/csv/factors_enriched.csv")
    else:
        db_path = Path("data/comparefactors.db")
        csv_path = Path("data/csv/factors_enriched.csv")
    
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)
    
    fix_database(db_path, csv_path)
