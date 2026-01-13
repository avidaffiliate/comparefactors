#!/usr/bin/env python3
"""
Recover original tribunal outcome data from comparefactors.db
and update tribunal_enriched.db with correct values.

The AI extraction corrupted many outcomes (559 Dismissed vs 23 real).
The original scraped data in comparefactors.db is reliable.
"""

import sqlite3
from pathlib import Path

# Paths - UPDATE THESE to match your local setup
SOURCE_DB = Path(r"C:\Users\david\Downloads\comparefactors-fixed_1\comparefactors.db")
TARGET_DB = Path("data/tribunal/tribunal_enriched.db")  # Corrupted AI extraction

def map_outcome_type(outcome_type: str, outcome: str, has_pfeo: bool) -> dict:
    """
    Map original scraped outcome to AI extraction schema.
    
    CRITICAL: pfeo_issued should only be True when outcome_type = 'PFEO Issued'
    The has_pfeo field means "case involved PFEO at some point" which includes
    compliance hearings - NOT the same as a new PFEO being issued.
    
    outcome_type meanings:
    - 'PFEO Issued': A new PFEO was actually issued in this case
    - 'Complied': This is a compliance check for a PREVIOUS PFEO (not a new one)
    - 'Compensation Only': Breach found, compensation ordered, no PFEO
    - 'Upheld No Order': Breach found, no enforcement needed
    - 'Dismissed': Homeowner lost
    - 'Rejected': Application rejected (procedural)
    - 'Withdrawn': Case withdrawn
    
    Returns dict with:
        - ai_outcome: Upheld/Partially Upheld/Dismissed/Withdrawn/Referred to Ministers
        - pfeo_issued: bool - True ONLY if new PFEO issued in THIS case
        - pfeo_proposed: bool  
        - pfeo_complied: bool or None
    """
    outcome_type = (outcome_type or "").strip()
    outcome = (outcome or "").strip()
    
    result = {
        'ai_outcome': '',
        'pfeo_issued': False,
        'pfeo_proposed': False,
        'pfeo_complied': None,
    }
    
    # Map based on outcome_type (more specific)
    if outcome_type == 'Dismissed':
        result['ai_outcome'] = 'Dismissed'
    
    elif outcome_type == 'PFEO Issued':
        # This is the ONLY outcome_type where pfeo_issued should be True
        result['ai_outcome'] = 'Upheld'
        result['pfeo_issued'] = True
        # Check compliance status from outcome field
        if 'Non-Compliant' in outcome:
            result['pfeo_complied'] = False
        elif 'Complied' in outcome:
            result['pfeo_complied'] = True
    
    elif outcome_type == 'Complied':
        # Compliance hearing for a PREVIOUS case's PFEO - NOT a new PFEO
        result['ai_outcome'] = 'Upheld'
        result['pfeo_issued'] = False  # This case didn't issue a new PFEO
        # This indicates they complied with a previous PFEO
    
    elif outcome_type == 'Rejected':
        # Application rejected (procedural) - treat as dismissed
        result['ai_outcome'] = 'Dismissed'
    
    elif outcome_type == 'Upheld No Order':
        # Breach found but no PFEO needed
        result['ai_outcome'] = 'Upheld'
    
    elif outcome_type == 'Compensation Only':
        result['ai_outcome'] = 'Upheld'
    
    elif outcome_type == 'Withdrawn':
        result['ai_outcome'] = 'Withdrawn'
    
    elif outcome_type == '':
        # Fall back to outcome field when outcome_type is blank
        if 'PFEO' in outcome and 'Non-Compliant' in outcome:
            result['ai_outcome'] = 'Referred to Ministers'
            result['pfeo_issued'] = True
            result['pfeo_complied'] = False
        elif 'PFEO Issued' in outcome and 'Non-Compliant' not in outcome:
            result['ai_outcome'] = 'Upheld'
            result['pfeo_issued'] = True
        elif 'PFEO Proposed' in outcome:
            result['ai_outcome'] = 'Upheld'
            result['pfeo_proposed'] = True
        elif 'Failure to Comply' in outcome:
            # Failure to comply with a previous PFEO - NOT a new PFEO
            result['ai_outcome'] = 'Upheld'
            result['pfeo_issued'] = False
        elif 'Complied' in outcome:
            # Compliance with previous PFEO - NOT a new PFEO
            result['ai_outcome'] = 'Upheld'
            result['pfeo_issued'] = False
        elif 'Decision Issued' in outcome:
            result['ai_outcome'] = 'Upheld'
            # Don't assume PFEO just because has_pfeo is True
        elif 'Rejected' in outcome:
            result['ai_outcome'] = 'Dismissed'
        elif 'Dismissed' in outcome:
            result['ai_outcome'] = 'Dismissed'
        else:
            # Unknown - leave as is
            result['ai_outcome'] = ''
    
    # NO has_pfeo override - it's unreliable for counting actual PFEOs
    
    return result


def main():
    print("=" * 70)
    print("TRIBUNAL DATA RECOVERY")
    print("=" * 70)
    
    if not SOURCE_DB.exists():
        print(f"❌ Source DB not found: {SOURCE_DB}")
        print("   Update SOURCE_DB path in script")
        return
    
    if not TARGET_DB.exists():
        print(f"❌ Target DB not found: {TARGET_DB}")
        print("   Update TARGET_DB path in script")
        return
    
    # Connect to both databases
    source_conn = sqlite3.connect(SOURCE_DB)
    source_conn.row_factory = sqlite3.Row
    target_conn = sqlite3.connect(TARGET_DB)
    
    # Get original data
    print(f"\n📖 Reading from {SOURCE_DB}...")
    cursor = source_conn.execute("""
        SELECT 
            case_reference,
            outcome,
            outcome_type,
            has_pfeo,
            compensation_amount
        FROM tribunal_cases
    """)
    original_data = {row['case_reference']: dict(row) for row in cursor.fetchall()}
    print(f"   Found {len(original_data)} cases in source")
    
    # Get current target data
    print(f"\n📖 Reading from {TARGET_DB}...")
    cursor = target_conn.execute("SELECT case_reference FROM cases")
    target_cases = {row[0] for row in cursor.fetchall()}
    print(f"   Found {len(target_cases)} cases in target")
    
    # Find matches
    matches = set(original_data.keys()) & target_cases
    print(f"   {len(matches)} cases match between databases")
    
    # Show current (corrupted) distribution
    print("\n📊 BEFORE recovery (corrupted):")
    cursor = target_conn.execute("""
        SELECT ai_outcome, COUNT(*), SUM(pfeo_issued) 
        FROM cases GROUP BY ai_outcome ORDER BY COUNT(*) DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0] or '(blank)'}: {row[1]} cases, {row[2] or 0} PFEOs")
    
    # Prepare updates
    print("\n🔄 Preparing updates...")
    updates = []
    stats = {'updated': 0, 'skipped': 0, 'outcome_changes': {}}
    
    for case_ref in matches:
        orig = original_data[case_ref]
        mapped = map_outcome_type(
            orig['outcome_type'],
            orig['outcome'],
            bool(orig['has_pfeo'])
        )
        
        if mapped['ai_outcome']:
            updates.append((
                mapped['ai_outcome'],
                1 if mapped['pfeo_issued'] else 0,
                1 if mapped['pfeo_proposed'] else 0,
                mapped['pfeo_complied'],
                orig['compensation_amount'] or 0,
                case_ref
            ))
            stats['updated'] += 1
            
            # Track changes
            key = f"{orig['outcome_type'] or orig['outcome']} -> {mapped['ai_outcome']}"
            stats['outcome_changes'][key] = stats['outcome_changes'].get(key, 0) + 1
        else:
            stats['skipped'] += 1
    
    print(f"   {stats['updated']} cases to update")
    print(f"   {stats['skipped']} cases skipped (no mapping)")
    
    # Show mapping distribution
    print("\n📊 Outcome mappings:")
    for mapping, count in sorted(stats['outcome_changes'].items(), key=lambda x: -x[1])[:15]:
        print(f"   {count:4d}: {mapping}")
    
    # Confirm before applying
    response = input("\n⚠️  Apply updates? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        source_conn.close()
        target_conn.close()
        return
    
    # Apply updates
    print("\n💾 Applying updates...")
    
    # Add pfeo_proposed column if missing
    try:
        target_conn.execute("ALTER TABLE cases ADD COLUMN pfeo_proposed INTEGER DEFAULT 0")
        print("   Added pfeo_proposed column")
    except sqlite3.OperationalError:
        pass  # Already exists
    
    target_conn.executemany("""
        UPDATE cases SET
            ai_outcome = ?,
            pfeo_issued = ?,
            pfeo_proposed = ?,
            pfeo_complied = ?,
            compensation_awarded = ?
        WHERE case_reference = ?
    """, updates)
    
    target_conn.commit()
    print(f"   ✅ Updated {len(updates)} cases")
    
    # Show new distribution
    print("\n📊 AFTER recovery:")
    cursor = target_conn.execute("""
        SELECT ai_outcome, COUNT(*), SUM(pfeo_issued), SUM(pfeo_proposed)
        FROM cases GROUP BY ai_outcome ORDER BY COUNT(*) DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0] or '(blank)'}: {row[1]} cases, {row[2] or 0} PFEOs issued, {row[3] or 0} proposed")
    
    source_conn.close()
    target_conn.close()
    
    print("\n✅ Recovery complete!")
    print("   Next: run 'python comparefactors_master_pipeline.py --step 7' to recalculate scores")


if __name__ == "__main__":
    main()
