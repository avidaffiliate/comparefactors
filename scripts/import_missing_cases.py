#!/usr/bin/env python3
"""
Import missing tribunal cases from comparefactors.db into tribunal_enriched.db.

Uses the good data from:
- tribunal_cases (outcomes, PFEO status, compensation)
- tribunal_extractions (summaries, severity scores)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# UPDATE THESE PATHS
SOURCE_DB = Path(r"C:\Users\david\Downloads\comparefactors-fixed_1\comparefactors.db")
TARGET_DB = Path(r"data\tribunal\tribunal_enriched.db")


def map_outcome(outcome_type, outcome, has_pfeo):
    """Map original scraped outcome to ai_outcome field."""
    outcome_type = (outcome_type or '').strip()
    outcome = (outcome or '').strip()
    
    result = {
        'ai_outcome': '',
        'pfeo_issued': False,
        'pfeo_proposed': False,
        'pfeo_complied': None,
    }
    
    if outcome_type == 'Dismissed':
        result['ai_outcome'] = 'Dismissed'
    elif outcome_type == 'PFEO Issued':
        result['ai_outcome'] = 'Upheld'
        result['pfeo_issued'] = True
    elif outcome_type == 'Complied':
        result['ai_outcome'] = 'Upheld'
        result['pfeo_issued'] = True
        result['pfeo_complied'] = True
    elif outcome_type == 'Rejected':
        result['ai_outcome'] = 'Dismissed'
    elif outcome_type == 'Upheld No Order':
        result['ai_outcome'] = 'Upheld'
    elif outcome_type == 'Compensation Only':
        result['ai_outcome'] = 'Upheld'
    elif outcome_type == 'Withdrawn':
        result['ai_outcome'] = 'Withdrawn'
    elif outcome_type == '':
        if 'PFEO' in outcome and 'Non-Compliant' in outcome:
            result['ai_outcome'] = 'Referred to Ministers'
            result['pfeo_issued'] = True
            result['pfeo_complied'] = False
        elif 'Failure to Comply' in outcome:
            result['ai_outcome'] = 'Upheld'
            result['pfeo_issued'] = True
            result['pfeo_complied'] = False
        elif 'PFEO Proposed' in outcome:
            result['ai_outcome'] = 'Upheld'
            result['pfeo_proposed'] = True
        elif 'PFEO Issued' in outcome:
            result['ai_outcome'] = 'Upheld'
            result['pfeo_issued'] = True
        elif 'Complied' in outcome:
            result['ai_outcome'] = 'Upheld'
            result['pfeo_issued'] = True
            result['pfeo_complied'] = True
        elif 'Decision Issued' in outcome:
            result['ai_outcome'] = 'Upheld'
            if has_pfeo:
                result['pfeo_issued'] = True
        elif 'Rejected' in outcome:
            result['ai_outcome'] = 'Dismissed'
        elif 'Dismissed' in outcome:
            result['ai_outcome'] = 'Dismissed'
    
    # PFEO override
    if has_pfeo and not result['pfeo_issued'] and not result['pfeo_proposed']:
        result['pfeo_issued'] = True
    
    # Can't have PFEO and Dismissed
    if has_pfeo and result['ai_outcome'] == 'Dismissed':
        result['ai_outcome'] = 'Upheld'
    
    return result


def main():
    print("=" * 70)
    print("IMPORT MISSING TRIBUNAL CASES")
    print("=" * 70)
    
    if not SOURCE_DB.exists():
        print(f"❌ Source DB not found: {SOURCE_DB}")
        return
    
    if not TARGET_DB.exists():
        print(f"❌ Target DB not found: {TARGET_DB}")
        return
    
    source_conn = sqlite3.connect(SOURCE_DB)
    source_conn.row_factory = sqlite3.Row
    target_conn = sqlite3.connect(TARGET_DB)
    target_conn.row_factory = sqlite3.Row
    
    # Get existing cases in target
    cursor = target_conn.execute("SELECT case_reference FROM cases")
    existing = {row[0] for row in cursor.fetchall()}
    print(f"\n📖 Target has {len(existing)} existing cases")
    
    # Get source cases with extractions
    cursor = source_conn.execute("""
        SELECT 
            tc.case_reference,
            tc.factor_registration_number,
            tc.hearing_date,
            tc.outcome,
            tc.outcome_type,
            tc.has_pfeo,
            tc.compensation_amount,
            tc.pdf_urls,
            te.summary,
            te.severity_score,
            te.key_quote
        FROM tribunal_cases tc
        LEFT JOIN tribunal_extractions te ON tc.case_reference = te.case_reference
    """)
    source_cases = list(cursor.fetchall())
    print(f"📖 Source has {len(source_cases)} cases")
    
    # Find missing
    missing = [c for c in source_cases if c['case_reference'] not in existing]
    print(f"📊 {len(missing)} cases to import")
    
    if not missing:
        print("\n✅ No missing cases - all synced!")
        
        # But let's update summaries for existing cases
        print("\n🔄 Updating summaries for existing cases...")
        updates = 0
        for case in source_cases:
            if case['case_reference'] in existing and case['summary']:
                target_conn.execute("""
                    UPDATE cases SET summary = ?, severity_score = ?
                    WHERE case_reference = ? AND (summary IS NULL OR summary = '')
                """, (case['summary'], case['severity_score'] or 0, case['case_reference']))
                updates += target_conn.total_changes
        
        target_conn.commit()
        print(f"   Updated {updates} summaries")
        
        source_conn.close()
        target_conn.close()
        return
    
    # Show sample
    print("\n📋 Sample missing cases:")
    for case in missing[:5]:
        print(f"   {case['case_reference']}: {case['outcome_type'] or case['outcome']}")
    
    # Prepare inserts
    inserts = []
    now = datetime.now().isoformat()
    
    for case in missing:
        mapped = map_outcome(
            case['outcome_type'],
            case['outcome'],
            bool(case['has_pfeo'])
        )
        
        inserts.append((
            case['case_reference'],
            case['factor_registration_number'] or '',
            case['hearing_date'] or '',
            mapped['ai_outcome'],
            1 if mapped['pfeo_issued'] else 0,
            1 if mapped['pfeo_proposed'] else 0,
            mapped['pfeo_complied'],
            case['compensation_amount'] or 0,
            case['summary'] or '',
            case['severity_score'] or 0,
            case['key_quote'] or '',
            case['pdf_urls'] or '',
            now,
            1  # extraction_success
        ))
    
    response = input(f"\n⚠️  Import {len(inserts)} cases? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        source_conn.close()
        target_conn.close()
        return
    
    # Ensure columns exist
    try:
        target_conn.execute("ALTER TABLE cases ADD COLUMN pfeo_proposed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    
    # Insert
    print("\n💾 Importing...")
    target_conn.executemany("""
        INSERT OR IGNORE INTO cases (
            case_reference,
            matched_registration_number,
            decision_date,
            ai_outcome,
            pfeo_issued,
            pfeo_proposed,
            pfeo_complied,
            compensation_awarded,
            summary,
            severity_score,
            key_quote,
            pdf_url,
            extracted_at,
            extraction_success
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, inserts)
    
    target_conn.commit()
    print(f"   ✅ Imported {len(inserts)} cases")
    
    # Final count
    cursor = target_conn.execute("SELECT COUNT(*) FROM cases")
    total = cursor.fetchone()[0]
    print(f"\n📊 Target now has {total} cases")
    
    # Distribution
    print("\n📊 Final outcome distribution:")
    cursor = target_conn.execute("""
        SELECT ai_outcome, COUNT(*), SUM(pfeo_issued)
        FROM cases GROUP BY ai_outcome ORDER BY COUNT(*) DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0] or '(blank)'}: {row[1]} cases, {row[2] or 0} PFEOs")
    
    source_conn.close()
    target_conn.close()
    
    print("\n✅ Import complete!")
    print("   Next: run 'python comparefactors_master_pipeline.py --step 7'")


if __name__ == "__main__":
    main()
