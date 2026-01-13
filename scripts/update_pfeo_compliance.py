#!/usr/bin/env python3
"""
Update PFEO compliance status based on age.

Logic: If a PFEO is >90 days old and there's no "Failure to Comply" or 
"Referred to Ministers" record, assume the factor complied.

Rationale: Non-compliance would trigger follow-up proceedings which would
appear in the tribunal records. Absence of such records after 90 days 
strongly suggests compliance.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# UPDATE THIS PATH
TARGET_DB = Path(r"data\tribunal\tribunal_enriched.db")

COMPLIANCE_WINDOW_DAYS = 90

def main():
    print("=" * 70)
    print("UPDATE PFEO COMPLIANCE STATUS")
    print("=" * 70)
    
    if not TARGET_DB.exists():
        print(f"❌ Database not found: {TARGET_DB}")
        return
    
    conn = sqlite3.connect(TARGET_DB)
    cursor = conn.cursor()
    
    cutoff_date = (datetime.now() - timedelta(days=COMPLIANCE_WINDOW_DAYS)).strftime('%Y-%m-%d')
    
    print(f"\n📋 Logic: PFEOs issued before {cutoff_date} with unknown compliance")
    print(f"   will be assumed complied (no non-compliance record found)")
    
    # Count current status
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN pfeo_complied = 1 THEN 1 ELSE 0 END) as complied,
            SUM(CASE WHEN pfeo_complied = 0 THEN 1 ELSE 0 END) as not_complied,
            SUM(CASE WHEN pfeo_complied IS NULL THEN 1 ELSE 0 END) as unknown
        FROM cases 
        WHERE pfeo_issued = 1
    """)
    row = cursor.fetchone()
    print(f"\n📊 BEFORE:")
    print(f"   Complied: {row[0]}")
    print(f"   Not complied: {row[1]}")
    print(f"   Unknown: {row[2]}")
    
    # Find cases to update
    cursor.execute("""
        SELECT case_reference, matched_registration_number, decision_date
        FROM cases
        WHERE pfeo_issued = 1
          AND pfeo_complied IS NULL
          AND decision_date < ?
          AND decision_date != ''
        ORDER BY decision_date DESC
    """, (cutoff_date,))
    
    to_update = cursor.fetchall()
    print(f"\n🔄 Found {len(to_update)} old PFEOs with unknown compliance to update")
    
    if to_update:
        print(f"\n📋 Sample cases to update:")
        for case in to_update[:5]:
            print(f"   {case[0]} ({case[1]}) - {case[2]}")
        if len(to_update) > 5:
            print(f"   ... and {len(to_update) - 5} more")
    
    # Also check for recent ones that will remain unknown
    cursor.execute("""
        SELECT case_reference, matched_registration_number, decision_date
        FROM cases
        WHERE pfeo_issued = 1
          AND pfeo_complied IS NULL
          AND (decision_date >= ? OR decision_date = '' OR decision_date IS NULL)
        ORDER BY decision_date DESC
    """, (cutoff_date,))
    
    remaining = cursor.fetchall()
    print(f"\n⏳ {len(remaining)} recent PFEOs will remain as 'unknown' (within {COMPLIANCE_WINDOW_DAYS} days)")
    if remaining:
        for case in remaining[:5]:
            print(f"   {case[0]} ({case[1]}) - {case[2] or 'no date'}")
    
    if not to_update:
        print("\n✅ No updates needed!")
        conn.close()
        return
    
    response = input(f"\n⚠️  Update {len(to_update)} cases to complied? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        conn.close()
        return
    
    # Apply updates
    cursor.execute("""
        UPDATE cases
        SET pfeo_complied = 1
        WHERE pfeo_issued = 1
          AND pfeo_complied IS NULL
          AND decision_date < ?
          AND decision_date != ''
    """, (cutoff_date,))
    
    conn.commit()
    print(f"\n💾 Updated {cursor.rowcount} cases")
    
    # Show new status
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN pfeo_complied = 1 THEN 1 ELSE 0 END) as complied,
            SUM(CASE WHEN pfeo_complied = 0 THEN 1 ELSE 0 END) as not_complied,
            SUM(CASE WHEN pfeo_complied IS NULL THEN 1 ELSE 0 END) as unknown
        FROM cases 
        WHERE pfeo_issued = 1
    """)
    row = cursor.fetchone()
    print(f"\n📊 AFTER:")
    print(f"   Complied: {row[0]}")
    print(f"   Not complied: {row[1]}")
    print(f"   Unknown: {row[2]}")
    
    conn.close()
    
    print("\n✅ Update complete!")
    print("   Next: run 'python comparefactors_master_pipeline.py --step 3,7,9'")


if __name__ == "__main__":
    main()
