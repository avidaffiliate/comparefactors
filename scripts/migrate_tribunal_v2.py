#!/usr/bin/env python3
"""
Migrate tribunal_cases from v1 to v2 schema.

Usage:
    python scripts/migrate_tribunal_v2.py --dry-run    # Preview changes
    python scripts/migrate_tribunal_v2.py              # Execute migration

This script:
1. Backs up existing tribunal_cases table
2. Creates new schema with v2 fields
3. Copies data from tribunal_enriched_v2.db
4. Validates the migration
"""

import sqlite3
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Paths (adjust if needed)
MAIN_DB = Path("data/database/comparefactors.db")
V2_DB = Path("data/tribunal/tribunal_enriched_v2.db")
BACKUP_DIR = Path("data/backups")


def backup_database(db_path: Path) -> Path:
    """Create timestamped backup of database."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{db_path.stem}_{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    return backup_path


def get_v2_case_count(v2_db: Path) -> int:
    """Get count of cases in v2 database."""
    conn = sqlite3.connect(v2_db)
    count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    conn.close()
    return count


def migrate(dry_run: bool = False):
    """Execute the migration."""
    
    print("=" * 70)
    print("TRIBUNAL V2 MIGRATION")
    print("=" * 70)
    
    # Validate paths
    if not MAIN_DB.exists():
        print(f"❌ Main database not found: {MAIN_DB}")
        return False
    
    if not V2_DB.exists():
        print(f"❌ V2 database not found: {V2_DB}")
        return False
    
    v2_count = get_v2_case_count(V2_DB)
    print(f"\n📊 Source: {V2_DB}")
    print(f"   Cases to migrate: {v2_count}")
    
    # Connect to main DB and get current state
    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row
    
    current_count = conn.execute("SELECT COUNT(*) FROM tribunal_cases").fetchone()[0]
    print(f"\n📊 Target: {MAIN_DB}")
    print(f"   Current cases: {current_count}")
    
    # Show schema changes
    print("\n📋 SCHEMA CHANGES:")
    print("   DROPPING:")
    print("     - complaints_made")
    print("     - complaints_upheld")
    print("     - severity_score")
    print("     - complaint_categories")
    print("   ADDING:")
    print("     - application_dismissed (BOOLEAN)")
    print("     - application_withdrawn (BOOLEAN)")
    print("     - breach_found (BOOLEAN)")
    print("     - pfeo_proposed (BOOLEAN)")
    print("     - pfeo_breached (BOOLEAN)")
    print("     - refund_ordered (REAL)")
    print("     - outcome_reasoning (TEXT)")
    print("     - validation_errors (TEXT)")
    print("   KEEPING:")
    print("     - case_reference, factor_registration_number, decision_date")
    print("     - outcome, pdf_url, summary, key_quote")
    print("     - compensation_awarded, pfeo_issued, pfeo_complied")
    
    if dry_run:
        print("\n🔍 DRY RUN - No changes made")
        conn.close()
        return True
    
    # Backup
    print("\n💾 Creating backup...")
    backup_path = backup_database(MAIN_DB)
    print(f"   Saved to: {backup_path}")
    
    # Create new table with v2 schema
    print("\n🔧 Migrating schema...")
    
    try:
        # Rename old table
        conn.execute("ALTER TABLE tribunal_cases RENAME TO tribunal_cases_v1_backup")
        
        # Create new table with v2 schema
        conn.execute("""
            CREATE TABLE tribunal_cases (
                case_reference TEXT PRIMARY KEY,
                factor_registration_number TEXT,
                decision_date TEXT,
                outcome TEXT,
                
                -- v2 boolean outcome fields
                application_dismissed BOOLEAN DEFAULT 0,
                application_withdrawn BOOLEAN DEFAULT 0,
                breach_found BOOLEAN DEFAULT 0,
                pfeo_proposed BOOLEAN DEFAULT 0,
                pfeo_issued BOOLEAN DEFAULT 0,
                pfeo_complied BOOLEAN DEFAULT 0,
                pfeo_breached BOOLEAN DEFAULT 0,
                
                -- Financial
                compensation_awarded REAL DEFAULT 0,
                refund_ordered REAL DEFAULT 0,
                
                -- Content
                pdf_url TEXT,
                summary TEXT,
                key_quote TEXT,
                outcome_reasoning TEXT,
                
                -- Metadata
                validation_errors TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index
        conn.execute("""
            CREATE INDEX idx_tribunal_factor 
            ON tribunal_cases(factor_registration_number)
        """)
        
        conn.commit()
        print("   ✅ New schema created")
        
    except Exception as e:
        print(f"   ❌ Schema migration failed: {e}")
        conn.execute("ALTER TABLE tribunal_cases_v1_backup RENAME TO tribunal_cases")
        conn.commit()
        conn.close()
        return False
    
    # Copy data from v2 database
    print("\n📥 Copying data from v2...")
    
    try:
        conn.execute(f"ATTACH DATABASE '{V2_DB}' AS v2")
        
        conn.execute("""
            INSERT INTO tribunal_cases (
                case_reference,
                factor_registration_number,
                decision_date,
                outcome,
                application_dismissed,
                application_withdrawn,
                breach_found,
                pfeo_proposed,
                pfeo_issued,
                pfeo_complied,
                pfeo_breached,
                compensation_awarded,
                refund_ordered,
                pdf_url,
                summary,
                key_quote,
                outcome_reasoning,
                validation_errors
            )
            SELECT 
                case_reference,
                matched_registration_number,
                decision_date,
                outcome,
                application_dismissed,
                application_withdrawn,
                breach_found,
                pfeo_proposed,
                pfeo_issued,
                pfeo_complied,
                pfeo_breached,
                COALESCE(compensation_awarded, 0),
                COALESCE(refund_ordered, 0),
                pdf_url,
                summary,
                key_quote,
                outcome_reasoning,
                validation_errors
            FROM v2.cases
            WHERE outcome IS NOT NULL
        """)
        
        migrated = conn.execute("SELECT COUNT(*) FROM tribunal_cases").fetchone()[0]
        conn.commit()
        
        conn.execute("DETACH DATABASE v2")
        
        print(f"   ✅ Migrated {migrated} cases")
        
    except Exception as e:
        print(f"   ❌ Data copy failed: {e}")
        # Restore backup
        conn.close()
        shutil.copy2(backup_path, MAIN_DB)
        print(f"   Restored from backup")
        return False
    
    # Validate
    print("\n🔍 Validating migration...")
    
    outcome_dist = conn.execute("""
        SELECT outcome, COUNT(*) as cnt 
        FROM tribunal_cases 
        GROUP BY outcome 
        ORDER BY cnt DESC
    """).fetchall()
    
    print("   Outcome distribution:")
    for row in outcome_dist:
        print(f"     {row[0]}: {row[1]}")
    
    # Check for any validation errors
    errors = conn.execute("""
        SELECT COUNT(*) FROM tribunal_cases 
        WHERE validation_errors IS NOT NULL 
        AND validation_errors != '[]'
    """).fetchone()[0]
    print(f"   Cases with validation warnings: {errors}")
    
    # Final counts
    final_count = conn.execute("SELECT COUNT(*) FROM tribunal_cases").fetchone()[0]
    print(f"\n✅ MIGRATION COMPLETE")
    print(f"   Before: {current_count} cases")
    print(f"   After: {final_count} cases")
    print(f"   Backup: {backup_path}")
    
    # Cleanup prompt
    print(f"\n🧹 Old table preserved as 'tribunal_cases_v1_backup'")
    print(f"   To remove: DROP TABLE tribunal_cases_v1_backup;")
    
    conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate tribunal_cases to v2 schema")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    args = parser.parse_args()
    
    success = migrate(dry_run=args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())