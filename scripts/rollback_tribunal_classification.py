"""
Rollback script for tribunal case reclassification.

This script reverses the classification changes made to tribunal_enriched_v2.db.
It restores outcome_detailed to match outcome_original and resets classification fields.

Usage:
    python scripts/rollback_tribunal_classification.py [--dry-run]

Options:
    --dry-run    Show what would be changed without making changes
"""

import sqlite3
import sys

DB_PATH = 'data/tribunal/tribunal_enriched_v2.db'

def rollback(dry_run=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check backup exists
    cols = [c[1] for c in conn.execute('PRAGMA table_info(cases)').fetchall()]
    if 'outcome_original' not in cols:
        print('ERROR: No backup column (outcome_original) found. Cannot rollback.')
        return False

    # Count affected rows
    affected = conn.execute('''
        SELECT COUNT(*) FROM cases
        WHERE outcome_detailed != outcome_original
        OR outcome_detailed IS NOT NULL
    ''').fetchone()[0]

    print(f'Rollback will affect {affected} rows')

    if dry_run:
        print('\n[DRY RUN] No changes made.')
        print('\nSample of what would be restored:')
        samples = conn.execute('''
            SELECT case_reference, outcome_original, outcome_detailed, outcome_category
            FROM cases
            WHERE outcome_detailed != outcome_original
            LIMIT 10
        ''').fetchall()
        for s in samples:
            print(f'  {s["case_reference"]}: {s["outcome_detailed"]} -> {s["outcome_original"]}')
        return True

    # Perform rollback
    conn.execute('''
        UPDATE cases SET
            outcome_detailed = outcome_original,
            outcome_category = NULL,
            is_substantive = NULL,
            classification_confidence = NULL,
            classification_method = NULL,
            classification_timestamp = NULL
    ''')

    conn.commit()
    print(f'\nRollback complete. {conn.total_changes} rows restored.')

    # Verify
    check = conn.execute('''
        SELECT COUNT(*) FROM cases WHERE outcome_detailed != outcome_original
    ''').fetchone()[0]
    print(f'Verification: {check} rows still differ (should be 0)')

    return True

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    rollback(dry_run)
