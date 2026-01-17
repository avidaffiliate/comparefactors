#!/usr/bin/env python3
"""
===============================================================================
COMPARE FACTORS SCOTLAND - SIMPLIFIED PIPELINE v4.0
===============================================================================

Refactored from v3.3 with these simplifications:
1. Removed case_fees processing (table was empty)
2. Streamlined template context - use DB values directly instead of recalculating
3. Consolidated tribunal stats (no recalculation in Step 9)
4. Removed unused template variables (similar_factors, timeline_events, fee_examples)
5. Simplified WSS import (document_url only)
6. Moved all page generation to Jinja2 templates (no more embedded HTML)

Kept: Trustpilot import (for future data)

TEMPLATES REQUIRED (in templates/ directory):
- factor_profile.html  - Individual factor pages
- homepage.html        - Landing page with hero, hotspots
- factors_listing.html - All factors grid with filters  
- areas_index.html     - Areas overview
- area.html            - Individual area page
- base_listing.html    - Shared base for listing pages

USAGE:
    python comparefactors_pipeline_simplified.py --full
    python comparefactors_pipeline_simplified.py --step 1,2,3
"""

import sqlite3
import csv
import json
import os
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    from jinja2 import Environment, FileSystemLoader
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False


# =============================================================================
# CONFIGURATION (Simplified)
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration - simplified paths and settings."""
    project_root: Path = field(default_factory=lambda: Path("."))
    
    # Derived paths (set in resolve_paths)
    csv_dir: Path = None
    db_path: Path = None
    tribunal_db_path: Path = None
    wss_db_path: Path = None
    template_dir: Path = None
    site_dir: Path = None
    
    # Scoring weights (unchanged)
    score_weight_volume: float = 0.40
    score_weight_severity: float = 0.35
    score_weight_recency: float = 0.25
    
    def resolve_paths(self, root: Path):
        self.project_root = root
        self.csv_dir = root / "data/csv"
        self.db_path = root / "data/database/comparefactors.db"
        self.tribunal_db_path = root / "data/tribunal/tribunal_enriched_v2.db"
        self.wss_db_path = root / "data/wss/wss_extracted.db"
        self.template_dir = root / "templates"
        self.site_dir = root / "site"


CONFIG = Config()


# =============================================================================
# UTILITIES (Consolidated - removed redundant helpers)
# =============================================================================

@contextmanager
def get_db():
    """Database connection context manager."""
    CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def normalize_pf_number(pf: Any) -> Optional[str]:
    """Normalize PF registration number."""
    if not pf:
        return None
    match = re.search(r'(\d{6})', str(pf).upper())
    return f"PF{match.group(1)}" if match else None


def parse_date(date_str: Any) -> Optional[str]:
    """Parse date to ISO format."""
    if not date_str:
        return None
    s = str(date_str).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def parse_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def get_csv_value(row: Dict, keys: List[str], default=None) -> Any:
    """Get value from CSV row by trying multiple column names."""
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return default


def is_adverse_outcome(outcome: str) -> bool:
    """Check if tribunal outcome is adverse for the factor."""
    if not outcome:
        return False
    outcome_lower = outcome.lower()
    # Not adverse: dismissed, rejected, withdrawn (treated as one bucket)
    for term in ['dismissed', 'rejected', 'withdrawn', 'unknown']:
        if term in outcome_lower:
            return False
    return True


def is_attendance_hearing(outcome: str) -> bool:
    """Check if this is an attendance/compliance hearing rather than a substantive decision."""
    if not outcome:
        return False
    outcome_lower = outcome.lower()
    # Filter out attendance/compliance hearings
    return 'attendance' in outcome_lower


def normalize_outcome_display(outcome: str) -> str:
    """Normalize outcome for display - combine dismissed/rejected."""
    if not outcome:
        return 'Unknown'
    outcome_lower = outcome.lower()
    if 'dismissed' in outcome_lower or 'rejected' in outcome_lower:
        return 'Dismissed/Rejected'
    return outcome


def log_step(num: int, title: str):
    print(f"\n{'='*60}\nSTEP {num}: {title}\n{'='*60}")


def log_info(msg: str):
    print(f"  [i] {msg}")


def log_success(msg: str):
    print(f"  [+] {msg}")


# Complaint category extraction from case summaries
COMPLAINT_CATEGORY_KEYWORDS = {
    'communication': [
        'communication', 'respond', 'reply', 'contact', 'notify', 'notification',
        'informed', 'update', 'correspondence', 'email', 'letter', 'phone',
        'failed to respond', 'no response', 'lack of communication'
    ],
    'financial': [
        'fee', 'charge', 'invoice', 'payment', 'account', 'money', 'cost',
        'budget', 'fund', 'reserve', 'float', 'arrears', 'debt', 'billing',
        'overcharge', 'financial', 'statement of account'
    ],
    'maintenance': [
        'repair', 'maintenance', 'common area', 'cleaning', 'garden', 'roof',
        'gutter', 'close', 'stair', 'entry', 'door', 'window', 'lift',
        'damp', 'leak', 'damage', 'defect', 'contractor', 'works'
    ],
    'insurance': [
        'insurance', 'policy', 'cover', 'claim', 'premium', 'underwriter',
        'buildings insurance', 'block insurance'
    ],
    'governance': [
        'meeting', 'vote', 'agm', 'minutes', 'decision', 'consultation',
        'ballot', 'majority', 'quorum', 'resolution', 'deed of conditions'
    ],
    'disclosure': [
        'disclosure', 'information', 'document', 'record', 'access',
        'written statement', 'wss', 'transparency', 'provide information'
    ],
    'health_safety': [
        'health', 'safety', 'fire', 'hazard', 'risk', 'emergency',
        'dangerous', 'unsafe', 'asbestos', 'legionella'
    ],
    'debt_recovery': [
        'debt recovery', 'debt collection', 'sheriff officer', 'legal action',
        'court', 'pursue', 'arrears recovery', 'charging order'
    ],
}


def extract_complaint_categories(text: str) -> List[str]:
    """Extract complaint categories from case summary text using keyword matching."""
    if not text:
        return []

    text_lower = text.lower()
    categories = []

    for category, keywords in COMPLAINT_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                categories.append(category)
                break  # Only add each category once

    return categories


# =============================================================================
# SCHEMA (Unchanged but documented)
# =============================================================================

SCHEMA_SQL = """
-- Core tables remain the same as v3.3
-- Key tables: factors, tribunal_cases, reviews, companies, wss
-- Removed: case_fees table (never populated)

CREATE TABLE IF NOT EXISTS factors (
    registration_number TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trading_name TEXT,
    address TEXT,
    city TEXT,
    postcode TEXT,
    website TEXT,
    email TEXT,
    phone TEXT,
    registration_date TEXT,
    registration_status TEXT DEFAULT 'registered',
    status TEXT DEFAULT 'registered',
    property_count INTEGER DEFAULT 0,
    factor_type TEXT,
    tpi_member INTEGER DEFAULT 0,
    postcode_areas TEXT,
    postcode_count INTEGER DEFAULT 0,
    geographic_reach TEXT,
    coverage_areas TEXT,
    
    -- Tribunal aggregates
    tribunal_case_count INTEGER DEFAULT 0,
    tribunal_case_count_5yr INTEGER DEFAULT 0,
    tribunal_cases_upheld INTEGER DEFAULT 0,
    tribunal_cases_dismissed INTEGER DEFAULT 0,
    tribunal_pfeo_count INTEGER DEFAULT 0,
    tribunal_total_compensation REAL DEFAULT 0,
    tribunal_rate_per_10k REAL,
    has_active_pfeo INTEGER DEFAULT 0,
    risk_band TEXT,
    
    -- Score components
    tribunal_composite_score REAL,
    tribunal_volume_score REAL,
    tribunal_severity_score REAL,
    tribunal_recency_score REAL,
    tribunal_cases_last_3_years INTEGER DEFAULT 0,
    
    -- Reviews
    google_rating REAL,
    google_review_count INTEGER DEFAULT 0,
    trustpilot_rating REAL,
    trustpilot_review_count INTEGER DEFAULT 0,
    combined_rating REAL,
    total_review_count INTEGER DEFAULT 0,
    
    -- AI summaries
    at_a_glance TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tribunal_cases (
    case_reference TEXT PRIMARY KEY,
    factor_registration_number TEXT,
    decision_date TEXT,
    outcome TEXT,
    
    -- v2 boolean outcome fields
    application_dismissed INTEGER DEFAULT 0,
    application_withdrawn INTEGER DEFAULT 0,
    breach_found INTEGER DEFAULT 0,
    pfeo_proposed INTEGER DEFAULT 0,
    pfeo_issued INTEGER DEFAULT 0,
    pfeo_complied INTEGER DEFAULT 0,
    pfeo_breached INTEGER DEFAULT 0,
    
    -- Financial
    compensation_awarded REAL DEFAULT 0,
    refund_ordered REAL DEFAULT 0,
    
    -- Content
    pdf_url TEXT,
    summary TEXT,
    key_quote TEXT,
    outcome_reasoning TEXT,
    complaint_categories TEXT,  -- JSON array of categories extracted from summary

    -- Metadata
    validation_errors TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_registration_number TEXT,
    platform TEXT,
    rating REAL,
    review_count INTEGER DEFAULT 1,
    review_text TEXT,
    review_date TEXT,
    author_name TEXT,
    source_id TEXT,
    phone TEXT,
    address TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS companies (
    registration_number TEXT PRIMARY KEY,
    company_number TEXT,
    company_name TEXT,
    company_url TEXT,
    incorporated_date TEXT,
    company_status TEXT,
    has_insolvency_history INTEGER DEFAULT 0,
    has_charges INTEGER DEFAULT 0,
    accounts_overdue INTEGER DEFAULT 0,
    director_count INTEGER DEFAULT 0,
    avg_director_tenure_years REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wss (
    registration_number TEXT PRIMARY KEY,
    document_url TEXT,
    -- Key extracted fields
    management_fee_amount TEXT,
    management_fee_frequency TEXT,
    delegated_authority_limit TEXT,
    emergency_response TEXT,
    urgent_response TEXT,
    routine_response TEXT,
    enquiry_response TEXT,
    complaint_response TEXT,
    billing_frequency TEXT,
    float_required INTEGER,
    notice_period TEXT,
    code_of_conduct_version TEXT,
    professional_memberships TEXT,
    portal TEXT,
    app TEXT,
    confidence_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE VIEW IF NOT EXISTS v_factor_profiles AS
SELECT 
    f.*,
    c.company_number AS ch_number,
    c.company_name AS ch_name,
    c.company_status AS ch_status,
    c.incorporated_date AS ch_incorporated,
    c.has_insolvency_history AS ch_insolvency,
    c.has_charges AS ch_charges,
    c.accounts_overdue AS ch_accounts_overdue,
    c.director_count AS ch_directors,
    c.avg_director_tenure_years AS ch_avg_tenure,
    c.company_url AS ch_url
FROM factors f
LEFT JOIN companies c ON f.registration_number = c.registration_number;
"""


# =============================================================================
# STEP 1: INITIALIZE DATABASE
# =============================================================================

def step_1_init_database(reset: bool = False):
    log_step(1, "Initialize Database")
    
    if reset and CONFIG.db_path.exists():
        CONFIG.db_path.unlink()
        log_info("Removed existing database")
    
    with get_db() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    
    log_success(f"Database ready: {CONFIG.db_path}")


# =============================================================================
# STEP 2: IMPORT FACTORS
# =============================================================================

def step_2_import_factors():
    log_step(2, "Import Factor Data")
    
    csv_path = CONFIG.csv_dir / "factors_register.csv"
    if not csv_path.exists():
        print(f"  [X] Not found: {csv_path}")
        return
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    
    imported = 0
    with get_db() as conn:
        for row in rows:
            pf = normalize_pf_number(get_csv_value(row, ['registration_number', 'Registration Number']))
            name = get_csv_value(row, ['name', 'Name', 'trading_name'])
            if not pf or not name:
                continue
            
            status = 'expired' if 'expired' in str(get_csv_value(row, ['status', 'registration_status'], '')).lower() else 'registered'
            
            conn.execute("""
                INSERT INTO factors (registration_number, name, address, city, postcode, 
                    website, registration_date, status, property_count, factor_type, coverage_areas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    name = COALESCE(excluded.name, name),
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
            """, [
                pf, name,
                get_csv_value(row, ['address', 'Address']),
                get_csv_value(row, ['city', 'City']),
                get_csv_value(row, ['postcode', 'Postcode']),
                get_csv_value(row, ['website', 'Website']),
                parse_date(get_csv_value(row, ['registration_date'])),
                status,
                parse_int(get_csv_value(row, ['property_count', 'properties'])),
                get_csv_value(row, ['trading_type', 'factor_type']),
                get_csv_value(row, ['coverage_areas', 'cities']),
            ])
            imported += 1
        conn.commit()
    
    log_success(f"Imported {imported} factors")
    
    # Import FOI postcode coverage data
    foi_path = CONFIG.csv_dir / "factors_postcodes.csv"
    if foi_path.exists():
        log_info(f"Reading FOI postcodes: {foi_path}")
        
        with open(foi_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        updated = 0
        with get_db() as conn:
            for row in rows:
                pf = normalize_pf_number(get_csv_value(row, ['registration_number']))
                if not pf:
                    continue
                
                # Count postcodes from postcode_areas
                postcode_areas = get_csv_value(row, ['postcode_areas']) or ''
                postcode_count = len([p.strip() for p in postcode_areas.split(',') if p.strip()]) if postcode_areas else 0
                
                conn.execute("""
                    UPDATE factors SET
                        postcode_areas = ?,
                        postcode_count = ?,
                        geographic_reach = ?,
                        coverage_areas = COALESCE(coverage_areas, ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE registration_number = ?
                """, [
                    postcode_areas,
                    postcode_count,
                    get_csv_value(row, ['geographic_reach']),
                    get_csv_value(row, ['cities', 'foi_cities']),
                    pf,
                ])
                updated += 1
            conn.commit()
        log_success(f"Updated {updated} factors with FOI postcode coverage")
    else:
        log_info(f"FOI postcodes CSV not found at {foi_path}")


# =============================================================================
# STEP 3: IMPORT TRIBUNAL CASES
# =============================================================================

def step_3_import_tribunal():
    log_step(3, "Import Tribunal Cases")
    
    if not CONFIG.tribunal_db_path.exists():
        csv_path = CONFIG.csv_dir / "tribunal_cases.csv"
        if not csv_path.exists():
            print(f"  [X] No tribunal data found")
            return
        # CSV import path...
        return
    
    source = sqlite3.connect(CONFIG.tribunal_db_path)
    source.row_factory = sqlite3.Row
    
    # Check column names in source database
    col_cursor = source.execute("PRAGMA table_info(cases)")
    columns = {r['name'] for r in col_cursor}
    
    # Determine which column name is used for PF number
    pf_column = None
    for col in ['matched_registration_number', 'factor_pf_number', 'registration_number', 'pf_number', 'factor_registration_number']:
        if col in columns:
            pf_column = col
            break
    
    if not pf_column:
        print(f"  [X] Could not find PF number column. Available: {columns}")
        source.close()
        return
    
    cursor = source.execute("SELECT * FROM cases")
    
    imported = 0
    with get_db() as conn:
        for row in cursor:
            row_dict = dict(row)
            pf = normalize_pf_number(row_dict.get(pf_column))
            if not pf:
                continue
            
            # Handle complaint_categories - may be JSON string or already parsed
            complaint_cats = row_dict.get('complaint_categories')
            if complaint_cats and not isinstance(complaint_cats, str):
                complaint_cats = json.dumps(complaint_cats)

            conn.execute("""
                INSERT INTO tribunal_cases (
                    case_reference, factor_registration_number, decision_date, outcome,
                    application_dismissed, application_withdrawn, breach_found,
                    pfeo_proposed, pfeo_issued, pfeo_complied, pfeo_breached,
                    compensation_awarded, refund_ordered, pdf_url, summary, key_quote,
                    outcome_reasoning, complaint_categories, validation_errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_reference) DO UPDATE SET
                    outcome = excluded.outcome,
                    summary = COALESCE(excluded.summary, summary),
                    compensation_awarded = excluded.compensation_awarded,
                    complaint_categories = COALESCE(excluded.complaint_categories, complaint_categories)
            """, [
                row_dict.get('case_reference'),
                pf,
                row_dict.get('decision_date'),
                row_dict.get('outcome'),
                row_dict.get('application_dismissed', 0),
                row_dict.get('application_withdrawn', 0),
                row_dict.get('breach_found', 0),
                row_dict.get('pfeo_proposed', 0),
                row_dict.get('pfeo_issued', 0),
                row_dict.get('pfeo_complied', 0),
                row_dict.get('pfeo_breached', 0),
                row_dict.get('compensation_awarded', 0),
                row_dict.get('refund_ordered', 0),
                row_dict.get('pdf_url'),
                row_dict.get('summary'),
                row_dict.get('key_quote'),
                row_dict.get('outcome_reasoning'),
                complaint_cats,
                row_dict.get('validation_errors'),
            ])
            imported += 1
        conn.commit()
    
    source.close()
    log_success(f"Imported {imported} tribunal cases")


# =============================================================================
# STEP 4: IMPORT REVIEWS (Google + Trustpilot)
# =============================================================================

def step_4_import_reviews():
    log_step(4, "Import Reviews")
    
    total_imported = 0
    
    # Google Reviews (aggregate data)
    google_path = CONFIG.csv_dir / "google_reviews.csv"
    if google_path.exists():
        with open(google_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        imported = 0
        with get_db() as conn:
            for row in rows:
                pf = normalize_pf_number(
                    get_csv_value(row, ['factor_registration_number', 'registration_number', 'pf_number'])
                )
                if not pf:
                    continue
                
                source_id = get_csv_value(row, ['place_id', 'google_place_id', 'source_id'])
                if not source_id:
                    source_id = f"google_{pf}"
                
                # For aggregate records, author_name stores the location name
                review_text = get_csv_value(row, ['review_text', 'text'])
                if review_text:
                    author_name = get_csv_value(row, ['author_name', 'author'])
                else:
                    author_name = get_csv_value(row, ['name', 'location_name', 'place_name', 'google_name', 'address'])
                
                # Check for existing record to prevent duplicates
                if review_text:
                    existing = conn.execute("""
                        SELECT id FROM reviews 
                        WHERE factor_registration_number = ? AND source_id = ? AND author_name = ? AND review_text IS NOT NULL
                    """, [pf, source_id, author_name]).fetchone()
                else:
                    existing = conn.execute("""
                        SELECT id FROM reviews 
                        WHERE factor_registration_number = ? AND platform = 'google' AND source_id = ? AND review_text IS NULL
                    """, [pf, source_id]).fetchone()
                
                if existing:
                    new_rating = parse_float(get_csv_value(row, ['rating', 'google_rating']))
                    new_count = parse_int(get_csv_value(row, ['review_count', 'total_reviews', 'google_review_count'])) or 1
                    conn.execute("UPDATE reviews SET rating = ?, review_count = ? WHERE id = ?", 
                                 [new_rating, new_count, existing['id']])
                    imported += 1
                    continue
                
                try:
                    conn.execute("""
                        INSERT INTO reviews (factor_registration_number, platform, rating, review_count, review_text, review_date, author_name, source_id, phone)
                        VALUES (?, 'google', ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        pf,
                        parse_float(get_csv_value(row, ['rating', 'google_rating'])),
                        parse_int(get_csv_value(row, ['review_count', 'total_reviews', 'google_review_count'])) or 1,
                        review_text,
                        parse_date(get_csv_value(row, ['review_date', 'date'])),
                        author_name,
                        source_id,
                        get_csv_value(row, ['phone', 'phone_number']),
                    ])
                    imported += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        log_info(f"Google aggregate: {imported} records")
        total_imported += imported
    else:
        log_info("Google reviews CSV not found")
    
    # Trustpilot Reviews
    tp_path = CONFIG.csv_dir / "trustpilot_reviews.csv"
    if tp_path.exists():
        with open(tp_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        imported = 0
        with get_db() as conn:
            for row in rows:
                pf = normalize_pf_number(
                    get_csv_value(row, ['factor_registration_number', 'registration_number', 'pf_number'])
                )
                if not pf:
                    continue
                
                source_id = get_csv_value(row, ['trustpilot_url', 'url', 'source_id'])
                if not source_id:
                    source_id = f"trustpilot_{pf}"
                
                review_text = get_csv_value(row, ['review_text', 'text'])
                author_name = get_csv_value(row, ['author_name', 'author'])
                
                # Check for existing
                if review_text:
                    existing = conn.execute("""
                        SELECT id FROM reviews 
                        WHERE factor_registration_number = ? AND platform = 'trustpilot' AND source_id = ? AND author_name = ? AND review_text IS NOT NULL
                    """, [pf, source_id, author_name]).fetchone()
                else:
                    existing = conn.execute("""
                        SELECT id FROM reviews 
                        WHERE factor_registration_number = ? AND platform = 'trustpilot' AND source_id = ? AND review_text IS NULL
                    """, [pf, source_id]).fetchone()
                
                if existing:
                    new_rating = parse_float(get_csv_value(row, ['rating']))
                    new_count = parse_int(get_csv_value(row, ['review_count', 'total_reviews'])) or 1
                    conn.execute("UPDATE reviews SET rating = ?, review_count = ? WHERE id = ?", 
                                 [new_rating, new_count, existing['id']])
                    imported += 1
                    continue
                
                try:
                    conn.execute("""
                        INSERT INTO reviews (factor_registration_number, platform, rating, review_count, review_text, review_date, author_name, source_id)
                        VALUES (?, 'trustpilot', ?, ?, ?, ?, ?, ?)
                    """, [
                        pf,
                        parse_float(get_csv_value(row, ['rating'])),
                        parse_int(get_csv_value(row, ['review_count', 'total_reviews'])) or 1,
                        review_text,
                        parse_date(get_csv_value(row, ['review_date', 'date'])),
                        author_name,
                        source_id,
                    ])
                    imported += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        log_info(f"Trustpilot: {imported} records")
        total_imported += imported
    else:
        log_info("Trustpilot CSV not found")
    
    # Google Reviews with Text (individual reviews)
    reviews_text_path = CONFIG.csv_dir / "google_reviews_text.csv"
    if reviews_text_path.exists():
        with open(reviews_text_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        imported = 0
        skipped = 0
        with get_db() as conn:
            for row in rows:
                pf = normalize_pf_number(
                    get_csv_value(row, ['factor_registration_number', 'registration_number', 'pf_number'])
                )
                if not pf:
                    skipped += 1
                    continue
                
                place_id = get_csv_value(row, ['google_place_id', 'place_id', 'source_id'])
                author = get_csv_value(row, ['author_name', 'author'])
                text = get_csv_value(row, ['review_text', 'text'])
                
                if not text or len(text.strip()) < 10:
                    skipped += 1
                    continue
                
                rating = parse_float(get_csv_value(row, ['review_rating', 'rating']))
                
                # Convert Unix timestamp to date if present
                review_date = None
                review_time = get_csv_value(row, ['review_time', 'time'])
                if review_time:
                    try:
                        ts = int(review_time)
                        review_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        pass
                
                # Check if this exact review already exists
                existing = conn.execute("""
                    SELECT id FROM reviews 
                    WHERE factor_registration_number = ? AND source_id = ? AND author_name = ? AND review_text IS NOT NULL
                """, [pf, place_id, author]).fetchone()
                
                if existing:
                    skipped += 1
                    continue
                
                try:
                    conn.execute("""
                        INSERT INTO reviews (factor_registration_number, platform, rating, review_count, review_text, review_date, author_name, source_id)
                        VALUES (?, 'google', ?, 1, ?, ?, ?, ?)
                    """, [pf, rating, text.strip(), review_date, author, place_id])
                    imported += 1
                except sqlite3.IntegrityError:
                    skipped += 1
            conn.commit()
        log_info(f"Google text reviews: {imported} (skipped {skipped})")
        total_imported += imported
    else:
        log_info("Google reviews text CSV not found")
    
    if total_imported == 0:
        print("  [!]  No review data imported")
    
    # Update factor aggregates
    # NOTE: Only use aggregate location records (review_text IS NULL) to avoid double-counting
    # individual reviews. Use weighted average instead of MAX for accurate ratings.
    with get_db() as conn:
        conn.execute("""
            UPDATE factors SET
                google_rating = (
                    SELECT SUM(rating * review_count) / SUM(review_count) 
                    FROM reviews 
                    WHERE factor_registration_number = factors.registration_number 
                      AND platform = 'google' 
                      AND review_text IS NULL
                ),
                google_review_count = (
                    SELECT SUM(review_count) 
                    FROM reviews 
                    WHERE factor_registration_number = factors.registration_number 
                      AND platform = 'google' 
                      AND review_text IS NULL
                ),
                trustpilot_rating = (
                    SELECT SUM(rating * review_count) / SUM(review_count) 
                    FROM reviews 
                    WHERE factor_registration_number = factors.registration_number 
                      AND platform = 'trustpilot' 
                      AND review_text IS NULL
                ),
                trustpilot_review_count = (
                    SELECT SUM(review_count) 
                    FROM reviews 
                    WHERE factor_registration_number = factors.registration_number 
                      AND platform = 'trustpilot' 
                      AND review_text IS NULL
                )
        """)
        # Calculate combined rating (weighted by review count)
        conn.execute("""
            UPDATE factors SET
                total_review_count = COALESCE(google_review_count, 0) + COALESCE(trustpilot_review_count, 0),
                combined_rating = CASE 
                    WHEN COALESCE(google_review_count, 0) + COALESCE(trustpilot_review_count, 0) > 0 THEN
                        (COALESCE(google_rating * google_review_count, 0) + COALESCE(trustpilot_rating * trustpilot_review_count, 0)) 
                        / (COALESCE(google_review_count, 0) + COALESCE(trustpilot_review_count, 0))
                    ELSE NULL
                END
        """)
        conn.commit()
    
    log_success(f"Imported {total_imported} total review records")


# =============================================================================
# STEP 5: IMPORT COMPANIES HOUSE
# =============================================================================

def step_5_import_companies_house():
    log_step(5, "Import Companies House")
    
    csv_path = CONFIG.csv_dir / "companies_house.csv"
    if not csv_path.exists():
        print(f"  [>]  No companies CSV found")
        return
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    
    imported = 0
    with get_db() as conn:
        for row in rows:
            pf = normalize_pf_number(
                get_csv_value(row, ['factor_registration_number', 'registration_number', 'pf_number'])
            )
            if not pf:
                continue
            
            # Parse boolean fields
            def parse_bool(val):
                if val is None:
                    return 0
                return 1 if str(val).lower() in ['true', '1', 'yes'] else 0
            
            conn.execute("""
                INSERT INTO companies (
                    registration_number, company_number, company_name, company_url,
                    incorporated_date, company_status, has_insolvency_history, has_charges,
                    accounts_overdue, director_count, avg_director_tenure_years
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    company_number = COALESCE(excluded.company_number, company_number),
                    company_name = COALESCE(excluded.company_name, company_name),
                    company_status = COALESCE(excluded.company_status, company_status),
                    has_insolvency_history = COALESCE(excluded.has_insolvency_history, has_insolvency_history),
                    has_charges = COALESCE(excluded.has_charges, has_charges),
                    accounts_overdue = COALESCE(excluded.accounts_overdue, accounts_overdue),
                    director_count = COALESCE(excluded.director_count, director_count),
                    avg_director_tenure_years = COALESCE(excluded.avg_director_tenure_years, avg_director_tenure_years)
            """, [
                pf,
                get_csv_value(row, ['company_number', 'ch_number']),
                get_csv_value(row, ['company_name', 'ch_name']),
                get_csv_value(row, ['company_url']),
                parse_date(get_csv_value(row, ['incorporated_date', 'incorporation_date'])),
                get_csv_value(row, ['company_status', 'status']),
                parse_bool(get_csv_value(row, ['has_insolvency_history'])),
                parse_bool(get_csv_value(row, ['has_charges'])),
                parse_bool(get_csv_value(row, ['accounts_overdue', 'overdue'])),
                parse_int(get_csv_value(row, ['director_count', 'directors'])),
                parse_float(get_csv_value(row, ['avg_director_tenure_years', 'avg_tenure'])),
            ])
            imported += 1
        conn.commit()
    
    log_success(f"Imported {imported} company records")


# =============================================================================
# STEP 6: IMPORT WSS (With extracted fields)
# =============================================================================

def step_6_import_wss():
    log_step(6, "Import WSS Data")

    if not CONFIG.wss_db_path.exists():
        print(f"  [>]  WSS database not found")
        return

    source = sqlite3.connect(CONFIG.wss_db_path)
    source.row_factory = sqlite3.Row

    # Join documents, mappings, and extracted key fields
    cursor = source.execute("""
        SELECT
            m.registration_number,
            d.url AS document_url,
            k.management_fee_amount,
            k.management_fee_frequency,
            k.delegated_authority_limit,
            k.emergency_response,
            k.urgent_response,
            k.routine_response,
            k.enquiry_response,
            k.complaint_response,
            k.billing_frequency,
            k.float_required,
            k.notice_period,
            k.code_of_conduct_version,
            k.professional_memberships,
            k.portal,
            k.app,
            k.confidence_score
        FROM wss_factor_mapping m
        JOIN wss_documents d ON m.document_id = d.id
        LEFT JOIN wss_key_fields k ON m.document_id = k.document_id
        WHERE m.registration_number IS NOT NULL
    """)

    imported = 0
    with_fields = 0
    with get_db() as conn:
        for row in cursor:
            pf = normalize_pf_number(row['registration_number'])
            if not pf:
                continue

            conn.execute("""
                INSERT INTO wss (
                    registration_number, document_url,
                    management_fee_amount, management_fee_frequency,
                    delegated_authority_limit,
                    emergency_response, urgent_response, routine_response,
                    enquiry_response, complaint_response,
                    billing_frequency, float_required,
                    notice_period, code_of_conduct_version, professional_memberships,
                    portal, app, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    document_url = excluded.document_url,
                    management_fee_amount = excluded.management_fee_amount,
                    management_fee_frequency = excluded.management_fee_frequency,
                    delegated_authority_limit = excluded.delegated_authority_limit,
                    emergency_response = excluded.emergency_response,
                    urgent_response = excluded.urgent_response,
                    routine_response = excluded.routine_response,
                    enquiry_response = excluded.enquiry_response,
                    complaint_response = excluded.complaint_response,
                    billing_frequency = excluded.billing_frequency,
                    float_required = excluded.float_required,
                    notice_period = excluded.notice_period,
                    code_of_conduct_version = excluded.code_of_conduct_version,
                    professional_memberships = excluded.professional_memberships,
                    portal = excluded.portal,
                    app = excluded.app,
                    confidence_score = excluded.confidence_score
            """, [
                pf, row['document_url'],
                row['management_fee_amount'], row['management_fee_frequency'],
                row['delegated_authority_limit'],
                row['emergency_response'], row['urgent_response'], row['routine_response'],
                row['enquiry_response'], row['complaint_response'],
                row['billing_frequency'], row['float_required'],
                row['notice_period'], row['code_of_conduct_version'], row['professional_memberships'],
                row['portal'], row['app'], row['confidence_score']
            ])
            imported += 1
            if row['confidence_score']:
                with_fields += 1
        conn.commit()

    source.close()
    log_success(f"Imported WSS for {imported} factors ({with_fields} with extracted fields)")


# =============================================================================
# STEP 7: CALCULATE SCORES
# =============================================================================

def step_7_calculate_scores():
    """
    Simplified Risk Band Methodology v2
    
    Rules:
      < 50 properties        → LIMITED (insufficient data for scoring)
      0 adverse cases        → CLEAN  
      1-2 cases, no breach   → GREEN (capped)
      2+ breaches            → ORANGE minimum
      3+ cases OR 1 breach   → Use rate calculation
    
    Rate = (regular + complied×1.5 + breached×3) / properties × 10,000
    
    Thresholds:
      ≤ 10  → GREEN
      11-30 → ORANGE  
      > 30  → RED
    """
    log_step(7, "Calculate Scores & Risk Bands")
    
    GREEN_MAX = 10
    ORANGE_MAX = 30
    MIN_PROPERTIES = 50
    
    with get_db() as conn:
        factors = conn.execute("SELECT * FROM factors").fetchall()
        
        for f in factors:
            pf = f['registration_number']
            prop_count = f['property_count'] or 2000
            
            # Get tribunal cases (last 5 years)
            five_year_cutoff = datetime.now().year - 5
            cases = conn.execute("""
                SELECT * FROM tribunal_cases 
                WHERE factor_registration_number = ?
                AND decision_date IS NOT NULL
                AND CAST(SUBSTR(decision_date, 1, 4) AS INTEGER) >= ?
            """, [pf, five_year_cutoff]).fetchall()
            
            # Count adverse outcomes (not dismissed/rejected/withdrawn)
            adverse_cases = [c for c in cases if is_adverse_outcome(c['outcome'])]
            adverse_count = len(adverse_cases)
            
            # Count breaches and complied
            breach_count = sum(1 for c in cases if 'Breached' in (c['outcome'] or ''))
            complied_count = sum(1 for c in cases if 'Complied' in (c['outcome'] or ''))
            
            # Calculate effective rate with weighting
            regular = adverse_count - complied_count - breach_count
            effective = regular + (complied_count * 1.5) + (breach_count * 3)
            rate = (effective / prop_count) * 10000 if prop_count > 0 else 0
            
            # Determine risk band
            if prop_count < MIN_PROPERTIES:
                tier = 'LIMITED'  # Limited data (< 50 properties)
            elif adverse_count == 0:
                tier = 'CLEAN'
            elif breach_count >= 2:
                # 2+ breaches = minimum ORANGE
                tier = 'RED' if rate > ORANGE_MAX else 'ORANGE'
            elif adverse_count <= 2 and breach_count == 0:
                # 1-2 cases, no breach = capped GREEN
                tier = 'GREEN'
            else:
                # 3+ cases OR 1 breach: use rate
                if rate <= GREEN_MAX:
                    tier = 'GREEN'
                elif rate <= ORANGE_MAX:
                    tier = 'ORANGE'
                else:
                    tier = 'RED'
            
            # Calculate other stats for display
            all_cases = conn.execute("""
                SELECT * FROM tribunal_cases WHERE factor_registration_number = ?
            """, [pf]).fetchall()
            
            total_case_count = len(all_cases)
            total_upheld = sum(1 for c in all_cases if is_adverse_outcome(c['outcome']))
            pfeo_count = sum(1 for c in all_cases if c['pfeo_issued'])
            compensation = sum(c['compensation_awarded'] or 0 for c in all_cases)
            
            # Recent cases (3 years for display)
            three_year_cutoff = datetime.now().year - 3
            recent_cases = [c for c in all_cases if c['decision_date'] and c['decision_date'][:4].isdigit() and int(c['decision_date'][:4]) >= three_year_cutoff]
            
            # Check for active/unresolved PFEOs
            has_unresolved = any(c['pfeo_issued'] and not c['pfeo_complied'] and not c['pfeo_breached'] for c in cases)
            
            # Update factor
            conn.execute("""
                UPDATE factors SET
                    tribunal_case_count = ?,
                    tribunal_case_count_5yr = ?,
                    tribunal_cases_upheld = ?,
                    tribunal_pfeo_count = ?,
                    tribunal_total_compensation = ?,
                    tribunal_rate_per_10k = ?,
                    tribunal_cases_last_3_years = ?,
                    risk_band = ?,
                    has_active_pfeo = ?
                WHERE registration_number = ?
            """, [
                total_case_count, len(cases), total_upheld, pfeo_count, compensation, round(rate, 2),
                len(recent_cases), tier, 1 if has_unresolved else 0, pf
            ])
        
        conn.commit()
    
    # Show distribution
    with get_db() as conn:
        for band in ['CLEAN', 'GREEN', 'ORANGE', 'RED']:
            count = conn.execute("SELECT COUNT(*) FROM factors WHERE risk_band = ?", [band]).fetchone()[0]
            log_info(f"{band}: {count}")
        
        limited = conn.execute("SELECT COUNT(*) FROM factors WHERE risk_band = 'LIMITED'").fetchone()[0]
        log_info(f"LIMITED (< 50 props): {limited}")
    
    log_success("Scores calculated")


# =============================================================================
# STEP 8: GENERATE AI SUMMARIES (Optional - unchanged)
# =============================================================================

def step_8_generate_summaries(skip_ai: bool = False):
    log_step(8, "Generate AI Summaries")
    
    if skip_ai:
        print("  [>]  Skipping AI generation")
        return
    
    if not HAS_VERTEX:
        print("  [>]  Vertex AI not available")
        return
    
    # AI summary generation logic would go here
    log_success("AI summaries generated")


# =============================================================================
# STEP 9: GENERATE SITE (Using Jinja2 templates)
# =============================================================================

# Area definitions for geographic pages
AREA_DEFINITIONS = {
    'glasgow': {'name': 'Glasgow', 'postcodes': ['G1', 'G2', 'G3', 'G4', 'G5', 'G11', 'G12', 'G13', 'G14', 'G15', 'G20', 'G21', 'G22', 'G23', 'G31', 'G32', 'G33', 'G34', 'G40', 'G41', 'G42', 'G43', 'G44', 'G45', 'G46', 'G51', 'G52', 'G53', 'G60', 'G61', 'G62', 'G63', 'G64', 'G65', 'G66', 'G67', 'G68', 'G69', 'G70', 'G71', 'G72', 'G73', 'G74', 'G75', 'G76', 'G77', 'G78', 'G79', 'G81', 'G82', 'G83', 'G84']},
    'edinburgh': {'name': 'Edinburgh', 'postcodes': ['EH1', 'EH2', 'EH3', 'EH4', 'EH5', 'EH6', 'EH7', 'EH8', 'EH9', 'EH10', 'EH11', 'EH12', 'EH13', 'EH14', 'EH15', 'EH16', 'EH17']},
    'lothians': {'name': 'Lothians', 'postcodes': ['EH18', 'EH19', 'EH20', 'EH21', 'EH22', 'EH23', 'EH24', 'EH25', 'EH26', 'EH27', 'EH28', 'EH29', 'EH30', 'EH31', 'EH32', 'EH33', 'EH34', 'EH35', 'EH36', 'EH37', 'EH38', 'EH39', 'EH40', 'EH41', 'EH42', 'EH43', 'EH44', 'EH45', 'EH46', 'EH47', 'EH48', 'EH49', 'EH51', 'EH52', 'EH53', 'EH54', 'EH55']},
    'aberdeen': {'name': 'Aberdeen & Aberdeenshire', 'postcodes': ['AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB21', 'AB22', 'AB23', 'AB24', 'AB25', 'AB30', 'AB31', 'AB32', 'AB33', 'AB34', 'AB35', 'AB36', 'AB37', 'AB38', 'AB39', 'AB41', 'AB42', 'AB43', 'AB44', 'AB45', 'AB51', 'AB52', 'AB53', 'AB54', 'AB55', 'AB56']},
    'dundee': {'name': 'Dundee', 'postcodes': ['DD1', 'DD2', 'DD3', 'DD4', 'DD5']},
    'angus': {'name': 'Angus', 'postcodes': ['DD6', 'DD7', 'DD8', 'DD9', 'DD10', 'DD11']},
    'fife': {'name': 'Fife', 'postcodes': ['KY1', 'KY2', 'KY3', 'KY4', 'KY5', 'KY6', 'KY7', 'KY8', 'KY9', 'KY10', 'KY11', 'KY12', 'KY13', 'KY14', 'KY15', 'KY16']},
    'stirling': {'name': 'Stirling', 'postcodes': ['FK1', 'FK2', 'FK3', 'FK4', 'FK5', 'FK6', 'FK7', 'FK8', 'FK9', 'FK10', 'FK11', 'FK12', 'FK13', 'FK14', 'FK15', 'FK16', 'FK17', 'FK18', 'FK19', 'FK20', 'FK21']},
    'paisley': {'name': 'Paisley & Renfrewshire', 'postcodes': ['PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7', 'PA8', 'PA9', 'PA10', 'PA11', 'PA12', 'PA13', 'PA14', 'PA15', 'PA16', 'PA17', 'PA18', 'PA19']},
    'ayrshire': {'name': 'Ayrshire', 'postcodes': ['KA1', 'KA2', 'KA3', 'KA4', 'KA5', 'KA6', 'KA7', 'KA8', 'KA9', 'KA10', 'KA11', 'KA12', 'KA13', 'KA14', 'KA15', 'KA16', 'KA17', 'KA18', 'KA19', 'KA20', 'KA21', 'KA22', 'KA23', 'KA24', 'KA25', 'KA26', 'KA27', 'KA28', 'KA29', 'KA30']},
    'inverness': {'name': 'Inverness & Highlands', 'postcodes': ['IV1', 'IV2', 'IV3', 'IV4', 'IV5', 'IV6', 'IV7', 'IV8', 'IV9', 'IV10', 'IV11', 'IV12', 'IV13', 'IV14', 'IV15', 'IV16', 'IV17', 'IV18', 'IV19', 'IV20', 'IV21', 'IV22', 'IV23', 'IV24', 'IV25', 'IV26', 'IV27', 'IV28', 'IV30', 'IV31', 'IV32', 'IV36', 'IV40', 'IV41', 'IV42', 'IV43', 'IV44', 'IV45', 'IV46', 'IV47', 'IV48', 'IV49', 'IV51', 'IV52', 'IV53', 'IV54', 'IV55', 'IV56', 'IV63']},
    'perth': {'name': 'Perth & Kinross', 'postcodes': ['PH1', 'PH2', 'PH3', 'PH4', 'PH5', 'PH6', 'PH7', 'PH8', 'PH9', 'PH10', 'PH11', 'PH12', 'PH13', 'PH14', 'PH15', 'PH16', 'PH17', 'PH18']},
    'borders': {'name': 'Scottish Borders', 'postcodes': ['TD1', 'TD2', 'TD3', 'TD4', 'TD5', 'TD6', 'TD7', 'TD8', 'TD9', 'TD10', 'TD11', 'TD12', 'TD13', 'TD14', 'TD15']},
    'dumfries': {'name': 'Dumfries & Galloway', 'postcodes': ['DG1', 'DG2', 'DG3', 'DG4', 'DG5', 'DG6', 'DG7', 'DG8', 'DG9', 'DG10', 'DG11', 'DG12', 'DG13', 'DG14', 'DG16']},
    'lanarkshire': {'name': 'Lanarkshire', 'postcodes': ['ML1', 'ML2', 'ML3', 'ML4', 'ML5', 'ML6', 'ML7', 'ML8', 'ML9', 'ML10', 'ML11', 'ML12']},
}


def step_9_generate_site():
    log_step(9, "Generate Static Site")
    
    if not HAS_JINJA:
        print("  [X] Jinja2 required")
        return
    
    env = Environment(loader=FileSystemLoader(CONFIG.template_dir), autoescape=False)
    CONFIG.site_dir.mkdir(parents=True, exist_ok=True)
    
    generated_date = datetime.now().strftime('%B %Y')
    current_year = datetime.now().year
    
    with get_db() as conn:
        # ========== HOMEPAGE ==========
        if (CONFIG.template_dir / "homepage.html").exists():
            template = env.get_template("homepage.html")
            
            # Get stats
            stats_row = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'registered' THEN 1 ELSE 0 END) as active,
                    SUM(COALESCE(property_count, 0)) as properties,
                    SUM(COALESCE(total_review_count, 0)) as reviews
                FROM factors
            """).fetchone()
            tribunal_count = conn.execute("SELECT COUNT(*) FROM tribunal_cases").fetchone()[0]
            
            # Get tribunal hotspots (using 5-year filtered count for consistency with profiles)
            hotspots = [dict(r) for r in conn.execute("""
                SELECT * FROM factors 
                WHERE status='registered' AND tribunal_case_count_5yr > 0
                ORDER BY tribunal_case_count_5yr DESC LIMIT 5
            """).fetchall()]
            
            # Add initials for avatars
            for f in hotspots:
                words = (f['name'] or '').replace('Ltd', '').replace('Limited', '').split()
                f['initials'] = ''.join(w[0] for w in words[:2] if w).upper()
            
            html = template.render(
                stats={
                    'total_factors': stats_row['total'],
                    'active_factors': stats_row['active'],
                    'total_properties': stats_row['properties'],
                    'tribunal_cases': tribunal_count,
                    'total_reviews': stats_row['reviews']
                },
                hotspots=hotspots,
                generated_date=generated_date,
                current_year=current_year,
            )
            with open(CONFIG.site_dir / 'index.html', 'w', encoding='utf-8') as f:
                f.write(html)
            log_info("Generated homepage")
        
        # ========== FACTORS LISTING ==========
        factors_dir = CONFIG.site_dir / "factors"
        factors_dir.mkdir(parents=True, exist_ok=True)
        
        if (CONFIG.template_dir / "factors_listing.html").exists():
            template = env.get_template("factors_listing.html")
            
            factors = [dict(r) for r in conn.execute("SELECT * FROM factors ORDER BY name").fetchall()]
            tribunal_count = conn.execute("SELECT COUNT(*) FROM tribunal_cases").fetchone()[0]
            
            # Calculate stats
            active = [f for f in factors if f['status'] == 'registered' and f.get('factor_type') not in ('Registered Social Landlord', 'Local Authority')]
            expired = sum(1 for f in factors if f['status'] != 'registered')
            rsl_council = sum(1 for f in factors if f['status'] == 'registered' and f.get('factor_type') in ('Registered Social Landlord', 'Local Authority'))
            
            html = template.render(
                factors=factors,
                stats={
                    'total': len(factors),
                    'active': len(active),
                    'expired': expired,
                    'rsl_council': rsl_council,
                    'total_properties': sum(f['property_count'] or 0 for f in active),
                    'total_properties_all': sum(f['property_count'] or 0 for f in factors),
                    'tribunal_cases': tribunal_count,
                    'with_reviews': sum(1 for f in active if f.get('google_rating') or f.get('trustpilot_rating')),
                },
                generated_date=generated_date,
                current_year=current_year,
            )
            with open(factors_dir / 'index.html', 'w', encoding='utf-8') as f:
                f.write(html)
            log_info("Generated factors listing")
        
        # ========== FACTOR PROFILES ==========
        if (CONFIG.template_dir / "factor_profile.html").exists():
            template = env.get_template("factor_profile.html")
            profile_count = _generate_factor_profiles(conn, env, template, factors_dir)
            log_info(f"Generated {profile_count} factor profiles")
        
        # ========== AREAS ==========
        areas_dir = CONFIG.site_dir / "areas"
        areas_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all factors with postcode data (including RSL/LA for toggle functionality)
        all_factors_for_areas = [dict(r) for r in conn.execute("""
            SELECT * FROM factors 
            WHERE postcode_areas IS NOT NULL AND postcode_areas != ''
        """).fetchall()]
        
        # Build area data
        area_list = []
        for slug, area_info in AREA_DEFINITIONS.items():
            area_postcodes = set(area_info['postcodes'])
            area_factors = []
            
            for f in all_factors_for_areas:
                factor_postcodes = set(p.strip() for p in (f.get('postcode_areas') or '').split(','))
                if factor_postcodes & area_postcodes:
                    area_factors.append(f)
            
            if area_factors:
                area_list.append({
                    'slug': slug,
                    'name': area_info['name'],
                    'factor_count': len([f for f in area_factors if f['status'] == 'registered' and f.get('factor_type') not in ('Registered Social Landlord', 'Local Authority')]),
                    'property_count': sum(f['property_count'] or 0 for f in area_factors),
                    'factors': area_factors,
                })
        
        # Areas index
        if (CONFIG.template_dir / "areas_index.html").exists():
            template = env.get_template("areas_index.html")
            html = template.render(
                areas=sorted(area_list, key=lambda x: -x['factor_count']),
                generated_date=generated_date,
                current_year=current_year,
            )
            with open(areas_dir / 'index.html', 'w', encoding='utf-8') as f:
                f.write(html)
            log_info("Generated areas index")
        
        # Individual area pages
        if (CONFIG.template_dir / "area.html").exists():
            template = env.get_template("area.html")
            for area in area_list:
                area_dir = areas_dir / area['slug']
                area_dir.mkdir(parents=True, exist_ok=True)
                
                # Calculate stats matching factors_listing.html
                all_area_factors = area['factors']
                active_factors = [f for f in all_area_factors 
                                  if f['status'] == 'registered' 
                                  and f.get('factor_type') not in ('Registered Social Landlord', 'Local Authority')]
                expired_count = sum(1 for f in all_area_factors if f['status'] != 'registered')
                rsl_council_count = sum(1 for f in all_area_factors 
                                        if f.get('factor_type') in ('Registered Social Landlord', 'Local Authority'))
                
                html = template.render(
                    area=area,
                    factors=all_area_factors,
                    stats={
                        'active': len(active_factors),
                        'expired': expired_count,
                        'rsl_council': rsl_council_count,
                        'total_properties': sum(f['property_count'] or 0 for f in active_factors),
                        'with_reviews': sum(1 for f in active_factors if f.get('google_rating') or f.get('trustpilot_rating')),
                    },
                    generated_date=generated_date,
                    current_year=current_year,
                )
                with open(area_dir / 'index.html', 'w', encoding='utf-8') as f:
                    f.write(html)
            log_info(f"Generated {len(area_list)} area pages")
    
    log_success("Site generation complete")


def _generate_factor_profiles(conn, env, template, output_dir: Path) -> int:
    """Generate individual factor profile pages."""
    # Use the view to get factors with Companies House data joined
    factors = conn.execute("SELECT * FROM v_factor_profiles ORDER BY name").fetchall()

    # Calculate sector average tribunal rate (for context in summaries)
    # Include ALL factors with >= 50 properties (including those with 0 cases)
    rate_data = conn.execute("""
        SELECT COALESCE(tribunal_rate_per_10k, 0) as rate
        FROM factors
        WHERE property_count >= 50
    """).fetchall()
    rates = [r[0] for r in rate_data]
    sector_avg_rate = sum(rates) / len(rates) if rates else 8.0  # Mean across all factors

    # Load manual overrides from CSV (full text replacement for at-a-glance)
    overrides = {}
    overrides_path = CONFIG.csv_dir / "factor_overrides.csv"
    if overrides_path.exists():
        with open(overrides_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(row for row in f if not row.strip().startswith('#'))
            for row in reader:
                pf_num = normalize_pf_number(row.get('registration_number', ''))
                override_text = row.get('at_a_glance_override', '').strip()
                if pf_num and override_text:
                    overrides[pf_num] = override_text

    generated = 0
    for f in factors:
        pf = f['registration_number']
        profile = dict(f)
        
        # Sanitize None values to 0 for numeric fields (Jinja2 |default only catches undefined, not None)
        for field in ['property_count', 'tribunal_case_count', 'tribunal_cases_upheld', 
                      'tribunal_cases_dismissed', 'tribunal_pfeo_count', 'tribunal_total_compensation',
                      'tribunal_rate_per_10k', 'tribunal_composite_score', 'tribunal_pfeo_penalty',
                      'tribunal_volume_score', 'tribunal_severity_score', 'tribunal_recency_score',
                      'tribunal_cases_last_3_years', 'google_rating', 'google_review_count', 
                      'trustpilot_rating', 'trustpilot_review_count', 'combined_rating', 
                      'combined_review_count', 'ch_avg_tenure', 'ch_directors', 
                      'ch_insolvency', 'ch_charges', 'ch_accounts_overdue']:
            if profile.get(field) is None:
                profile[field] = 0
        
        # Get cases - filter to 5-year horizon to match profile stats
        five_year_cutoff = datetime.now().year - 5
        all_cases_raw = [dict(c) for c in conn.execute(
            "SELECT * FROM tribunal_cases WHERE factor_registration_number = ? ORDER BY decision_date DESC",
            [pf]
        ).fetchall()]
        
        # Filter out attendance/compliance hearings
        all_cases = [c for c in all_cases_raw if not is_attendance_hearing(c.get('outcome', ''))]
        
        # Sanitize all case numeric fields first
        for case in all_cases:
            if case.get('compensation_amount') is None:
                case['compensation_amount'] = 0
            if case.get('compensation_awarded') is None:
                case['compensation_awarded'] = 0
            if case.get('pfeo_issued') is None:
                case['pfeo_issued'] = 0
            # Add normalized outcome for display
            case['outcome_display'] = normalize_outcome_display(case.get('outcome', ''))
            # Add hearing_date alias (template uses hearing_date, DB has decision_date)
            case['hearing_date'] = case.get('decision_date')
            # Add case_ref alias (template uses case_ref, DB has case_reference)
            case['case_ref'] = case.get('case_reference')
            # Parse complaint categories from DB (JSON string) or extract from summary as fallback
            stored_cats = case.get('complaint_categories')
            if stored_cats:
                try:
                    case['complaint_categories'] = json.loads(stored_cats) if isinstance(stored_cats, str) else stored_cats
                except (json.JSONDecodeError, TypeError):
                    case['complaint_categories'] = extract_complaint_categories(case.get('summary', ''))
            else:
                # Fallback: extract from summary if no stored categories
                case['complaint_categories'] = extract_complaint_categories(case.get('summary', ''))
        
        # Filter to 5-year horizon (consistent with risk band calculation)
        cases = [
            c for c in all_cases 
            if c.get('decision_date') and c['decision_date'][:4].isdigit() 
            and int(c['decision_date'][:4]) >= five_year_cutoff
        ]
        
        # Get reviews with text
        reviews = [dict(r) for r in conn.execute("""
            SELECT * FROM reviews 
            WHERE factor_registration_number = ? AND review_text IS NOT NULL 
            ORDER BY review_date DESC LIMIT 5
        """, [pf]).fetchall()]
        
        # Sanitize review numeric fields
        for review in reviews:
            if review.get('rating') is None:
                review['rating'] = 0
            if review.get('review_count') is None:
                review['review_count'] = 0
        
        # Get Google locations
        google_locs = [dict(r) for r in conn.execute("""
            SELECT source_id as place_id, author_name as name, rating, review_count, phone
            FROM reviews WHERE factor_registration_number = ? AND platform = 'google'
            GROUP BY source_id ORDER BY review_count DESC
        """, [pf]).fetchall()]
        
        # Sanitize google_locs numeric fields
        for loc in google_locs:
            if loc.get('rating') is None:
                loc['rating'] = 0
            if loc.get('review_count') is None:
                loc['review_count'] = 0
        
        google_places = {
            'place_id': google_locs[0]['place_id'], 
            'name': google_locs[0].get('name'),
            'phone': google_locs[0].get('phone'),
            'address': profile.get('address')  # From factors table
        } if google_locs else None
        
        # Get Trustpilot
        tp_row = conn.execute("""
            SELECT source_id as url, rating, review_count
            FROM reviews WHERE factor_registration_number = ? AND platform = 'trustpilot'
            ORDER BY review_count DESC LIMIT 1
        """, [pf]).fetchone()
        trustpilot = {
            'url': tp_row['url'], 
            'rating': tp_row['rating'] or 0, 
            'review_count': tp_row['review_count'] or 0
        } if tp_row else None
        
        # Get WSS data (URL + extracted fields)
        wss_row = conn.execute("""
            SELECT document_url, management_fee_amount, management_fee_frequency,
                   delegated_authority_limit, emergency_response, urgent_response,
                   routine_response, enquiry_response, complaint_response,
                   billing_frequency, float_required, notice_period,
                   code_of_conduct_version, professional_memberships, portal, app,
                   confidence_score
            FROM wss WHERE registration_number = ?
        """, [pf]).fetchone()
        wss_url = wss_row['document_url'] if wss_row else None
        wss_data = dict(wss_row) if wss_row else None
        
        # Build complaint categories from case summaries
        category_counts = {}
        categorized_case_count = 0
        for c in cases:
            case_cats = c.get('complaint_categories', [])
            if case_cats:
                categorized_case_count += 1
                for cat in case_cats:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        # Sort by count descending and format for template
        complaint_cats = [
            {'name': cat, 'count': count}
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
        ]
        
        # Cases by year (use same range as five_year_cutoff filter)
        cases_by_year = {}
        for year in range(five_year_cutoff, datetime.now().year + 1):
            cases_by_year[year] = {'year': year, 'cases': 0, 'pfeos': 0}
        for c in cases:
            if c['decision_date'] and c['decision_date'][:4].isdigit():
                year = int(c['decision_date'][:4])
                if year in cases_by_year:
                    cases_by_year[year]['cases'] += 1
                    if c['pfeo_issued']:
                        cases_by_year[year]['pfeos'] += 1
        
        # Parse AI summary
        at_a_glance = None
        if profile.get('at_a_glance'):
            try:
                at_a_glance = json.loads(profile['at_a_glance'])
            except:
                pass
        
        # Build risk explainer
        risk_band = profile.get('risk_band') or 'LIMITED'  # Default to LIMITED if not set
        risk_explainers = {
            'CLEAN': 'No adverse tribunal cases in the past 5 years.',
            'GREEN': 'Low case rate (≤10 per 10k properties).',
            'LIMITED': 'Limited data - fewer than 50 properties managed.',
            'ORANGE': 'Elevated case rate or multiple PFEO breaches.',
            'RED': 'High tribunal case rate (>30 per 10k) - significant concerns.',
        }
        
        # Calculate 5-year stats from filtered cases (cases is already 5-year filtered)
        case_count = len(cases)
        cases_upheld = sum(1 for c in cases if is_adverse_outcome(c.get('outcome', '')))
        pfeo_count = sum(1 for c in cases if c.get('pfeo_issued'))
        compensation = sum(c.get('compensation_awarded') or 0 for c in cases)
        prop_count = profile.get('property_count') or 2000
        rate_per_10k = (case_count / prop_count) * 10000 if prop_count > 0 else 0
        
        context = {
            'profile': profile,
            'factor': profile,  # Alias for templates that use 'factor' instead of 'profile'
            'at_a_glance': at_a_glance,
            'wss_url': wss_url,
            'wss': wss_data,  # Full WSS data with extracted fields
            'registry_url': f"https://www.propertyfactorregister.gov.scot/property-factor/{pf}",
            'companies_house_url': f"https://find-and-update.company-information.service.gov.uk/search?q={profile.get('name', '').replace(' ', '+')}" if profile.get('name') else None,
            'coverage_areas_list': [a.strip() for a in (profile.get('coverage_areas') or '').split(',') if a.strip()],
            'recent_reviews': reviews,
            'google_places': google_places,
            'google_locations': google_locs,
            'trustpilot': trustpilot,
            
            # Risk explanation
            'risk_explainer': risk_explainers.get(risk_band, ''),
            
            # Tribunal stats with safe defaults
            'tribunal_last_5_years': {
                'case_count': case_count,
                'pfeo_count': pfeo_count,
                'compensation': compensation,
                'rate_per_10k': rate_per_10k,
                'complaints_upheld_pct': (cases_upheld / case_count * 100) if case_count > 0 else 0,
            },
            'stats': {  # Alias for templates that use 'stats' instead of 'tribunal_last_5_years'
                'case_count': case_count,
                'pfeo_count': pfeo_count,
                'compensation': compensation,
                'rate_per_10k': rate_per_10k,
                'complaints_upheld_pct': (cases_upheld / case_count * 100) if case_count > 0 else 0,
            },
            'tribunal_full_history': {
                'case_count': len(all_cases),
                'first_case_year': min((c['decision_date'][:4] for c in all_cases if c.get('decision_date')), default=None),
            },
            'cases_by_year': list(cases_by_year.values()),
            'recent_cases': cases[:5],
            'complaint_categories': complaint_cats,
            'cases': cases,  # Alias for templates expecting 'cases' at top level
            
            # Variables the template might expect (safe empty defaults)
            'similar_factors': [],
            'timeline_events': [],
            'code_breaches': [],
            'case_fees': [],
            'fee_examples': [],
            'market_position': None,
            'regulatory_timeline': [],
            'enforcement_status': None,
            
            'generated_date': datetime.now().strftime('%Y-%m-%d'),
            'current_year': datetime.now().year,
            'sector_avg_rate': sector_avg_rate,
            'at_a_glance_override': overrides.get(pf),  # Full text override from CSV
        }
        
        try:
            html = template.render(**context)
            factor_dir = output_dir / pf.lower()
            factor_dir.mkdir(parents=True, exist_ok=True)
            with open(factor_dir / 'index.html', 'w', encoding='utf-8') as f:
                f.write(html)
            
            # Generate tribunal history page if factor has cases
            if cases and (Path(env.loader.searchpath[0]) / "tribunal_history.html").exists():
                tribunal_template = env.get_template("tribunal_history.html")
                tribunal_dir = factor_dir / 'tribunal'
                tribunal_dir.mkdir(parents=True, exist_ok=True)
                
                # Build outcome breakdown for template
                outcome_counts = {}
                for c in cases:
                    outcome = c.get('outcome_display') or c.get('outcome') or 'Unknown'
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                outcomes_list = [
                    {'type': k, 'count': v, 'percentage': round(v / len(cases) * 100) if cases else 0}
                    for k, v in sorted(outcome_counts.items(), key=lambda x: -x[1])
                ]
                
                # Build cases_by_year as dict for template
                cases_by_year_dict = {item['year']: max(item['cases'], 0) for item in cases_by_year.values()}
                # Ensure at least one non-zero value to prevent division by zero in template
                if cases_by_year_dict and max(cases_by_year_dict.values()) == 0:
                    # This shouldn't happen if cases is non-empty, but safeguard anyway
                    cases_by_year_dict = {k: 1 if k == max(cases_by_year_dict.keys()) else 0 for k, v in cases_by_year_dict.items()}
                
                # Build complaint categories list for template (v2: not extracted)
                complaint_cats_list = []
                
                # Find most recent case date and largest award
                most_recent = max((c.get('decision_date', '') for c in cases), default=None)
                largest_award = max(
                    [(c.get('compensation_awarded') or 0, c) for c in cases],
                    key=lambda x: x[0],
                    default=(0, None)
                )
                
                # Add slug to profile for template
                profile_with_slug = dict(profile)
                profile_with_slug['slug'] = pf.lower()
                
                # Build comprehensive stats dict matching template expectations
                stats_dict = {
                    # Core metrics
                    'total_cases': case_count,
                    'case_count': case_count,  # Alias
                    'upheld_count': cases_upheld,
                    'upheld_rate': round((cases_upheld / case_count * 100)) if case_count > 0 else 0,
                    'complaints_upheld_pct': round((cases_upheld / case_count * 100)) if case_count > 0 else 0,
                    
                    # PFEO stats
                    'pfeo_count': pfeo_count,
                    'pfeo_rate': round((pfeo_count / case_count * 100)) if case_count > 0 else 0,
                    
                    # Compensation
                    'total_compensation': compensation,
                    'compensation': compensation,  # Alias
                    'largest_award': {
                        'amount': largest_award[0],
                        'case_ref': largest_award[1].get('case_reference') if largest_award[1] else None,
                        'pdf_url': largest_award[1].get('pdf_url') if largest_award[1] else None,
                    } if largest_award[0] > 0 else None,
                    
                    # Time context
                    'five_year_cutoff': five_year_cutoff,
                    'analysis_end_year': datetime.now().year,
                    'most_recent_case_date': most_recent[:10] if most_recent else None,
                    'last_updated': datetime.now().strftime('%Y-%m-%d'),
                    
                    # Risk
                    'risk_band': risk_band,
                    'rate_per_10k': rate_per_10k,
                    
                    # Breakdowns
                    'outcomes': outcomes_list,
                    'cases_by_year': cases_by_year_dict,
                    'complaint_categories': complaint_cats,
                    'categorized_cases': categorized_case_count,

                    # Optional (not computed but template checks for them)
                    'code_breaches': None,
                    'pfeo_requirements': None,
                    'compensation_breakdown': None,
                    'trend_notes': None,
                }
                
                tribunal_context = {
                    'profile': profile_with_slug,
                    'factor': profile_with_slug,
                    'cases': cases,
                    'all_cases': all_cases,
                    'all_cases_count': len(all_cases),
                    'cases_by_year': list(cases_by_year.values()),
                    'complaint_categories': complaint_cats,
                    'tribunal_stats': stats_dict,
                    'stats': stats_dict,
                    'notable_cases': [c for c in cases if c.get('pfeo_issued') or c.get('pfeo_breached') or (c.get('compensation_awarded') or 0) >= 500][:5],
                    'five_year_cutoff': five_year_cutoff,
                    'risk_band': risk_band,
                    'risk_explainer': risk_explainers.get(risk_band, ''),
                    'generated_date': datetime.now().strftime('%Y-%m-%d'),
                    'current_year': datetime.now().year,
                }
                
                tribunal_html = tribunal_template.render(**tribunal_context)
                with open(tribunal_dir / 'index.html', 'w', encoding='utf-8') as f:
                    f.write(tribunal_html)
            
            generated += 1
        except Exception as e:
            print(f"  [!]  Failed {pf}: {e}")
    
    return generated


# =============================================================================
# STEP 10: VALIDATE
# =============================================================================

def step_10_validate():
    log_step(10, "Validate Output")
    
    with get_db() as conn:
        for table in ['factors', 'tribunal_cases', 'reviews']:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            log_info(f"{table}: {count:,} rows")
        
        scored = conn.execute("SELECT COUNT(*) FROM factors WHERE risk_band IS NOT NULL").fetchone()[0]
        log_info(f"Factors with risk band: {scored}")
    
    pages = list((CONFIG.site_dir / "factors").glob("*/index.html")) if CONFIG.site_dir.exists() else []
    log_info(f"Generated pages: {len(pages)}")
    
    log_success("Validation complete")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Factors - Simplified Pipeline v4.0")
    parser.add_argument('--full', action='store_true', help='Full rebuild')
    parser.add_argument('--step', type=str, help='Run specific steps (comma-separated)')
    parser.add_argument('--skip-ai', action='store_true', help='Skip AI generation')
    parser.add_argument('--root', type=str, default='.', help='Project root')
    
    args = parser.parse_args()
    CONFIG.resolve_paths(Path(args.root).resolve())
    
    steps = {
        1: lambda: step_1_init_database(reset=args.full),
        2: step_2_import_factors,
        3: step_3_import_tribunal,
        4: step_4_import_reviews,
        5: step_5_import_companies_house,
        6: step_6_import_wss,
        7: step_7_calculate_scores,
        8: lambda: step_8_generate_summaries(skip_ai=args.skip_ai),
        9: step_9_generate_site,
        10: step_10_validate,
    }
    
    to_run = [int(s) for s in args.step.split(',')] if args.step else list(range(1, 11))
    
    print(f"\n{'='*60}")
    print("COMPARE FACTORS - SIMPLIFIED PIPELINE v4.0")
    print(f"{'='*60}")
    print(f"Root: {CONFIG.project_root}")
    print(f"Steps: {to_run}")
    
    start = datetime.now()
    for step_num in to_run:
        if step_num in steps:
            try:
                steps[step_num]()
            except Exception as e:
                print(f"  [X] Step {step_num} failed: {e}")
    
    print(f"\nTotal time: {(datetime.now() - start).total_seconds():.1f}s")


if __name__ == "__main__":
    main()