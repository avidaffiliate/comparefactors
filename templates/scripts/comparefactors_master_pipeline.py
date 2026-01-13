#!/usr/bin/env python3
"""
===============================================================================
COMPARE FACTORS SCOTLAND - MASTER PIPELINE v2.9
===============================================================================

A clean, reproducible pipeline that runs end-to-end without manual intervention.
Execute from your project root directory.

v2.9 CHANGES:
- Step 9 now generates factors directory index page (/factors/index.html)
  * Shows all 675 factors with client-side filtering
  * Checkboxes to include/exclude expired and RSL/Council factors
  * Search, risk filter, and sort options
  * Dynamic stats update based on filters
- Step 9 now generates area pages (/areas/{area}/ and /areas/index.html)
  * 16 areas defined: Glasgow, Edinburgh, Lothians, Aberdeen, Aberdeenshire,
    Dundee, Angus, Fife, Stirling, Paisley, Ayrshire, Inverness, Perth, 
    Borders, Dumfries, Lanarkshire
  * Factors matched by postcode_areas field
  * Each area page shows factors serving that area with filtering
  * Areas index shows all areas with factor counts
- Refined tier logic for unresolved PFEOs:
  * RULE 1a: Unresolved PFEO (5yr) + 3+ cases = RED
  * RULE 1b: Unresolved PFEO (5yr) + <3 cases = ORANGE (mitigated by low volume)
- This prevents factors with a single unresolved PFEO but otherwise clean records
  from being rated the same as factors with systemic tribunal problems
- Factors like Aberdeen City Council (1 case, unresolved PFEO) now get ORANGE not RED

v2.8 CHANGES:
- Step 8 At a Glance prompt updated to be factual, not editorial
- Step 8 At a Glance now uses rich context including:
  * Recent tribunal case summaries and outcomes
  * Top complaint categories from cases
  * Sample review text from Google and Trustpilot
  * WSS fee data (management, late, insurance)
  * Tribunal fee examples
  * Company information (incorporation date, status)
- Step 8 output now includes data_sources field for transparency
- Step 9 re-enabled tribunal history page generation (/factors/{pf}/tribunal/)
- Step 9 case_fees now grouped by type with 'details' array for expandable display
  (renamed from 'items' to avoid conflict with Python dict.items() method)
- Step 9 case_fees now has intelligent frequency defaults when not in source data:
  * Management/Factor fees → per year
  * Insurance fees → per year
  * Reserve/Sinking fund → per year
  * Late payment/Arrears → per incident
  * Admin/Transfer/Registration → one-time
- Step 9 now passes google_locations to template (list of Google Places per factor)
- Step 9 now passes google_places with primary location contact info
- Step 9 now passes trustpilot with URL for Trustpilot link
- Step 9 case_fees filtering with type-specific caps
- Step 9 removed invalid PDF URL fallback (template handles missing PDFs)
- Step 4 Google import now stores location name in author_name for aggregate records
- Template v5.18.2 support: expandable tribunal fees with frequency defaults

v2.7 CHANGES:
- Step 1 now migrates existing reviews table to remove UNIQUE constraint
- This allows importing individual reviews with text from google_reviews_text.csv
- Step 4 imports google_reviews_text.csv automatically (data/csv/google_reviews_text.csv)
- Step 9 no longer generates tribunal pages (links to official HPC PDFs instead)

v2.6 CHANGES:
- New tier determination logic with clearer rules:
  * RULE 1: Any unresolved PFEO (5yr) = RED (always wins)
  * RULE 2: 2+ PFEOs in 5yr = ORANGE max
  * RULE 3: < 3 cases in 5yr = CLEAN
  * RULE 4: Score-based for 3+ cases (65+ GREEN, 40+ AMBER, 20+ ORANGE, else RED)
- Score scale changed from 0-10 to 0-100 for more granularity
- _determine_tier() now examines individual cases for PFEO resolution status
- Step 4 now imports google_reviews_text.csv with individual review text
- Step 7 refactored to fetch individual cases per factor (not just aggregates)
- Step 4 now imports google_reviews_text.csv with individual review text
- Reviews table schema updated to allow multiple reviews per Google Place
- Tribunal page generation disabled (linking to official PDFs instead)

v2.5 CHANGES:
- Fixed reviews query: removed 'review_text IS NOT NULL' filter that was excluding 
  all aggregate review records (rating + count data without individual text)
- Reviews now ordered by review_count DESC to show most significant data first
- Template updated to display aggregate review data when no individual text exists

v2.4 CHANGES:
- Fixed property count fallback from 1 to 2000 for factors with unknown size
- This prevents massively inflated case rates for councils and factors without FOI data
- Added property_count_estimated flag to score output

v2.3 CHANGES:
- Fixed registration_status normalization (now 'registered' or 'expired')
- Added factor_type auto-detection from name patterns
- Added 'status' column for template compatibility
- Added has_active_pfeo flag
- Fixed review snippet data structure for template
- Fixed cases_by_year population for trend charts
- Fixed complaint_categories dict population

v2.2 CHANGES:
- Recency score now uses rate per 10k (like volume) instead of absolute count
- This fixes unfair penalization of large factors with many properties
- Added recent_per_10k to score output

v2.1 CHANGES:
- Composite tribunal scoring with Volume (40%), Severity (35%), Recency (25%)
- PFEO penalty system with automatic tier caps
- New database columns for score components
- Enhanced risk band methodology

USAGE:
    # Full rebuild (drops existing data)
    python comparefactors_master_pipeline.py --full

    # Incremental update (preserves existing, adds new)
    python comparefactors_master_pipeline.py --update

    # Specific steps only
    python comparefactors_master_pipeline.py --step 1,2,3
    python comparefactors_master_pipeline.py --step 8  # Just summaries

    # Dry run (shows what would happen)
    python comparefactors_master_pipeline.py --dry-run
    
    # Skip AI generation (faster rebuild)
    python comparefactors_master_pipeline.py --full --skip-ai

DIRECTORY STRUCTURE (expected):
    comparefactors/
    ├── data/
    │   ├── csv/                        # Input CSVs
    │   │   ├── factors_register.csv    # From Scottish Gov registry
    │   │   ├── factors_postcodes.csv   # FOI postcode coverage
    │   │   ├── tribunal_cases.csv      # Matched tribunal cases
    │   │   ├── google_reviews.csv      # Google Places aggregate data
    │   │   ├── google_reviews_text.csv # Individual reviews with text
    │   │   ├── trustpilot_reviews.csv  # Trustpilot data
    │   │   └── companies_house.csv     # CH company data
    │   ├── tribunal/
    │   │   └── tribunal_enriched.db    # AI-extracted case details
    │   ├── wss/
    │   │   └── wss_extracted.db        # WSS fee structures
    │   └── database/
    │       └── comparefactors.db       # Consolidated output database
    ├── templates/
    │   ├── factor_profile.html         # Jinja2 profile template
    │   ├── factors_listing.html        # Listing page template
    │   ├── tribunal_history.html       # Factor tribunal analysis
    │   └── tribunal_case.html          # Individual case page
    ├── site/                           # Generated static site
    │   ├── factors/
    │   │   └── pf000xxx/index.html
    │   ├── tribunal/
    │   │   └── case-ref/index.html
    │   └── areas/
    │       └── edinburgh/index.html
    └── comparefactors_master_pipeline.py

STEPS:
    1. Initialize Database Schema
    2. Import Core Factor Data (registry + FOI postcodes)
    3. Import Tribunal Data (cases + AI extractions)
    4. Import Review Data (Google + Trustpilot)
    5. Import Companies House Data
    6. Import WSS Data (fee structures)
    7. Calculate Scores & Risk Bands (COMPOSITE METHODOLOGY v2.1)
    8. Generate AI Summaries (At a Glance + Tribunal Analysis)
    9. Generate Static Site
    10. Validate Output

===============================================================================
"""

import sqlite3
import csv
import json
import os
import sys
import argparse
import re
import shutil
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Try to import optional dependencies
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

# Vertex AI for summary generation
HAS_VERTEX = False
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    HAS_VERTEX = True
except ImportError:
    pass

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration."""
    # Paths (relative to project root)
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    csv_dir: Path = Path("data/csv")
    db_path: Path = Path("data/database/comparefactors.db")
    tribunal_db_path: Path = Path("data/tribunal/tribunal_enriched.db")
    wss_db_path: Path = Path("data/wss/wss_extracted.db")
    template_dir: Path = Path("templates")
    site_dir: Path = Path("site")
    
    # CSV file names (within csv_dir)
    factors_csv: str = "factors_register.csv"
    factors_postcodes_csv: str = "factors_postcodes.csv"
    tribunal_csv: str = "tribunal_cases.csv"
    tribunal_enriched_csv: str = "tribunal_enriched.csv"  # Fallback if no .db
    google_reviews_csv: str = "google_reviews.csv"
    google_reviews_text_csv: str = "google_reviews_text.csv"  # Individual reviews with text
    trustpilot_csv: str = "trustpilot_reviews.csv"
    companies_house_csv: str = "companies_house.csv"
    
    # Vertex AI settings
    gcp_project: str = os.getenv("GCP_PROJECT_ID", "scotland-factors-places")
    gcp_location: str = "us-central1"
    gemini_model: str = "gemini-2.0-flash-lite-001"
    
    # Composite scoring weights (v2.1)
    score_weight_volume: float = 0.40
    score_weight_severity: float = 0.35
    score_weight_recency: float = 0.25
    
    # PFEO penalty settings
    pfeo_penalty_per_order: float = 1.5
    pfeo_penalty_max: float = 4.0
    pfeo_auto_cap_threshold: int = 2  # 2+ PFEOs = capped at ORANGE
    
    # Recency window (years)
    recency_window_years: int = 3
    
    # Risk band thresholds (composite score 0-10)
    risk_thresholds: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        # Composite score to tier mapping (inverted: higher score = better)
        self.risk_thresholds = {
            'CLEAN': (8.0, 10.0),   # "No significant tribunal history"
            'GREEN': (6.0, 8.0),    # "Minor tribunal activity"
            'AMBER': (4.0, 6.0),    # "Some tribunal concerns"
            'ORANGE': (2.0, 4.0),   # "Significant tribunal issues"
            'RED': (0.0, 2.0),      # "Serious tribunal record"
        }
    
    def resolve_paths(self, root: Path):
        """Resolve all paths relative to project root."""
        self.project_root = root
        self.data_dir = root / self.data_dir
        self.csv_dir = root / self.csv_dir
        self.db_path = root / self.db_path
        self.tribunal_db_path = root / self.tribunal_db_path
        self.wss_db_path = root / self.wss_db_path
        self.template_dir = root / self.template_dir
        self.site_dir = root / self.site_dir


# Global config
CONFIG = Config()


# =============================================================================
# LOGGING & UTILITIES
# =============================================================================

class PipelineLogger:
    """Simple logger with step tracking."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_step = 0
        self.total_steps = 10
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def step(self, num: int, title: str):
        """Start a new step."""
        self.current_step = num
        print(f"\n{'='*70}")
        print(f"STEP {num}/{self.total_steps}: {title}")
        print(f"{'='*70}")
    
    def info(self, msg: str):
        if self.verbose:
            print(f"  ℹ️  {msg}")
    
    def success(self, msg: str):
        print(f"  ✅ {msg}")
    
    def warning(self, msg: str):
        self.warnings.append(f"Step {self.current_step}: {msg}")
        print(f"  ⚠️  {msg}")
    
    def error(self, msg: str):
        self.errors.append(f"Step {self.current_step}: {msg}")
        print(f"  ❌ {msg}")
    
    def skip(self, msg: str):
        print(f"  ⏭️  SKIP: {msg}")
    
    def progress(self, current: int, total: int, item: str = ""):
        if self.verbose and total > 0:
            pct = (current / total) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {current}/{total} {item[:30]}", end="", flush=True)
            if current == total:
                print()
    
    def summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        if self.errors:
            print(f"\n❌ {len(self.errors)} ERRORS:")
            for e in self.errors:
                print(f"   - {e}")
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNINGS:")
            for w in self.warnings:
                print(f"   - {w}")
        if not self.errors and not self.warnings:
            print("\n✅ All steps completed successfully!")


LOG = PipelineLogger()


# =============================================================================
# DATABASE HELPERS
# =============================================================================

@contextmanager
def get_db():
    """Context manager for database connections."""
    CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def normalize_pf_number(pf: Any) -> Optional[str]:
    """Normalize PF registration number to standard format."""
    if not pf:
        return None
    s = str(pf).strip().upper()
    match = re.search(r'(\d{6})', s)
    if match:
        return f"PF{match.group(1)}"
    return None


def parse_date(date_str: Any) -> Optional[str]:
    """Parse various date formats to ISO format."""
    if not date_str:
        return None
    
    date_str = str(date_str).strip()
    
    # Handle Unix timestamps
    if date_str.isdigit() and len(date_str) >= 10:
        try:
            return datetime.fromtimestamp(int(date_str)).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    
    # Handle standard date formats
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_registration_status(raw_status: str) -> str:
    """Normalize registration status to 'registered' or 'expired'."""
    if not raw_status:
        return 'registered'
    raw_lower = raw_status.lower().strip()
    if 'expired' in raw_lower:
        return 'expired'
    if 'active' in raw_lower or 'registered' in raw_lower:
        return 'registered'
    return 'registered'


def detect_factor_type(name: str, raw_type: str = None) -> str:
    """Detect factor type from name patterns."""
    if raw_type and raw_type.strip():
        return raw_type.strip()
    
    name_lower = name.lower()
    
    # RSL / Housing Association patterns
    if any(x in name_lower for x in ['housing association', 'housing society', 'housing co-op']):
        return 'Housing Association'
    if 'rsl' in name_lower or 'registered social landlord' in name_lower:
        return 'RSL'
    
    # Local authority patterns
    if any(x in name_lower for x in ['council', 'local authority', 'city of']):
        return 'Local Authority'
    
    # Company patterns
    if any(x in name_lower for x in [' ltd', ' limited', ' plc', ' llp']):
        return 'Commercial'
    
    # Sole proprietor patterns
    if 'trading as' in name_lower or 't/a' in name_lower:
        return 'Sole Proprietor'
    
    return 'Commercial'


def parse_int(val: Any) -> Optional[int]:
    """Safely parse integer."""
    if val is None or val == "":
        return None
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def parse_float(val: Any) -> Optional[float]:
    """Safely parse float."""
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_bool(val: Any) -> bool:
    """Parse boolean from various formats."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ('true', 'yes', '1', 'y', 't')


def find_csv_column(row: Dict, candidates: List[str], default: Any = None) -> Any:
    """Find a value from multiple possible column names."""
    for col in candidates:
        if col in row and row[col] is not None and row[col] != "":
            return row[col]
    return default


# =============================================================================
# SCHEMA DEFINITION (v2.1 with composite scoring)
# =============================================================================

SCHEMA_SQL = """
-- ============================================================================
-- COMPARE FACTORS SCOTLAND - DATABASE SCHEMA v2.1
-- With composite tribunal scoring columns
-- ============================================================================

-- Core factor table
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
    
    -- Property Factor Register data
    registration_date TEXT,
    registration_status TEXT DEFAULT 'registered',
    status TEXT DEFAULT 'registered',  -- Normalized: 'registered' or 'expired'
    property_count INTEGER DEFAULT 0,
    factor_type TEXT,  -- Commercial, RSL, Housing Association, Local Authority, Sole Proprietor
    
    -- Accreditation flags
    tpi_member INTEGER DEFAULT 0,
    
    -- FOI postcode coverage data
    postcode_areas TEXT,
    postcode_count INTEGER DEFAULT 0,
    geographic_reach TEXT,
    foi_cities TEXT,
    coverage_areas TEXT,  -- For template compatibility
    
    -- Tribunal aggregates (populated in Step 7)
    tribunal_case_count INTEGER DEFAULT 0,
    tribunal_cases_upheld INTEGER DEFAULT 0,
    tribunal_cases_dismissed INTEGER DEFAULT 0,
    tribunal_pfeo_count INTEGER DEFAULT 0,
    tribunal_total_compensation REAL DEFAULT 0,
    tribunal_rate_per_10k REAL,
    has_enforcement_order INTEGER DEFAULT 0,
    has_active_pfeo INTEGER DEFAULT 0,  -- Outstanding PFEO not yet complied with
    risk_band TEXT,
    
    -- COMPOSITE SCORE COMPONENTS (v2.1)
    tribunal_composite_score REAL,       -- Final 0-10 score
    tribunal_volume_score REAL,          -- Volume component (0-10)
    tribunal_severity_score REAL,        -- Severity component (0-10)
    tribunal_recency_score REAL,         -- Recency component (0-10)
    tribunal_avg_severity REAL,          -- Average case severity (1-10)
    tribunal_cases_last_3_years INTEGER DEFAULT 0,  -- Recent case count
    tribunal_pfeo_penalty REAL DEFAULT 0,           -- PFEO deduction applied
    
    -- Review aggregates
    google_rating REAL,
    google_review_count INTEGER DEFAULT 0,
    trustpilot_rating REAL,
    trustpilot_review_count INTEGER DEFAULT 0,
    combined_rating REAL,
    total_review_count INTEGER DEFAULT 0,
    
    -- AI-generated summaries (populated in Step 8)
    at_a_glance TEXT,              -- JSON: {bullets: [...], generated_at: ...}
    tribunal_analysis TEXT,        -- JSON: {patterns: [...], trend: ...}
    
    -- Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Tribunal cases (basic listing data)
CREATE TABLE IF NOT EXISTS tribunal_cases (
    case_reference TEXT PRIMARY KEY,
    factor_registration_number TEXT,
    decision_date TEXT,
    outcome TEXT,
    complaints_made INTEGER,
    complaints_upheld INTEGER,
    has_enforcement_order INTEGER DEFAULT 0,
    pdf_url TEXT,
    
    -- AI-extracted fields
    summary TEXT,
    key_quote TEXT,
    complaint_categories TEXT,
    severity_score REAL,
    compensation_awarded REAL,
    pfeo_issued INTEGER DEFAULT 0,
    pfeo_complied INTEGER,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_registration_number) REFERENCES factors(registration_number)
);

-- Reviews (aggregate data + individual reviews with text)
-- Note: No UNIQUE constraint to allow multiple individual reviews from same Google Place
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
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_registration_number) REFERENCES factors(registration_number)
);

-- Companies House data
CREATE TABLE IF NOT EXISTS companies (
    registration_number TEXT PRIMARY KEY,
    company_number TEXT,
    company_name TEXT,
    company_url TEXT,
    incorporated_date TEXT,
    company_status TEXT,
    company_type TEXT,
    jurisdiction TEXT,
    sic_codes TEXT,
    registered_address TEXT,
    
    -- Risk indicators
    has_insolvency_history INTEGER DEFAULT 0,
    has_charges INTEGER DEFAULT 0,
    registered_office_in_dispute INTEGER DEFAULT 0,
    undeliverable_address INTEGER DEFAULT 0,
    
    -- Corporate history
    previous_names_count INTEGER DEFAULT 0,
    previous_names TEXT,
    
    -- Directors
    director_count INTEGER DEFAULT 0,
    secretary_count INTEGER DEFAULT 0,
    avg_director_tenure_years REAL,
    min_director_tenure_years REAL,
    max_director_tenure_years REAL,
    directors TEXT,  -- JSON
    
    -- PSCs
    psc_count INTEGER DEFAULT 0,
    corporate_psc_count INTEGER DEFAULT 0,
    pscs TEXT,  -- JSON
    
    -- Accounts
    last_accounts_date TEXT,
    last_accounts_type TEXT,
    next_accounts_due TEXT,
    accounts_overdue INTEGER DEFAULT 0,
    accounts_days_overdue INTEGER,
    
    -- Confirmation statement
    last_confirmation_date TEXT,
    next_confirmation_due TEXT,
    confirmation_overdue INTEGER DEFAULT 0,
    
    -- Filing history
    total_filings INTEGER DEFAULT 0,
    filings_last_year INTEGER DEFAULT 0,
    late_filings_count INTEGER DEFAULT 0,
    last_filing_date TEXT,
    last_filing_type TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (registration_number) REFERENCES factors(registration_number)
);

-- WSS (Written Statement of Services) data
CREATE TABLE IF NOT EXISTS wss (
    registration_number TEXT PRIMARY KEY,
    document_url TEXT,
    document_year INTEGER,
    
    -- Fee structure
    management_fee_amount TEXT,
    management_fee_frequency TEXT,
    management_fee_vat TEXT,
    billing_frequency TEXT,
    late_penalty TEXT,
    float_required TEXT,
    sinking_fund TEXT,
    nopli INTEGER,  -- Notice of Potential Liability for Costs
    
    -- Insurance
    insurance_provider TEXT,
    insurance_broker TEXT,
    commission_disclosure TEXT,
    
    -- Response times
    emergency_response TEXT,
    urgent_response TEXT,
    routine_response TEXT,
    enquiry_response TEXT,
    complaint_response TEXT,
    
    -- Technology
    portal TEXT,
    app TEXT,
    
    -- Contract
    notice_period TEXT,
    
    -- Accreditation
    code_of_conduct_version TEXT,
    professional_memberships TEXT,
    
    confidence_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (registration_number) REFERENCES factors(registration_number)
);

-- Case fees extracted from tribunal cases
CREATE TABLE IF NOT EXISTS case_fees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_reference TEXT,
    factor_registration_number TEXT,
    fee_type TEXT,
    amount REAL,
    frequency TEXT,
    vat_included INTEGER,
    evidence TEXT,
    context TEXT,
    disputed INTEGER DEFAULT 0,
    tribunal_finding TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_reference) REFERENCES tribunal_cases(case_reference),
    FOREIGN KEY (factor_registration_number) REFERENCES factors(registration_number)
);

-- Create useful views
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

-- Create indices
CREATE INDEX IF NOT EXISTS idx_tribunal_factor ON tribunal_cases(factor_registration_number);
CREATE INDEX IF NOT EXISTS idx_tribunal_date ON tribunal_cases(decision_date);
CREATE INDEX IF NOT EXISTS idx_reviews_factor ON reviews(factor_registration_number);
CREATE INDEX IF NOT EXISTS idx_factors_risk ON factors(risk_band);
CREATE INDEX IF NOT EXISTS idx_factors_composite ON factors(tribunal_composite_score);
"""


# =============================================================================
# STEP 1: INITIALIZE DATABASE
# =============================================================================

def migrate_reviews_table(conn):
    """
    Migrate reviews table to remove UNIQUE constraint.
    This allows multiple individual reviews from the same Google Place ID.
    """
    # Check if table has UNIQUE constraint
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='reviews'"
    ).fetchone()
    
    if not schema:
        return False  # Table doesn't exist yet
    
    if 'UNIQUE' in schema[0]:
        LOG.info("Migrating reviews table to remove UNIQUE constraint...")
        
        conn.executescript("""
            -- Create new table without UNIQUE constraint
            CREATE TABLE reviews_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_registration_number TEXT,
                platform TEXT,
                rating REAL,
                review_count INTEGER DEFAULT 1,
                review_text TEXT,
                review_date TEXT,
                author_name TEXT,
                source_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (factor_registration_number) REFERENCES factors(registration_number)
            );
            
            -- Copy existing data
            INSERT INTO reviews_new SELECT * FROM reviews;
            
            -- Drop old table
            DROP TABLE reviews;
            
            -- Rename new table
            ALTER TABLE reviews_new RENAME TO reviews;
            
            -- Create index
            CREATE INDEX IF NOT EXISTS idx_reviews_factor ON reviews(factor_registration_number);
        """)
        
        LOG.success("Reviews table migrated successfully")
        return True
    
    return False


def step_1_init_database(reset: bool = False):
    """Initialize or reset the database schema."""
    LOG.step(1, "Initialize Database Schema")
    
    if reset and CONFIG.db_path.exists():
        LOG.info(f"Removing existing database: {CONFIG.db_path}")
        CONFIG.db_path.unlink()
    
    CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db() as conn:
        # Run migrations on existing database
        if CONFIG.db_path.exists():
            migrate_reviews_table(conn)
        
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    
    LOG.success(f"Database ready: {CONFIG.db_path}")
    
    # Show schema version info
    LOG.info("Schema v2.6: Reviews table without UNIQUE constraint for individual reviews")


# =============================================================================
# STEP 2: IMPORT CORE FACTOR DATA
# =============================================================================

def step_2_import_factors():
    """Import factors from registry CSV and FOI postcodes."""
    LOG.step(2, "Import Core Factor Data")
    
    # Import main registry
    csv_path = CONFIG.csv_dir / CONFIG.factors_csv
    if not csv_path.exists():
        LOG.error(f"Factors CSV not found: {csv_path}")
        return
    
    LOG.info(f"Reading: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    
    imported = 0
    with get_db() as conn:
        for i, row in enumerate(rows):
            LOG.progress(i + 1, len(rows))
            
            pf = normalize_pf_number(
                find_csv_column(row, ['registration_number', 'Registration Number', 'pf_number'])
            )
            if not pf:
                continue
            
            name = find_csv_column(row, ['name', 'Name', 'factor_name', 'trading_name'])
            if not name:
                continue
            
            # Normalize status to 'registered' or 'expired'
            raw_status = find_csv_column(row, ['registration_status', 'status', 'Status']) or 'registered'
            status = normalize_registration_status(raw_status)
            
            # Detect factor type from name patterns
            raw_type = find_csv_column(row, ['factor_type', 'type', 'Type'])
            factor_type = detect_factor_type(name, raw_type)
            
            # Get coverage areas for template
            coverage = find_csv_column(row, ['coverage_areas', 'foi_cities', 'cities', 'areas'])
            
            conn.execute("""
                INSERT INTO factors (
                    registration_number, name, trading_name, address, city, postcode,
                    website, email, phone, registration_date, registration_status, status,
                    property_count, factor_type, coverage_areas
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    name = COALESCE(excluded.name, name),
                    trading_name = COALESCE(excluded.trading_name, trading_name),
                    address = COALESCE(excluded.address, address),
                    website = COALESCE(excluded.website, website),
                    property_count = COALESCE(excluded.property_count, property_count),
                    status = excluded.status,
                    factor_type = COALESCE(excluded.factor_type, factor_type),
                    coverage_areas = COALESCE(excluded.coverage_areas, coverage_areas),
                    updated_at = CURRENT_TIMESTAMP
            """, [
                pf,
                name,
                find_csv_column(row, ['trading_name', 'Trading Name']),
                find_csv_column(row, ['address', 'Address']),
                find_csv_column(row, ['city', 'City']),
                find_csv_column(row, ['postcode', 'Postcode']),
                find_csv_column(row, ['website', 'Website']),
                find_csv_column(row, ['email', 'Email']),
                find_csv_column(row, ['phone', 'Phone']),
                parse_date(find_csv_column(row, ['registration_date', 'Registration Date'])),
                raw_status,  # Keep original for reference
                status,      # Normalized status
                parse_int(find_csv_column(row, ['property_count', 'properties', 'Properties Managed'])),
                factor_type,
                coverage,
            ])
            imported += 1
        
        conn.commit()
    
    LOG.success(f"Imported {imported} factors from registry")
    
    # Import FOI postcode coverage
    foi_path = CONFIG.csv_dir / CONFIG.factors_postcodes_csv
    if foi_path.exists():
        LOG.info(f"Reading FOI data: {foi_path}")
        
        with open(foi_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        updated = 0
        with get_db() as conn:
            for row in rows:
                pf = normalize_pf_number(find_csv_column(row, ['registration_number']))
                if not pf:
                    continue
                
                conn.execute("""
                    UPDATE factors SET
                        postcode_areas = ?,
                        postcode_count = ?,
                        geographic_reach = ?,
                        foi_cities = ?,
                        coverage_areas = COALESCE(coverage_areas, ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE registration_number = ?
                """, [
                    find_csv_column(row, ['postcode_areas']),
                    parse_int(find_csv_column(row, ['postcode_count'])),
                    find_csv_column(row, ['geographic_reach']),
                    find_csv_column(row, ['cities', 'foi_cities']),
                    find_csv_column(row, ['cities', 'foi_cities']),  # Also use for coverage_areas
                    pf,
                ])
                updated += 1
            conn.commit()
        LOG.success(f"Updated {updated} factors with FOI postcode coverage")
    else:
        LOG.skip(f"FOI postcodes CSV not found")


# =============================================================================
# STEP 3: IMPORT TRIBUNAL DATA
# =============================================================================

def step_3_import_tribunal():
    """Import tribunal cases from enriched database or CSV."""
    LOG.step(3, "Import Tribunal Data")
    
    imported = 0
    
    # Prefer SQLite database from tribunal enrichment
    if CONFIG.tribunal_db_path.exists():
        LOG.info(f"Reading from: {CONFIG.tribunal_db_path}")
        
        source_conn = sqlite3.connect(CONFIG.tribunal_db_path)
        source_conn.row_factory = sqlite3.Row
        
        # Check what table exists
        tables = [r[0] for r in source_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        
        if 'cases' in tables:
            cursor = source_conn.execute("SELECT * FROM cases")
        else:
            LOG.warning("No 'cases' table found in tribunal DB")
            source_conn.close()
            return
        
        with get_db() as conn:
            skipped = 0
            for row in cursor:
                pf = normalize_pf_number(row['matched_registration_number'])
                
                # Skip cases where factor doesn't exist (FK constraint)
                if pf:
                    exists = conn.execute(
                        "SELECT 1 FROM factors WHERE registration_number = ?", [pf]
                    ).fetchone()
                    if not exists:
                        skipped += 1
                        continue
                
                conn.execute("""
                    INSERT INTO tribunal_cases (
                        case_reference, factor_registration_number, decision_date,
                        outcome, complaints_made, complaints_upheld,
                        has_enforcement_order, pdf_url,
                        summary, key_quote, complaint_categories,
                        severity_score, compensation_awarded, pfeo_issued, pfeo_complied
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(case_reference) DO UPDATE SET
                        factor_registration_number = COALESCE(excluded.factor_registration_number, factor_registration_number),
                        decision_date = COALESCE(excluded.decision_date, decision_date),
                        outcome = COALESCE(excluded.outcome, outcome),
                        complaints_made = COALESCE(excluded.complaints_made, complaints_made),
                        complaints_upheld = COALESCE(excluded.complaints_upheld, complaints_upheld),
                        has_enforcement_order = COALESCE(excluded.has_enforcement_order, has_enforcement_order),
                        pdf_url = COALESCE(excluded.pdf_url, pdf_url),
                        summary = COALESCE(excluded.summary, summary),
                        key_quote = COALESCE(excluded.key_quote, key_quote),
                        complaint_categories = COALESCE(excluded.complaint_categories, complaint_categories),
                        severity_score = COALESCE(excluded.severity_score, severity_score),
                        compensation_awarded = COALESCE(excluded.compensation_awarded, compensation_awarded),
                        pfeo_issued = COALESCE(excluded.pfeo_issued, pfeo_issued),
                        pfeo_complied = COALESCE(excluded.pfeo_complied, pfeo_complied)
                """, [
                    row['case_reference'],
                    pf,
                    row['decision_date'] if 'decision_date' in row.keys() else None,
                    row['ai_outcome'] if 'ai_outcome' in row.keys() else None,
                    row['complaints_made'] if 'complaints_made' in row.keys() else None,
                    row['complaints_upheld'] if 'complaints_upheld' in row.keys() else None,
                    1 if ('pfeo_issued' in row.keys() and row['pfeo_issued']) else 0,
                    row['pdf_url'] if 'pdf_url' in row.keys() else None,
                    row['summary'] if 'summary' in row.keys() else None,
                    row['key_quote'] if 'key_quote' in row.keys() else None,
                    row['complaint_categories'] if 'complaint_categories' in row.keys() else None,
                    row['severity_score'] if 'severity_score' in row.keys() else None,
                    next((row[c] for c in ['compensation_total', 'compensation_awarded', 'compensation', 'compensation_amount'] if c in row.keys() and row[c]), None),
                    1 if ('pfeo_issued' in row.keys() and row['pfeo_issued']) else 0,
                    1 if ('pfeo_complied' in row.keys() and row['pfeo_complied']) else None,
                ])
                imported += 1
            
            if skipped:
                LOG.info(f"Skipped {skipped} cases (factor not in register)")
            
            conn.commit()
        
        source_conn.close()
        LOG.success(f"Imported {imported} tribunal cases from enriched DB")
    
    # Fallback to CSV
    elif (CONFIG.csv_dir / CONFIG.tribunal_csv).exists():
        csv_path = CONFIG.csv_dir / CONFIG.tribunal_csv
        LOG.info(f"Reading from CSV: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        with get_db() as conn:
            for row in rows:
                case_ref = find_csv_column(row, ['case_reference', 'case_references'])
                if not case_ref:
                    continue
                
                pf = normalize_pf_number(
                    find_csv_column(row, ['matched_registration_number', 'factor_registration_number'])
                )
                
                conn.execute("""
                    INSERT INTO tribunal_cases (
                        case_reference, factor_registration_number, decision_date,
                        outcome, has_enforcement_order, pdf_url
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(case_reference) DO NOTHING
                """, [
                    case_ref.split(',')[0].strip(),
                    pf,
                    parse_date(find_csv_column(row, ['hearing_date', 'decision_date'])),
                    find_csv_column(row, ['outcome']),
                    0,
                    find_csv_column(row, ['pdf_urls', 'pdf_url']),
                ])
                imported += 1
            
            conn.commit()
        LOG.success(f"Imported {imported} tribunal cases from CSV")
    else:
        LOG.skip("No tribunal data found")
    
    # Import case fees from tribunal_enriched.db
    if CONFIG.tribunal_db_path.exists():
        source_conn = sqlite3.connect(CONFIG.tribunal_db_path)
        source_conn.row_factory = sqlite3.Row
        
        tables = [r[0] for r in source_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        
        if 'case_fees' in tables:
            LOG.info(f"Reading case fees from: {CONFIG.tribunal_db_path}")
            
            cursor = source_conn.execute("SELECT * FROM case_fees")
            
            fees_imported = 0
            with get_db() as conn:
                for row in cursor:
                    case_ref = row['case_reference'] if 'case_reference' in row.keys() else None
                    if not case_ref:
                        continue
                    
                    factor_row = conn.execute(
                        "SELECT factor_registration_number FROM tribunal_cases WHERE case_reference = ?",
                        [case_ref]
                    ).fetchone()
                    
                    pf = factor_row[0] if factor_row else None
                    
                    conn.execute("""
                        INSERT INTO case_fees (
                            case_reference, factor_registration_number, fee_type,
                            amount, frequency, vat_included, evidence, context,
                            disputed, tribunal_finding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        case_ref,
                        pf,
                        row['fee_type'] if 'fee_type' in row.keys() else None,
                        row['amount'] if 'amount' in row.keys() else None,
                        row['frequency'] if 'frequency' in row.keys() else None,
                        row['vat_included'] if 'vat_included' in row.keys() else None,
                        row['evidence'] if 'evidence' in row.keys() else None,
                        row['context'] if 'context' in row.keys() else None,
                        row['disputed'] if 'disputed' in row.keys() else 0,
                        row['tribunal_finding'] if 'tribunal_finding' in row.keys() else None,
                    ])
                    fees_imported += 1
                
                conn.commit()
            
            LOG.success(f"Imported {fees_imported} fee records from tribunal cases")
        
        source_conn.close()


# =============================================================================
# STEP 4: IMPORT REVIEWS
# =============================================================================

def step_4_import_reviews():
    """Import Google and Trustpilot reviews."""
    LOG.step(4, "Import Review Data")
    
    total_imported = 0
    
    # Google Reviews
    google_path = CONFIG.csv_dir / CONFIG.google_reviews_csv
    if google_path.exists():
        LOG.info(f"Reading: {google_path}")
        
        with open(google_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        imported = 0
        with get_db() as conn:
            for i, row in enumerate(rows):
                LOG.progress(i + 1, len(rows))
                
                pf = normalize_pf_number(
                    find_csv_column(row, ['factor_registration_number', 'registration_number', 'pf_number'])
                )
                if not pf:
                    continue
                
                source_id = find_csv_column(row, ['place_id', 'google_place_id', 'source_id'])
                if not source_id:
                    source_id = f"google_{pf}"
                
                # For aggregate records (no review_text), author_name stores the location name
                review_text = find_csv_column(row, ['review_text', 'text'])
                if review_text:
                    # Individual review - use author name
                    author_name = find_csv_column(row, ['author_name', 'author'])
                else:
                    # Aggregate record - use location/place name
                    author_name = find_csv_column(row, ['name', 'location_name', 'place_name', 'google_name', 'address'])
                
                try:
                    conn.execute("""
                        INSERT INTO reviews (
                            factor_registration_number, platform, rating, review_count,
                            review_text, review_date, author_name, source_id
                        ) VALUES (?, 'google', ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                    """, [
                        pf,
                        parse_float(find_csv_column(row, ['rating', 'google_rating'])),
                        parse_int(find_csv_column(row, ['review_count', 'total_reviews', 'google_review_count'])) or 1,
                        review_text,
                        parse_date(find_csv_column(row, ['review_date', 'date'])),
                        author_name,
                        source_id,
                    ])
                    imported += 1
                except sqlite3.IntegrityError:
                    pass
            
            conn.commit()
        LOG.success(f"Imported {imported} Google reviews")
        total_imported += imported
    else:
        LOG.skip("Google reviews CSV not found")
    
    # Trustpilot Reviews
    trustpilot_path = CONFIG.csv_dir / CONFIG.trustpilot_csv
    if trustpilot_path.exists():
        LOG.info(f"Reading: {trustpilot_path}")
        
        with open(trustpilot_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        imported = 0
        with get_db() as conn:
            for i, row in enumerate(rows):
                LOG.progress(i + 1, len(rows))
                
                pf = normalize_pf_number(
                    find_csv_column(row, ['factor_registration_number', 'registration_number', 'pf_number'])
                )
                if not pf:
                    continue
                
                source_id = find_csv_column(row, ['trustpilot_url', 'url', 'source_id'])
                if not source_id:
                    source_id = f"trustpilot_{pf}"
                
                try:
                    conn.execute("""
                        INSERT INTO reviews (
                            factor_registration_number, platform, rating, review_count,
                            review_text, review_date, author_name, source_id
                        ) VALUES (?, 'trustpilot', ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                    """, [
                        pf,
                        parse_float(find_csv_column(row, ['rating'])),
                        parse_int(find_csv_column(row, ['review_count', 'total_reviews'])) or 1,
                        find_csv_column(row, ['review_text', 'text']),
                        parse_date(find_csv_column(row, ['review_date', 'date'])),
                        find_csv_column(row, ['author_name', 'author']),
                        source_id,
                    ])
                    imported += 1
                except sqlite3.IntegrityError:
                    pass
            
            conn.commit()
        LOG.success(f"Imported {imported} Trustpilot reviews")
        total_imported += imported
    else:
        LOG.skip("Trustpilot CSV not found")
    
    # Google Reviews with Text (individual reviews scraped from Google)
    reviews_text_path = CONFIG.csv_dir / CONFIG.google_reviews_text_csv
    if reviews_text_path.exists():
        LOG.info(f"Reading individual reviews: {reviews_text_path}")
        
        with open(reviews_text_path, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        
        imported = 0
        skipped = 0
        with get_db() as conn:
            for i, row in enumerate(rows):
                if i % 500 == 0 and i > 0:
                    LOG.progress(i, len(rows), f"reviews text")
                
                pf = normalize_pf_number(
                    find_csv_column(row, ['factor_registration_number', 'registration_number', 'pf_number'])
                )
                if not pf:
                    skipped += 1
                    continue
                
                place_id = find_csv_column(row, ['google_place_id', 'place_id', 'source_id'])
                author = find_csv_column(row, ['author_name', 'author'])
                text = find_csv_column(row, ['review_text', 'text'])
                
                # Skip if no meaningful text
                if not text or len(text.strip()) < 10:
                    skipped += 1
                    continue
                
                # Parse rating
                rating = parse_float(find_csv_column(row, ['review_rating', 'rating']))
                
                # Convert Unix timestamp to date if present
                review_date = None
                review_time = find_csv_column(row, ['review_time', 'time'])
                if review_time:
                    try:
                        ts = int(review_time)
                        from datetime import datetime as dt
                        review_date = dt.fromtimestamp(ts).strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        pass
                
                # Check if this exact review already exists
                existing = conn.execute("""
                    SELECT id FROM reviews 
                    WHERE factor_registration_number = ? 
                      AND source_id = ?
                      AND author_name = ?
                      AND review_text IS NOT NULL
                """, [pf, place_id, author]).fetchone()
                
                if existing:
                    skipped += 1
                    continue
                
                # Insert individual review with text
                try:
                    conn.execute("""
                        INSERT INTO reviews (
                            factor_registration_number, platform, rating, review_count,
                            review_text, review_date, author_name, source_id
                        ) VALUES (?, 'google', ?, 1, ?, ?, ?, ?)
                    """, [pf, rating, text.strip(), review_date, author, place_id])
                    imported += 1
                except sqlite3.IntegrityError:
                    skipped += 1
            
            conn.commit()
        LOG.success(f"Imported {imported} Google reviews with text (skipped {skipped})")
        total_imported += imported
    else:
        LOG.skip("Google reviews text CSV not found")
    
    if total_imported == 0:
        LOG.warning("No review data imported")


# =============================================================================
# STEP 5: IMPORT COMPANIES HOUSE DATA
# =============================================================================

def step_5_import_companies_house():
    """Import Companies House data."""
    LOG.step(5, "Import Companies House Data")
    
    csv_path = CONFIG.csv_dir / CONFIG.companies_house_csv
    if not csv_path.exists():
        LOG.skip(f"Companies House CSV not found")
        return
    
    LOG.info(f"Reading: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    
    imported = 0
    with get_db() as conn:
        for i, row in enumerate(rows):
            LOG.progress(i + 1, len(rows))
            
            pf = normalize_pf_number(
                find_csv_column(row, ['factor_registration_number', 'registration_number', 'pf_number'])
            )
            if not pf:
                continue
            
            conn.execute("""
                INSERT INTO companies (
                    registration_number, company_number, company_name, company_url,
                    incorporated_date, company_status, company_type, jurisdiction,
                    sic_codes, registered_address,
                    has_insolvency_history, has_charges, registered_office_in_dispute, undeliverable_address,
                    previous_names_count, previous_names,
                    director_count, secretary_count, avg_director_tenure_years,
                    min_director_tenure_years, max_director_tenure_years, directors,
                    psc_count, corporate_psc_count, pscs,
                    last_accounts_date, last_accounts_type, next_accounts_due,
                    accounts_overdue, accounts_days_overdue,
                    last_confirmation_date, next_confirmation_due, confirmation_overdue,
                    total_filings, filings_last_year, late_filings_count,
                    last_filing_date, last_filing_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    company_number = COALESCE(excluded.company_number, company_number),
                    company_name = COALESCE(excluded.company_name, company_name),
                    company_status = COALESCE(excluded.company_status, company_status),
                    has_insolvency_history = COALESCE(excluded.has_insolvency_history, has_insolvency_history),
                    has_charges = COALESCE(excluded.has_charges, has_charges),
                    director_count = COALESCE(excluded.director_count, director_count),
                    accounts_overdue = COALESCE(excluded.accounts_overdue, accounts_overdue),
                    late_filings_count = COALESCE(excluded.late_filings_count, late_filings_count)
            """, [
                pf,
                find_csv_column(row, ['company_number', 'ch_number']),
                find_csv_column(row, ['company_name', 'ch_name']),
                find_csv_column(row, ['company_url']),
                parse_date(find_csv_column(row, ['incorporated_date', 'incorporation_date'])),
                find_csv_column(row, ['company_status', 'status']),
                find_csv_column(row, ['company_type', 'type']),
                find_csv_column(row, ['jurisdiction']),
                find_csv_column(row, ['sic_codes']),
                find_csv_column(row, ['registered_address']),
                parse_bool(find_csv_column(row, ['has_insolvency_history'])),
                parse_bool(find_csv_column(row, ['has_charges'])),
                parse_bool(find_csv_column(row, ['registered_office_in_dispute'])),
                parse_bool(find_csv_column(row, ['undeliverable_address'])),
                parse_int(find_csv_column(row, ['previous_names_count'])),
                find_csv_column(row, ['previous_names']),
                parse_int(find_csv_column(row, ['director_count', 'directors'])),
                parse_int(find_csv_column(row, ['secretary_count'])),
                parse_float(find_csv_column(row, ['avg_director_tenure_years', 'avg_tenure'])),
                parse_float(find_csv_column(row, ['min_director_tenure_years'])),
                parse_float(find_csv_column(row, ['max_director_tenure_years'])),
                find_csv_column(row, ['directors']),
                parse_int(find_csv_column(row, ['psc_count'])),
                parse_int(find_csv_column(row, ['corporate_psc_count'])),
                find_csv_column(row, ['pscs']),
                parse_date(find_csv_column(row, ['last_accounts_date', 'last_accounts'])),
                find_csv_column(row, ['last_accounts_type']),
                parse_date(find_csv_column(row, ['next_accounts_due'])),
                parse_bool(find_csv_column(row, ['accounts_overdue', 'overdue'])),
                parse_int(find_csv_column(row, ['accounts_days_overdue'])),
                parse_date(find_csv_column(row, ['last_confirmation_date'])),
                parse_date(find_csv_column(row, ['next_confirmation_due'])),
                parse_bool(find_csv_column(row, ['confirmation_overdue'])),
                parse_int(find_csv_column(row, ['total_filings'])),
                parse_int(find_csv_column(row, ['filings_last_year'])),
                parse_int(find_csv_column(row, ['late_filings_count'])),
                parse_date(find_csv_column(row, ['last_filing_date'])),
                find_csv_column(row, ['last_filing_type']),
            ])
            imported += 1
        
        conn.commit()
    
    LOG.success(f"Imported {imported} company records")


# =============================================================================
# STEP 6: IMPORT WSS DATA
# =============================================================================

def step_6_import_wss():
    """Import WSS (Written Statement of Services) data."""
    LOG.step(6, "Import WSS Data")
    
    if not CONFIG.wss_db_path.exists():
        LOG.skip(f"WSS database not found: {CONFIG.wss_db_path}")
        return
    
    LOG.info(f"Reading from: {CONFIG.wss_db_path}")
    
    source_conn = sqlite3.connect(CONFIG.wss_db_path)
    source_conn.row_factory = sqlite3.Row
    
    cursor = source_conn.execute("""
        SELECT 
            m.registration_number,
            d.url AS document_url,
            d.document_year,
            k.*
        FROM wss_factor_mapping m
        JOIN wss_documents d ON m.document_id = d.id
        LEFT JOIN wss_key_fields k ON m.document_id = k.document_id
        WHERE m.registration_number IS NOT NULL AND m.registration_number != ''
    """)
    
    imported = 0
    with get_db() as conn:
        for row in cursor:
            pf = normalize_pf_number(row['registration_number'])
            if not pf:
                continue
            
            conn.execute("""
                INSERT INTO wss (
                    registration_number, document_url, document_year,
                    management_fee_amount, management_fee_frequency, management_fee_vat,
                    billing_frequency, late_penalty, float_required, sinking_fund, nopli,
                    insurance_provider, insurance_broker, commission_disclosure,
                    emergency_response, urgent_response, routine_response,
                    enquiry_response, complaint_response,
                    portal, app, notice_period,
                    code_of_conduct_version, professional_memberships, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(registration_number) DO UPDATE SET
                    document_url = excluded.document_url,
                    management_fee_amount = COALESCE(excluded.management_fee_amount, management_fee_amount),
                    late_penalty = COALESCE(excluded.late_penalty, late_penalty),
                    nopli = COALESCE(excluded.nopli, nopli)
            """, [
                pf,
                row['document_url'],
                row['document_year'],
                row['management_fee_amount'] if 'management_fee_amount' in row.keys() else None,
                row['management_fee_frequency'] if 'management_fee_frequency' in row.keys() else None,
                row['management_fee_vat'] if 'management_fee_vat' in row.keys() else None,
                row['billing_frequency'] if 'billing_frequency' in row.keys() else None,
                row['late_penalty'] if 'late_penalty' in row.keys() else None,
                row['float_required'] if 'float_required' in row.keys() else None,
                row['sinking_fund'] if 'sinking_fund' in row.keys() else None,
                row['nopli'] if 'nopli' in row.keys() else None,
                row['insurance_provider'] if 'insurance_provider' in row.keys() else None,
                row['insurance_broker'] if 'insurance_broker' in row.keys() else None,
                row['commission_disclosure'] if 'commission_disclosure' in row.keys() else None,
                row['emergency_response'] if 'emergency_response' in row.keys() else None,
                row['urgent_response'] if 'urgent_response' in row.keys() else None,
                row['routine_response'] if 'routine_response' in row.keys() else None,
                row['enquiry_response'] if 'enquiry_response' in row.keys() else None,
                row['complaint_response'] if 'complaint_response' in row.keys() else None,
                row['portal'] if 'portal' in row.keys() else None,
                row['app'] if 'app' in row.keys() else None,
                row['notice_period'] if 'notice_period' in row.keys() else None,
                row['code_of_conduct_version'] if 'code_of_conduct_version' in row.keys() else None,
                row['professional_memberships'] if 'professional_memberships' in row.keys() else None,
                row['confidence_score'] if 'confidence_score' in row.keys() else None,
            ])
            imported += 1
        
        conn.commit()
    
    source_conn.close()
    LOG.success(f"Imported WSS data for {imported} factors")


# =============================================================================
# STEP 7: CALCULATE SCORES & RISK BANDS (v2.6 - New Tier Logic)
# =============================================================================

def _is_within_years(case: dict, years: int) -> bool:
    """Check if a case's decision_date is within the last N years."""
    decision_date = case.get('decision_date')
    if not decision_date:
        return False
    try:
        # Handle various date formats
        if isinstance(decision_date, str):
            # Extract year from YYYY-MM-DD or similar
            year_str = decision_date[:4]
            if year_str.isdigit():
                case_year = int(year_str)
                cutoff_year = datetime.now().year - years
                return case_year >= cutoff_year
    except (ValueError, TypeError):
        pass
    return False


def _determine_tier(score: float, cases: list, factor_data: dict) -> str:
    """
    Determine display tier from score with special rules.
    
    v2.9 Tier Logic:
    - RULE 1a: Unresolved PFEO (5yr) + 3+ cases = RED
    - RULE 1b: Unresolved PFEO (5yr) + <3 cases = ORANGE (mitigated by low volume)
    - RULE 2: 2+ resolved PFEOs in 5yr = ORANGE max
    - RULE 3: < 3 cases in 5yr (no unresolved PFEO) = CLEAN
    - RULE 4: Score-based for 3+ cases (65+ GREEN, 40+ AMBER, 20+ ORANGE, else RED)
    
    Args:
        score: Composite score (0-100 scale)
        cases: List of case dicts with pfeo_issued, pfeo_complied, decision_date
        factor_data: Factor dict (for potential future rules)
    
    Returns:
        Tier string: 'CLEAN', 'GREEN', 'AMBER', 'ORANGE', or 'RED'
    """
    # Filter to 5-year window
    recent_5yr = [c for c in cases if _is_within_years(c, 5)]
    recent_count = len(recent_5yr)
    
    # Check for unresolved PFEO
    # Unresolved = pfeo_issued AND (pfeo_complied is NULL or 0)
    has_unresolved_pfeo = False
    for c in recent_5yr:
        if c.get('pfeo_issued'):
            pfeo_complied = c.get('pfeo_complied')
            if pfeo_complied is None or pfeo_complied == 0:
                has_unresolved_pfeo = True
                break
    
    # RULE 1a: Unresolved PFEO + high volume = RED
    if has_unresolved_pfeo and recent_count >= 3:
        return 'RED'
    
    # RULE 1b: Unresolved PFEO + low volume = ORANGE (mitigated)
    if has_unresolved_pfeo:
        return 'ORANGE'
    
    # RULE 2: 2+ resolved PFEOs in 5yr = ORANGE max
    pfeo_count = sum(1 for c in recent_5yr if c.get('pfeo_issued'))
    if pfeo_count >= 2:
        return 'ORANGE'
    
    # RULE 3: < 3 cases in 5yr = CLEAN
    if recent_count < 3:
        return 'CLEAN'
    
    # RULE 4: Score-based for 3+ cases
    if score >= 65:
        return 'GREEN'
    elif score >= 40:
        return 'AMBER'
    elif score >= 20:
        return 'ORANGE'
    else:
        return 'RED'


def calculate_composite_tribunal_score(
    case_count: int,
    property_count: int,
    avg_severity: Optional[float],
    cases_last_3_years: int,
    pfeo_count: int,
    cases: list = None,
    factor_data: dict = None
) -> Dict[str, Any]:
    """
    Calculate composite tribunal score using weighted components.
    
    v2.6 CHANGES:
    - Score now on 0-100 scale (was 0-10)
    - Tier determination moved to _determine_tier() with new rules
    - Accepts cases list for PFEO resolution checking
    
    Components:
    - Volume (40%): Cases per 10k properties (all time)
    - Severity (35%): Average severity of cases
    - Recency (25%): Recent cases per 10k properties (last 3 years)
    
    Returns dict with score components and final tier.
    """
    cases = cases or []
    factor_data = factor_data or {}
    
    # Normalize property count - use median factor size as fallback
    property_count_estimated = False
    if property_count is None or property_count <= 0:
        property_count = 2000  # Median-ish factor size as reasonable fallback
        property_count_estimated = True
    
    # Calculate cases per 10k (all time)
    cases_per_10k = (case_count / property_count) * 10000
    
    # Volume score (0-100): 0 per 10k = 100, 5+ per 10k = 0
    # Linear scale: each case per 10k reduces score by 20 points
    volume_score = max(0, 100 - (cases_per_10k * 20))
    
    # Severity score (0-100, inverted): avg_severity of 1 = 100, avg_severity of 10 = 0
    if avg_severity and avg_severity > 0:
        severity_score = max(0, 100 - (avg_severity * 10))
    else:
        severity_score = 100  # No cases = perfect severity score
    
    # Recency score (0-100): Rate-based like volume. 0 per 10k = 100, 10+ per 10k = 0
    recent_per_10k = (cases_last_3_years / property_count) * 10000
    recency_score = max(0, 100 - (recent_per_10k * 10))
    
    # PFEO penalty - hard modifier (max 40 point penalty on 0-100 scale)
    pfeo_penalty = min(pfeo_count * 15, 40)  # 15 points per PFEO, max 40
    
    # Weighted composite (0-100 scale)
    composite = (
        volume_score * CONFIG.score_weight_volume +
        severity_score * CONFIG.score_weight_severity +
        recency_score * CONFIG.score_weight_recency
    ) - pfeo_penalty
    
    # Clamp to 0-100
    composite = max(0, min(100, composite))
    
    # Determine tier using new rules
    tier = _determine_tier(composite, cases, factor_data)
    
    return {
        'composite_score': round(composite, 2),
        'volume_score': round(volume_score, 2),
        'severity_score': round(severity_score, 2),
        'recency_score': round(recency_score, 2),
        'pfeo_penalty': round(pfeo_penalty, 2),
        'cases_per_10k': round(cases_per_10k, 2),
        'recent_per_10k': round(recent_per_10k, 2),
        'tier': tier,
        'property_count_estimated': property_count_estimated,
    }


def get_tier_description(tier: str) -> str:
    """Get human-readable description for a risk tier."""
    descriptions = {
        'CLEAN': 'No significant tribunal history',
        'GREEN': 'Minor tribunal activity',
        'AMBER': 'Some tribunal concerns',
        'ORANGE': 'Significant tribunal issues',
        'RED': 'Serious tribunal record',
    }
    return descriptions.get(tier, 'Unknown')


def step_7_calculate_scores():
    """Calculate composite tribunal scores, review scores, and risk bands."""
    LOG.step(7, "Calculate Scores & Risk Bands (v2.9 - Refined PFEO Logic)")
    
    with get_db() as conn:
        LOG.info("Calculating scores with v2.9 tier rules...")
        LOG.info("  Rule 1a: Unresolved PFEO (5yr) + 3+ cases = RED")
        LOG.info("  Rule 1b: Unresolved PFEO (5yr) + <3 cases = ORANGE")
        LOG.info("  Rule 2: 2+ resolved PFEOs (5yr) = ORANGE max")
        LOG.info("  Rule 3: <3 cases (5yr) = CLEAN")
        LOG.info("  Rule 4: Score-based for 3+ cases")
        
        # Get all factors
        factors = conn.execute("SELECT registration_number, property_count FROM factors").fetchall()
        
        updates = 0
        score_distribution = {'CLEAN': 0, 'GREEN': 0, 'AMBER': 0, 'ORANGE': 0, 'RED': 0}
        
        for factor in factors:
            pf = factor['registration_number']
            properties = factor['property_count'] or 0
            
            # Get all tribunal cases for this factor
            cases = conn.execute("""
                SELECT 
                    case_reference, decision_date, outcome,
                    complaints_made, complaints_upheld, 
                    pfeo_issued, pfeo_complied,
                    severity_score, compensation_awarded
                FROM tribunal_cases 
                WHERE factor_registration_number = ?
            """, [pf]).fetchall()
            
            # Convert to list of dicts for processing
            cases_list = [dict(c) for c in cases]
            
            # Calculate aggregates
            case_count = len(cases_list)
            total_upheld = sum(c['complaints_upheld'] or 0 for c in cases_list)
            cases_with_upheld = sum(1 for c in cases_list if (c['complaints_upheld'] or 0) > 0)
            has_enforcement = any(c['pfeo_issued'] for c in cases_list)
            total_compensation = sum(c['compensation_awarded'] or 0 for c in cases_list)
            pfeo_count = sum(1 for c in cases_list if c['pfeo_issued'])
            
            # Average severity
            severities = [c['severity_score'] for c in cases_list if c['severity_score']]
            avg_severity = sum(severities) / len(severities) if severities else None
            
            # Cases in last 3 years (for recency component)
            cases_last_3_years = sum(1 for c in cases_list if _is_within_years(c, 3))
            
            # Check for active/unresolved PFEO
            has_active_pfeo = any(
                c['pfeo_issued'] and not c.get('pfeo_complied')
                for c in cases_list
                if _is_within_years(c, 5)
            )
            
            # Calculate composite score with new tier logic
            score_result = calculate_composite_tribunal_score(
                case_count=case_count,
                property_count=properties,
                avg_severity=avg_severity,
                cases_last_3_years=cases_last_3_years,
                pfeo_count=pfeo_count,
                cases=cases_list,
                factor_data={'registration_number': pf}
            )
            
            tier = score_result['tier']
            score_distribution[tier] = score_distribution.get(tier, 0) + 1
            
            conn.execute("""
                UPDATE factors SET
                    tribunal_case_count = ?,
                    tribunal_cases_upheld = ?,
                    tribunal_pfeo_count = ?,
                    tribunal_total_compensation = ?,
                    has_enforcement_order = ?,
                    tribunal_rate_per_10k = ?,
                    risk_band = ?,
                    tribunal_composite_score = ?,
                    tribunal_volume_score = ?,
                    tribunal_severity_score = ?,
                    tribunal_recency_score = ?,
                    tribunal_avg_severity = ?,
                    tribunal_cases_last_3_years = ?,
                    tribunal_pfeo_penalty = ?,
                    has_active_pfeo = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE registration_number = ?
            """, [
                case_count, 
                cases_with_upheld,
                pfeo_count,
                total_compensation,
                has_enforcement,
                score_result['cases_per_10k'],
                tier,
                score_result['composite_score'],
                score_result['volume_score'],
                score_result['severity_score'],
                score_result['recency_score'],
                round(avg_severity, 2) if avg_severity else None,
                cases_last_3_years,
                score_result['pfeo_penalty'],
                has_active_pfeo,
                pf,
            ])
            updates += 1
        
        conn.commit()
        
        LOG.success(f"Calculated scores for {updates} factors")
        LOG.info("Score distribution (v2.6 - 0-100 scale):")
        for tier in ['CLEAN', 'GREEN', 'AMBER', 'ORANGE', 'RED']:
            count = score_distribution.get(tier, 0)
            desc = get_tier_description(tier)
            LOG.info(f"  {tier:8} {count:4} factors - {desc}")
        
        # Review aggregates (unchanged from v2)
        LOG.info("Aggregating review scores...")
        
        cursor = conn.execute("""
            SELECT 
                factor_registration_number,
                platform,
                SUM(rating * review_count) / SUM(review_count) AS avg_rating,
                SUM(review_count) AS review_count
            FROM reviews
            WHERE rating IS NOT NULL 
              AND review_count > 0
              AND review_text IS NULL
            GROUP BY factor_registration_number, platform
        """)
        
        review_data = {}
        for row in cursor:
            pf = row['factor_registration_number']
            if pf not in review_data:
                review_data[pf] = {'google': None, 'trustpilot': None,
                                   'google_count': 0, 'trustpilot_count': 0}
            
            if row['platform'] == 'google':
                review_data[pf]['google'] = row['avg_rating']
                review_data[pf]['google_count'] = row['review_count']
            elif row['platform'] == 'trustpilot':
                review_data[pf]['trustpilot'] = row['avg_rating']
                review_data[pf]['trustpilot_count'] = row['review_count']
        
        for pf, data in review_data.items():
            total = data['google_count'] + data['trustpilot_count']
            combined = None
            if total > 0:
                weighted = 0
                if data['google']:
                    weighted += data['google'] * data['google_count']
                if data['trustpilot']:
                    weighted += data['trustpilot'] * data['trustpilot_count']
                combined = weighted / total
            
            conn.execute("""
                UPDATE factors SET
                    google_rating = ?,
                    google_review_count = ?,
                    trustpilot_rating = ?,
                    trustpilot_review_count = ?,
                    combined_rating = ?,
                    total_review_count = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE registration_number = ?
            """, [
                round(data['google'], 2) if data['google'] else None,
                data['google_count'],
                round(data['trustpilot'], 2) if data['trustpilot'] else None,
                data['trustpilot_count'],
                round(combined, 2) if combined else None,
                total,
                pf
            ])
        
        conn.commit()
        LOG.success(f"Updated review scores for {len(review_data)} factors")


# =============================================================================
# STEP 8: GENERATE AI SUMMARIES
# =============================================================================

AT_A_GLANCE_PROMPT = """You are generating a factual "At a Glance" summary for a Scottish property factor profile page.
Based on the data provided, create 3-5 bullet points summarising the key facts a homeowner would want to know.

Guidelines:
- State facts, not opinions (e.g. "45 tribunal cases since 2012" not "troubled history")
- Include specific numbers where available (ratings, case counts, fees)
- Note patterns from tribunal cases if data exists (e.g. "Most common complaints relate to communication and fees")
- Mention review ratings and volume without characterising them as good/bad
- Include fee information if available
- If data is limited, simply note what is known

Avoid:
- Value judgments ("poor", "excellent", "concerning")
- Recommendations ("consider alternatives", "proceed with caution")  
- Speculation beyond the data provided
- Emotional language

Return JSON in this exact format:
{
  "bullets": [
    "Factual point with specific data",
    "Another factual point",
    "Third factual point"
  ],
  "overall_sentiment": "positive|neutral|mixed|negative",
  "confidence": 0.0-1.0
}

FACTOR DATA:
"""

TRIBUNAL_ANALYSIS_PROMPT = """You are summarising the tribunal history for a Scottish property factor.
Based on the case data provided, create a factual summary of the tribunal record.

Include:
- Total case count and timeframe
- Most common complaint categories
- Outcome breakdown (upheld vs dismissed) if available
- Any Property Factor Enforcement Orders (PFEOs)
- Compensation amounts if significant

Keep it factual - state what the data shows without characterising it as good or bad.
If there are few cases, simply note the limited data.

Return JSON in this exact format:
{
  "complaint_patterns": ["category 1", "category 2"],
  "trend": "increasing|stable|decreasing|insufficient_data",
  "key_facts": ["fact 1", "fact 2"],
  "summary": "One paragraph factual summary of tribunal history"
}

TRIBUNAL CASES:
"""


def build_factor_context(conn, factor: dict) -> dict:
    """Build rich context for AI summary generation."""
    pf = factor['registration_number']
    
    context = {
        "name": factor['name'],
        "registration_number": pf,
        "status": factor.get('status', 'registered'),
        "factor_type": factor.get('factor_type', 'Unknown'),
        "risk_band": factor.get('risk_band'),
        "tribunal_score": factor.get('tribunal_composite_score'),
    }
    
    # Tribunal data
    cases = conn.execute("""
        SELECT 
            case_reference,
            decision_date,
            outcome,
            pfeo_issued,
            compensation_awarded,
            summary,
            complaint_categories,
            severity_score
        FROM tribunal_cases
        WHERE factor_registration_number = ?
        ORDER BY decision_date DESC
        LIMIT 10
    """, [pf]).fetchall()
    
    if cases:
        # Calculate stats
        total_cases = conn.execute(
            "SELECT COUNT(*) FROM tribunal_cases WHERE factor_registration_number = ?", [pf]
        ).fetchone()[0]
        
        pfeo_count = sum(1 for c in cases if c['pfeo_issued'])
        upheld_count = sum(1 for c in cases if c['outcome'] and 'upheld' in c['outcome'].lower() and 'not' not in c['outcome'].lower())
        
        # Extract complaint categories
        all_categories = []
        for case in cases:
            if case['complaint_categories']:
                try:
                    cats = json.loads(case['complaint_categories'])
                    if isinstance(cats, list):
                        all_categories.extend(cats)
                except:
                    pass
        
        # Count top categories
        from collections import Counter
        category_counts = Counter(all_categories)
        top_categories = [cat for cat, _ in category_counts.most_common(5)]
        
        # Recent case summaries
        recent_cases = []
        for case in cases[:5]:
            if case['summary']:
                recent_cases.append({
                    "date": case['decision_date'],
                    "outcome": case['outcome'],
                    "summary": case['summary'][:300] if case['summary'] else None,
                    "pfeo": bool(case['pfeo_issued']),
                    "compensation": case['compensation_awarded']
                })
        
        context["tribunal"] = {
            "total_cases": total_cases,
            "pfeo_count": pfeo_count,
            "recent_upheld": upheld_count,
            "top_complaint_types": top_categories,
            "recent_cases": recent_cases
        }
    else:
        context["tribunal"] = {"total_cases": 0, "message": "No tribunal cases on record"}
    
    # Review data
    reviews_data = {}
    
    # Google reviews
    google_reviews = conn.execute("""
        SELECT rating, review_count, review_text
        FROM reviews
        WHERE factor_registration_number = ? AND platform = 'google'
        ORDER BY review_count DESC, review_date DESC
        LIMIT 10
    """, [pf]).fetchall()
    
    if google_reviews:
        # Get aggregate stats
        aggregate = next((r for r in google_reviews if r['review_count'] and r['review_count'] > 1 and not r['review_text']), None)
        samples = [r['review_text'][:200] for r in google_reviews if r['review_text']][:5]
        
        reviews_data["google"] = {
            "rating": aggregate['rating'] if aggregate else factor.get('google_rating'),
            "count": aggregate['review_count'] if aggregate else factor.get('google_review_count'),
            "sample_reviews": samples if samples else None
        }
    
    # Trustpilot reviews
    trustpilot_reviews = conn.execute("""
        SELECT rating, review_count, review_text
        FROM reviews
        WHERE factor_registration_number = ? AND platform = 'trustpilot'
        ORDER BY review_count DESC, review_date DESC
        LIMIT 10
    """, [pf]).fetchall()
    
    if trustpilot_reviews:
        aggregate = next((r for r in trustpilot_reviews if r['review_count'] and r['review_count'] > 1 and not r['review_text']), None)
        samples = [r['review_text'][:200] for r in trustpilot_reviews if r['review_text']][:5]
        
        reviews_data["trustpilot"] = {
            "rating": aggregate['rating'] if aggregate else factor.get('trustpilot_rating'),
            "count": aggregate['review_count'] if aggregate else factor.get('trustpilot_review_count'),
            "sample_reviews": samples if samples else None
        }
    
    if reviews_data:
        context["reviews"] = reviews_data
    else:
        context["reviews"] = {"message": "No reviews found"}
    
    # WSS fee data
    wss = conn.execute("""
        SELECT management_fee_amount, insurance_admin_fee, late_penalty,
               response_time_emergency, response_time_routine
        FROM wss
        WHERE registration_number = ?
    """, [pf]).fetchone()
    
    if wss:
        context["fees_wss"] = {
            "management_fee": wss['management_fee_amount'],
            "insurance_admin": wss['insurance_admin_fee'],
            "late_penalty": wss['late_penalty'],
            "emergency_response": wss['response_time_emergency'],
            "routine_response": wss['response_time_routine']
        }
    
    # Tribunal fee examples
    try:
        case_fees = conn.execute("""
            SELECT fee_type, amount, frequency
            FROM case_fees
            WHERE factor_registration_number = ?
              AND amount > 0 AND amount < 1000
            ORDER BY fee_type
            LIMIT 6
        """, [pf]).fetchall()
        
        if case_fees:
            context["fees_tribunal"] = [
                {"type": f['fee_type'], "amount": f['amount'], "frequency": f['frequency']}
                for f in case_fees
            ]
    except:
        pass
    
    # Company info
    company = conn.execute("""
        SELECT company_name, company_number, incorporation_date, company_status
        FROM companies_house
        WHERE registration_number = ?
    """, [pf]).fetchone()
    
    if company:
        context["company"] = {
            "name": company['company_name'],
            "number": company['company_number'],
            "incorporated": company['incorporation_date'],
            "status": company['company_status']
        }
    
    return context


def step_8_generate_summaries(skip_ai: bool = False):
    """Generate AI summaries (At a Glance + Tribunal Analysis)."""
    LOG.step(8, "Generate AI Summaries")
    
    if skip_ai:
        LOG.skip("AI generation skipped (--skip-ai)")
        return
    
    if not HAS_VERTEX:
        LOG.warning("Vertex AI not available. Install: pip install google-cloud-aiplatform")
        return
    
    try:
        vertexai.init(project=CONFIG.gcp_project, location=CONFIG.gcp_location)
        model = GenerativeModel(CONFIG.gemini_model)
    except Exception as e:
        LOG.error(f"Failed to initialize Vertex AI: {e}")
        return
    
    with get_db() as conn:
        # Get factors needing summaries
        cursor = conn.execute("""
            SELECT registration_number, name, status, factor_type,
                   tribunal_case_count, tribunal_composite_score, risk_band,
                   google_rating, trustpilot_rating, combined_rating,
                   google_review_count, trustpilot_review_count,
                   property_count
            FROM factors
            WHERE at_a_glance IS NULL
            ORDER BY tribunal_case_count DESC
            LIMIT 50
        """)
        
        factors = [dict(row) for row in cursor]
        
        if not factors:
            LOG.info("No factors need summaries")
            return
        
        LOG.info(f"Generating summaries for {len(factors)} factors...")
        
        generated = 0
        for i, factor in enumerate(factors):
            LOG.progress(i + 1, len(factors), factor['name'][:20])
            
            try:
                # Build rich context
                context = build_factor_context(conn, factor)
                
                prompt = AT_A_GLANCE_PROMPT + json.dumps(context, indent=2, default=str)
                response = model.generate_content(prompt)
                
                # Parse response
                text = response.text.strip()
                if text.startswith('```'):
                    text = text.split('```')[1]
                    if text.startswith('json'):
                        text = text[4:]
                
                result = json.loads(text)
                result['generated_at'] = datetime.now().isoformat()
                result['data_sources'] = list(context.keys())
                
                conn.execute("""
                    UPDATE factors SET
                        at_a_glance = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE registration_number = ?
                """, [json.dumps(result), factor['registration_number']])
                
                generated += 1
                
            except Exception as e:
                LOG.warning(f"Failed to generate summary for {factor['name']}: {e}")
        
        conn.commit()
        LOG.success(f"Generated {generated} At a Glance summaries")


# =============================================================================
# STEP 9: GENERATE STATIC SITE
# =============================================================================

def setup_jinja_env() -> Optional[Environment]:
    """Set up Jinja2 environment with custom filters."""
    if not HAS_JINJA:
        return None
    
    if not CONFIG.template_dir.exists():
        return None
    
    env = Environment(
        loader=FileSystemLoader(CONFIG.template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    
    # Add custom filters
    def has_numbers(value):
        if not value:
            return False
        return bool(re.search(r'\d', str(value)))
    
    env.filters['has_numbers'] = has_numbers
    
    return env


def generate_factor_profiles(conn, env, output_dir: Path) -> int:
    """Generate individual factor profile pages."""
    try:
        template = env.get_template('factor_profile.html')
    except Exception as e:
        LOG.warning(f"factor_profile.html template not found: {e}")
        return 0
    
    cursor = conn.execute("SELECT * FROM v_factor_profiles ORDER BY name")
    factors = [dict(row) for row in cursor]
    
    generated = 0
    for i, profile in enumerate(factors):
        LOG.progress(i + 1, len(factors), profile['name'][:20])
        
        pf = profile['registration_number']
        
        # Get WSS data
        wss = conn.execute(
            "SELECT * FROM wss WHERE registration_number = ?", [pf]
        ).fetchone()
        
        # Get recent reviews (includes aggregate records without individual text)
        reviews = conn.execute("""
            SELECT * FROM reviews 
            WHERE factor_registration_number = ?
            ORDER BY review_count DESC, review_date DESC LIMIT 10
        """, [pf]).fetchall()
        
        # Get Google locations (aggregate records for each place_id)
        # These are the location-level summaries, not individual reviews
        google_locations_raw = conn.execute("""
            SELECT 
                source_id as place_id,
                rating,
                review_count,
                author_name as name
            FROM reviews 
            WHERE factor_registration_number = ?
              AND platform = 'google'
              AND review_text IS NULL
              AND review_count > 1
            ORDER BY review_count DESC
        """, [pf]).fetchall()
        google_locations = [dict(loc) for loc in google_locations_raw]
        
        # Get primary Google place for contact info (highest review count)
        google_places = None
        if google_locations:
            primary = google_locations[0]
            google_places = {
                'place_id': primary.get('place_id'),
                'rating': primary.get('rating'),
                'review_count': primary.get('review_count'),
                'name': primary.get('name'),
            }
        
        # Get Trustpilot URL
        trustpilot = None
        trustpilot_row = conn.execute("""
            SELECT source_id as url, rating, review_count
            FROM reviews 
            WHERE factor_registration_number = ?
              AND platform = 'trustpilot'
              AND review_text IS NULL
            ORDER BY review_count DESC
            LIMIT 1
        """, [pf]).fetchone()
        if trustpilot_row:
            trustpilot = dict(trustpilot_row)
        
        # Get tribunal cases
        cases = conn.execute("""
            SELECT * FROM tribunal_cases 
            WHERE factor_registration_number = ?
            ORDER BY decision_date DESC
        """, [pf]).fetchall()
        
        # Get fee examples from tribunal cases with reasonable bounds
        # Group by fee type and show representative examples
        try:
            # Get all fees, then filter and group in Python
            case_fees_raw = conn.execute("""
                SELECT 
                    cf.fee_type, 
                    cf.amount,
                    cf.frequency,
                    cf.case_reference,
                    tc.pdf_url,
                    tc.decision_date
                FROM case_fees cf
                LEFT JOIN tribunal_cases tc ON cf.case_reference = tc.case_reference
                WHERE cf.factor_registration_number = ?
                  AND cf.amount IS NOT NULL
                  AND cf.amount > 0
                ORDER BY cf.fee_type, cf.amount ASC
            """, [pf]).fetchall()
            
            # Filter outliers with type-specific caps
            fee_caps = {
                'management': 500,
                'late': 100,
                'admin': 200,
                'insurance': 300,
                'transfer': 200,
                'debt': 200,
            }
            default_cap = 1000
            
            # Group fees by type
            fee_groups = {}
            for fee in case_fees_raw:
                fee_type = fee['fee_type'] or 'Other'
                fee_type_lower = fee_type.lower()
                amount = fee['amount'] or 0
                
                # Determine cap for this fee type
                cap = default_cap
                for key, limit in fee_caps.items():
                    if key in fee_type_lower:
                        cap = limit
                        break
                
                # Skip if over cap
                if amount > cap:
                    continue
                
                # Normalize fee type for grouping
                normalized_type = fee_type.replace('_', ' ').title()
                
                if normalized_type not in fee_groups:
                    fee_groups[normalized_type] = {
                        'fee_type': normalized_type,
                        'details': [],
                        'frequencies': set(),
                    }
                
                # Normalize frequency - use intelligent defaults if not provided
                freq = fee['frequency']
                freq_label = None
                
                # If frequency provided, normalize it
                if freq:
                    freq_lower = freq.lower()
                    if 'annual' in freq_lower or 'year' in freq_lower or 'pa' in freq_lower:
                        freq_label = 'per year'
                    elif 'month' in freq_lower:
                        freq_label = 'per month'
                    elif 'quarter' in freq_lower:
                        freq_label = 'per quarter'
                    elif 'one' in freq_lower or 'once' in freq_lower or 'per incident' in freq_lower:
                        freq_label = 'one-time'
                    else:
                        freq_label = freq
                else:
                    # Intelligent defaults based on fee type
                    fee_type_lower = fee_type.lower()
                    if 'management' in fee_type_lower or 'factor' in fee_type_lower:
                        freq_label = 'per year'
                    elif 'insurance' in fee_type_lower:
                        freq_label = 'per year'
                    elif 'late' in fee_type_lower or 'arrears' in fee_type_lower:
                        freq_label = 'per invoice'
                    elif 'admin' in fee_type_lower or 'transfer' in fee_type_lower or 'registration' in fee_type_lower:
                        freq_label = 'one-time'
                    elif 'reserve' in fee_type_lower or 'sinking' in fee_type_lower:
                        freq_label = 'per year'
                    # else: leave as None for unknown types
                
                if freq_label:
                    fee_groups[normalized_type]['frequencies'].add(freq_label)
                
                fee_groups[normalized_type]['details'].append({
                    'amount': amount,
                    'frequency': freq_label,
                    'case_reference': fee['case_reference'],
                    'pdf_url': fee['pdf_url'],
                    'decision_date': fee['decision_date']
                })
            
            # Build final list with ranges and details
            case_fees = []
            for fee_type, data in fee_groups.items():
                details = data['details']
                amounts = [i['amount'] for i in details]
                frequencies = list(data['frequencies'])
                
                # Primary frequency (from data or first item's inferred value)
                freq_label = frequencies[0] if frequencies else (details[0]['frequency'] if details else None)
                
                case_fees.append({
                    'fee_type': fee_type,
                    'amount_min': min(amounts),
                    'amount_max': max(amounts),
                    'frequency': freq_label,
                    'example_count': len(details),
                    'details': details  # Individual fee details for expansion
                })
            
            # Sort by fee type
            case_fees.sort(key=lambda x: x['fee_type'])
                        
        except sqlite3.OperationalError:
            # Table may not exist
            case_fees = []
        
        # Calculate tribunal stats
        cutoff_5y = (datetime.now().year - 5)
        cases_5y = [c for c in cases if c['decision_date'] and c['decision_date'][:4].isdigit() and int(c['decision_date'][:4]) >= cutoff_5y]
        
        tribunal_last_5_years = {
            'case_count': len(cases_5y),
            'pfeo_count': sum(1 for c in cases_5y if c['pfeo_issued']),
            'compensation': sum(c['compensation_awarded'] or 0 for c in cases_5y),
            'rate_per_10k': profile['tribunal_rate_per_10k'],
            'complaints_upheld_pct': None,
        }
        
        if cases_5y:
            total_made = sum(c['complaints_made'] or 0 for c in cases_5y)
            total_upheld = sum(c['complaints_upheld'] or 0 for c in cases_5y)
            if total_made > 0:
                tribunal_last_5_years['complaints_upheld_pct'] = (total_upheld / total_made) * 100
        
        first_case_year = None
        if cases:
            dates = [c['decision_date'] for c in cases if c['decision_date']]
            if dates:
                first_case_year = min(dates)[:4]
        
        tribunal_full_history = {
            'case_count': len(cases),
            'pfeo_count': sum(1 for c in cases if c['pfeo_issued']),
            'compensation': sum(c['compensation_awarded'] or 0 for c in cases),
            'first_case_year': first_case_year,
        }
        
        # Parse at_a_glance JSON
        at_a_glance = None
        if profile['at_a_glance']:
            try:
                at_a_glance = json.loads(profile['at_a_glance'])
            except:
                pass
        
        # Render template
        try:
            html = template.render(
                profile=profile,
                at_a_glance=at_a_glance,
                wss=dict(wss) if wss else None,
                recent_reviews=[dict(r) for r in reviews],
                google_places=google_places,
                google_locations=google_locations,
                trustpilot=trustpilot,
                fee_examples=[],
                fee_summary=None,
                case_fees=case_fees,
                tribunal_last_5_years=tribunal_last_5_years,
                tribunal_full_history=tribunal_full_history,
                cases_by_year=[],
                recent_cases=[dict(c) for c in cases[:5]],
                complaint_categories={},
                similar_factors=[],
                timeline_events=[],
                generated_date=datetime.now().strftime('%Y-%m-%d'),
                reviews_updated=datetime.now().strftime('%Y-%m-%d'),
                tribunal_updated=datetime.now().strftime('%Y-%m-%d'),
            )
            
            factor_dir = output_dir / pf.lower()
            factor_dir.mkdir(parents=True, exist_ok=True)
            
            with open(factor_dir / 'index.html', 'w', encoding='utf-8') as f:
                f.write(html)
            
            generated += 1
        except Exception as e:
            LOG.warning(f"Failed to render {pf}: {e}")
    
    return generated


def get_analysis_period_dates(conn) -> dict:
    """Calculate analysis period dates automatically from the data."""
    result = conn.execute("""
        SELECT MAX(SUBSTR(decision_date, 1, 4)) AS max_year
        FROM tribunal_cases
        WHERE decision_date IS NOT NULL 
          AND decision_date != 'null'
          AND decision_date != ''
          AND SUBSTR(decision_date, 1, 4) GLOB '[0-9][0-9][0-9][0-9]'
    """).fetchone()
    
    # Handle None, empty, or 'null' string values
    max_year = result['max_year'] if result else None
    if max_year and max_year not in ('null', 'None', ''):
        try:
            analysis_end_year = int(max_year)
        except (ValueError, TypeError):
            analysis_end_year = datetime.now().year
    else:
        analysis_end_year = datetime.now().year
    
    now = datetime.now()
    current_quarter = (now.month - 1) // 3 + 1
    last_updated = f"Q{current_quarter} {now.year}"
    
    quarter_months = {
        1: ('April', 0),
        2: ('July', 0),
        3: ('October', 0),
        4: ('January', 1),
    }
    next_month, year_offset = quarter_months[current_quarter]
    next_update = f"{next_month} {now.year + year_offset}"
    
    return {
        'analysis_end_year': analysis_end_year,
        'last_updated': last_updated,
        'next_update': next_update,
    }


def generate_factor_tribunal_pages(conn, env, factors_dir: Path) -> int:
    """Generate tribunal history pages for factors with cases."""
    
    template_path = CONFIG.template_dir / "tribunal_history.html"
    if not template_path.exists():
        LOG.skip("tribunal_history.html template not found")
        return 0
    
    template = env.get_template("tribunal_history.html")
    
    # Get factors with tribunal cases
    cursor = conn.execute("""
        SELECT DISTINCT f.* 
        FROM v_factor_profiles f
        JOIN tribunal_cases t ON f.registration_number = t.factor_registration_number
        ORDER BY f.tribunal_case_count DESC
    """)
    factors_with_cases = [dict(row) for row in cursor]
    
    # Calculate industry-wide stats for peer comparison
    industry_stats = conn.execute("""
        SELECT 
            AVG(tribunal_rate_per_10k) AS avg_case_rate,
            AVG(CAST(tribunal_cases_upheld AS FLOAT) / NULLIF(tribunal_case_count, 0) * 100) AS avg_upheld_rate
        FROM factors
        WHERE tribunal_case_count > 0
    """).fetchone()
    industry_avg_upheld_rate = round(industry_stats['avg_upheld_rate'] or 45, 0)
    
    # Get peer comparison by size bands
    size_bands = {}
    for band_name, (min_props, max_props) in [
        ('0-500', (0, 500)),
        ('500-2,000', (500, 2000)),
        ('2,000-5,000', (2000, 5000)),
        ('5,000-10,000', (5000, 10000)),
        ('10,000+', (10000, 1000000)),
    ]:
        peer_stats = conn.execute("""
            SELECT 
                AVG(tribunal_case_count) AS avg_cases,
                AVG(tribunal_rate_per_10k) AS avg_case_rate,
                AVG(CAST(tribunal_cases_upheld AS FLOAT) / NULLIF(tribunal_case_count, 0) * 100) AS avg_upheld_rate
            FROM factors
            WHERE property_count >= ? AND property_count < ?
              AND tribunal_case_count > 0
        """, [min_props, max_props]).fetchone()
        
        size_bands[band_name] = {
            'avg_cases': round(peer_stats['avg_cases'] or 0, 1),
            'avg_case_rate': round(peer_stats['avg_case_rate'] or 0, 1),
            'avg_upheld_rate': round(peer_stats['avg_upheld_rate'] or 45, 0),
        }
    
    LOG.info(f"Generating {len(factors_with_cases)} tribunal history pages...")
    
    # Get analysis period dates
    period_dates = get_analysis_period_dates(conn)
    
    generated = 0
    for i, factor in enumerate(factors_with_cases):
        LOG.progress(i + 1, len(factors_with_cases), "tribunal")
        
        pf = factor['registration_number'].lower()
        output_dir = factors_dir / pf / "tribunal"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get all cases for this factor
            cases_rows = conn.execute("""
                SELECT * FROM tribunal_cases
                WHERE factor_registration_number = ?
                ORDER BY decision_date DESC
            """, [factor['registration_number']]).fetchall()
            cases = [dict(c) for c in cases_rows]
            
            # Parse JSON fields in cases
            for case in cases:
                if case.get('complaint_categories'):
                    try:
                        case['complaint_categories_list'] = json.loads(case['complaint_categories'])
                    except:
                        case['complaint_categories_list'] = []
                else:
                    case['complaint_categories_list'] = []
            
            # Calculate comprehensive stats
            total_cases = len(cases)
            upheld_count = sum(1 for c in cases if c.get('outcome') in ['Upheld', 'Partially Upheld'] or (c.get('complaints_upheld') and c.get('complaints_upheld') > 0))
            upheld_rate = round((upheld_count / total_cases * 100) if total_cases > 0 else 0, 0)
            
            pfeo_count = sum(1 for c in cases if c.get('pfeo_issued'))
            pfeo_rate = round((pfeo_count / total_cases * 100) if total_cases > 0 else 0, 0)
            
            total_compensation = sum(c.get('compensation_awarded') or 0 for c in cases)
            
            # Attendance rate
            attended = sum(1 for c in cases if c.get('factor_attended') in ['yes', 'Yes', True, 1])
            attendance_rate = round((attended / total_cases * 100) if total_cases > 0 else 0, 0)
            
            # Case rate (from factor profile)
            case_rate = factor.get('tribunal_rate_per_10k') or 0
            risk_band = factor.get('risk_band') or 'UNKNOWN'
            
            # Most recent case date
            most_recent_case_date = None
            if cases and cases[0].get('decision_date'):
                try:
                    dt = datetime.strptime(cases[0]['decision_date'][:10], '%Y-%m-%d')
                    most_recent_case_date = dt.strftime('%B %Y')
                except:
                    most_recent_case_date = cases[0]['decision_date'][:7]
            
            # Outcomes breakdown
            outcome_counts = {}
            for c in cases:
                outcome = c.get('outcome') or 'Unknown'
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            outcomes = []
            for outcome_type in ['Upheld', 'Partially Upheld', 'Dismissed', 'Withdrawn']:
                count = outcome_counts.get(outcome_type, 0)
                outcomes.append({
                    'type': outcome_type,
                    'count': count,
                    'percentage': round((count / total_cases * 100) if total_cases > 0 else 0, 0),
                })
            
            # Cases by year (just counts for the chart)
            cases_by_year = {}
            for case in cases:
                year = case['decision_date'][:4] if case.get('decision_date') else 'Unknown'
                if year != 'Unknown':
                    cases_by_year[year] = cases_by_year.get(year, 0) + 1
            cases_by_year = dict(sorted(cases_by_year.items()))
            
            # Complaint categories aggregation
            all_categories = []
            for case in cases:
                all_categories.extend(case.get('complaint_categories_list', []))
            
            category_counts = {}
            for cat in all_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            complaint_categories = []
            if category_counts:
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:8]:
                    complaint_categories.append({
                        'name': cat,
                        'count': count,
                        'percentage': round((count / total_cases * 100) if total_cases > 0 else 0, 0),
                    })
            
            # Code breaches aggregation
            section_descriptions = {
                '1.1': 'Written Statement of Services',
                '2.1': 'Communication standards',
                '2.5': 'Responding to enquiries',
                '3.1': 'Financial obligations transparency',
                '3.3': 'Clear breakdown of charges',
                '4.1': 'Fair debt recovery',
                '5.3': 'Insurance disclosure',
                '6.1': 'Carrying out repairs',
                '7.1': 'Complaints procedure',
            }
            all_breaches = []
            for case in cases:
                breaches = case.get('code_breaches')
                if breaches:
                    if isinstance(breaches, str):
                        try:
                            breaches = json.loads(breaches)
                        except:
                            breaches = []
                    if isinstance(breaches, list):
                        for b in breaches:
                            section = b.get('section', str(b)) if isinstance(b, dict) else str(b)
                            all_breaches.append(section)
            
            breach_counts = {}
            for section in all_breaches:
                breach_counts[section] = breach_counts.get(section, 0) + 1
            
            code_breaches = []
            for section, count in sorted(breach_counts.items(), key=lambda x: -x[1])[:8]:
                code_breaches.append({
                    'section': section,
                    'description': section_descriptions.get(section, 'Code of Conduct breach'),
                    'count': count,
                })
            
            # Compensation breakdown
            comp_ranges = [
                ('£0', 0, 0),
                ('£1–500', 1, 500),
                ('£501–1,000', 501, 1000),
                ('£1,000+', 1001, 1000000),
            ]
            compensation_breakdown = []
            for label, min_val, max_val in comp_ranges:
                count = sum(1 for c in cases if min_val <= (c.get('compensation_awarded') or 0) <= max_val)
                if count > 0 or min_val == 0:
                    compensation_breakdown.append({'range': label, 'count': count})
            
            # Largest award
            largest_award = None
            max_comp = 0
            for c in cases:
                comp = c.get('compensation_awarded') or 0
                if comp > max_comp:
                    max_comp = comp
                    largest_award = {
                        'amount': comp,
                        'case_ref': c.get('case_reference', 'Unknown'),
                        'pdf_url': c.get('pdf_url', ''),
                    }
            
            # Notable cases (top 3 by severity or compensation)
            notable_cases = []
            scored_cases = [(c, c.get('severity_score') or 0) for c in cases]
            scored_cases.sort(key=lambda x: (-x[1], -(x[0].get('compensation_awarded') or 0)))
            for c, score in scored_cases[:3]:
                if score >= 5 or (c.get('compensation_awarded') or 0) >= 500 or c.get('pfeo_issued'):
                    notable_cases.append({
                        'case_ref': c.get('case_reference', 'Unknown'),
                        'hearing_date': c.get('decision_date', ''),
                        'outcome_type': c.get('outcome', 'Unknown'),
                        'severity_score': c.get('severity_score'),
                        'key_quote': c.get('key_quote', ''),
                        'pfeo_issued': c.get('pfeo_issued', False),
                        'compensation_awarded': c.get('compensation_awarded', 0),
                        'pdf_url': c.get('pdf_url', ''),
                    })
            
            # Peer comparison
            properties_managed = factor.get('property_count') or 0
            peer_band = None
            for band_name, (min_props, max_props) in [
                ('0-500', (0, 500)),
                ('500-2,000', (500, 2000)),
                ('2,000-5,000', (2000, 5000)),
                ('5,000-10,000', (5000, 10000)),
                ('10,000+', (10000, 1000000)),
            ]:
                if min_props <= properties_managed < max_props:
                    peer_band = band_name
                    break
            
            peer_comparison = None
            case_rate_percentile = 50  # Default
            if peer_band and peer_band in size_bands:
                peer_data = size_bands[peer_band]
                if upheld_rate > peer_data['avg_upheld_rate'] + 10:
                    relative = 'above'
                elif upheld_rate < peer_data['avg_upheld_rate'] - 10:
                    relative = 'below'
                else:
                    relative = 'around'
                
                peer_comparison = {
                    'size_band': peer_band,
                    'avg_cases': peer_data['avg_cases'],
                    'avg_upheld_rate': peer_data['avg_upheld_rate'],
                    'relative': relative,
                }
                
                # Calculate case_rate_percentile based on peer comparison
                # Higher than 120% of peer avg = 75th+ percentile (higher than average)
                # Lower than 80% of peer avg = 25th- percentile (lower than average)
                # Otherwise = around average (25-75th)
                peer_avg_cases = peer_data['avg_cases']
                if peer_avg_cases > 0:
                    if total_cases > peer_avg_cases * 1.2:
                        case_rate_percentile = 75  # Higher than average
                    elif total_cases < peer_avg_cases * 0.8:
                        case_rate_percentile = 25  # Lower than average
                    else:
                        case_rate_percentile = 50  # Around average
            
            # Build factor dict with slug
            factor_data = dict(factor)
            factor_data['slug'] = factor['registration_number'].lower()
            
            # Build comprehensive stats dict
            stats = {
                'total_cases': total_cases,
                'upheld_count': upheld_count,
                'upheld_rate': int(upheld_rate),
                'pfeo_count': pfeo_count,
                'pfeo_rate': int(pfeo_rate),
                'total_compensation': total_compensation,
                'attendance_rate': int(attendance_rate),
                'case_rate': round(case_rate, 1),
                'risk_band': risk_band,
                'most_recent_case_date': most_recent_case_date,
                'analysis_end_year': period_dates['analysis_end_year'],
                'last_updated': period_dates['last_updated'],
                'next_update': period_dates['next_update'],
                'outcomes': outcomes,
                'cases_by_year': cases_by_year,
                'complaint_categories': complaint_categories,
                'categorized_cases': len([c for c in cases if c.get('complaint_categories_list')]),
                'code_breaches': code_breaches,
                'compensation_breakdown': compensation_breakdown,
                'largest_award': largest_award,
                'peer_comparison': peer_comparison,
                'case_rate_percentile': case_rate_percentile,
            }
            
            # Normalize case field names for template
            for case in cases:
                case['case_ref'] = case.get('case_reference', '')
                case['hearing_date'] = case.get('decision_date', '')
                case['outcome_type'] = case.get('outcome', 'Unknown')
            
            html = template.render(
                factor=factor_data,
                cases=cases,
                stats=stats,
                notable_cases=notable_cases,
                current_year=datetime.now().year,
            )
            
            with open(output_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html)
            
            generated += 1
            
        except Exception as e:
            LOG.warning(f"Failed to generate tribunal page {pf}: {e}")
    
    return generated


def generate_case_pages(conn, env, tribunal_dir: Path) -> int:
    """Generate individual tribunal case pages."""
    
    template_path = CONFIG.template_dir / "tribunal_case.html"
    if not template_path.exists():
        LOG.skip("tribunal_case.html template not found")
        return 0
    
    template = env.get_template("tribunal_case.html")
    
    cursor = conn.execute("""
        SELECT t.*, f.name AS factor_name, f.risk_band, f.registration_number AS factor_reg
        FROM tribunal_cases t
        LEFT JOIN factors f ON t.factor_registration_number = f.registration_number
        ORDER BY t.decision_date DESC
    """)
    cases = cursor.fetchall()
    
    LOG.info(f"Generating {len(cases)} individual case pages...")
    
    # Section descriptions for code breaches
    section_descriptions = {
        '1.1': 'Written Statement of Services',
        '2.1': 'Communication standards',
        '2.5': 'Responding to enquiries',
        '3.1': 'Financial obligations transparency',
        '3.3': 'Clear breakdown of charges',
        '4.1': 'Fair debt recovery',
        '5.3': 'Insurance disclosure',
        '6.1': 'Carrying out repairs',
        '7.1': 'Complaints procedure',
    }
    
    generated = 0
    for i, case_row in enumerate(cases):
        LOG.progress(i + 1, len(cases), "cases")
        
        case = dict(case_row)
        
        # Create slug from case reference - skip if None
        case_ref = case.get('case_reference')
        if not case_ref:
            continue
            
        slug = re.sub(r'[^a-z0-9]+', '-', case_ref.lower()).strip('-')
        if not slug:
            continue
        
        output_dir = tribunal_dir / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Normalize field names for template - ensure no None values for string fields
            case['case_ref'] = case_ref
            case['hearing_date'] = case.get('decision_date') or ''
            case['outcome'] = case.get('outcome') or 'Unknown'
            case['outcome_type'] = case.get('outcome') or 'Unknown'  # Template uses both
            case['pdf_url'] = case.get('pdf_url') or ''  # Leave empty if no PDF - template handles fallback
            case['factor_slug'] = (case.get('factor_reg') or '').lower()
            case['factor_name'] = case.get('factor_name') or 'Unknown Factor'
            case['risk_band'] = case.get('risk_band') or ''
            case['summary'] = case.get('summary') or ''
            case['key_quote'] = case.get('key_quote') or ''
            
            # Parse complaint categories - template iterates over case.complaint_categories
            categories = []
            if case.get('complaint_categories'):
                try:
                    parsed = json.loads(case['complaint_categories'])
                    if isinstance(parsed, list):
                        categories = parsed
                except:
                    pass
            case['complaint_categories'] = categories  # Overwrite raw JSON with parsed list
            
            # Parse code breaches - template iterates over case.code_breaches
            breaches = []
            if case.get('code_breaches'):
                try:
                    parsed = json.loads(case['code_breaches'])
                    if isinstance(parsed, list):
                        # Normalize to list of strings for simple template iteration
                        for b in parsed:
                            if isinstance(b, dict):
                                section = b.get('section', '')
                                desc = b.get('description', '')
                                breaches.append(f"Section {section}: {desc}" if desc else f"Section {section}")
                            else:
                                breaches.append(str(b))
                except:
                    pass
            case['code_breaches'] = breaches  # Overwrite raw JSON with parsed list
            
            html = template.render(
                case=case,
                current_year=datetime.now().year,
            )
            
            with open(output_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html)
            
            generated += 1
            
        except Exception as e:
            LOG.warning(f"Failed to generate case page {case_ref}: {e}")
    
    return generated


def generate_factors_directory(conn, factors_dir: Path) -> int:
    """Generate the main factors directory index page with all factors."""
    
    cursor = conn.execute("""
        SELECT 
            f.registration_number, f.name, f.status, f.factor_type, f.risk_band,
            f.tribunal_case_count, f.tribunal_pfeo_count, f.property_count,
            f.google_rating, f.google_review_count, f.trustpilot_rating,
            f.trustpilot_review_count, f.combined_rating, f.tpi_member
        FROM factors f ORDER BY f.name
    """)
    factors = [dict(row) for row in cursor]
    
    # Stats for default view (active commercial only)
    default_factors = [f for f in factors if f['status'] == 'registered' and f['factor_type'] not in ('Housing Association', 'Local Authority')]
    total_default = len(default_factors)
    total_properties_default = sum(f['property_count'] or 0 for f in default_factors)
    with_reviews_default = sum(1 for f in default_factors if f['google_rating'] or f['trustpilot_rating'])
    
    total_all = len(factors)
    expired_count = sum(1 for f in factors if f['status'] != 'registered')
    rsl_council_count = sum(1 for f in factors if f['factor_type'] in ('Housing Association', 'Local Authority'))
    
    def fmt_num(n):
        if n is None: return '—'
        if n >= 1000000: return f'{n/1000000:.1f}M'
        if n >= 1000: return f'{n/1000:.0f}K'
        return str(int(n))
    
    def get_stars(rating):
        if not rating: return ''
        full = int(rating)
        half = 1 if rating - full >= 0.5 else 0
        return '★' * full + ('½' if half else '') + '☆' * (5 - full - half)
    
    badge_classes = {'RED': 'badge-red', 'ORANGE': 'badge-orange', 'AMBER': 'badge-amber', 'GREEN': 'badge-green', 'CLEAN': 'badge-clean'}
    
    cards = []
    for f in factors:
        pf = f['registration_number'].lower()
        name = f['name'] or 'Unknown'
        status = f['status'] or 'registered'
        factor_type = f['factor_type'] or 'Commercial'
        is_rsl = 1 if factor_type in ('Housing Association', 'Local Authority') else 0
        is_expired = 1 if status != 'registered' else 0
        band = f['risk_band'] or 'CLEAN'
        props = f['property_count'] or 0
        tribunal = f['tribunal_case_count'] or 0
        pfeo = f['tribunal_pfeo_count'] or 0
        g_rating, g_count = f['google_rating'], f['google_review_count'] or 0
        tp_rating, tp_count = f['trustpilot_rating'], f['trustpilot_review_count'] or 0
        combined = f['combined_rating'] or 0
        tpi = f['tpi_member']
        
        reviews_html = ''
        if tp_rating:
            reviews_html += f'<div class="review-item"><svg class="review-icon" viewBox="0 0 24 24" fill="#00b67a"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg><span class="stars">{get_stars(tp_rating)}</span><span class="rating-num">{tp_rating:.1f}</span><span class="review-count">({tp_count:,})</span></div>'
        if g_rating:
            reviews_html += f'<div class="review-item"><svg class="review-icon" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg><span class="stars">{get_stars(g_rating)}</span><span class="rating-num">{g_rating:.1f}</span><span class="review-count">({g_count:,})</span></div>'
        if not reviews_html:
            reviews_html = '<span class="no-reviews">No reviews yet</span>'
        
        badges = f'<span class="badge {badge_classes.get(band, "badge-clean")}">{band}</span>'
        if is_expired: badges += '<span class="badge badge-expired">Expired</span>'
        if is_rsl: badges += f'<span class="badge badge-rsl">{factor_type[:3].upper()}</span>'
        if tpi: badges += '<span class="badge badge-tpi">TPI</span>'
        
        cards.append(f'''
            <div class="factor-card" data-name="{name.lower()}" data-risk="{band}" data-properties="{props}" data-rating="{combined}" data-tribunal="{tribunal}" data-expired="{is_expired}" data-rsl="{is_rsl}">
                <div class="factor-card-header">
                    <div><h3 class="factor-name"><a href="/factors/{pf}/">{name}</a></h3><div class="factor-reg">{f['registration_number']}</div></div>
                    <div class="factor-badges">{badges}</div>
                </div>
                <div class="factor-stats">
                    <div class="factor-stat"><div class="factor-stat-value">{fmt_num(props)}</div><div class="factor-stat-label">Properties</div></div>
                    <div class="factor-stat"><div class="factor-stat-value">{tribunal}</div><div class="factor-stat-label">Tribunal Cases</div></div>
                    <div class="factor-stat"><div class="factor-stat-value">{pfeo}</div><div class="factor-stat-label">PFEOs</div></div>
                </div>
                <div class="reviews-row">{reviews_html}</div>
                <div class="factor-footer"><span class="factor-meta"></span><a href="/factors/{pf}/" class="factor-link">View Profile →</a></div>
            </div>''')
    
    generated_date = datetime.now().strftime('%B %Y')
    
    html = f'''<!DOCTYPE html>
<html lang="en-GB">
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-P9QSNCJEBQ"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-P9QSNCJEBQ');</script>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" href="/favicon-48x48.png" sizes="48x48">
<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">
<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="/site.webmanifest">
<title>All Property Factors in Scotland | Compare Factors Scotland</title>
<meta name="description" content="Compare {total_all} registered property factors in Scotland. View ratings, tribunal records, and find the right factor for your property.">
<link rel="canonical" href="https://comparefactors.co.uk/factors/">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}
</style>
</head>
<body>
{SHARED_HEADER}
<div class="breadcrumb"><a href="/">Home</a><span class="breadcrumb-sep">›</span><span>All Factors</span></div>
<div class="page-header"><h1 class="page-title">All Property Factors</h1><p class="page-subtitle">Compare registered property factors serving Scottish homeowners</p></div>
<div class="main-content">
<div class="stats-bar"><div class="stat-item"><div class="stat-value" id="statFactors">{total_default}</div><div class="stat-label">Active Factors</div></div><div class="stat-item"><div class="stat-value" id="statProperties">{fmt_num(total_properties_default)}</div><div class="stat-label">Properties Managed</div></div><div class="stat-item"><div class="stat-value" id="statReviews">{with_reviews_default}</div><div class="stat-label">With Reviews</div></div></div>
<div class="filters-bar"><input type="text" class="search-input" id="searchInput" placeholder="Search by name..."><div class="filter-group"><label class="filter-label" for="riskFilter">Tribunal Risk:</label><select class="filter-select" id="riskFilter"><option value="">All</option><option value="CLEAN">Clean Record</option><option value="GREEN">Low Risk</option><option value="AMBER">Some Concerns</option><option value="ORANGE">Significant Issues</option><option value="RED">Serious Record</option></select></div><div class="filter-group"><label class="filter-label" for="sortFilter">Sort by:</label><select class="filter-select" id="sortFilter"><option value="name">Name (A-Z)</option><option value="properties">Properties (most)</option><option value="rating">Rating (highest)</option><option value="tribunal">Tribunal Cases (fewest)</option></select></div><div class="filter-divider"></div><label class="filter-checkbox"><input type="checkbox" id="includeExpired">Include expired ({expired_count})</label><label class="filter-checkbox"><input type="checkbox" id="includeRsl">Include RSL / Council ({rsl_council_count})</label></div>
<div class="results-count" id="resultsCount">Showing <strong>{total_default}</strong> factors</div>
<div class="factor-grid" id="factorGrid">{''.join(cards)}</div>
</div>
{get_shared_footer(generated_date)}
<script>
const searchInput=document.getElementById('searchInput'),riskFilter=document.getElementById('riskFilter'),sortFilter=document.getElementById('sortFilter'),includeExpired=document.getElementById('includeExpired'),includeRsl=document.getElementById('includeRsl'),grid=document.getElementById('factorGrid'),cards=Array.from(grid.querySelectorAll('.factor-card')),resultsCount=document.getElementById('resultsCount');
function formatNumber(n){{if(n>=1000000)return(n/1000000).toFixed(1)+'M';if(n>=1000)return Math.round(n/1000)+'K';return n.toString()}}
function filterAndSort(){{const search=searchInput.value.toLowerCase(),risk=riskFilter.value,sort=sortFilter.value,showExpired=includeExpired.checked,showRsl=includeRsl.checked;let visibleCount=0,totalProps=0,withReviews=0;cards.forEach(card=>{{const name=card.dataset.name,cardRisk=card.dataset.risk,isExpired=card.dataset.expired==='1',isRsl=card.dataset.rsl==='1',props=parseInt(card.dataset.properties)||0,rating=parseFloat(card.dataset.rating)||0,visible=name.includes(search)&&(!risk||cardRisk===risk)&&(showExpired||!isExpired)&&(showRsl||!isRsl);card.style.display=visible?'':'none';if(visible){{visibleCount++;totalProps+=props;if(rating>0)withReviews++}}}});document.getElementById('statFactors').textContent=visibleCount;document.getElementById('statProperties').textContent=formatNumber(totalProps);document.getElementById('statReviews').textContent=withReviews;resultsCount.innerHTML='Showing <strong>'+visibleCount+'</strong> factors';const visibleCards=cards.filter(c=>c.style.display!=='none');visibleCards.sort((a,b)=>{{switch(sort){{case'properties':return(parseInt(b.dataset.properties)||0)-(parseInt(a.dataset.properties)||0);case'rating':return(parseFloat(b.dataset.rating)||0)-(parseFloat(a.dataset.rating)||0);case'tribunal':return(parseInt(a.dataset.tribunal)||0)-(parseInt(b.dataset.tribunal)||0);default:return a.dataset.name.localeCompare(b.dataset.name)}}}});visibleCards.forEach(card=>grid.appendChild(card))}}
filterAndSort();searchInput.addEventListener('input',filterAndSort);riskFilter.addEventListener('change',filterAndSort);sortFilter.addEventListener('change',filterAndSort);includeExpired.addEventListener('change',filterAndSort);includeRsl.addEventListener('change',filterAndSort);
</script>
{SHARED_SCRIPTS}
</body>
</html>'''
    
    factors_dir.mkdir(parents=True, exist_ok=True)
    with open(factors_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    return total_all


# ============================================================================
# SHARED HTML COMPONENTS (matching base.html)
# ============================================================================

SHARED_CSS = '''
:root{--navy-950:#0a0f1a;--navy-900:#0f172a;--navy-800:#1e293b;--navy-700:#334155;--slate-600:#475569;--slate-500:#64748b;--slate-400:#94a3b8;--slate-300:#cbd5e1;--slate-200:#e2e8f0;--slate-100:#f1f5f9;--slate-50:#f8fafc;--white:#ffffff;--red-700:#b91c1c;--red-600:#dc2626;--red-100:#fee2e2;--orange-600:#ea580c;--orange-100:#ffedd5;--amber-600:#d97706;--amber-100:#fef3c7;--green-700:#15803d;--green-600:#16a34a;--green-100:#dcfce7;--green-50:#f0fdf4;--blue-700:#1d4ed8;--blue-600:#2563eb;--blue-100:#dbeafe;--font-display:'Fraunces',Georgia,serif;--font-body:'Source Sans 3',-apple-system,BlinkMacSystemFont,sans-serif;--space-xs:0.25rem;--space-sm:0.5rem;--space-md:1rem;--space-lg:1.5rem;--space-xl:2rem;--space-2xl:3rem}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font-body);font-size:16px;line-height:1.6;color:var(--navy-800);background:var(--slate-50);-webkit-font-smoothing:antialiased}
a{color:var(--blue-600);text-decoration:none}a:hover{text-decoration:underline}
.site-header{background:var(--white);border-bottom:1px solid var(--slate-200);position:sticky;top:0;z-index:100}
.header-inner{max-width:1200px;margin:0 auto;padding:0 24px;height:68px;display:flex;align-items:center;justify-content:space-between}
.logo{display:flex;align-items:center;gap:10px;font-family:var(--font-display);font-size:1.25rem;font-weight:600;color:var(--navy-950);text-decoration:none}.logo:hover{text-decoration:none}
.logo-mark{width:32px;height:32px;background:var(--navy-950);border-radius:6px;display:flex;align-items:center;justify-content:center;color:white;font-size:0.75rem;font-weight:700}
.nav{display:flex;align-items:center;gap:24px}
.nav-link{font-size:0.9rem;font-weight:500;color:var(--slate-600);text-decoration:none;transition:color 0.15s}.nav-link:hover{color:var(--navy-950);text-decoration:none}
.nav-cta{background:var(--navy-950);color:white;padding:8px 16px;border-radius:6px;font-size:0.9rem;font-weight:600;text-decoration:none;transition:background 0.15s}.nav-cta:hover{background:var(--navy-800);text-decoration:none}
.burger{display:none;flex-direction:column;justify-content:center;gap:5px;width:28px;height:28px;background:none;border:none;cursor:pointer;padding:0}
.burger span{display:block;width:100%;height:2px;background:var(--navy-800);border-radius:2px;transition:all 0.3s}
.burger.active span:nth-child(1){transform:rotate(45deg) translate(5px,5px)}.burger.active span:nth-child(2){opacity:0}.burger.active span:nth-child(3){transform:rotate(-45deg) translate(5px,-5px)}
.mobile-nav{display:none;position:absolute;top:68px;left:0;right:0;background:var(--white);border-bottom:1px solid var(--slate-200);padding:16px 24px;box-shadow:0 4px 12px rgba(0,0,0,0.1)}.mobile-nav.active{display:block}
.mobile-nav a{display:block;padding:12px 0;color:var(--slate-600);text-decoration:none;font-size:1rem;font-weight:500;border-bottom:1px solid var(--slate-100)}.mobile-nav a:last-child{border-bottom:none}.mobile-nav a:hover{color:var(--navy-900)}
.breadcrumb{max-width:1200px;margin:0 auto;padding:16px 24px;font-size:0.85rem;color:var(--slate-500)}.breadcrumb a{color:var(--slate-500)}.breadcrumb-sep{margin:0 8px;color:var(--slate-300)}
.page-header{max-width:1200px;margin:0 auto;padding:0 24px 32px}.page-title{font-family:var(--font-display);font-size:2.25rem;font-weight:700;color:var(--navy-950);margin:0 0 8px 0}.page-subtitle{font-size:1.1rem;color:var(--slate-600);margin:0}
.main-content{max-width:1200px;margin:0 auto;padding:0 24px 64px}
.stats-bar{background:var(--white);border:1px solid var(--slate-200);border-radius:12px;padding:20px 28px;margin-bottom:24px;display:flex;gap:40px;flex-wrap:wrap}.stat-value{font-family:var(--font-display);font-size:1.75rem;font-weight:700;color:var(--navy-950);line-height:1}.stat-label{font-size:0.8rem;color:var(--slate-500);margin-top:4px}
.filters-bar{background:var(--white);border:1px solid var(--slate-200);border-radius:12px;padding:16px 20px;margin-bottom:24px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.filter-group{display:flex;align-items:center;gap:8px}.filter-label{font-size:0.85rem;color:var(--slate-600);font-weight:500}.filter-select{padding:8px 12px;border:1px solid var(--slate-200);border-radius:6px;font-size:0.9rem;background:white;cursor:pointer}.filter-select:focus{outline:none;border-color:var(--blue-600)}
.search-input{flex:1;min-width:200px;padding:8px 12px;border:1px solid var(--slate-200);border-radius:6px;font-size:0.9rem}.search-input:focus{outline:none;border-color:var(--blue-600)}
.filter-checkbox{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:0.85rem;color:var(--slate-600);user-select:none}.filter-checkbox input{width:16px;height:16px;cursor:pointer;accent-color:var(--blue-600)}
.filter-divider{width:1px;height:24px;background:var(--slate-200);margin:0 8px}
.results-count{font-size:0.9rem;color:var(--slate-500);margin-bottom:16px}.results-count strong{color:var(--slate-700)}
.factor-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:20px}
.factor-card{background:var(--white);border:1px solid var(--slate-200);border-radius:12px;padding:20px;transition:border-color 0.15s,box-shadow 0.15s}.factor-card:hover{border-color:var(--slate-300);box-shadow:0 4px 12px rgba(0,0,0,0.05)}
.factor-card-header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:12px}
.factor-name{font-family:var(--font-display);font-size:1.1rem;font-weight:600;color:var(--navy-950);margin:0;line-height:1.3}.factor-name a{color:inherit}.factor-name a:hover{text-decoration:underline}
.factor-reg{font-size:0.75rem;color:var(--slate-400);margin-top:2px}.factor-badges{display:flex;flex-wrap:wrap;gap:4px}
.badge{display:inline-flex;align-items:center;padding:3px 8px;border-radius:4px;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.02em;white-space:nowrap}
.badge-red{background:var(--red-100);color:var(--red-700)}.badge-orange{background:var(--orange-100);color:var(--orange-600)}.badge-amber{background:var(--amber-100);color:var(--amber-600)}.badge-green{background:var(--green-100);color:var(--green-700)}.badge-clean{background:var(--green-50);color:var(--green-600)}.badge-expired{background:var(--slate-100);color:var(--slate-500)}.badge-rsl{background:#e0e7ff;color:#4338ca}.badge-tpi{background:var(--blue-100);color:var(--blue-700)}
.factor-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}.factor-stat{text-align:center;padding:8px;background:var(--slate-50);border-radius:6px}.factor-stat-value{font-family:var(--font-display);font-size:1.1rem;font-weight:700;color:var(--navy-950)}.factor-stat-label{font-size:0.7rem;color:var(--slate-500);margin-top:2px}
.reviews-row{display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap}.review-item{display:flex;align-items:center;gap:4px;font-size:0.85rem}.review-icon{width:16px;height:16px}.stars{color:#fbbf24;letter-spacing:-1px}.rating-num{font-weight:600;color:var(--slate-700)}.review-count{color:var(--slate-400);font-size:0.8rem}.no-reviews{color:var(--slate-400);font-size:0.85rem;font-style:italic}
.factor-footer{display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid var(--slate-100)}.factor-meta{font-size:0.8rem;color:var(--slate-400)}.factor-link{font-size:0.85rem;font-weight:600;color:var(--blue-600)}
.area-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:20px}
.area-card{background:var(--white);border:1px solid var(--slate-200);border-radius:12px;padding:24px;transition:border-color 0.15s,box-shadow 0.15s,transform 0.15s;text-decoration:none}.area-card:hover{border-color:var(--slate-300);box-shadow:0 4px 12px rgba(0,0,0,0.08);transform:translateY(-2px);text-decoration:none}
.area-name{font-family:var(--font-display);font-size:1.25rem;font-weight:600;color:var(--navy-950);margin:0 0 16px 0}
.area-stats{display:flex;gap:24px}.area-stat{display:flex;flex-direction:column}.area-stat-value{font-family:var(--font-display);font-size:1.5rem;font-weight:700;color:var(--blue-600)}.area-stat-label{font-size:0.75rem;color:var(--slate-500);text-transform:uppercase;letter-spacing:0.03em}
.site-footer{background:var(--navy-950);color:rgba(255,255,255,0.7);padding:48px 24px 24px;margin-top:var(--space-2xl)}
.footer-inner{max-width:1200px;margin:0 auto}
.footer-grid{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:48px;margin-bottom:32px}
.footer-brand{display:flex;align-items:center;gap:10px;font-family:var(--font-display);font-size:1.25rem;font-weight:600;color:white;margin-bottom:12px}.footer-brand .logo-mark{background:white;color:var(--navy-950)}
.footer-about{font-size:0.9rem;line-height:1.6;color:rgba(255,255,255,0.6)}
.footer-col h4{font-family:var(--font-body);font-weight:600;color:white;font-size:0.9rem;margin-bottom:16px}
.footer-col a{display:block;color:rgba(255,255,255,0.7);font-size:0.9rem;margin-bottom:10px;text-decoration:none}.footer-col a:hover{color:white;text-decoration:none}
.footer-bottom{padding-top:24px;border-top:1px solid rgba(255,255,255,0.1);display:flex;justify-content:space-between;align-items:center;font-size:0.8rem;color:rgba(255,255,255,0.5)}.footer-bottom a{color:rgba(255,255,255,0.5)}.footer-bottom a:hover{color:white}
@media(max-width:768px){.nav{display:none}.burger{display:flex}.footer-grid{grid-template-columns:1fr 1fr;gap:32px}.footer-bottom{flex-direction:column;gap:8px;text-align:center}.stats-bar{gap:24px}.filters-bar{flex-direction:column;align-items:stretch}.filter-divider{display:none}.page-title{font-size:1.75rem}}
@media(max-width:480px){.header-inner{padding:0 16px}.main-content{padding:0 16px var(--space-xl)}.footer-grid{grid-template-columns:1fr;gap:24px}.factor-grid{grid-template-columns:1fr}.area-grid{grid-template-columns:1fr}}
'''

SHARED_HEADER = '''<header class="site-header">
    <div class="header-inner">
        <a href="/" class="logo"><div class="logo-mark">CF</div>Compare Factors</a>
        <nav class="nav">
            <a href="/factors/" class="nav-link">All Factors</a>
            <a href="/areas/" class="nav-link">By Area</a>
            <a href="/methodology/" class="nav-link">How We Score</a>
            <a href="/guides/" class="nav-link">Guides</a>
            <a href="/get-quotes/" class="nav-link nav-cta">Get Quotes</a>
        </nav>
        <button class="burger" onclick="toggleMenu()" aria-label="Menu"><span></span><span></span><span></span></button>
    </div>
    <div class="mobile-nav" id="mobileNav">
        <a href="/factors/">All Factors</a>
        <a href="/areas/">By Area</a>
        <a href="/methodology/">How We Score</a>
        <a href="/guides/">Guides</a>
        <a href="/get-quotes/" class="nav-cta">Get Quotes</a>
    </div>
</header>'''

def get_shared_footer(generated_date: str) -> str:
    return f'''<footer class="site-footer">
    <div class="footer-inner">
        <div class="footer-grid">
            <div class="footer-col">
                <div class="footer-brand"><div class="logo-mark">CF</div>Compare Factors</div>
                <p class="footer-about">Helping Scottish homeowners make informed decisions with transparent, independent data.</p>
            </div>
            <div class="footer-col">
                <h4>Explore</h4>
                <a href="/factors/">All Factors</a>
                <a href="/areas/">By Area</a>
                <a href="/rankings/">Rankings</a>
                <a href="/tribunal/">Tribunal Cases</a>
            </div>
            <div class="footer-col">
                <h4>Guides</h4>
                <a href="/guides/how-to-switch-factors-scotland/">Switching Factors</a>
                <a href="/guides/complaints/">Making a Complaint</a>
                <a href="/guides/history-of-property-factoring-scotland/">History</a>
            </div>
            <div class="footer-col">
                <h4>About</h4>
                <a href="/methodology/">Our Methodology</a>
                <a href="/about/">About Us</a>
                <a href="/contact/">Contact</a>
                <a href="/privacy/">Privacy Policy</a>
            </div>
        </div>
        <div class="footer-bottom">
            <div>© 2026 Compare Factors. Not affiliated with the Scottish Government.</div>
            <div><a href="/privacy/">Privacy</a> · <a href="/terms/">Terms</a></div>
        </div>
    </div>
</footer>'''

SHARED_SCRIPTS = '''<script>function toggleMenu(){document.querySelector('.burger').classList.toggle('active');document.getElementById('mobileNav').classList.toggle('active')}</script>'''


# Area definitions: slug -> {name, postcodes}
AREA_DEFINITIONS = {
    'glasgow': {'name': 'Glasgow', 'postcodes': ['G1', 'G2', 'G3', 'G4', 'G5', 'G11', 'G12', 'G13', 'G14', 'G15', 'G20', 'G21', 'G22', 'G23', 'G31', 'G32', 'G33', 'G34', 'G40', 'G41', 'G42', 'G43', 'G44', 'G45', 'G46', 'G51', 'G52', 'G53', 'G60', 'G61', 'G62', 'G63', 'G64', 'G65', 'G66', 'G67', 'G68', 'G69', 'G71', 'G72', 'G73', 'G74', 'G75', 'G76', 'G77', 'G78', 'G79', 'G81', 'G82', 'G83', 'G84']},
    'edinburgh': {'name': 'Edinburgh', 'postcodes': ['EH1', 'EH2', 'EH3', 'EH4', 'EH5', 'EH6', 'EH7', 'EH8', 'EH9', 'EH10', 'EH11', 'EH12', 'EH13', 'EH14', 'EH15', 'EH16', 'EH17']},
    'lothians': {'name': 'Lothians', 'postcodes': ['EH18', 'EH19', 'EH20', 'EH21', 'EH22', 'EH23', 'EH24', 'EH25', 'EH26', 'EH27', 'EH28', 'EH29', 'EH30', 'EH31', 'EH32', 'EH33', 'EH34', 'EH35', 'EH36', 'EH37', 'EH38', 'EH39', 'EH40', 'EH41', 'EH42', 'EH43', 'EH44', 'EH45', 'EH46', 'EH47', 'EH48', 'EH49', 'EH51', 'EH52', 'EH53', 'EH54', 'EH55']},
    'aberdeen': {'name': 'Aberdeen', 'postcodes': ['AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB21', 'AB22', 'AB23', 'AB24', 'AB25']},
    'aberdeenshire': {'name': 'Aberdeenshire', 'postcodes': ['AB30', 'AB31', 'AB32', 'AB33', 'AB34', 'AB35', 'AB36', 'AB37', 'AB38', 'AB39', 'AB41', 'AB42', 'AB43', 'AB44', 'AB45', 'AB51', 'AB52', 'AB53', 'AB54', 'AB55', 'AB56']},
    'dundee': {'name': 'Dundee', 'postcodes': ['DD1', 'DD2', 'DD3', 'DD4', 'DD5']},
    'angus': {'name': 'Angus & Tayside', 'postcodes': ['DD6', 'DD7', 'DD8', 'DD9', 'DD10', 'DD11']},
    'fife': {'name': 'Fife', 'postcodes': ['KY1', 'KY2', 'KY3', 'KY4', 'KY5', 'KY6', 'KY7', 'KY8', 'KY9', 'KY10', 'KY11', 'KY12', 'KY13', 'KY14', 'KY15', 'KY16']},
    'stirling': {'name': 'Stirling & Falkirk', 'postcodes': ['FK1', 'FK2', 'FK3', 'FK4', 'FK5', 'FK6', 'FK7', 'FK8', 'FK9', 'FK10', 'FK11', 'FK12', 'FK13', 'FK14', 'FK15', 'FK16', 'FK17', 'FK18', 'FK19', 'FK20', 'FK21']},
    'paisley': {'name': 'Paisley & Renfrewshire', 'postcodes': ['PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7', 'PA8', 'PA9', 'PA10', 'PA11', 'PA12', 'PA13', 'PA14', 'PA15', 'PA16', 'PA17', 'PA18', 'PA19']},
    'ayrshire': {'name': 'Ayrshire', 'postcodes': ['KA1', 'KA2', 'KA3', 'KA4', 'KA5', 'KA6', 'KA7', 'KA8', 'KA9', 'KA10', 'KA11', 'KA12', 'KA13', 'KA14', 'KA15', 'KA16', 'KA17', 'KA18', 'KA19', 'KA20', 'KA21', 'KA22', 'KA23', 'KA24', 'KA25', 'KA26', 'KA27', 'KA28', 'KA29', 'KA30']},
    'inverness': {'name': 'Inverness & Highlands', 'postcodes': ['IV1', 'IV2', 'IV3', 'IV4', 'IV5', 'IV6', 'IV7', 'IV8', 'IV9', 'IV10', 'IV11', 'IV12', 'IV13', 'IV14', 'IV15', 'IV16', 'IV17', 'IV18', 'IV19', 'IV20', 'IV21', 'IV22', 'IV23', 'IV24', 'IV25', 'IV26', 'IV27', 'IV28', 'IV30', 'IV31', 'IV32', 'IV36', 'IV40', 'IV41', 'IV42', 'IV43', 'IV44', 'IV45', 'IV46', 'IV47', 'IV48', 'IV49', 'IV51', 'IV52', 'IV53', 'IV54', 'IV55', 'IV56', 'IV63']},
    'perth': {'name': 'Perth & Kinross', 'postcodes': ['PH1', 'PH2', 'PH3', 'PH4', 'PH5', 'PH6', 'PH7', 'PH8', 'PH9', 'PH10', 'PH11', 'PH12', 'PH13', 'PH14', 'PH15', 'PH16', 'PH17', 'PH18']},
    'borders': {'name': 'Scottish Borders', 'postcodes': ['TD1', 'TD2', 'TD3', 'TD4', 'TD5', 'TD6', 'TD7', 'TD8', 'TD9', 'TD10', 'TD11', 'TD12', 'TD13', 'TD14', 'TD15']},
    'dumfries': {'name': 'Dumfries & Galloway', 'postcodes': ['DG1', 'DG2', 'DG3', 'DG4', 'DG5', 'DG6', 'DG7', 'DG8', 'DG9', 'DG10', 'DG11', 'DG12', 'DG13', 'DG14', 'DG16']},
    'lanarkshire': {'name': 'Lanarkshire', 'postcodes': ['ML1', 'ML2', 'ML3', 'ML4', 'ML5', 'ML6', 'ML7', 'ML8', 'ML9', 'ML10', 'ML11', 'ML12']},
}


def generate_area_pages(conn, areas_dir: Path) -> int:
    """Generate individual area pages like /areas/glasgow/."""
    
    # Get ALL factors with postcode data (filtered client-side)
    cursor = conn.execute("""
        SELECT 
            f.registration_number, f.name, f.status, f.factor_type, f.risk_band,
            f.tribunal_case_count, f.tribunal_pfeo_count, f.property_count,
            f.google_rating, f.google_review_count, f.trustpilot_rating,
            f.trustpilot_review_count, f.combined_rating, f.tpi_member,
            f.postcode_areas
        FROM factors f
        WHERE f.postcode_areas IS NOT NULL AND f.postcode_areas != ''
        ORDER BY f.name
    """)
    all_factors = [dict(row) for row in cursor]
    
    def fmt_num(n):
        if n is None: return '—'
        if n >= 1000000: return f'{n/1000000:.1f}M'
        if n >= 1000: return f'{n/1000:.0f}K'
        return str(int(n))
    
    def get_stars(rating):
        if not rating: return ''
        full = int(rating)
        half = 1 if rating - full >= 0.5 else 0
        return '★' * full + ('½' if half else '') + '☆' * (5 - full - half)
    
    badge_classes = {'RED': 'badge-red', 'ORANGE': 'badge-orange', 'AMBER': 'badge-amber', 'GREEN': 'badge-green', 'CLEAN': 'badge-clean'}
    generated_date = datetime.now().strftime('%B %Y')
    generated = 0
    
    for slug, area_info in AREA_DEFINITIONS.items():
        area_name = area_info['name']
        area_postcodes = set(area_info['postcodes'])
        
        # Find ALL factors serving this area
        area_factors = []
        for f in all_factors:
            factor_postcodes = set(p.strip() for p in (f['postcode_areas'] or '').split(','))
            if factor_postcodes & area_postcodes:  # Any overlap
                area_factors.append(f)
        
        if not area_factors:
            continue
        
        # Stats for default view (active commercial only)
        default_factors = [f for f in area_factors if f['status'] == 'registered' and f['factor_type'] not in ('Housing Association', 'Local Authority')]
        total_default = len(default_factors)
        total_properties_default = sum(f['property_count'] or 0 for f in default_factors)
        with_reviews_default = sum(1 for f in default_factors if f['google_rating'] or f['trustpilot_rating'])
        
        # Counts for toggles
        expired_count = sum(1 for f in area_factors if f['status'] != 'registered')
        rsl_count = sum(1 for f in area_factors if f['factor_type'] in ('Housing Association', 'Local Authority'))
        
        # Generate cards for ALL factors
        cards = []
        for f in area_factors:
            pf = f['registration_number'].lower()
            name = f['name'] or 'Unknown'
            status = f['status'] or 'registered'
            factor_type = f['factor_type'] or 'Commercial'
            is_rsl = 1 if factor_type in ('Housing Association', 'Local Authority') else 0
            is_expired = 1 if status != 'registered' else 0
            band = f['risk_band'] or 'CLEAN'
            props = f['property_count'] or 0
            tribunal = f['tribunal_case_count'] or 0
            pfeo = f['tribunal_pfeo_count'] or 0
            g_rating, g_count = f['google_rating'], f['google_review_count'] or 0
            tp_rating, tp_count = f['trustpilot_rating'], f['trustpilot_review_count'] or 0
            combined = f['combined_rating'] or 0
            tpi = f['tpi_member']
            
            reviews_html = ''
            if tp_rating:
                reviews_html += f'<div class="review-item"><svg class="review-icon" viewBox="0 0 24 24" fill="#00b67a"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg><span class="stars">{get_stars(tp_rating)}</span><span class="rating-num">{tp_rating:.1f}</span><span class="review-count">({tp_count:,})</span></div>'
            if g_rating:
                reviews_html += f'<div class="review-item"><svg class="review-icon" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg><span class="stars">{get_stars(g_rating)}</span><span class="rating-num">{g_rating:.1f}</span><span class="review-count">({g_count:,})</span></div>'
            if not reviews_html:
                reviews_html = '<span class="no-reviews">No reviews yet</span>'
            
            badges = f'<span class="badge {badge_classes.get(band, "badge-clean")}">{band}</span>'
            if is_expired: badges += '<span class="badge badge-expired">Expired</span>'
            if is_rsl: badges += f'<span class="badge badge-rsl">{factor_type[:3].upper()}</span>'
            if tpi: badges += '<span class="badge badge-tpi">TPI</span>'
            
            cards.append(f'''
            <div class="factor-card" data-name="{name.lower()}" data-risk="{band}" data-properties="{props}" data-rating="{combined}" data-tribunal="{tribunal}" data-expired="{is_expired}" data-rsl="{is_rsl}">
                <div class="factor-card-header">
                    <div><h3 class="factor-name"><a href="/factors/{pf}/">{name}</a></h3><div class="factor-reg">{f['registration_number']}</div></div>
                    <div class="factor-badges">{badges}</div>
                </div>
                <div class="factor-stats">
                    <div class="factor-stat"><div class="factor-stat-value">{fmt_num(props)}</div><div class="factor-stat-label">Properties</div></div>
                    <div class="factor-stat"><div class="factor-stat-value">{tribunal}</div><div class="factor-stat-label">Tribunal Cases</div></div>
                    <div class="factor-stat"><div class="factor-stat-value">{pfeo}</div><div class="factor-stat-label">PFEOs</div></div>
                </div>
                <div class="reviews-row">{reviews_html}</div>
                <div class="factor-footer"><span class="factor-meta"></span><a href="/factors/{pf}/" class="factor-link">View Profile →</a></div>
            </div>''')
        
        html = f'''<!DOCTYPE html>
<html lang="en-GB">
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-P9QSNCJEBQ"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-P9QSNCJEBQ');</script>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" href="/favicon-48x48.png" sizes="48x48">
<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">
<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="/site.webmanifest">
<title>Property Factors in {area_name} | Compare Factors Scotland</title>
<meta name="description" content="Compare {total_default} property factors serving {area_name}. View ratings, tribunal records, and reviews.">
<link rel="canonical" href="https://comparefactors.co.uk/areas/{slug}/">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}
</style>
</head>
<body>
{SHARED_HEADER}
<div class="breadcrumb"><a href="/">Home</a><span class="breadcrumb-sep">›</span><a href="/areas/">Areas</a><span class="breadcrumb-sep">›</span><span>{area_name}</span></div>
<div class="page-header"><h1 class="page-title">Property Factors in {area_name}</h1><p class="page-subtitle">Compare factors serving the {area_name} area</p></div>
<div class="main-content">
<div class="stats-bar"><div class="stat-item"><div class="stat-value" id="statFactors">{total_default}</div><div class="stat-label">Active Factors</div></div><div class="stat-item"><div class="stat-value" id="statProperties">{fmt_num(total_properties_default)}</div><div class="stat-label">Properties Managed</div></div><div class="stat-item"><div class="stat-value" id="statReviews">{with_reviews_default}</div><div class="stat-label">With Reviews</div></div></div>
<div class="filters-bar"><input type="text" class="search-input" id="searchInput" placeholder="Search by name..."><div class="filter-group"><label class="filter-label" for="riskFilter">Tribunal Risk:</label><select class="filter-select" id="riskFilter"><option value="">All</option><option value="CLEAN">Clean Record</option><option value="GREEN">Low Risk</option><option value="AMBER">Some Concerns</option><option value="ORANGE">Significant Issues</option><option value="RED">Serious Record</option></select></div><div class="filter-group"><label class="filter-label" for="sortFilter">Sort by:</label><select class="filter-select" id="sortFilter"><option value="name">Name (A-Z)</option><option value="properties">Properties (most)</option><option value="rating">Rating (highest)</option><option value="tribunal">Tribunal Cases (fewest)</option></select></div><div class="filter-divider"></div><label class="filter-checkbox"><input type="checkbox" id="includeExpired">Include expired ({expired_count})</label><label class="filter-checkbox"><input type="checkbox" id="includeRsl">Include RSL / Council ({rsl_count})</label></div>
<div class="results-count" id="resultsCount">Showing <strong>{total_default}</strong> factors</div>
<div class="factor-grid" id="factorGrid">{''.join(cards)}</div>
</div>
{get_shared_footer(generated_date)}
<script>
const searchInput=document.getElementById('searchInput'),riskFilter=document.getElementById('riskFilter'),sortFilter=document.getElementById('sortFilter'),includeExpired=document.getElementById('includeExpired'),includeRsl=document.getElementById('includeRsl'),grid=document.getElementById('factorGrid'),cards=Array.from(grid.querySelectorAll('.factor-card')),resultsCount=document.getElementById('resultsCount');
function formatNumber(n){{if(n>=1000000)return(n/1000000).toFixed(1)+'M';if(n>=1000)return Math.round(n/1000)+'K';return n.toString()}}
function filterAndSort(){{const search=searchInput.value.toLowerCase(),risk=riskFilter.value,sort=sortFilter.value,showExpired=includeExpired.checked,showRsl=includeRsl.checked;let visibleCount=0,totalProps=0,withReviews=0;cards.forEach(card=>{{const name=card.dataset.name,cardRisk=card.dataset.risk,isExpired=card.dataset.expired==='1',isRsl=card.dataset.rsl==='1',props=parseInt(card.dataset.properties)||0,rating=parseFloat(card.dataset.rating)||0,visible=name.includes(search)&&(!risk||cardRisk===risk)&&(showExpired||!isExpired)&&(showRsl||!isRsl);card.style.display=visible?'':'none';if(visible){{visibleCount++;totalProps+=props;if(rating>0)withReviews++}}}});document.getElementById('statFactors').textContent=visibleCount;document.getElementById('statProperties').textContent=formatNumber(totalProps);document.getElementById('statReviews').textContent=withReviews;resultsCount.innerHTML='Showing <strong>'+visibleCount+'</strong> factors';const visibleCards=cards.filter(c=>c.style.display!=='none');visibleCards.sort((a,b)=>{{switch(sort){{case'properties':return(parseInt(b.dataset.properties)||0)-(parseInt(a.dataset.properties)||0);case'rating':return(parseFloat(b.dataset.rating)||0)-(parseFloat(a.dataset.rating)||0);case'tribunal':return(parseInt(a.dataset.tribunal)||0)-(parseInt(b.dataset.tribunal)||0);default:return a.dataset.name.localeCompare(b.dataset.name)}}}});visibleCards.forEach(card=>grid.appendChild(card))}}
filterAndSort();searchInput.addEventListener('input',filterAndSort);riskFilter.addEventListener('change',filterAndSort);sortFilter.addEventListener('change',filterAndSort);includeExpired.addEventListener('change',filterAndSort);includeRsl.addEventListener('change',filterAndSort);
</script>
{SHARED_SCRIPTS}
</body>
</html>'''
        
        # Write area page
        area_dir = areas_dir / slug
        area_dir.mkdir(parents=True, exist_ok=True)
        with open(area_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html)
        generated += 1
    
    return generated


def generate_areas_index(conn, areas_dir: Path) -> int:
    """Generate the areas index page listing all areas."""
    
    # Get all active commercial factors with postcode data
    cursor = conn.execute("""
        SELECT registration_number, name, postcode_areas, property_count, combined_rating
        FROM factors
        WHERE status = 'registered'
        AND (factor_type IS NULL OR factor_type NOT IN ('Housing Association', 'Local Authority'))
        AND postcode_areas IS NOT NULL AND postcode_areas != ''
    """)
    all_factors = [dict(row) for row in cursor]
    
    # Calculate stats per area
    area_stats = []
    for slug, area_info in AREA_DEFINITIONS.items():
        area_postcodes = set(area_info['postcodes'])
        factors_in_area = []
        for f in all_factors:
            factor_postcodes = set(p.strip() for p in (f['postcode_areas'] or '').split(','))
            if factor_postcodes & area_postcodes:
                factors_in_area.append(f)
        
        if factors_in_area:
            area_stats.append({
                'slug': slug,
                'name': area_info['name'],
                'factor_count': len(factors_in_area),
                'property_count': sum(f['property_count'] or 0 for f in factors_in_area),
            })
    
    # Sort by factor count descending
    area_stats.sort(key=lambda x: x['factor_count'], reverse=True)
    
    def fmt_num(n):
        if n is None: return '—'
        if n >= 1000000: return f'{n/1000000:.1f}M'
        if n >= 1000: return f'{n/1000:.0f}K'
        return str(int(n))
    
    generated_date = datetime.now().strftime('%B %Y')
    
    # Generate area cards
    cards = []
    for a in area_stats:
        cards.append(f'''
        <a href="/areas/{a['slug']}/" class="area-card">
            <h3 class="area-name">{a['name']}</h3>
            <div class="area-stats">
                <div class="area-stat"><span class="area-stat-value">{a['factor_count']}</span><span class="area-stat-label">Factors</span></div>
                <div class="area-stat"><span class="area-stat-value">{fmt_num(a['property_count'])}</span><span class="area-stat-label">Properties</span></div>
            </div>
        </a>''')
    
    html = f'''<!DOCTYPE html>
<html lang="en-GB">
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-P9QSNCJEBQ"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-P9QSNCJEBQ');</script>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" href="/favicon-48x48.png" sizes="48x48">
<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">
<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="/site.webmanifest">
<title>Property Factors by Area | Compare Factors Scotland</title>
<meta name="description" content="Find property factors by area in Scotland. Browse factors serving Glasgow, Edinburgh, Aberdeen, Dundee and more.">
<link rel="canonical" href="https://comparefactors.co.uk/areas/">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}
</style>
</head>
<body>
{SHARED_HEADER}
<div class="breadcrumb"><a href="/">Home</a><span class="breadcrumb-sep">›</span><span>Areas</span></div>
<div class="page-header"><h1 class="page-title">Property Factors by Area</h1><p class="page-subtitle">Find factors serving your area of Scotland</p></div>
<div class="main-content">
<div class="area-grid">{''.join(cards)}</div>
</div>
{get_shared_footer(generated_date)}
{SHARED_SCRIPTS}
</body>
</html>'''
    
    areas_dir.mkdir(parents=True, exist_ok=True)
    with open(areas_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    return len(area_stats)


def generate_comparison_page(conn, site_dir: Path) -> bool:
    """Generate the /compare/ page for side-by-side factor comparison."""
    import html as html_module
    
    # Get all active commercial factors with postcode data
    cursor = conn.execute('''
        SELECT registration_number, name, status, factor_type, risk_band,
               tribunal_case_count, tribunal_pfeo_count, property_count,
               google_rating, google_review_count, trustpilot_rating,
               trustpilot_review_count, tpi_member, postcode_areas, city
        FROM factors 
        WHERE status = 'registered'
        AND (factor_type IS NULL OR factor_type NOT IN ('Housing Association', 'Local Authority'))
        AND postcode_areas IS NOT NULL AND postcode_areas != ''
        ORDER BY name
    ''')
    factors = [dict(row) for row in cursor]
    
    # Build factor data with area assignments
    factor_data = {}
    area_factors = {slug: [] for slug in AREA_DEFINITIONS}
    
    for f in factors:
        postcodes = set(p.strip() for p in (f['postcode_areas'] or '').split(','))
        
        # Determine which areas this factor serves
        factor_areas = []
        for slug, area_info in AREA_DEFINITIONS.items():
            if postcodes & set(area_info['postcodes']):
                factor_areas.append(slug)
                area_factors[slug].append(f['registration_number'])
        
        # Summarize coverage for display
        areas = set()
        for pc in postcodes:
            if pc.startswith('G'): areas.add('Glasgow')
            elif pc.startswith('EH'): areas.add('Edinburgh/Lothians')
            elif pc.startswith('AB'): areas.add('Aberdeen')
            elif pc.startswith('DD'): areas.add('Dundee/Angus')
            elif pc.startswith('KY'): areas.add('Fife')
            elif pc.startswith('FK'): areas.add('Stirling')
            elif pc.startswith('PA'): areas.add('Renfrewshire')
            elif pc.startswith('KA'): areas.add('Ayrshire')
            elif pc.startswith('IV'): areas.add('Highlands')
            elif pc.startswith('PH'): areas.add('Perth')
            elif pc.startswith('TD'): areas.add('Borders')
            elif pc.startswith('DG'): areas.add('Dumfries')
            elif pc.startswith('ML'): areas.add('Lanarkshire')
        coverage = ', '.join(sorted(areas)[:4])
        if len(areas) > 4:
            coverage += f' +{len(areas)-4} more'
        if not coverage:
            coverage = 'Scotland'
        
        factor_data[f['registration_number']] = {
            'name': f['name'] or 'Unknown',
            'registration_number': f['registration_number'],
            'risk_band': f['risk_band'] or 'CLEAN',
            'tribunal_cases': f['tribunal_case_count'] or 0,
            'pfeo_count': f['tribunal_pfeo_count'] or 0,
            'properties': f['property_count'] or 0,
            'google_rating': f['google_rating'],
            'google_count': f['google_review_count'] or 0,
            'trustpilot_rating': f['trustpilot_rating'],
            'trustpilot_count': f['trustpilot_review_count'] or 0,
            'tpi_member': bool(f['tpi_member']),
            'city': f['city'] or 'Scotland',
            'coverage': coverage,
            'areas': factor_areas
        }
    
    # Build area options sorted by factor count
    area_options = ['<option value="">Select your area first...</option>']
    for slug, info in sorted(AREA_DEFINITIONS.items(), key=lambda x: -len(area_factors[x[0]])):
        count = len(area_factors[slug])
        if count > 0:
            area_options.append(f'<option value="{slug}">{info["name"]} ({count} factors)</option>')
    
    area_options_html = '\n                        '.join(area_options)
    factor_json = json.dumps(factor_data)
    area_factors_json = json.dumps(area_factors)
    generated_date = datetime.now().strftime('%B %Y')
    
    html = f'''<!DOCTYPE html>
<html lang="en-GB">
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-P9QSNCJEBQ"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-P9QSNCJEBQ');</script>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" href="/favicon-48x48.png" sizes="48x48">
<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">
<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="/site.webmanifest">
<title>Compare Factors Side by Side | Compare Factors Scotland</title>
<meta name="description" content="Compare Scottish property factors side-by-side on tribunal record, reviews, coverage areas, and company information.">
<link rel="canonical" href="https://comparefactors.co.uk/compare/">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}
.main{{max-width:1200px;margin:0 auto;padding:0 24px 96px}}
.page-header{{text-align:center;margin-bottom:40px}}
.page-title{{font-family:var(--font-display);font-size:2.25rem;font-weight:700;color:var(--navy-950);margin:0 0 12px 0}}
.page-desc{{font-size:1.1rem;color:var(--slate-600);margin:0;max-width:600px;margin-left:auto;margin-right:auto}}
.selector-section{{background:var(--white);border:1px solid var(--slate-200);border-radius:12px;padding:24px;margin-bottom:32px}}
.selector-title{{font-weight:600;color:var(--navy-950);margin-bottom:16px;font-size:1rem}}
.selector-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
.selector-label{{font-size:0.85rem;color:var(--slate-500);margin-bottom:6px}}
.selector-dropdown{{width:100%;padding:10px 14px;font-size:1rem;font-family:inherit;border:1px solid var(--slate-200);border-radius:8px;background:white;color:var(--slate-700);cursor:pointer}}
.selector-dropdown:focus{{outline:none;border-color:var(--blue-600);box-shadow:0 0 0 3px rgba(37,99,235,0.1)}}
.selector-dropdown:disabled{{background:var(--slate-50);color:var(--slate-400);cursor:not-allowed}}
.area-selector{{margin-bottom:20px}}
.area-selector .selector-label{{font-size:0.9rem;font-weight:500;color:var(--navy-950)}}
.area-dropdown{{max-width:320px}}
.area-hint{{font-size:0.85rem;color:var(--slate-500);margin-top:8px}}
.factor-selectors{{border-top:1px solid var(--slate-100);padding-top:20px}}
.factor-selectors.disabled{{opacity:0.5;pointer-events:none}}
.comparison-wrapper{{overflow-x:auto}}
.comparison-table{{width:100%;border-collapse:collapse;background:var(--white);border-radius:12px;overflow:hidden;border:1px solid var(--slate-200);min-width:700px}}
.comparison-table th,.comparison-table td{{padding:16px 20px;text-align:left;border-bottom:1px solid var(--slate-100)}}
.comparison-table th{{background:var(--slate-50);font-weight:600;color:var(--navy-950)}}
.comparison-table th:first-child{{width:180px}}
.comparison-table tr:last-child td{{border-bottom:none}}
.comparison-table .row-label{{font-weight:500;color:var(--slate-600);background:var(--slate-50)}}
.comparison-table .factor-name{{font-family:var(--font-display);font-weight:600;font-size:1.1rem;color:var(--navy-950)}}
.comparison-table .factor-link{{display:block;font-size:0.85rem;color:var(--blue-600);margin-top:4px}}
.score-badge{{display:inline-block;padding:4px 10px;border-radius:4px;font-size:0.8rem;font-weight:600;text-transform:uppercase}}
.score-clean{{background:var(--green-100);color:var(--green-600)}}.score-green{{background:var(--green-100);color:var(--green-700)}}.score-amber{{background:var(--amber-100);color:var(--amber-600)}}.score-orange{{background:var(--orange-100);color:var(--orange-600)}}.score-red{{background:var(--red-100);color:var(--red-700)}}
.rating-stars{{color:#f59e0b;letter-spacing:-1px}}.rating-value{{font-weight:600;color:var(--slate-700);margin-left:4px}}.rating-count{{font-size:0.85rem;color:var(--slate-500);margin-left:4px}}.no-rating{{color:var(--slate-400);font-style:italic}}
.check-yes{{color:var(--green-600);font-weight:500}}.check-no{{color:var(--slate-400)}}
.empty-state{{text-align:center;padding:80px 40px;background:var(--white);border:2px dashed var(--slate-200);border-radius:12px}}
.empty-icon{{font-size:3rem;margin-bottom:16px}}.empty-title{{font-family:var(--font-display);font-size:1.25rem;font-weight:600;color:var(--navy-950);margin-bottom:8px}}.empty-desc{{color:var(--slate-500);margin:0}}
.back-link{{display:inline-block;margin-top:32px;color:var(--blue-600);font-weight:500}}
@media(max-width:768px){{.selector-grid{{grid-template-columns:1fr}}.page-title{{font-size:1.75rem}}}}
</style>
</head>
<body>
{SHARED_HEADER}
<div class="breadcrumb"><a href="/">Home</a><span class="breadcrumb-sep">›</span><span>Compare Factors</span></div>
<main class="main">
<div class="page-header">
<h1 class="page-title">Compare Factors Side by Side</h1>
<p class="page-desc">Select your area, then choose factors to compare their tribunal records, ratings, and coverage.</p>
</div>
<div class="selector-section">
<div class="area-selector">
<div class="selector-label">Step 1: Select your area</div>
<select class="selector-dropdown area-dropdown" id="areaSelect">
{area_options_html}
</select>
<p class="area-hint" id="areaHint">Choose an area to see factors serving that location</p>
</div>
<div class="factor-selectors disabled" id="factorSelectors">
<div class="selector-title">Step 2: Choose factors to compare</div>
<div class="selector-grid">
<div class="selector-item"><div class="selector-label">Factor 1</div><select class="selector-dropdown" id="factor1" disabled><option value="">Select area first...</option></select></div>
<div class="selector-item"><div class="selector-label">Factor 2</div><select class="selector-dropdown" id="factor2" disabled><option value="">Select area first...</option></select></div>
<div class="selector-item"><div class="selector-label">Factor 3 (optional)</div><select class="selector-dropdown" id="factor3" disabled><option value="">Select area first...</option></select></div>
</div>
</div>
</div>
<div id="emptyState" class="empty-state">
<div class="empty-icon">📍</div>
<div class="empty-title">Start by selecting your area</div>
<p class="empty-desc">Choose your area above to see a filtered list of factors serving that location.</p>
</div>
<div id="comparisonTable" class="comparison-wrapper" style="display:none;">
<table class="comparison-table">
<thead><tr><th></th><th id="header1">Factor 1</th><th id="header2">Factor 2</th><th id="header3" style="display:none;">Factor 3</th></tr></thead>
<tbody id="comparisonBody"></tbody>
</table>
</div>
<a href="/factors/" class="back-link">← Browse all factors</a>
</main>
{get_shared_footer(generated_date)}
<script>
const factorData={factor_json};
const areaFactors={area_factors_json};
function getStars(r){{if(!r)return'';const f=Math.floor(r),h=r-f>=0.5?1:0;return'★'.repeat(f)+(h?'½':'')+'☆'.repeat(5-f-h)}}
function getRiskBadgeClass(r){{return{{'CLEAN':'score-clean','GREEN':'score-green','AMBER':'score-amber','ORANGE':'score-orange','RED':'score-red'}}[r]||'score-clean'}}
function formatRating(r,c){{if(!r)return'<span class="no-rating">No reviews</span>';return`<span class="rating-stars">${{getStars(r)}}</span><span class="rating-value">${{r.toFixed(1)}}</span><span class="rating-count">(${{c.toLocaleString()}})</span>`}}
function updateFactorDropdowns(area){{const ids=areaFactors[area]||[],factors=ids.map(id=>factorData[id]).filter(f=>f).sort((a,b)=>a.name.localeCompare(b.name));let opts='<option value="">Select a factor...</option>';for(const f of factors)opts+=`<option value="${{f.registration_number}}">${{f.name}}</option>`;['factor1','factor2','factor3'].forEach(id=>{{const s=document.getElementById(id);s.innerHTML=opts;s.disabled=false}});document.getElementById('factorSelectors').classList.remove('disabled');document.getElementById('areaHint').textContent=`${{factors.length}} factors serve this area`;document.getElementById('emptyState').innerHTML=`<div class="empty-icon">⚖️</div><div class="empty-title">Select factors to compare</div><p class="empty-desc">Choose at least two factors from the ${{factors.length}} available in this area.</p>`}}
function updateComparison(){{const f1=document.getElementById('factor1').value,f2=document.getElementById('factor2').value,f3=document.getElementById('factor3').value,selected=[f1,f2,f3].filter(f=>f&&factorData[f]);if(selected.length<2){{document.getElementById('emptyState').style.display='block';document.getElementById('comparisonTable').style.display='none';return}}document.getElementById('emptyState').style.display='none';document.getElementById('comparisonTable').style.display='block';const factors=selected.map(id=>factorData[id]);document.getElementById('header1').innerHTML=factors[0]?`<div class="factor-name">${{factors[0].name}}</div><a href="/factors/${{factors[0].registration_number.toLowerCase()}}/" class="factor-link">View full profile →</a>`:'';document.getElementById('header2').innerHTML=factors[1]?`<div class="factor-name">${{factors[1].name}}</div><a href="/factors/${{factors[1].registration_number.toLowerCase()}}/" class="factor-link">View full profile →</a>`:'';const h3=document.getElementById('header3');if(factors[2]){{h3.style.display='';h3.innerHTML=`<div class="factor-name">${{factors[2].name}}</div><a href="/factors/${{factors[2].registration_number.toLowerCase()}}/" class="factor-link">View full profile →</a>`}}else{{h3.style.display='none'}}const rows=[['Tribunal risk score',f=>`<span class="score-badge ${{getRiskBadgeClass(f.risk_band)}}">${{f.risk_band}}</span>`],['Tribunal cases',f=>`${{f.tribunal_cases}} case${{f.tribunal_cases!==1?'s':''}}`],['Enforcement orders',f=>f.pfeo_count>0?`<span class="check-no">${{f.pfeo_count}} PFEO${{f.pfeo_count!==1?'s':''}}</span>`:'<span class="check-yes">None</span>'],['Properties managed',f=>f.properties?f.properties.toLocaleString():'—'],['Google rating',f=>formatRating(f.google_rating,f.google_count)],['Trustpilot rating',f=>formatRating(f.trustpilot_rating,f.trustpilot_count)],['TPI Scotland member',f=>f.tpi_member?'<span class="check-yes">✓ Yes</span>':'<span class="check-no">✗ No</span>'],['Headquarters',f=>f.city||'—'],['Coverage',f=>f.coverage||'—']];let tbody='';for(const[label,formatter]of rows){{tbody+=`<tr><td class="row-label">${{label}}</td>`;for(let i=0;i<factors.length;i++){{const style=i===2&&!factors[2]?' style="display:none;"':'';tbody+=`<td${{style}}>${{formatter(factors[i])}}</td>`}}if(factors.length<3)tbody+='<td style="display:none;"></td>';tbody+='</tr>'}}document.getElementById('comparisonBody').innerHTML=tbody}}
document.getElementById('areaSelect').addEventListener('change',function(){{if(this.value){{updateFactorDropdowns(this.value);['factor1','factor2','factor3'].forEach(id=>{{document.getElementById(id).value=''}});updateComparison()}}else{{document.getElementById('factorSelectors').classList.add('disabled');['factor1','factor2','factor3'].forEach(id=>{{const s=document.getElementById(id);s.innerHTML='<option value="">Select area first...</option>';s.disabled=true}});document.getElementById('areaHint').textContent='Choose an area to see factors serving that location';document.getElementById('emptyState').innerHTML=`<div class="empty-icon">📍</div><div class="empty-title">Start by selecting your area</div><p class="empty-desc">Choose your area above to see a filtered list of factors serving that location.</p>`;document.getElementById('emptyState').style.display='block';document.getElementById('comparisonTable').style.display='none'}}}});
document.getElementById('factor1').addEventListener('change',updateComparison);
document.getElementById('factor2').addEventListener('change',updateComparison);
document.getElementById('factor3').addEventListener('change',updateComparison);
</script>
{SHARED_SCRIPTS}
</body>
</html>'''
    
    # Write the comparison page
    compare_dir = site_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    with open(compare_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    return True


def generate_homepage(conn, site_dir: Path) -> bool:
    """Generate the homepage with dynamic counts and tribunal hotspots."""
    
    # Get counts
    active_factors = conn.execute("""
        SELECT COUNT(*) FROM factors 
        WHERE status='registered' 
        AND (factor_type IS NULL OR factor_type NOT IN ('Housing Association', 'Local Authority'))
    """).fetchone()[0]
    
    total_factors = conn.execute("SELECT COUNT(*) FROM factors WHERE status='registered'").fetchone()[0]
    tribunal_cases = conn.execute("SELECT COUNT(*) FROM tribunal_cases").fetchone()[0]
    total_reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    
    # Get top tribunal factors for hotspots
    cursor = conn.execute("""
        SELECT name, registration_number, property_count, tribunal_case_count, risk_band, city
        FROM factors 
        WHERE status='registered' 
        AND tribunal_case_count > 0
        ORDER BY tribunal_case_count DESC 
        LIMIT 5
    """)
    top_factors = [dict(row) for row in cursor]
    
    generated_date = datetime.now().strftime('%B %Y')
    
    # Build tribunal hotspots rows
    def get_initials(name):
        words = name.replace('Ltd', '').replace('Limited', '').split()
        return ''.join(w[0] for w in words[:2] if w).upper()
    
    def fmt_props(n):
        if not n: return '—'
        return f'{n:,}'
    
    hotspot_rows = []
    for f in top_factors:
        initials = get_initials(f['name'])
        name = f['name']
        if len(name) > 35:
            name = name[:32] + '...'
        city = f['city'] or 'Scotland'
        props = fmt_props(f['property_count'])
        cases = f['tribunal_case_count']
        risk = (f['risk_band'] or 'CLEAN').lower()
        reg = f['registration_number'].lower()
        
        hotspot_rows.append(f'''            <div class="table-row">
                <div class="factor-name">
                    <div class="factor-avatar">{initials}</div>
                    <div class="factor-info">
                        <h3>{name}</h3>
                        <span>{city}</span>
                    </div>
                </div>
                <div class="table-cell">{props}</div>
                <div class="table-cell"><strong>{cases}</strong></div>
                <div class="table-cell">
                    <span class="risk-badge risk-badge--{risk}">{risk.title()}</span>
                </div>
                <a href="/factors/{reg}/" class="view-btn">View</a>
            </div>''')
    
    hotspots_html = '\n\n'.join(hotspot_rows)
    
    html = f'''<!DOCTYPE html>
<html lang="en-GB">
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-P9QSNCJEBQ"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-P9QSNCJEBQ');</script>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" href="/favicon-48x48.png" sizes="48x48">
<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">
<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">
<link rel="manifest" href="/site.webmanifest">
<title>Compare Property Factors | Scotland's Independent Factor Directory</title>
<meta name="description" content="Compare {active_factors} Scottish property factors by tribunal record, customer reviews, and regulatory status. Free, independent, no sponsored rankings.">
<link rel="canonical" href="https://comparefactors.co.uk/">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
    --navy-950: #0a0f1a;--navy-900: #0f172a;--navy-800: #1e293b;--navy-700: #334155;
    --slate-600: #475569;--slate-500: #64748b;--slate-400: #94a3b8;--slate-300: #cbd5e1;--slate-200: #e2e8f0;--slate-100: #f1f5f9;--slate-50: #f8fafc;--white: #ffffff;
    --red-700: #b91c1c;--red-600: #dc2626;--red-100: #fee2e2;--red-50: #fef2f2;
    --orange-600: #ea580c;--orange-100: #ffedd5;
    --amber-600: #d97706;--amber-100: #fef3c7;
    --green-700: #15803d;--green-600: #16a34a;--green-100: #dcfce7;--green-50: #f0fdf4;
    --blue-700: #1d4ed8;--blue-600: #2563eb;--blue-100: #dbeafe;
    --indigo-600: #4f46e5;--indigo-100: #e0e7ff;--indigo-50: #eef2ff;
    --font-display: 'Fraunces', Georgia, serif;
    --font-body: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--font-body);background:var(--slate-50);color:var(--navy-800);line-height:1.6;-webkit-font-smoothing:antialiased}}
.site-header{{background:var(--white);border-bottom:1px solid var(--slate-200);position:sticky;top:0;z-index:100}}
.header-inner{{max-width:1200px;margin:0 auto;padding:0 24px;display:flex;align-items:center;justify-content:space-between;height:68px}}
.logo{{font-family:var(--font-display);font-size:1.35rem;font-weight:600;color:var(--navy-900);text-decoration:none;display:flex;align-items:center;gap:10px}}
.logo-mark{{width:36px;height:36px;background:var(--navy-900);border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--white);font-weight:700;font-size:1.1rem;letter-spacing:-0.5px}}
.nav{{display:flex;align-items:center;gap:32px}}
.nav-link{{color:var(--slate-600);text-decoration:none;font-size:0.95rem;font-weight:500;transition:color 0.15s}}
.nav-link:hover{{color:var(--navy-900)}}
.nav-cta{{background:var(--navy-900);color:var(--white);padding:10px 20px;border-radius:8px;font-weight:600;transition:background 0.15s}}
.nav-cta:hover{{background:var(--navy-800);color:var(--white)}}
.burger{{display:none;flex-direction:column;justify-content:center;gap:5px;width:28px;height:28px;background:none;border:none;cursor:pointer;padding:0}}
.burger span{{display:block;width:100%;height:2px;background:var(--navy-900);border-radius:2px;transition:all 0.3s}}
.burger.active span:nth-child(1){{transform:rotate(45deg) translate(5px,5px)}}.burger.active span:nth-child(2){{opacity:0}}.burger.active span:nth-child(3){{transform:rotate(-45deg) translate(5px,-5px)}}
.mobile-nav{{display:none;position:absolute;top:68px;left:0;right:0;background:var(--white);border-bottom:1px solid var(--slate-200);padding:16px 24px;box-shadow:0 4px 12px rgba(0,0,0,0.1)}}
.mobile-nav.active{{display:block}}
.mobile-nav a{{display:block;padding:12px 0;color:var(--slate-700);text-decoration:none;font-size:1rem;font-weight:500;border-bottom:1px solid var(--slate-100)}}
.mobile-nav a:last-child{{border-bottom:none}}.mobile-nav a:hover{{color:var(--navy-900)}}.mobile-nav .nav-cta{{display:inline-block;margin-top:12px;text-align:center}}
@media(max-width:768px){{.nav{{display:none}}.burger{{display:flex}}}}

.hero{{background:linear-gradient(135deg,var(--navy-950) 0%,var(--navy-900) 100%);color:var(--white);padding:80px 24px 100px;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")}}
.hero-inner{{max-width:1200px;margin:0 auto;position:relative;z-index:1}}
.hero-badge{{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);padding:8px 16px;border-radius:100px;font-size:0.85rem;margin-bottom:24px}}
.hero-badge svg{{width:16px;height:16px}}
.hero-title{{font-family:var(--font-display);font-size:clamp(2.5rem,5vw,3.75rem);font-weight:700;line-height:1.1;margin-bottom:20px;max-width:800px}}
.hero-subtitle{{font-size:1.25rem;color:rgba(255,255,255,0.8);max-width:600px;margin-bottom:40px;line-height:1.5}}
.search-box{{background:var(--white);border-radius:12px;padding:8px;display:flex;gap:8px;max-width:500px;box-shadow:0 4px 24px rgba(0,0,0,0.2)}}
.search-input{{flex:1;border:none;padding:12px 16px;font-size:1rem;font-family:inherit;background:transparent;outline:none}}
.search-btn{{background:var(--navy-900);color:var(--white);border:none;padding:12px 24px;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;transition:background 0.15s;display:flex;align-items:center;gap:8px}}
.search-btn:hover{{background:var(--navy-800)}}
.hero-stats{{display:flex;gap:48px;margin-top:48px;padding-top:32px;border-top:1px solid rgba(255,255,255,0.1)}}
.hero-stat{{text-align:left}}
.hero-stat-value{{font-family:var(--font-display);font-size:2rem;font-weight:700;color:var(--white)}}
.hero-stat-label{{font-size:0.9rem;color:rgba(255,255,255,0.6);margin-top:4px}}
@media(max-width:768px){{.hero{{padding:60px 24px 80px}}.hero-stats{{flex-wrap:wrap;gap:24px}}.hero-stat{{flex:1;min-width:120px}}.search-box{{flex-direction:column}}.search-btn{{justify-content:center}}}}

.tribunal{{background:var(--white);padding:80px 24px}}
.tribunal-inner{{max-width:1200px;margin:0 auto}}
.section-header{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:32px;flex-wrap:wrap;gap:16px}}
.section-title{{font-family:var(--font-display);font-size:2rem;font-weight:700;color:var(--navy-950)}}
.section-subtitle{{color:var(--slate-500);margin-top:4px}}
.section-link{{color:var(--blue-600);font-weight:500;text-decoration:none;display:flex;align-items:center;gap:4px}}
.section-link:hover{{text-decoration:underline}}
.table-container{{border:1px solid var(--slate-200);border-radius:12px;overflow:hidden}}
.table-header{{display:grid;grid-template-columns:2fr 1fr 1fr 1fr 80px;gap:16px;padding:16px 20px;background:var(--slate-50);font-size:0.85rem;font-weight:600;color:var(--slate-500);text-transform:uppercase;letter-spacing:0.5px}}
.table-row{{display:grid;grid-template-columns:2fr 1fr 1fr 1fr 80px;gap:16px;padding:16px 20px;align-items:center;border-bottom:1px solid var(--slate-100)}}
.table-row:last-of-type{{border-bottom:none}}
.factor-name{{display:flex;align-items:center;gap:12px}}
.factor-avatar{{width:40px;height:40px;background:var(--navy-900);border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--white);font-weight:600;font-size:0.85rem;flex-shrink:0}}
.factor-info h3{{font-size:0.95rem;font-weight:600;color:var(--navy-900);margin-bottom:2px}}
.factor-info span{{font-size:0.8rem;color:var(--slate-500)}}
.table-cell{{font-size:0.95rem;color:var(--slate-700)}}
.risk-badge{{display:inline-block;padding:4px 10px;border-radius:4px;font-size:0.75rem;font-weight:600;text-transform:uppercase}}
.risk-badge--red{{background:var(--red-100);color:var(--red-700)}}.risk-badge--orange{{background:var(--orange-100);color:var(--orange-600)}}.risk-badge--amber{{background:var(--amber-100);color:var(--amber-600)}}.risk-badge--green{{background:var(--green-100);color:var(--green-700)}}.risk-badge--clean{{background:var(--green-100);color:var(--green-600)}}
.view-btn{{background:var(--slate-100);color:var(--navy-900);padding:8px 16px;border-radius:6px;font-size:0.85rem;font-weight:500;text-decoration:none;text-align:center;transition:background 0.15s}}
.view-btn:hover{{background:var(--slate-200)}}
.table-footer{{padding:16px 20px;background:var(--slate-50);text-align:center}}
.table-footer a{{color:var(--blue-600);font-weight:500;text-decoration:none}}
.table-footer a:hover{{text-decoration:underline}}
@media(max-width:900px){{.table-header{{display:none}}.table-row{{grid-template-columns:1fr auto;gap:12px;padding:16px}}.table-row .table-cell:nth-child(2),.table-row .table-cell:nth-child(4){{display:none}}.factor-info h3{{font-size:0.9rem}}}}

.problems{{background:var(--slate-50);padding:80px 24px}}
.problems-inner{{max-width:1200px;margin:0 auto}}
.problems-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}}
.problem-card{{background:var(--white);border:1px solid var(--slate-200);border-radius:12px;padding:24px;display:flex;gap:16px}}
.problem-icon{{font-size:1.5rem;flex-shrink:0}}
.problem-content h3{{font-size:1rem;font-weight:600;color:var(--navy-900);margin-bottom:8px}}
.problem-content p{{font-size:0.95rem;color:var(--slate-600);margin-bottom:12px}}
.problem-stat{{font-size:0.85rem;color:var(--blue-600);font-weight:500;padding:8px 12px;background:var(--blue-100);border-radius:6px}}
@media(max-width:768px){{.problems-grid{{grid-template-columns:1fr}}}}

.why{{background:var(--navy-950);color:var(--white);padding:80px 24px}}
.why-inner{{max-width:800px;margin:0 auto;text-align:center}}
.why-title{{font-family:var(--font-display);font-size:2rem;font-weight:700;margin-bottom:16px}}
.why-text{{color:rgba(255,255,255,0.8);font-size:1.1rem;line-height:1.6;margin-bottom:40px}}
.why-sources{{display:flex;justify-content:center;gap:32px;flex-wrap:wrap}}
.source-item{{display:flex;align-items:center;gap:10px;color:rgba(255,255,255,0.9);font-size:0.95rem}}
.source-icon{{font-size:1.25rem}}

.cta{{background:linear-gradient(135deg,var(--indigo-50) 0%,var(--blue-100) 100%);padding:80px 24px}}
.cta-inner{{max-width:700px;margin:0 auto;text-align:center}}
.cta-title{{font-family:var(--font-display);font-size:2rem;font-weight:700;color:var(--navy-950);margin-bottom:16px}}
.cta-text{{color:var(--slate-600);font-size:1.1rem;margin-bottom:32px}}
.cta-buttons{{display:flex;justify-content:center;gap:16px;flex-wrap:wrap}}
.btn{{display:inline-flex;align-items:center;gap:8px;padding:14px 28px;border-radius:8px;font-size:1rem;font-weight:600;text-decoration:none;transition:all 0.15s}}
.btn--primary{{background:var(--navy-900);color:var(--white)}}.btn--primary:hover{{background:var(--navy-800)}}
.btn--secondary{{background:var(--white);color:var(--navy-900);border:1px solid var(--slate-200)}}.btn--secondary:hover{{border-color:var(--slate-300);background:var(--slate-50)}}

.footer{{background:var(--navy-950);color:rgba(255,255,255,0.7);padding:64px 24px 32px}}
.footer-inner{{max-width:1200px;margin:0 auto}}
.footer-grid{{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:48px;margin-bottom:48px}}
.footer-brand p{{margin-top:16px;font-size:0.9rem;line-height:1.6;color:rgba(255,255,255,0.6)}}
.footer-title{{font-weight:600;color:var(--white);margin-bottom:16px;font-size:0.9rem}}
.footer-links{{list-style:none}}
.footer-links li{{margin-bottom:10px}}
.footer-links a{{color:rgba(255,255,255,0.7);text-decoration:none;font-size:0.9rem;transition:color 0.15s}}
.footer-links a:hover{{color:var(--white)}}
.footer-bottom{{padding-top:32px;border-top:1px solid rgba(255,255,255,0.1);display:flex;justify-content:space-between;font-size:0.85rem;color:rgba(255,255,255,0.5)}}
@media(max-width:768px){{.footer-grid{{grid-template-columns:1fr 1fr;gap:32px}}.footer-bottom{{flex-direction:column;gap:8px;text-align:center}}}}
@media(max-width:480px){{.footer-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header class="site-header">
    <div class="header-inner">
        <a href="/" class="logo"><div class="logo-mark">CF</div>Compare Factors</a>
        <nav class="nav">
            <a href="/factors/" class="nav-link">All Factors</a>
            <a href="/areas/" class="nav-link">By Area</a>
            <a href="/methodology/" class="nav-link">How We Score</a>
            <a href="/guides/" class="nav-link">Guides</a>
            <a href="/get-quotes/" class="nav-link nav-cta">Get Quotes</a>
        </nav>
        <button class="burger" onclick="toggleMenu()" aria-label="Menu"><span></span><span></span><span></span></button>
    </div>
    <div class="mobile-nav" id="mobileNav">
        <a href="/factors/">All Factors</a>
        <a href="/areas/">By Area</a>
        <a href="/methodology/">How We Score</a>
        <a href="/guides/">Guides</a>
        <a href="/get-quotes/" class="nav-cta">Get Quotes</a>
    </div>
</header>

<section class="hero">
    <div class="hero-inner">
        <div class="hero-badge">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            Independent &amp; Transparent
        </div>
        <h1 class="hero-title">Compare Scotland's property factors</h1>
        <p class="hero-subtitle">Search {active_factors} factors by tribunal record, reviews, and regulatory status. No sponsored rankings. No hidden agendas.</p>
        <div class="search-box">
            <input type="text" class="search-input" placeholder="Enter your postcode or factor name...">
            <button class="search-btn">
                Search
                <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </button>
        </div>
        <div class="hero-stats">
            <div class="hero-stat"><div class="hero-stat-value">{active_factors}</div><div class="hero-stat-label">Factors tracked</div></div>
            <div class="hero-stat"><div class="hero-stat-value">{tribunal_cases:,}</div><div class="hero-stat-label">Tribunal decisions</div></div>
            <div class="hero-stat"><div class="hero-stat-value">{total_reviews:,}</div><div class="hero-stat-label">Reviews analysed</div></div>
        </div>
    </div>
</section>

<section class="tribunal">
    <div class="tribunal-inner">
        <div class="section-header">
            <div>
                <h2 class="section-title">Tribunal hotspots</h2>
                <p class="section-subtitle">Factors with the most Housing & Property Chamber cases</p>
            </div>
            <a href="/factors/?sort=tribunal" class="section-link">View all by tribunal record <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7"/></svg></a>
        </div>
        <div class="table-container">
            <div class="table-header">
                <div>Factor</div>
                <div>Properties</div>
                <div>Tribunal Cases</div>
                <div>Risk Level</div>
                <div></div>
            </div>
{hotspots_html}
            <div class="table-footer">
                <a href="/factors/">Browse all {active_factors} factors →</a>
            </div>
        </div>
    </div>
</section>

<section class="problems">
    <div class="problems-inner">
        <div class="section-header">
            <div>
                <h2 class="section-title">Why this matters</h2>
                <p class="section-subtitle">Common issues hidden from most property owners</p>
            </div>
        </div>
        <div class="problems-grid">
            <div class="problem-card">
                <div class="problem-icon">💰</div>
                <div class="problem-content">
                    <h3>Hidden insurance commissions</h3>
                    <p>Most factors take 20–40% commission on buildings insurance without disclosing it. You could be overpaying hundreds of pounds per year.</p>
                    <div class="problem-stat">Ask your factor: "What commission do you take on our insurance?"</div>
                </div>
            </div>
            <div class="problem-card">
                <div class="problem-icon">📋</div>
                <div class="problem-content">
                    <h3>Code of Conduct breaches</h3>
                    <p>Factors must follow a statutory Code of Conduct, but enforcement is complaint-driven. Many violations go unreported.</p>
                    <div class="problem-stat">We've analysed {tribunal_cases:,} tribunal decisions to surface patterns</div>
                </div>
            </div>
            <div class="problem-card">
                <div class="problem-icon">🔒</div>
                <div class="problem-content">
                    <h3>Switching seems impossible</h3>
                    <p>Most owners don't realise they can change factors. If over 50% of owners agree, you can appoint a new factor—often with no notice period.</p>
                    <div class="problem-stat">We can connect you with factors looking for new developments</div>
                </div>
            </div>
            <div class="problem-card">
                <div class="problem-icon">❓</div>
                <div class="problem-content">
                    <h3>No way to compare</h3>
                    <p>The government register tells you who's registered—but nothing about their track record. Until now, there's been no independent comparison.</p>
                    <div class="problem-stat">Compare Factors exists to fix this</div>
                </div>
            </div>
        </div>
    </div>
</section>

<section class="why">
    <div class="why-inner">
        <h2 class="why-title">Where our data comes from</h2>
        <p class="why-text">We combine official government sources with independent research to build the most comprehensive picture of Scotland's property factors. Every score is transparent and explainable.</p>
        <div class="why-sources">
            <div class="source-item"><div class="source-icon">🏛️</div><span>Scottish Property Factor Register</span></div>
            <div class="source-item"><div class="source-icon">⚖️</div><span>Housing & Property Chamber</span></div>
            <div class="source-item"><div class="source-icon">🏢</div><span>Companies House</span></div>
            <div class="source-item"><div class="source-icon">⭐</div><span>Trustpilot & Google Reviews</span></div>
        </div>
    </div>
</section>

<section class="cta">
    <div class="cta-inner">
        <h2 class="cta-title">Ready to switch factors?</h2>
        <p class="cta-text">Most developments can change factor with a simple majority vote. Get free, no-obligation quotes from factors actively looking for new business.</p>
        <div class="cta-buttons">
            <a href="/get-quotes/" class="btn btn--primary">Get Quotes <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7"/></svg></a>
            <a href="/guides/how-to-switch-factors-scotland/" class="btn btn--secondary">How switching works</a>
        </div>
    </div>
</section>

<footer class="footer">
    <div class="footer-inner">
        <div class="footer-grid">
            <div class="footer-brand">
                <a href="/" class="logo" style="color:white;"><div class="logo-mark">CF</div>Compare Factors</a>
                <p>Helping Scottish homeowners make informed decisions with transparent, independent data.</p>
            </div>
            <div>
                <h4 class="footer-title">Explore</h4>
                <ul class="footer-links">
                    <li><a href="/factors/">All Factors</a></li>
                    <li><a href="/areas/">By Area</a></li>
                    <li><a href="/compare/">Compare</a></li>
                    <li><a href="/tribunal/">Tribunal Cases</a></li>
                </ul>
            </div>
            <div>
                <h4 class="footer-title">Guides</h4>
                <ul class="footer-links">
                    <li><a href="/guides/how-to-switch-factors-scotland/">Switching Factors</a></li>
                    <li><a href="/guides/complaints/">Making a Complaint</a></li>
                    <li><a href="/guides/history-of-property-factoring-scotland/">History</a></li>
                </ul>
            </div>
            <div>
                <h4 class="footer-title">About</h4>
                <ul class="footer-links">
                    <li><a href="/methodology/">Our Methodology</a></li>
                    <li><a href="/about/">About Us</a></li>
                    <li><a href="/contact/">Contact</a></li>
                    <li><a href="/privacy/">Privacy Policy</a></li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            <div>© 2026 Compare Factors. Not affiliated with the Scottish Government.</div>
            <div>Data last updated: {generated_date}</div>
        </div>
    </div>
</footer>

<script>
function toggleMenu(){{document.querySelector('.burger').classList.toggle('active');document.getElementById('mobileNav').classList.toggle('active')}}
(function(){{
    const searchInput=document.querySelector('.search-input'),searchBtn=document.querySelector('.search-btn');
    const postcodeMap={{'G':'glasgow','EH':'edinburgh','AB':'aberdeen','DD':'dundee','FK':'stirling','PH':'perth','IV':'inverness','PA':'paisley','ML':'lanarkshire','KA':'ayrshire','KY':'fife','DG':'dumfries','TD':'borders','ZE':'aberdeen','KW':'inverness','HS':'inverness'}};
    function handleSearch(){{const query=searchInput.value.trim().toUpperCase();if(!query){{searchInput.focus();return}}const match=query.match(/^([A-Z]{{1,2}})/);if(match){{const prefix=match[1],area=postcodeMap[prefix]||postcodeMap[prefix[0]];if(area){{window.location.href='/areas/'+area+'/';return}}}}window.location.href='/factors/?search='+encodeURIComponent(query.toLowerCase())}}
    searchBtn.addEventListener('click',handleSearch);searchInput.addEventListener('keypress',function(e){{if(e.key==='Enter')handleSearch()}});
}})();
</script>
</body>
</html>'''
    
    # Write homepage
    with open(site_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    return True


def generate_listing_pages(conn, env) -> int:
    """Generate listing pages (all factors, by area, etc.)."""
    try:
        template = env.get_template('factors_listing.html')
    except:
        return 0
    
    generated = 0
    
    # All factors listing
    cursor = conn.execute("SELECT * FROM v_factor_profiles ORDER BY name")
    factors = [dict(row) for row in cursor]
    
    factors_dir = CONFIG.site_dir / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        html = template.render(
            title="All Property Factors",
            factors=factors,
            generated_date=datetime.now().strftime('%Y-%m-%d'),
        )
        
        with open(factors_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html)
        
        LOG.success(f"Generated factors listing ({len(factors)} factors)")
        generated += 1
    except Exception as e:
        LOG.warning(f"Failed to generate factors listing: {e}")
    
    # Tribunal listing
    try:
        tribunal_template = env.get_template('tribunal_listing.html')
        
        cursor = conn.execute("""
            SELECT t.*, f.name AS factor_name 
            FROM tribunal_cases t
            LEFT JOIN factors f ON t.factor_registration_number = f.registration_number
            ORDER BY t.decision_date DESC
        """)
        cases = [dict(row) for row in cursor]
        
        tribunal_dir = CONFIG.site_dir / "tribunal"
        tribunal_dir.mkdir(parents=True, exist_ok=True)
        
        html = tribunal_template.render(
            title="Tribunal Cases",
            cases=cases,
            generated_date=datetime.now().strftime('%Y-%m-%d'),
        )
        
        with open(tribunal_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html)
        
        LOG.success(f"Generated tribunal listing ({len(cases)} cases)")
        generated += 1
    except:
        pass
    
    return generated


def step_9_generate_site():
    """Generate static HTML site from database."""
    LOG.step(9, "Generate Static Site")
    
    if not HAS_JINJA:
        LOG.error("Jinja2 not installed. Run: pip install jinja2")
        return
    
    env = setup_jinja_env()
    if not env:
        LOG.error(f"Template directory not found: {CONFIG.template_dir}")
        return
    
    # Create output directories
    factors_dir = CONFIG.site_dir / "factors"
    tribunal_dir = CONFIG.site_dir / "tribunal"
    areas_dir = CONFIG.site_dir / "areas"
    factors_dir.mkdir(parents=True, exist_ok=True)
    tribunal_dir.mkdir(parents=True, exist_ok=True)
    areas_dir.mkdir(parents=True, exist_ok=True)
    
    with get_db() as conn:
        # 1. Factor profiles
        profile_count = generate_factor_profiles(conn, env, factors_dir)
        if profile_count > 0:
            LOG.success(f"Generated {profile_count} factor profile pages")
        
        # 2. Factor tribunal history pages
        tribunal_history_count = generate_factor_tribunal_pages(conn, env, factors_dir)
        if tribunal_history_count > 0:
            LOG.success(f"Generated {tribunal_history_count} tribunal history pages")
        
        # 3. Individual case pages - DISABLED (v2.6: linking to PDFs instead)
        case_count = 0  # Initialize since generation is disabled
        # case_count = generate_case_pages(conn, env, tribunal_dir)
        # if case_count > 0:
        #     LOG.success(f"Generated {case_count} individual case pages")
        LOG.info("Skipping individual case pages (linking to official PDFs)")
        
        # 4. Factors directory index page
        directory_count = generate_factors_directory(conn, factors_dir)
        LOG.success(f"Generated factors directory index ({directory_count} factors)")
        
        # 5. Area pages
        area_page_count = generate_area_pages(conn, areas_dir)
        LOG.success(f"Generated {area_page_count} area pages")
        
        area_index_count = generate_areas_index(conn, areas_dir)
        LOG.success(f"Generated areas index ({area_index_count} areas)")
        
        # 6. Comparison page
        generate_comparison_page(conn, CONFIG.site_dir)
        LOG.success("Generated comparison page (/compare/)")
        
        # 7. Homepage with dynamic counts
        generate_homepage(conn, CONFIG.site_dir)
        LOG.success("Generated homepage with current data")
        
        # 8. Listing pages (template-based, if available)
        listing_count = generate_listing_pages(conn, env)
        
        # 9. Export JSON data for frontend/API use
        cursor = conn.execute("SELECT * FROM v_factor_profiles ORDER BY name")
        factors = [dict(row) for row in cursor]
        
        json_path = CONFIG.site_dir / "data" / "factors.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(factors, f, indent=2, default=str)
        LOG.success(f"Exported factors.json ({len(factors)} factors)")
        
        # Export cases JSON
        cursor = conn.execute("""
            SELECT t.*, f.name AS factor_name 
            FROM tribunal_cases t
            LEFT JOIN factors f ON t.factor_registration_number = f.registration_number
            ORDER BY t.decision_date DESC
        """)
        cases = [dict(row) for row in cursor]
        
        cases_json_path = CONFIG.site_dir / "data" / "cases.json"
        with open(cases_json_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, default=str)
        LOG.success(f"Exported cases.json ({len(cases)} cases)")
    
    # Summary
    total_pages = profile_count + tribunal_history_count + case_count + listing_count
    LOG.info(f"Total pages generated: {total_pages}")


# =============================================================================
# STEP 10: VALIDATE OUTPUT
# =============================================================================

def step_10_validate():
    """Validate the pipeline output."""
    LOG.step(10, "Validate Output")
    
    issues = []
    
    if not CONFIG.db_path.exists():
        issues.append("Database file not created")
    else:
        with get_db() as conn:
            for table in ['factors', 'tribunal_cases', 'reviews', 'wss']:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                LOG.info(f"{table}: {count:,} rows")
                if table == 'factors' and count == 0:
                    issues.append("No factors imported")
            
            # Check composite scores
            with_score = conn.execute(
                "SELECT COUNT(*) FROM factors WHERE tribunal_composite_score IS NOT NULL"
            ).fetchone()[0]
            LOG.info(f"Factors with composite score: {with_score}")
            
            # Check summaries
            with_summary = conn.execute(
                "SELECT COUNT(*) FROM factors WHERE at_a_glance IS NOT NULL"
            ).fetchone()[0]
            LOG.info(f"Factors with At a Glance: {with_summary}")
            
            # Risk distribution
            LOG.info("Risk band distribution (v2.9):")
            cursor = conn.execute("""
                SELECT risk_band, COUNT(*) as cnt,
                       ROUND(AVG(tribunal_composite_score), 2) as avg_score
                FROM factors 
                WHERE risk_band IS NOT NULL 
                GROUP BY risk_band
                ORDER BY CASE risk_band 
                    WHEN 'CLEAN' THEN 1 WHEN 'GREEN' THEN 2 
                    WHEN 'AMBER' THEN 3 WHEN 'ORANGE' THEN 4 WHEN 'RED' THEN 5
                END
            """)
            for row in cursor:
                desc = get_tier_description(row['risk_band'])
                LOG.info(f"  {row['risk_band']:8} {row['cnt']:4} factors (avg score: {row['avg_score']}) - {desc}")
    
    # Check site
    if CONFIG.site_dir.exists():
        factors_dir = CONFIG.site_dir / "factors"
        if factors_dir.exists():
            pages = list(factors_dir.glob("*/index.html"))
            LOG.info(f"Generated pages: {len(pages)}")
    
    if issues:
        for issue in issues:
            LOG.warning(issue)
    else:
        LOG.success("All validations passed!")
    
    # Final stats
    if CONFIG.db_path.exists():
        size_mb = CONFIG.db_path.stat().st_size / (1024 * 1024)
        LOG.info(f"Database size: {size_mb:.1f} MB")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare Factors Scotland - Master Pipeline v2.9",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--full', action='store_true',
                        help='Full rebuild (drops existing data)')
    parser.add_argument('--update', action='store_true',
                        help='Incremental update (preserves existing)')
    parser.add_argument('--step', type=str,
                        help='Run specific steps (comma-separated)')
    parser.add_argument('--skip-ai', action='store_true',
                        help='Skip AI summary generation (faster)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done')
    parser.add_argument('--root', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    CONFIG.resolve_paths(Path(args.root).resolve())
    LOG.verbose = not args.quiet
    
    all_steps = list(range(1, 11))
    
    if args.step:
        steps_to_run = [int(s.strip()) for s in args.step.split(',')]
    elif args.full or args.update:
        steps_to_run = all_steps
    else:
        steps_to_run = all_steps
    
    reset_db = args.full
    
    print("\n" + "="*70)
    print("COMPARE FACTORS SCOTLAND - MASTER PIPELINE v2.9")
    print("Tier Logic: Unresolved PFEO+high volume=RED, +low volume=ORANGE, <3 cases=CLEAN")
    print("Reviews: Imports google_reviews_text.csv for individual review text")
    print("="*70)
    print(f"Project root: {CONFIG.project_root}")
    print(f"Database: {CONFIG.db_path}")
    print(f"Mode: {'FULL REBUILD' if reset_db else 'INCREMENTAL UPDATE'}")
    print(f"Steps: {steps_to_run}")
    if args.skip_ai:
        print("AI generation: SKIPPED")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        step_names = {
            1: "Initialize Database Schema (v2.1 with composite columns)",
            2: "Import Core Factor Data",
            3: "Import Tribunal Data",
            4: "Import Review Data",
            5: "Import Companies House Data",
            6: "Import WSS Data",
            7: "Calculate Scores & Risk Bands (v2.9 - Refined PFEO Logic)",
            8: "Generate AI Summaries",
            9: "Generate Static Site",
            10: "Validate Output",
        }
        for s in steps_to_run:
            print(f"  Step {s}: {step_names[s]}")
        return
    
    step_functions = {
        1: lambda: step_1_init_database(reset=reset_db),
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
    
    start_time = datetime.now()
    
    for step_num in steps_to_run:
        if step_num in step_functions:
            try:
                step_functions[step_num]()
            except Exception as e:
                LOG.error(f"Step {step_num} failed: {e}")
                import traceback
                traceback.print_exc()
    
    elapsed = datetime.now() - start_time
    LOG.summary()
    print(f"\n⏱️  Total time: {elapsed.total_seconds():.1f} seconds")


if __name__ == "__main__":
    main()