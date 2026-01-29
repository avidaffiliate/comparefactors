#!/usr/bin/env python3
"""
SCRIPT 05b: TRIBUNAL PDF ENRICHMENT v2.1

PURPOSE: Extract key details from tribunal decision PDFs using AI.
         Focused on accuracy and defensibility over comprehensiveness.

PHILOSOPHY:
  - Extract FACTUAL, VERIFIABLE data only
  - No subjective severity scoring - outcomes speak for themselves
  - Validation rules catch contradictions before they reach production
  - Store raw booleans for audit trail, derive outcome deterministically

WHAT WE EXTRACT:
  - Factor identification (for unmatched cases)
  - Outcome determination (7 boolean flags → 1 outcome category)
  - Financial orders (compensation + refunds, kept separate)
  - Key findings (quote + summary)

WHAT WE DROPPED (from v1):
  - Severity scores (arbitrary, hard to defend)
  - Complaint counts (unreliable extraction)
  - Complaint categories (nice-to-have, not essential)
  - Detailed fee extraction (complex, error-prone)

OUTCOMES (mutually exclusive):
  - Dismissed: Application dismissed on merits (factor cleared)
  - Rejected: Application rejected on procedural grounds
  - Withdrawn: Homeowner withdrew before determination
  - Breach - No Order: Breach found, no PFEO issued
  - PFEO Proposed: Tribunal proposed PFEO
  - PFEO Pending: PFEO issued, awaiting compliance
  - PFEO Complied: Factor complied with PFEO
  - PFEO Breached: Factor failed to comply with PFEO

RISK TIERS (factor-level, based on rate per 10k properties):
  - CLEAN: No adverse findings
  - GREEN: < 5 per 10k properties
  - AMBER: 5-20 per 10k properties
  - RED: > 20 per 10k properties

USAGE:
  python 05b_tribunal_enrich_v2.py                    # Process new cases
  python 05b_tribunal_enrich_v2.py --force            # Reprocess all
  python 05b_tribunal_enrich_v2.py --limit 10         # Test with 10 cases
  python 05b_tribunal_enrich_v2.py --export-csv       # Export to CSV
  python 05b_tribunal_enrich_v2.py --dry-run          # Show what would run
  python 05b_tribunal_enrich_v2.py --case "FTS/HPC/PF/25/2411"
  python 05b_tribunal_enrich_v2.py --validate-only    # Check existing data

DEPENDENCIES:
  pip install google-cloud-aiplatform pymupdf requests python-dotenv
"""

import os
import sys
import csv
import json
import time
import re
import argparse
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# PDF extraction
try:
    import fitz  # pymupdf
except ImportError:
    print("❌ pymupdf not installed. Run: pip install pymupdf")
    sys.exit(1)

# Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
except ImportError:
    print("❌ google-cloud-aiplatform not installed.")
    print("   Run: pip install google-cloud-aiplatform")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = Path("data/tribunal/tribunal_cases.csv")
OUTPUT_DB = Path("data/tribunal/tribunal_enriched_v2.db")
OUTPUT_CSV = Path("data/tribunal/tribunal_enriched_v2.csv")
PDF_CACHE_DIR = Path("data/cache/pdfs")

# Vertex AI settings
PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "scotland-factors-places"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"

# Rate limiting
REQUESTS_PER_MINUTE = 15
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Only process matched cases by default
MATCHED_ONLY = True


# =============================================================================
# EXTRACTION PROMPT
# =============================================================================

EXTRACTION_PROMPT = """You are analyzing a Scottish Property Factor tribunal decision. Extract key details accurately.

IMPORTANT: Read the ENTIRE document carefully before answering. Tribunal decisions often have multiple sections - the final determination may be near the end.

Return ONLY valid JSON (no markdown, no explanation):

{
    "factor_identification": {
        "factor_name": "<exact legal name of the property factor/respondent>",
        "registration_number": "<PF number if found, e.g. 'PF000103', or empty string>",
        "confidence": "<high/medium/low>"
    },
    
    "case_outcome": {
        "application_dismissed": <boolean - was the homeowner's application dismissed or rejected entirely?>,
        "application_withdrawn": <boolean - did the homeowner withdraw before determination?>,
        "breach_found": <boolean - did the tribunal find ANY breach of the Code of Conduct?>,
        
        "pfeo_proposed": <boolean - did the tribunal PROPOSE a Property Factor Enforcement Order?>,
        "pfeo_issued": <boolean - was a PFEO formally ISSUED/MADE?>,
        "pfeo_complied": <boolean - if PFEO was issued, did the factor comply with it?>,
        "pfeo_breached": <boolean - if PFEO was issued, did the factor FAIL to comply?>,
        
        "outcome_reasoning": "<1-2 sentences explaining how you determined the outcome>"
    },
    
    "financial_orders": {
        "compensation_awarded": <number - £ amount for distress/inconvenience/time, 0 if none>,
        "refund_ordered": <number - £ amount factor ordered to refund/credit, 0 if none>
    },
    
    "key_findings": {
        "key_quote": "<most significant tribunal finding, max 50 words, verbatim>",
        "summary": "<1-2 sentences: what happened and what did the tribunal decide?>"
    }
}

OUTCOME LOGIC - these are MUTUALLY EXCLUSIVE paths:

PATH A - Application fails (factor wins):
  - "dismissed" = tribunal reviewed merits, found no breach
  - "rejected" = application failed on procedural grounds  
  - "withdrawn" = homeowner withdrew before determination
  → breach_found=false, all PFEO fields=false

PATH B - Breach found, no enforcement:
  - Tribunal found Code breach BUT did not issue PFEO
  - Often because issue was minor or resolved during proceedings
  - ONLY use this if there is NO mention of PFEO anywhere in the document
  → breach_found=true, pfeo_proposed=false, pfeo_issued=false

PATH C - PFEO path:
  - pfeo_proposed=true means tribunal PROPOSED making an order
  - pfeo_issued=true means the order was formally MADE
  - Then check: did factor comply or breach it?
  → breach_found=true (PFEO requires breach finding)

CRITICAL - PFEO DETECTION:
Look for these phrases that indicate pfeo_proposed=true:
  - "proposed to make a Property Factor Enforcement Order"
  - "proposes to make a PFEO"
  - "the Tribunal proposed a PFEO"
  - "before making a PFEO, the Tribunal must give Notice"
  - Document title contains "Proposed PFEO"
  
Look for these phrases that indicate pfeo_issued=true:
  - "makes a Property Factor Enforcement Order"
  - "the PFEO is made"
  - "Property Factor Enforcement Order dated..."
  - Document title contains "PFEO" without "Proposed"

If the document mentions ANY PFEO language, it is NOT "Breach - No Order".

VALIDATION RULES (your response must satisfy ALL of these):
  - If application_dismissed=true → breach_found MUST be false
  - If breach_found=false → all PFEO fields MUST be false
  - If pfeo_complied=true → pfeo_issued MUST be true
  - If pfeo_breached=true → pfeo_issued MUST be true
  - pfeo_complied and pfeo_breached cannot BOTH be true

FINANCIAL ORDERS:
  - compensation_awarded = money for distress, inconvenience, time
  - refund_ordered = money factor must PAY BACK (overcharges, credits)
  - These are DIFFERENT - keep them separate

DOCUMENT TEXT:
"""


# =============================================================================
# VALID OUTCOMES
# =============================================================================

VALID_OUTCOMES = [
    "Dismissed",
    "Rejected",
    "Withdrawn",
    "Breach - No Order",
    "PFEO Proposed",
    "PFEO Pending",
    "PFEO Complied",
    "PFEO Breached",
]

# =============================================================================
# DETAILED OUTCOME CLASSIFICATION (for Dismissed/Rejected cases)
# =============================================================================

# Patterns indicating procedural dismissal (excluded from success rate)
PROCEDURAL_PATTERNS = [
    r'failed to provide.*(?:information|evidence|documentation)',
    r'did not provide.*(?:information|evidence|documentation|necessary)',
    r'failed to (?:appear|attend)',
    r'did not (?:appear|attend)',
    r'failed.*(?:comply|response)',
    r'no.*(?:participation|engagement|response)',
    r'lack of.*(?:participation|engagement|response)',
    r'applicant.*(?:failed to|did not|unable to).*(?:pursue|prosecute)',
    r'no jurisdiction',
    r'outwith.*jurisdiction',
    r'lacks jurisdiction',
    r'no basis for the application',
    r'no longer.*registered',
    r'not.*registered',
    r'no prima facie case',
    r'without further engagement',
    r'without further participation',
    r'non[-\s]?compliance by.*(?:applicant|homeowner)',
    r'not (?:competent|within)',
    r'incompetent',
    r'res judicata',
    r'time[-\s]?bar',
]

# Patterns indicating factor successfully defended (homeowner lost on merits)
FACTOR_COMPLIED_PATTERNS = [
    r'factor (?:had|has) (?:complied|not breached|not failed)',
    r'no breach.*(?:found|established)',
    r'no failure.*(?:found|established|to comply)',
    r'tribunal.*(?:found|determined).*no breach',
    r'factor.*(?:fulfilled|met|satisfied).*obligations',
    r'satisfied.*(?:on merits|on the merits)',
    r'not.*breach.*code',
    r'complaint.*(?:not|un)substantiated',
    r'allegations.*not.*(?:established|proven|made out)',
    r'evidence.*(?:does not|did not).*support',
    r'no evidence.*(?:of breach|to support|that)',
    r'no.*failure',
    r'not.*failed.*(?:duties|obligations)',
    r'no valid basis',
]

# Patterns indicating settlement/withdrawal (excluded from success rate)
SETTLEMENT_PATTERNS = [
    r'parties reached a settlement',
    r'parties.*agreed',
    r'settled',
    r'matter.*resolved',
    r'satisfactorily resolved',
    r'issue.*resolved',
    r'resolution.*reached',
    r'withdrawn',
]


def classify_outcome_detailed(
    outcome: str,
    summary: str,
    breach_found: bool = False
) -> tuple:
    """
    Classify Dismissed/Rejected cases into detailed categories.

    Returns:
        tuple: (outcome_detailed, outcome_category, is_substantive)

    outcome_detailed values:
        - "Factor Complied": Factor won on merits (homeowner lost)
        - "Rejected - Procedural": Dismissed for homeowner's non-participation
        - "Withdrawn - Settled": Parties reached settlement
        - "Breach - No Order": Misclassified - actually found breach
        - "Ambiguous": Couldn't determine from text

    outcome_category values:
        - "breach": Breach was found
        - "factor_complied": Factor won on merits
        - "procedural": Dismissed due to homeowner's failure
        - "withdrawn": Settlement or withdrawal
        - "ambiguous": Couldn't classify

    is_substantive: 1 if counts toward success rate, 0 if excluded
    """
    if not outcome:
        return (None, None, None)

    outcome_lower = outcome.lower()

    # Only classify Dismissed/Rejected cases
    if 'dismissed' not in outcome_lower and 'rejected' not in outcome_lower:
        # For non-dismissed cases, derive category from outcome
        if 'breach' in outcome_lower or 'pfeo' in outcome_lower:
            return (outcome, 'breach', 1)
        elif 'withdrawn' in outcome_lower:
            return (outcome, 'withdrawn', 0)
        else:
            return (outcome, None, None)

    # If breach_found flag is set, this is misclassified
    if breach_found:
        return ('Breach - No Order', 'breach', 1)

    text = (summary or '').lower()

    # Check for settlement first (takes priority)
    for pattern in SETTLEMENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ('Withdrawn - Settled', 'withdrawn', 0)

    # Check for factor complied (homeowner lost on merits)
    for pattern in FACTOR_COMPLIED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ('Factor Complied', 'factor_complied', 1)

    # Check for procedural dismissal
    for pattern in PROCEDURAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ('Rejected - Procedural', 'procedural', 0)

    # Ambiguous - couldn't determine
    return ('Ambiguous', 'ambiguous', None)


# =============================================================================
# OUTCOME DETERMINATION
# =============================================================================

def determine_outcome(ai_response: dict) -> str:
    """
    Map AI extraction to outcome category deterministically.
    Returns one of VALID_OUTCOMES.
    """
    outcome = ai_response.get('case_outcome', {})
    
    # Path A: Application failed
    if outcome.get('application_dismissed'):
        return "Dismissed"
    
    if outcome.get('application_withdrawn'):
        return "Withdrawn"
    
    # No breach = rejected (procedural failure)
    if not outcome.get('breach_found'):
        return "Rejected"
    
    # Path B & C: Breach found
    pfeo_issued = outcome.get('pfeo_issued', False)
    pfeo_proposed = outcome.get('pfeo_proposed', False)
    pfeo_complied = outcome.get('pfeo_complied', False)
    pfeo_breached = outcome.get('pfeo_breached', False)
    
    # Terminal PFEO states
    if pfeo_breached:
        return "PFEO Breached"
    
    if pfeo_complied:
        return "PFEO Complied"
    
    # Active PFEO states
    if pfeo_issued:
        return "PFEO Pending"
    
    if pfeo_proposed:
        return "PFEO Proposed"
    
    # Breach but no PFEO
    return "Breach - No Order"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_extraction(ai_response: dict) -> List[str]:
    """
    Check for logical inconsistencies.
    Returns list of validation errors (empty = valid).
    """
    errors = []
    outcome = ai_response.get('case_outcome', {})
    financial = ai_response.get('financial_orders', {})
    
    dismissed = outcome.get('application_dismissed', False)
    withdrawn = outcome.get('application_withdrawn', False)
    breach = outcome.get('breach_found', False)
    pfeo_proposed = outcome.get('pfeo_proposed', False)
    pfeo_issued = outcome.get('pfeo_issued', False)
    pfeo_complied = outcome.get('pfeo_complied', False)
    pfeo_breached = outcome.get('pfeo_breached', False)
    compensation = financial.get('compensation_awarded', 0)
    
    # Rule 1: Dismissed/withdrawn means no breach
    if dismissed and breach:
        errors.append("application_dismissed=true but breach_found=true")
    
    if withdrawn and breach:
        errors.append("application_withdrawn=true but breach_found=true")
    
    # Rule 2: No breach means no PFEO
    if not breach and (pfeo_proposed or pfeo_issued):
        errors.append("breach_found=false but PFEO fields are true")
    
    # Rule 3: PFEO compliance/breach requires PFEO to exist
    # Note: pfeo_complied only requires pfeo_proposed (compliance can happen before formal issuance)
    if pfeo_complied and not pfeo_proposed:
        errors.append("pfeo_complied=true but pfeo_proposed=false")

    if pfeo_breached and not pfeo_issued:
        errors.append("pfeo_breached=true but pfeo_issued=false")
    
    # Rule 4: Can't be both complied and breached
    if pfeo_complied and pfeo_breached:
        errors.append("both pfeo_complied and pfeo_breached are true")
    
    # Rule 5: Can't be both dismissed and withdrawn
    if dismissed and withdrawn:
        errors.append("both application_dismissed and application_withdrawn are true")
    
    # Rule 6: Suspicious - dismissed with compensation (warning only)
    if dismissed and compensation > 0:
        errors.append(f"WARNING: application_dismissed but compensation={compensation}")
    
    return errors


# =============================================================================
# DETERMINISTIC PFEO DETECTION (overrides AI when text is clear)
# =============================================================================

def detect_pfeo_from_text(pdf_text: str) -> dict:
    """
    Deterministic PFEO detection from PDF text.
    Returns overrides for AI extraction if PFEO language is found.
    The AI sometimes misses explicit PFEO language - this catches it.
    """
    text_lower = pdf_text.lower()
    
    overrides = {}
    
    # Check for PFEO Proposed language
    pfeo_proposed_phrases = [
        "proposed to make a property factor enforcement order",
        "proposes to make a pfeo",
        "proposes to make a property factor enforcement order",
        "before making a pfeo",
        "proposed pfeo",
        "the tribunal proposed a pfeo",
    ]
    
    for phrase in pfeo_proposed_phrases:
        if phrase in text_lower:
            overrides['pfeo_proposed'] = True
            overrides['breach_found'] = True
            break
    
    # Check for PFEO Issued/Made language (stronger than proposed)
    pfeo_issued_phrases = [
        "makes a property factor enforcement order",
        "the tribunal makes the following pfeo",
        "property factor enforcement order is made",
        "pfeo dated",
        "makes the following property factor enforcement order",
    ]
    
    for phrase in pfeo_issued_phrases:
        if phrase in text_lower:
            overrides['pfeo_issued'] = True
            overrides['pfeo_proposed'] = True  # Issued implies proposed
            overrides['breach_found'] = True
            break
    
    # Check for compliance/breach of PFEO
    if overrides.get('pfeo_issued'):
        compliance_phrases = ["complied with the pfeo", "compliance with the pfeo", 
                             "certificate of compliance", "complied with the property factor enforcement order"]
        breach_phrases = ["failed to comply with the pfeo", "failure to comply with the pfeo",
                         "failed to comply with the property factor enforcement order"]
        
        for phrase in compliance_phrases:
            if phrase in text_lower:
                overrides['pfeo_complied'] = True
                break
        
        for phrase in breach_phrases:
            if phrase in text_lower:
                overrides['pfeo_breached'] = True
                break
    
    return overrides


def apply_pfeo_overrides(ai_response: dict, pdf_text: str) -> dict:
    """
    Apply deterministic PFEO overrides to AI response.
    """
    overrides = detect_pfeo_from_text(pdf_text)
    
    if overrides:
        case_outcome = ai_response.get('case_outcome', {})
        for key, value in overrides.items():
            case_outcome[key] = value
        ai_response['case_outcome'] = case_outcome
    
    return ai_response


# =============================================================================
# RISK TIER CALCULATION (factor-level)
# =============================================================================

def calculate_risk_tier(summary: dict, properties_managed: int) -> str:
    """
    Calculate risk tier from tribunal summary.
    
    Based on adverse cases per 10,000 properties.
    
    Tiers:
      CLEAN  - No adverse tribunal findings
      GREEN  - Low rate of adverse findings (< 5 per 10k)
      AMBER  - Moderate rate (5-20 per 10k)
      RED    - High rate (> 20 per 10k)
    """
    if properties_managed == 0:
        return "UNKNOWN"
    
    # Count adverse outcomes (any case where breach was found)
    adverse_outcomes = [
        'PFEO Breached', 'PFEO Pending', 'PFEO Proposed',
        'PFEO Complied', 'Breach - No Order'
    ]
    
    adverse_cases = sum(
        summary['outcomes'].get(o, 0)
        for o in adverse_outcomes
    )
    
    # Rate per 10,000 properties
    rate = (adverse_cases / properties_managed) * 10000
    
    if adverse_cases == 0:
        return "CLEAN"
    elif rate < 5:
        return "GREEN"
    elif rate < 20:
        return "AMBER"
    else:
        return "RED"


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class CaseEnrichment:
    """Case enrichment data."""

    # Identifiers
    case_reference: str = ""  # Primary reference (first in combined cases)
    all_case_references: str = ""  # All references for combined cases (pipe-separated)
    matched_registration_number: str = ""

    # AI-extracted factor ID
    extracted_factor_name: str = ""
    extracted_registration_number: str = ""
    extraction_confidence: str = ""

    # Outcome (original AI-determined)
    outcome: str = ""
    outcome_reasoning: str = ""

    # Detailed outcome classification (post-processing)
    outcome_original: str = ""
    outcome_detailed: str = ""
    outcome_category: str = ""
    is_substantive: int = None  # 1 or 0 or None
    classification_confidence: str = ""
    classification_method: str = ""
    classification_timestamp: str = ""

    # Raw booleans (audit trail)
    application_dismissed: bool = False
    application_withdrawn: bool = False
    breach_found: bool = False
    pfeo_proposed: bool = False
    pfeo_issued: bool = False
    pfeo_complied: bool = False
    pfeo_breached: bool = False

    # Financial
    compensation_awarded: float = 0.0
    refund_ordered: float = 0.0

    # Key findings
    key_quote: str = ""
    summary: str = ""

    # Metadata
    decision_date: str = ""
    pdf_url: str = ""
    extraction_success: bool = False
    extraction_error: str = ""
    validation_errors: str = "[]"
    pdf_hash: str = ""
    extracted_at: str = ""


# =============================================================================
# DATABASE
# =============================================================================

def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            -- Identifiers
            case_reference TEXT PRIMARY KEY,
            all_case_references TEXT,  -- Full pipe-separated string for combined cases
            matched_registration_number TEXT,

            -- AI-extracted factor ID
            extracted_factor_name TEXT,
            extracted_registration_number TEXT,
            extraction_confidence TEXT,

            -- Outcome (original AI-determined)
            outcome TEXT,
            outcome_reasoning TEXT,

            -- Detailed outcome classification (post-processing)
            outcome_original TEXT,          -- Backup of outcome before reclassification
            outcome_detailed TEXT,          -- e.g., "Factor Complied", "Rejected - Procedural"
            outcome_category TEXT,          -- breach/factor_complied/procedural/withdrawn/ambiguous
            is_substantive INTEGER,         -- 1 if counts toward success rate, 0 if excluded
            classification_confidence TEXT, -- high/medium/low
            classification_method TEXT,     -- pattern_match/breach_flag/default
            classification_timestamp TEXT,

            -- Raw booleans (audit trail)
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

            -- Key findings
            key_quote TEXT,
            summary TEXT,

            -- Metadata
            decision_date TEXT,
            pdf_url TEXT,
            extraction_success INTEGER DEFAULT 0,
            extraction_error TEXT,
            validation_errors TEXT,
            pdf_hash TEXT,
            extracted_at TEXT,

            -- Full text for search
            full_text TEXT
        )
    """)

    # Add all_case_references column if it doesn't exist (for existing databases)
    try:
        conn.execute("ALTER TABLE cases ADD COLUMN all_case_references TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    return conn


def get_processed_cases(conn: sqlite3.Connection) -> set:
    """Get set of already-processed case references."""
    cursor = conn.execute(
        "SELECT case_reference FROM cases WHERE extraction_success = 1"
    )
    return {row[0] for row in cursor}


def save_case(conn: sqlite3.Connection, enrichment: CaseEnrichment, full_text: str = ""):
    """Save enrichment to database."""
    conn.execute("""
        INSERT OR REPLACE INTO cases (
            case_reference, all_case_references, matched_registration_number,
            extracted_factor_name, extracted_registration_number, extraction_confidence,
            outcome, outcome_reasoning,
            outcome_original, outcome_detailed, outcome_category, is_substantive,
            classification_confidence, classification_method, classification_timestamp,
            application_dismissed, application_withdrawn, breach_found,
            pfeo_proposed, pfeo_issued, pfeo_complied, pfeo_breached,
            compensation_awarded, refund_ordered,
            key_quote, summary,
            decision_date, pdf_url,
            extraction_success, extraction_error, validation_errors,
            pdf_hash, extracted_at, full_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        enrichment.case_reference,
        enrichment.all_case_references,
        enrichment.matched_registration_number,
        enrichment.extracted_factor_name,
        enrichment.extracted_registration_number,
        enrichment.extraction_confidence,
        enrichment.outcome,
        enrichment.outcome_reasoning,
        enrichment.outcome_original,
        enrichment.outcome_detailed,
        enrichment.outcome_category,
        enrichment.is_substantive,
        enrichment.classification_confidence,
        enrichment.classification_method,
        enrichment.classification_timestamp,
        1 if enrichment.application_dismissed else 0,
        1 if enrichment.application_withdrawn else 0,
        1 if enrichment.breach_found else 0,
        1 if enrichment.pfeo_proposed else 0,
        1 if enrichment.pfeo_issued else 0,
        1 if enrichment.pfeo_complied else 0,
        1 if enrichment.pfeo_breached else 0,
        enrichment.compensation_awarded,
        enrichment.refund_ordered,
        enrichment.key_quote,
        enrichment.summary,
        enrichment.decision_date,
        enrichment.pdf_url,
        1 if enrichment.extraction_success else 0,
        enrichment.extraction_error,
        enrichment.validation_errors,
        enrichment.pdf_hash,
        enrichment.extracted_at,
        full_text
    ])
    conn.commit()


# =============================================================================
# PDF HANDLING
# =============================================================================

def sanitize_filename(name: str) -> str:
    """Convert string to safe filename."""
    safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    safe = safe.replace('|', '_').replace(':', '_').replace('?', '')
    safe = safe.replace('"', '').replace('<', '').replace('>', '')
    return safe[:100]


def download_pdf(url: str, cache_dir: Path, registration_number: str = "", case_reference: str = "") -> Optional[Path]:
    """Download PDF and cache locally."""
    if not url:
        return None
    
    reg_folder = registration_number if registration_number else "_unmatched"
    case_folder = sanitize_filename(case_reference.split(' | ')[0]) if case_reference else "unknown"
    
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    original_filename = unquote(parsed.path.split('/')[-1])
    original_filename = sanitize_filename(original_filename)
    if not original_filename.endswith('.pdf'):
        original_filename += '.pdf'
    
    case_dir = cache_dir / reg_folder / case_folder
    cache_path = case_dir / original_filename
    
    if cache_path.exists():
        return cache_path
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        case_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return cache_path
    except Exception as e:
        print(f"      ⚠️ Download failed: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, str]:
    """Extract text from PDF. Returns (text, hash)."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        
        full_text = "\n".join(text_parts)
        pdf_hash = hashlib.md5(full_text.encode()).hexdigest()[:16]
        return full_text, pdf_hash
    except Exception:
        return "", ""


# =============================================================================
# AI EXTRACTION
# =============================================================================

def init_vertex_ai() -> GenerativeModel:
    """Initialize Vertex AI."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return GenerativeModel(MODEL_ID)


def extract_with_ai(model: GenerativeModel, pdf_text: str) -> Dict[str, Any]:
    """Extract enrichment fields using AI."""
    # Truncate to stay within limits
    truncated = pdf_text[:50000] if len(pdf_text) > 50000 else pdf_text
    
    prompt = EXTRACTION_PROMPT + truncated
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # Clean markdown if present
    if text.startswith('```'):
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
        text = text.strip()
    
    result = json.loads(text)
    
    # Normalize factor identification
    factor_id = result.get('factor_identification', {})
    factor_id['factor_name'] = str(factor_id.get('factor_name', ''))[:200]
    factor_id['registration_number'] = str(factor_id.get('registration_number', ''))[:20]
    factor_id['confidence'] = str(factor_id.get('confidence', 'low'))
    
    # Clean registration number format
    reg_num = factor_id['registration_number'].upper().strip()
    if reg_num and not reg_num.startswith('PF'):
        match = re.search(r'PF\d{6}', reg_num)
        reg_num = match.group(0) if match else ''
    factor_id['registration_number'] = reg_num
    result['factor_identification'] = factor_id
    
    # Normalize case outcome booleans
    outcome = result.get('case_outcome', {})
    for field in ['application_dismissed', 'application_withdrawn', 'breach_found',
                  'pfeo_proposed', 'pfeo_issued', 'pfeo_complied', 'pfeo_breached']:
        outcome[field] = bool(outcome.get(field, False))
    outcome['outcome_reasoning'] = str(outcome.get('outcome_reasoning', ''))[:500]
    result['case_outcome'] = outcome
    
    # Normalize financial orders
    financial = result.get('financial_orders', {})
    financial['compensation_awarded'] = max(0.0, float(financial.get('compensation_awarded', 0)))
    financial['refund_ordered'] = max(0.0, float(financial.get('refund_ordered', 0)))
    result['financial_orders'] = financial
    
    # Normalize key findings
    findings = result.get('key_findings', {})
    findings['key_quote'] = str(findings.get('key_quote', ''))[:500]
    findings['summary'] = str(findings.get('summary', ''))[:500]
    result['key_findings'] = findings
    
    return result


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_case(
    model: GenerativeModel,
    case: Dict[str, str],
    cache_dir: Path,
    conn: sqlite3.Connection
) -> CaseEnrichment:
    """Process a single case (may have multiple case references)."""

    # Get all case references - store the full string for searchability
    all_case_refs_str = case.get('case_references', '')
    # Split on | to get individual refs for processing
    all_case_refs = [r.strip() for r in all_case_refs_str.split(' | ') if r.strip()]
    case_ref = all_case_refs[0] if all_case_refs else ''
    reg_num = case.get('matched_registration_number', '')
    all_pdf_urls = case.get('pdf_urls', '').split(' | ')

    enrichment = CaseEnrichment(
        case_reference=case_ref,
        all_case_references=all_case_refs_str,  # Store full string for searching
        matched_registration_number=reg_num,
        decision_date=case.get('hearing_date', ''),
        pdf_url=all_pdf_urls[0] if all_pdf_urls else '',
        extracted_at=datetime.now().isoformat()
    )
    
    # Download ALL PDFs and concatenate text
    all_text = []
    all_hashes = []
    
    for pdf_url in all_pdf_urls:
        if not pdf_url.strip():
            continue
        
        pdf_path = download_pdf(
            pdf_url,
            cache_dir,
            registration_number=reg_num,
            case_reference=case_ref
        )
        if pdf_path:
            text, pdf_hash = extract_text_from_pdf(pdf_path)
            if text and len(text) > 50:
                all_text.append(f"--- Document: {pdf_path.name} ---\n{text}")
                all_hashes.append(pdf_hash)
    
    if not all_text:
        enrichment.extraction_error = f"No PDFs extracted (tried {len(all_pdf_urls)})"
        return enrichment
    
    pdf_text = "\n\n".join(all_text)
    enrichment.pdf_hash = "|".join(all_hashes[:3])
    
    if len(pdf_text) < 100:
        enrichment.extraction_error = "PDF text too short (may need OCR)"
        return enrichment
    
    # AI extraction
    try:
        ai_result = extract_with_ai(model, pdf_text)
        
        # Apply deterministic PFEO overrides (AI sometimes misses explicit language)
        ai_result = apply_pfeo_overrides(ai_result, pdf_text)
        
        # Factor identification
        factor_id = ai_result.get('factor_identification', {})
        enrichment.extracted_factor_name = factor_id.get('factor_name', '')
        enrichment.extracted_registration_number = factor_id.get('registration_number', '')
        enrichment.extraction_confidence = factor_id.get('confidence', '')
        
        # Case outcome booleans
        outcome_data = ai_result.get('case_outcome', {})
        enrichment.application_dismissed = outcome_data.get('application_dismissed', False)
        enrichment.application_withdrawn = outcome_data.get('application_withdrawn', False)
        enrichment.breach_found = outcome_data.get('breach_found', False)
        enrichment.pfeo_proposed = outcome_data.get('pfeo_proposed', False)
        enrichment.pfeo_issued = outcome_data.get('pfeo_issued', False)
        enrichment.pfeo_complied = outcome_data.get('pfeo_complied', False)
        enrichment.pfeo_breached = outcome_data.get('pfeo_breached', False)
        enrichment.outcome_reasoning = outcome_data.get('outcome_reasoning', '')
        
        # Financial
        financial = ai_result.get('financial_orders', {})
        enrichment.compensation_awarded = financial.get('compensation_awarded', 0)
        enrichment.refund_ordered = financial.get('refund_ordered', 0)
        
        # Key findings
        findings = ai_result.get('key_findings', {})
        enrichment.key_quote = findings.get('key_quote', '')
        enrichment.summary = findings.get('summary', '')
        
        # Validate
        validation_errors = validate_extraction(ai_result)
        enrichment.validation_errors = json.dumps(validation_errors)
        
        # Determine outcome deterministically
        enrichment.outcome = determine_outcome(ai_result)

        # Apply detailed classification for Dismissed/Rejected cases
        enrichment.outcome_original = enrichment.outcome
        detailed, category, is_subst = classify_outcome_detailed(
            enrichment.outcome,
            enrichment.summary,
            enrichment.breach_found
        )
        enrichment.outcome_detailed = detailed or enrichment.outcome
        enrichment.outcome_category = category or ''
        enrichment.is_substantive = is_subst
        enrichment.classification_method = 'pattern_match' if detailed else 'default'
        enrichment.classification_timestamp = datetime.now().isoformat()

        enrichment.extraction_success = True
        
    except json.JSONDecodeError as e:
        enrichment.extraction_error = f"JSON parse error: {e}"
    except Exception as e:
        enrichment.extraction_error = f"AI extraction failed: {e}"

    # Save to database (combined cases kept as single entry)
    save_case(conn, enrichment, pdf_text if enrichment.extraction_success else "")

    return enrichment


def export_to_csv(conn: sqlite3.Connection, output_path: Path):
    """Export cases table to CSV."""
    cursor = conn.execute("""
        SELECT 
            case_reference, matched_registration_number,
            extracted_factor_name, extracted_registration_number, extraction_confidence,
            outcome, outcome_reasoning,
            application_dismissed, application_withdrawn, breach_found,
            pfeo_proposed, pfeo_issued, pfeo_complied, pfeo_breached,
            compensation_awarded, refund_ordered,
            key_quote, summary,
            decision_date, pdf_url,
            extraction_success, validation_errors
        FROM cases
        ORDER BY decision_date DESC
    """)
    
    rows = cursor.fetchall()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'case_reference', 'matched_registration_number',
            'extracted_factor_name', 'extracted_registration_number', 'extraction_confidence',
            'outcome', 'outcome_reasoning',
            'application_dismissed', 'application_withdrawn', 'breach_found',
            'pfeo_proposed', 'pfeo_issued', 'pfeo_complied', 'pfeo_breached',
            'compensation_awarded', 'refund_ordered',
            'key_quote', 'summary',
            'decision_date', 'pdf_url',
            'extraction_success', 'validation_errors'
        ])
        
        for row in rows:
            writer.writerow(row)
    
    print(f"📄 Exported {len(rows)} cases to {output_path}")


def validate_existing_data(conn: sqlite3.Connection):
    """Validate existing data in database and report issues."""
    print("\n🔍 Validating existing data...")
    
    cursor = conn.execute("""
        SELECT case_reference, validation_errors, outcome,
               application_dismissed, breach_found, compensation_awarded
        FROM cases
        WHERE extraction_success = 1
    """)
    
    total = 0
    with_errors = 0
    error_types = {}
    
    for row in cursor:
        total += 1
        errors = json.loads(row[1] or '[]')
        if errors:
            with_errors += 1
            for e in errors:
                error_types[e] = error_types.get(e, 0) + 1
    
    print(f"\n   Total cases: {total}")
    print(f"   Cases with validation errors: {with_errors}")
    
    if error_types:
        print("\n   Error breakdown:")
        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"     {count}x {error}")
    
    # Outcome distribution
    print("\n   Outcome distribution:")
    for row in conn.execute("""
        SELECT outcome, COUNT(*) as cnt
        FROM cases WHERE extraction_success = 1
        GROUP BY outcome ORDER BY cnt DESC
    """):
        print(f"     {row[0]}: {row[1]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tribunal PDF Enrichment v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--force', action='store_true',
                        help='Reprocess all cases')
    parser.add_argument('--limit', type=int,
                        help='Limit number of cases')
    parser.add_argument('--export-csv', action='store_true',
                        help='Export to CSV')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed')
    parser.add_argument('--year', type=int,
                        help='Only process cases from specific year')
    parser.add_argument('--case', type=str,
                        help='Process specific case reference')
    parser.add_argument('--unmatched-only', action='store_true',
                        help='Only process unmatched cases')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRIBUNAL PDF ENRICHMENT v2.1")
    print("=" * 60)
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_DB}")
    print(f"Model: {MODEL_ID}")
    print()
    print("Extracts: factor ID, outcome (7 booleans), compensation, refund, quote, summary")
    print("Dropped: severity scores, complaint counts, fee details")
    print()
    
    # Initialize database
    conn = init_database(OUTPUT_DB)
    
    # Validate-only mode
    if args.validate_only:
        validate_existing_data(conn)
        conn.close()
        return
    
    # Load cases
    if not INPUT_CSV.exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        print("   Run 04_tribunal_scrape.py first.")
        return
    
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        all_cases = list(reader)
    
    print(f"📋 Loaded {len(all_cases)} cases from script 04")
    
    # Filters
    if args.case:
        all_cases = [c for c in all_cases if args.case in c.get('case_references', '')]
        print(f"   Filtered to {len(all_cases)} matching '{args.case}'")
    
    if args.unmatched_only:
        all_cases = [c for c in all_cases if not c.get('matched_registration_number')]
        print(f"   Filtered to {len(all_cases)} unmatched cases")
    elif MATCHED_ONLY and not args.case:
        all_cases = [c for c in all_cases if c.get('matched_registration_number')]
        print(f"   Filtered to {len(all_cases)} matched cases")
    
    if args.year:
        all_cases = [c for c in all_cases if c.get('hearing_date', '')[:4] == str(args.year)]
        print(f"   Filtered to {len(all_cases)} cases from {args.year}")
    
    cases_with_pdfs = [c for c in all_cases if c.get('pdf_urls')]
    print(f"   {len(cases_with_pdfs)} have PDF URLs")
    
    # Skip processed
    processed = set()
    if not args.force:
        processed = get_processed_cases(conn)
        print(f"   {len(processed)} already processed")
    
    pending = [c for c in cases_with_pdfs
               if c.get('case_references', '').split(' | ')[0] not in processed]
    
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} to process")
    
    # Dry run
    if args.dry_run:
        print("\n🔍 DRY RUN - would process:")
        for case in pending[:10]:
            ref = case.get('case_references', '').split(' | ')[0]
            factor = case.get('tribunal_property_factor', '')[:35]
            print(f"   {ref}: {factor}")
        if len(pending) > 10:
            print(f"   ... and {len(pending) - 10} more")
        return
    
    if not pending:
        print("\n✅ All cases already processed!")
        if args.export_csv:
            export_to_csv(conn, OUTPUT_CSV)
        validate_existing_data(conn)
        conn.close()
        return
    
    # Initialize AI
    print(f"\n🤖 Initializing Vertex AI...")
    model = init_vertex_ai()
    
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Processing {len(pending)} cases...")
    print(f"   Rate: {REQUESTS_PER_MINUTE}/min (~{len(pending) / REQUESTS_PER_MINUTE:.0f} minutes)")
    
    start_time = datetime.now()
    success_count = 0
    error_count = 0
    validation_issues = 0
    
    for i, case in enumerate(pending):
        ref = case.get('case_references', '').split(' | ')[0]
        
        print(f"[{i+1}/{len(pending)}] {ref}...", end=" ", flush=True)
        
        try:
            enrichment = process_case(model, case, PDF_CACHE_DIR, conn)
            
            if enrichment.extraction_success:
                errors = json.loads(enrichment.validation_errors)
                if errors:
                    validation_issues += 1
                    print(f"⚠️ {enrichment.outcome} (validation: {len(errors)} issues)")
                else:
                    comp_str = f" £{enrichment.compensation_awarded:.0f}" if enrichment.compensation_awarded else ""
                    print(f"✅ {enrichment.outcome}{comp_str}")
                success_count += 1
            else:
                print(f"❌ {enrichment.extraction_error[:50]}")
                error_count += 1
            
            time.sleep(REQUEST_DELAY)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted! Progress saved.")
            break
        except Exception as e:
            print(f"❌ {e}")
            error_count += 1
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"⚠️ With validation issues: {validation_issues}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️ Time: {elapsed}")
    print(f"📦 Database: {OUTPUT_DB}")
    
    if args.export_csv:
        export_to_csv(conn, OUTPUT_CSV)
    
    # Final stats
    validate_existing_data(conn)
    
    conn.close()


if __name__ == "__main__":
    main()