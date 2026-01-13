#!/usr/bin/env python3
"""
SCRIPT 05: TRIBUNAL PDF ENRICHMENT (Vertex AI) - v4.1

PURPOSE: Extract structured data from tribunal decision PDFs using
         Google Vertex AI (Gemini). Focused on actionable metrics.

INPUT:  data/tribunal/tribunal_cases.csv (from 03_tribunal_scrape.py)
OUTPUT: data/tribunal/tribunal_enriched.db (SQLite - primary)
        data/tribunal/tribunal_enriched.csv (optional with --export-csv)
        data/tribunal/tribunal_fees.csv (optional with --export-csv)

PDF SOURCE (checked in order):
  1. data/tribunal/pdfs/{PF_NUMBER}/{CASE_REF}/*.pdf - ALL PDFs in folder
  2. Download from tribunal website (cached in data/cache/pdfs/)

NOTE: The script now reads ALL PDFs in each case folder, not just the decision.
      This captures PFEO details, compliance outcomes, CMD information, etc.
      Documents are processed in priority order: Decision → PFEO → Compliance → CMD → Other

TABLES IN OUTPUT DB:
  - cases: Main extraction data with proper types (v4 schema)
  - case_texts: Full PDF text with FTS5 search
  - case_fees: Normalized fee data with quotes and context
  - extraction_log: Processing metadata and errors

v4.1 CHANGES:
  - CRITICAL FIX: PFEO_ISSUED now distinguishes between:
    * FINAL PFEOs actually made ("The Tribunal makes a PFEO") → pfeo_issued = true
    * PROPOSED PFEOs (warning stage, 14-day response period) → pfeo_proposed = true
    * Compliance hearings checking previous PFEOs → both false
  - CRITICAL FIX: OUTCOME prompt now much clearer:
    * "Dismissed" = homeowner LOST, factor did nothing wrong
    * Factor fixing issues during proceedings ≠ Dismissed
    * If PFEO issued/proposed or compensation ordered → Upheld
  - CRITICAL FIX: PDF sorting priority now correctly handles:
    * "No_PFEO" and "Certificate of Compliance" documents → LOWEST priority
    * "Proposed_PFEO" decisions → HIGHEST priority
    * This prevents compliance certificates from overwriting main decision data
  - Added pfeo_proposed field to track proposed-but-not-final PFEOs
  - Added --pfeo-only flag for targeted reextraction of PFEO-flagged cases
  - Added --dismissed-only flag for reextraction of possibly-wrong Dismissed cases
  - Severity scoring: +20 for issued PFEO, +10 for proposed PFEO

v4 ADDITIONS (from manual case analysis):
  - PFEO compliance lifecycle (complied, date, referred to ministers)
  - Case procedural details (CMDs, duration, first complaint date)
  - Authority/works issues (exceeded limits, consultation methods)
  - Factor behavior (admitted breaches, false statements, factor type)
  - Response time analysis (SLA stated vs actual failures)
  - Financial findings (overcharges found)
  - Enhanced complaint breakdown (not upheld, withdrawn, detail array)
  - Tribunal criticism language tracking
  - Key legal principles established

SETUP:
  1. Create GCP project and enable Vertex AI API
  2. Create .env file in project root with:
     GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account.json
     GCP_PROJECT_ID=your-project-id

  3. Or authenticate with gcloud:
     gcloud auth application-default login
     gcloud config set project YOUR_PROJECT_ID
     gcloud services enable aiplatform.googleapis.com

USAGE:
  python scripts/05_tribunal_enrich.py                   # Process new cases only
  python scripts/05_tribunal_enrich.py --force           # Reprocess all cases
  python scripts/05_tribunal_enrich.py --limit 10        # Process first 10 only
  python scripts/05_tribunal_enrich.py --year 2024       # Only 2024 cases
  python scripts/05_tribunal_enrich.py --export-csv      # Also export CSV
  python scripts/05_tribunal_enrich.py --no-full-text    # Skip storing PDF text
  python scripts/05_tribunal_enrich.py --reextract       # Re-run AI with new prompt
  python scripts/05_tribunal_enrich.py --reextract --pfeo-only  # Re-run AI only on PFEO-flagged cases
  python scripts/05_tribunal_enrich.py --reextract --dismissed-only  # Re-run AI only on Dismissed cases
  python scripts/05_tribunal_enrich.py --pdf-dir /path   # Use different PDF location

DEPENDENCIES:
  pip install google-cloud-aiplatform pymupdf requests python-dotenv

COST: ~$0.02-0.04 per PDF (~$50-70 for full corpus with v4 prompt)
TIME: ~45-60 minutes for 1000 cases (rate limited)
"""

import os
import sys
import csv
import json
import time
import re
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import hashlib

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_paths = [
        Path(".env"),
        Path(__file__).parent.parent / ".env",
        Path.home() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
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
    from vertexai.generative_models import GenerativeModel, Part
except ImportError:
    print("❌ google-cloud-aiplatform not installed.")
    print("   Run: pip install google-cloud-aiplatform")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = Path("data/tribunal/tribunal_cases.csv")
OUTPUT_DB = Path("data/tribunal/tribunal_enriched.db")
OUTPUT_CSV = Path("data/tribunal/tribunal_enriched.csv")
OUTPUT_FEES_CSV = Path("data/tribunal/tribunal_fees.csv")
DEFAULT_PDF_DIR = Path("data/tribunal/pdfs")
PDF_CACHE_DIR = Path("data/cache/pdfs")

# Vertex AI settings
PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "scotland-factors-places"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"

# Rate limiting
REQUESTS_PER_MINUTE = 15
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5

# Only process matched cases (skip unmatched)
MATCHED_ONLY = True


# ============================================================================
# CASE REFERENCE UTILITIES
# ============================================================================

def normalize_case_refs(refs_string: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Normalize case references for deduplication.
    
    Handles formats like:
        "FTS/HPC/PF/23/3590, FTS/HPC/PF/24/1449"
        "FTS/HPC/PF/24/1449 | FTS/HPC/PF/23/3590"
    
    Returns:
        (primary_ref, all_refs_sorted_tuple)
        
    The sorted tuple allows consistent deduplication regardless of order.
    """
    if not refs_string:
        return ("UNKNOWN", ("UNKNOWN",))
    
    # Split on comma, pipe, or semicolon
    refs = re.split(r'[,|;]\s*', refs_string)
    refs = [r.strip() for r in refs if r.strip()]
    
    if not refs:
        return ("UNKNOWN", ("UNKNOWN",))
    
    # Sort for consistent deduplication
    refs_sorted = tuple(sorted(refs))
    
    # Primary = first one listed (for display)
    primary = refs[0]
    
    return primary, refs_sorted


def get_all_case_ref_variants(refs_string: str) -> List[str]:
    """
    Get all case reference variants for PDF folder lookup.
    
    Returns list of possible folder names to check.
    """
    if not refs_string:
        return []
    
    refs = re.split(r'[,|;]\s*', refs_string)
    refs = [r.strip() for r in refs if r.strip()]
    
    # Convert each to folder name format
    variants = []
    for ref in refs:
        folder_name = ref.replace('/', '_').replace(' ', '_')
        variants.append(folder_name)
    
    return variants


# ============================================================================
# DATA STRUCTURES (v4)
# ============================================================================

@dataclass
class CaseExtraction:
    """Enhanced case extraction v4 with all fields from manual analysis."""
    
    # Identifiers
    case_reference: str = ""  # Primary reference for display
    all_case_references: str = ""  # All refs, pipe-separated (e.g., "FTS/.../23/3590 | FTS/.../24/1449")
    matched_registration_number: str = ""
    
    # Section A: Timeline
    decision_date: str = ""
    application_date: str = ""
    first_complaint_date: str = ""
    case_duration_months: Optional[int] = None
    hearing_count: int = 0
    cmds_held: int = 0
    referred_to_ministers: bool = False
    
    # Section B: Complaints & Outcome
    complaints_made: int = 0
    complaints_upheld: int = 0
    complaints_not_upheld: int = 0
    complaints_withdrawn: int = 0
    ai_outcome: str = ""
    outcome_evidence: str = ""
    complaints_detail: str = "[]"  # JSON array
    complaint_categories: str = "[]"  # JSON array
    
    # Section C: PFEO & Enforcement
    pfeo_issued: bool = False  # FINAL PFEO actually made
    pfeo_proposed: bool = False  # Proposed PFEO (warning stage, not yet final)
    pfeo_evidence: str = ""
    pfeo_requirements: str = ""
    pfeo_complied: Optional[bool] = None
    pfeo_compliance_date: str = ""
    pfeo_referred_to_ministers: bool = False
    compensation_awarded: float = 0.0
    compensation_evidence: str = ""
    
    # Section D: Factor Conduct
    factor_attended: str = "unknown"
    factor_cooperated: str = "unknown"
    factor_admitted_breaches: bool = False
    false_statement_to_tribunal: bool = False
    factor_type: str = "unknown"
    attended_evidence: str = ""
    cooperated_evidence: str = ""
    
    # Section E: Response Times
    factor_sla_stated: str = ""
    response_time_failures: str = "[]"  # JSON array
    unreturned_calls: Optional[int] = None
    
    # Section F: Major Works & Authority
    authority_exceeded: bool = False
    delegated_authority_limit: Optional[float] = None
    works_value_per_owner: Optional[float] = None
    consultation_method_invalid: str = ""
    independent_expert_ordered: bool = False
    contractor_supervision_failure: bool = False
    repair_attempts: Optional[int] = None
    
    # Section G: Financial
    overcharge_found: Optional[float] = None
    fees_extracted: str = "[]"  # JSON array, saved to case_fees table
    disputed_fee_type: str = "none"
    tribunal_fee_finding: str = ""
    
    # Section H: Code Breaches
    sections_breached: str = "[]"  # JSON array
    breach_category: str = "none"
    sections_evidence: str = ""
    
    # Section I: Quotes & Criticism
    tribunal_criticism: str = "[]"  # JSON array of critical terms
    criticism_quote: str = ""
    positive_quote: str = ""
    key_legal_principle: str = ""
    summary: str = ""
    
    # Section J: Metadata (from v3)
    property_unit_count: Optional[int] = None
    property_floors: Optional[int] = None
    property_has_lift: str = "unknown"
    property_facilities: str = "[]"
    property_location: str = ""
    property_address: str = ""  # Full address e.g. "1/38 Chapel Lane, Edinburgh, EH6 6ST"
    property_postcode: str = ""  # Just postcode e.g. "EH6 6ST"
    wss_response_time: str = ""
    
    # Legacy fields (v3 compatibility)
    previous_enforcement_mentioned: bool = False
    previous_tribunal_mentioned: bool = False
    key_quote: str = ""  # Kept for backward compatibility
    severity_factors: str = "[]"
    
    # Calculated
    severity_score: float = 0.0
    severity_reasoning: str = ""
    
    # Meta
    extraction_success: bool = False
    extraction_error: str = ""
    extracted_at: str = ""
    pdf_url: str = ""
    pdf_count: int = 1  # Number of PDFs analyzed for this case


# ============================================================================
# EXTRACTION PROMPT (v4)
# ============================================================================

EXTRACTION_PROMPT = """You are analyzing a Scottish Housing and Property Chamber tribunal decision about a property factor complaint.

Your task is to extract FACTS from this document. Do not interpret severity - just extract what the tribunal found and decided.

═══════════════════════════════════════════════════════════════════════════════
SECTION A: CASE IDENTIFICATION & TIMELINE
═══════════════════════════════════════════════════════════════════════════════

1. DECISION_DATE: The date the decision was issued.
   - Format: YYYY-MM-DD

2. APPLICATION_DATE: The date the homeowner submitted their complaint.
   - Format: YYYY-MM-DD or null if not stated

3. FIRST_COMPLAINT_DATE: If mentioned, when did the homeowner first complain to the factor?
   - This is often earlier than the tribunal application
   - Format: YYYY-MM-DD or null

4. CASE_DURATION_MONTHS: Calculate months from application to decision.
   - If application_date not found, return null
   - Return integer

5. HEARING_COUNT: How many hearings were held?
   - "Paper determination" = 0
   - Return integer

6. CMDS_HELD: How many Case Management Discussions (CMDs) were held?
   - CMDs are preliminary procedural hearings
   - Return integer or 0 if none mentioned

7. REFERRED_TO_MINISTERS: Was this case referred to Scottish Ministers?
   - This happens for PFEO non-compliance - a criminal matter
   - Return true/false

═══════════════════════════════════════════════════════════════════════════════
SECTION B: COMPLAINTS & OUTCOME
═══════════════════════════════════════════════════════════════════════════════

8. COMPLAINTS_MADE: Count distinct complaints raised.
   - Return integer

9. COMPLAINTS_UPHELD: How many upheld or partially upheld?
   - Return integer

10. COMPLAINTS_NOT_UPHELD: How many dismissed/not upheld?
    - Return integer

11. COMPLAINTS_WITHDRAWN: How many withdrawn by applicant?
    - Return integer

12. OUTCOME: Classify the OVERALL case outcome as exactly ONE of:
    - "Upheld" - the tribunal found the factor FAILED in their duties / breached the code
    - "Partially Upheld" - some complaints upheld, some dismissed
    - "Dismissed" - ALL complaints rejected, factor did nothing wrong
    - "Withdrawn" - case withdrawn by homeowner before decision
    - "Referred to Ministers" - PFEO non-compliance referred for prosecution
    
    IMPORTANT GUIDANCE:
    - If the tribunal says "failed to carry out property factor's duties" → Upheld
    - If a PFEO is issued OR proposed → Upheld (PFEOs only happen when factor found at fault)
    - If compensation is ordered → Upheld
    - If code breaches are found → Upheld
    - The factor fixing issues DURING tribunal proceedings does NOT make it "Dismissed"
    - "Dismissed" means the HOMEOWNER LOST - the factor was found to have done nothing wrong
    - Individual complaints being "not upheld" doesn't mean the whole case is Dismissed

13. OUTCOME_EVIDENCE: Quote the tribunal's KEY FINDING about whether the factor failed.
    - Look for: "failed to carry out", "in breach of", "failed to comply"
    - Max 50 words, verbatim

14. COMPLAINTS_DETAIL: Array of individual complaints with outcomes.
    Return array like:
    [
        {"section": "2.1", "issue": "communication failures", "outcome": "upheld"},
        {"section": "4.1", "issue": "accounts not provided", "outcome": "dismissed"},
        {"section": "6.1", "issue": "major works consultation", "outcome": "upheld"}
    ]

15. COMPLAINT_CATEGORIES: Array of applicable categories:
    - "communication", "financial", "maintenance", "code_breach", "disclosure"
    - "insurance", "governance", "health_safety", "exit_fees", "major_works"
    - "debt_recovery", "authority_exceeded", "consultation_failure", "other"

═══════════════════════════════════════════════════════════════════════════════
SECTION C: PFEO & ENFORCEMENT
═══════════════════════════════════════════════════════════════════════════════

16. PFEO_ISSUED: Was a FINAL Property Factor Enforcement Order ACTUALLY MADE in THIS decision?
    - Return true ONLY if the tribunal MAKES/ISSUES a final PFEO (not proposed)
    - Look for: "The Tribunal makes a PFEO", "A PFEO is made", "PFEO is issued"
    - Return FALSE if:
      * The document says "proposes to make a PFEO" or "proposed PFEO" - this is just a WARNING, not final
      * This is a COMPLIANCE hearing checking if a previous PFEO was complied with
      * The document discusses whether to comply with an existing PFEO  
      * A PFEO is merely mentioned, discussed, or referenced from a prior case
      * The phrase "failed to comply with the PFEO" appears (this means reviewing OLD PFEO)
    - IMPORTANT: "Proposed PFEO" ≠ actual PFEO. A proposed PFEO gives the factor 14 days to respond.
      The ACTUAL PFEO comes in a separate later document if the factor doesn't respond/comply.
    - Return true/false

17. PFEO_EVIDENCE: Quote the tribunal MAKING the final PFEO (not proposing one).
    - Max 50 words, verbatim
    - Leave empty if PFEO_ISSUED is false

17b. PFEO_PROPOSED: Did the tribunal PROPOSE to make a PFEO (but not yet final)?
    - Return true if document says "proposes to make a PFEO" or "proposed PFEO"
    - This is a warning stage - the factor has 14 days to respond before it becomes final
    - A proposed PFEO still indicates serious breaches were found
    - Return true/false

18. PFEO_REQUIREMENTS: What was the factor ordered to do?
    - One sentence, include timescales

19. PFEO_COMPLIED: Did the factor comply with the PFEO?
    - Return true, false, or null if not a compliance case
    - Look for "satisfied that the PFEO has been complied with"

20. PFEO_COMPLIANCE_DATE: If complied, when?
    - Format: YYYY-MM-DD or null

21. PFEO_REFERRED_TO_MINISTERS: Was PFEO non-compliance referred to Ministers?
    - This is serious - potential criminal prosecution
    - Return true/false

22. COMPENSATION_AWARDED: Compensation for distress, inconvenience, or time wasted.
    - Only include TRUE COMPENSATION, not refunds/reimbursements of overcharges
    - Compensation = remedy for harm caused (e.g., "£300 for distress")
    - Refund = returning improperly charged money (e.g., "refund the £500 overcharge")
    - Return number (no £ symbol), 0 if only refunds ordered or none

23. COMPENSATION_EVIDENCE: Quote awarding compensation (not refunds).
    - Max 40 words, verbatim

═══════════════════════════════════════════════════════════════════════════════
SECTION D: FACTOR CONDUCT & BEHAVIOR
═══════════════════════════════════════════════════════════════════════════════

24. FACTOR_ATTENDED: Did they participate?
    - Return "yes", "no", or "unknown"

25. FACTOR_COOPERATED: Did they engage constructively?
    - Return "yes", "no", or "unknown"

26. FACTOR_ADMITTED_BREACHES: Did the factor admit to any breaches?
    - Return true/false
    - Look for "the respondent accepted", "conceded", "admitted"

27. FALSE_STATEMENT_TO_TRIBUNAL: Did the factor make false statements?
    - Return true/false
    - Very serious if true

28. FACTOR_TYPE: What type of factor is this?
    - "commercial" - private factoring company
    - "rsl" - Registered Social Landlord / Housing Association
    - "council" - Local authority
    - "resident" - Resident-led factoring
    - "unknown"

29. ATTENDED_EVIDENCE: Quote about participation.
    - Max 30 words

30. COOPERATED_EVIDENCE: Quote about cooperation.
    - Max 30 words

═══════════════════════════════════════════════════════════════════════════════
SECTION E: RESPONSE TIME ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

31. FACTOR_SLA_STATED: What response times does the factor claim in their WSS?
    - Extract verbatim if mentioned, e.g., "emails within 5 working days"
    - Return string or null

32. RESPONSE_TIME_FAILURES: Array of specific response time failures found.
    Return array like:
    [
        {"type": "email", "promised": "5 working days", "actual": "18 days"},
        {"type": "complaint", "promised": "10 working days", "actual": "27 days"}
    ]
    Empty array if none found.

33. UNRETURNED_CALLS: Number of unreturned calls mentioned, or null.

═══════════════════════════════════════════════════════════════════════════════
SECTION F: MAJOR WORKS & AUTHORITY ISSUES
═══════════════════════════════════════════════════════════════════════════════

34. AUTHORITY_EXCEEDED: Did the factor exceed their delegated authority?
    - Return true/false
    - Common issue: instructing works over the limit without consultation

35. DELEGATED_AUTHORITY_LIMIT: What is the factor's spending limit per owner?
    - Return amount in £ or null if not mentioned
    - e.g., 150, 250

36. WORKS_VALUE_PER_OWNER: What was charged per owner for disputed works?
    - Return amount in £ or null

37. CONSULTATION_METHOD_INVALID: Was the consultation method found invalid?
    - Return the invalid method used, e.g., "objections only", "email without follow-up"
    - Return null if consultation was valid or not discussed

38. INDEPENDENT_EXPERT_ORDERED: Did tribunal order independent expert review?
    - Return true/false

39. CONTRACTOR_SUPERVISION_FAILURE: Did factor fail to supervise contractors?
    - Return true/false

40. REPAIR_ATTEMPTS: Number of failed repair attempts before tribunal, or null.

═══════════════════════════════════════════════════════════════════════════════
SECTION G: FINANCIAL FINDINGS
═══════════════════════════════════════════════════════════════════════════════

41. OVERCHARGE_FOUND: Total £ overcharge identified by tribunal.
    - Return number or null if no overcharge found/discussed

42. FEES_MENTIONED: All fee amounts mentioned in the document.
    Return array of fee objects:
    [
        {
            "type": "management_fee",
            "amount": 480.00,
            "period": "annual",
            "vat": "unknown",
            "quote": "The factor charges £480 per annum...",
            "notes": "2023 fee",
            "is_disputed": false,
            "tribunal_finding": ""
        }
    ]
    Fee types: "management_fee", "insurance_premium", "reserve_fund", "admin_fee",
               "exit_fee", "sale_pack", "arrears_fee", "major_works", "common_repairs",
               "abortive_call_out", "other"
    Periods: "annual", "semi_annual", "quarterly", "monthly", "one_off"

43. DISPUTED_FEE_TYPE: Primary fee type in dispute.
    - "management_fee", "insurance", "major_works", "exit_fee",
      "admin_charges", "reserve_fund", "debt_recovery", "multiple", "none"

44. TRIBUNAL_FEE_FINDING: What did tribunal find about the disputed fee?
    - One sentence summary

═══════════════════════════════════════════════════════════════════════════════
SECTION H: CODE BREACHES
═══════════════════════════════════════════════════════════════════════════════

45. SECTIONS_BREACHED: Specific Code sections found breached.
    - Array like ["Section 2.1", "Section 4.1", "Section 6.9"]
    - Empty array if none

46. BREACH_CATEGORY: Primary category of breach.
    - "communication", "transparency", "financial_management", "maintenance"
    - "consultation", "authority", "response_times", "complaints_handling"
    - "multiple", "none"

47. SECTIONS_EVIDENCE: Quote the tribunal's finding on breaches.
    - Max 50 words

═══════════════════════════════════════════════════════════════════════════════
SECTION I: QUOTES & CRITICISM
═══════════════════════════════════════════════════════════════════════════════

48. TRIBUNAL_CRITICISM: Array of critical terms used by tribunal.
    - Look for: "lackadaisical", "deplorable", "unacceptable", "fell below",
      "failed", "inadequate", "unreasonable", "dismissive", "appalling"
    - Return array like ["lackadaisical", "appalling"]

49. CRITICISM_QUOTE: Most damning quote about factor's conduct.
    - Max 60 words, verbatim

50. POSITIVE_QUOTE: Any positive comments about the factor.
    - Max 40 words, verbatim
    - Empty string if none

51. KEY_LEGAL_PRINCIPLE: If tribunal establishes important legal principle.
    - Quote verbatim, max 60 words
    - e.g., "The Respondent cannot assume they have authority to instruct works by asking for objections"

52. SUMMARY: 2-3 sentence summary of case and outcome.

═══════════════════════════════════════════════════════════════════════════════
SECTION J: CONTEXT
═══════════════════════════════════════════════════════════════════════════════

53. PROPERTY_CONTEXT: Extract any details about the property:
    - address: Full property address from the "Re:" line (e.g., "1/38 Chapel Lane, Edinburgh, EH6 6ST")
    - postcode: Just the postcode (e.g., "EH6 6ST") - extract from address
    - unit_count: Number of flats/units (integer or null)
    - floors: Number of storeys (integer or null)
    - has_lift: Is a lift/elevator mentioned? ("yes", "no", or "unknown")
    - facilities: List any common facilities mentioned
    - location: City/area if stated (string or null)

54. PREVIOUS_ENFORCEMENT_MENTIONED: Did tribunal mention previous enforcement orders? (true/false)

55. PREVIOUS_TRIBUNAL_MENTIONED: Did tribunal mention this factor appeared before? (true/false)

56. SEVERITY_FACTORS: List any aggravating factors present:
    - "health_safety", "vulnerable_residents", "financial_harm", "prolonged_duration"

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT - Valid JSON only, no markdown:
═══════════════════════════════════════════════════════════════════════════════

{
    "decision_date": "YYYY-MM-DD",
    "application_date": null,
    "first_complaint_date": null,
    "case_duration_months": null,
    "hearing_count": 0,
    "cmds_held": 0,
    "referred_to_ministers": false,
    
    "complaints_made": 0,
    "complaints_upheld": 0,
    "complaints_not_upheld": 0,
    "complaints_withdrawn": 0,
    "outcome": "",
    "outcome_evidence": "",
    "complaints_detail": [],
    "complaint_categories": [],
    
    "pfeo_issued": false,
    "pfeo_proposed": false,
    "pfeo_evidence": "",
    "pfeo_requirements": "",
    "pfeo_complied": null,
    "pfeo_compliance_date": null,
    "pfeo_referred_to_ministers": false,
    "compensation_awarded": 0.00,
    "compensation_evidence": "",
    
    "factor_attended": "unknown",
    "factor_cooperated": "unknown",
    "factor_admitted_breaches": false,
    "false_statement_to_tribunal": false,
    "factor_type": "unknown",
    "attended_evidence": "",
    "cooperated_evidence": "",
    
    "factor_sla_stated": null,
    "response_time_failures": [],
    "unreturned_calls": null,
    
    "authority_exceeded": false,
    "delegated_authority_limit": null,
    "works_value_per_owner": null,
    "consultation_method_invalid": null,
    "independent_expert_ordered": false,
    "contractor_supervision_failure": false,
    "repair_attempts": null,
    
    "overcharge_found": null,
    "fees": [],
    "disputed_fee_type": "none",
    "tribunal_fee_finding": "",
    
    "sections_breached": [],
    "breach_category": "none",
    "sections_evidence": "",
    
    "tribunal_criticism": [],
    "criticism_quote": "",
    "positive_quote": "",
    "key_legal_principle": "",
    "summary": "",
    
    "property_context": {
        "address": null,
        "postcode": null,
        "unit_count": null,
        "floors": null,
        "has_lift": "unknown",
        "facilities": [],
        "location": null
    },
    "previous_enforcement_mentioned": false,
    "previous_tribunal_mentioned": false,
    "severity_factors": []
}

IMPORTANT:
- Use null for missing data, not empty strings (except quote fields)
- All dates in YYYY-MM-DD format
- All amounts as numbers without £ symbol
- Boolean fields: true/false, not strings

TRIBUNAL DECISION DOCUMENT:
"""


# ============================================================================
# SEVERITY CALCULATION (v3)
# ============================================================================

def calculate_severity(extraction: CaseExtraction) -> Tuple[float, str]:
    """
    Calculate severity score v3.
    
    Designed to be averaged across a factor's cases for their profile page.
    
    Input fields used:
        - ai_outcome: string
        - pfeo_issued: bool
        - pfeo_complied: bool | null
        - compensation_awarded: float
        - false_statement_to_tribunal: bool
        - factor_attended / factor_cooperated: engagement (only if adverse outcome)
    
    Returns:
        - severity_score: 0.0-10.0 (100-point scale divided by 10)
        - severity_reasoning: Human-readable breakdown
    
    Score interpretation:
        0     Dismissed, no issues
        1-2   Minor issues or partial upholds
        3-4   Upheld, no enforcement
        5-6   PFEO issued or multiple issues
        7-10  Serious - PFEO non-compliance, false statements, ministerial referral
    
    Note: Engagement penalty only applies if case had adverse findings.
    No penalty for not attending a case that was dismissed.
    """
    score = 0
    reasons = []
    
    # Outcome (base score, mutually exclusive)
    outcome = extraction.ai_outcome.lower() if extraction.ai_outcome else ""
    
    # Track if case had adverse findings (for engagement penalty)
    adverse_outcome = False
    
    if 'referred to ministers' in outcome:
        score += 40
        reasons.append("Referred to Ministers (+40)")
        adverse_outcome = True
    elif 'upheld' in outcome and 'partially' not in outcome:
        score += 30
        reasons.append("Upheld (+30)")
        adverse_outcome = True
    elif 'partially' in outcome:
        score += 15
        reasons.append("Partially Upheld (+15)")
        adverse_outcome = True
    # Dismissed/Withdrawn = 0 points, no reason added
    
    # Enforcement
    if extraction.pfeo_issued:
        score += 20
        reasons.append("PFEO issued (+20)")
    elif extraction.pfeo_proposed:
        score += 10
        reasons.append("PFEO proposed (+10)")
    
    if extraction.pfeo_complied is False:  # Explicitly False, not None/unknown
        score += 20
        reasons.append("PFEO not complied (+20)")
    
    # Compensation
    if extraction.compensation_awarded and extraction.compensation_awarded > 0:
        score += 10
        reasons.append("Compensation awarded (+10)")
    
    # Conduct
    if extraction.false_statement_to_tribunal:
        score += 15
        reasons.append("False statement (+15)")
    
    # Engagement penalty only applies if case had adverse findings
    # (no penalty for not showing up to a case that was dismissed)
    if adverse_outcome and (extraction.factor_attended == "no" or extraction.factor_cooperated == "no"):
        score += 5
        reasons.append("Failed to engage (+5)")
    
    # Cap at 100
    score = min(score, 100)
    
    # Scale to 0-10 for display
    severity_score = score / 10
    
    # Build reasoning string
    if reasons:
        reasoning = " + ".join(reasons) + f" = {score} → {severity_score:.1f}"
    else:
        reasoning = "No issues found → 0.0"
    
    return severity_score, reasoning


# ============================================================================
# DATABASE SETUP (v4)
# ============================================================================

def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with v4 schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Main cases table (v4 schema)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_reference TEXT PRIMARY KEY,
            all_case_references TEXT,
            matched_registration_number TEXT,
            
            -- Timeline
            decision_date TEXT,
            application_date TEXT,
            first_complaint_date TEXT,
            case_duration_months INTEGER,
            hearing_count INTEGER DEFAULT 0,
            cmds_held INTEGER DEFAULT 0,
            referred_to_ministers INTEGER DEFAULT 0,
            
            -- Outcome breakdown
            complaints_made INTEGER DEFAULT 0,
            complaints_upheld INTEGER DEFAULT 0,
            complaints_not_upheld INTEGER DEFAULT 0,
            complaints_withdrawn INTEGER DEFAULT 0,
            ai_outcome TEXT,
            outcome_evidence TEXT,
            complaints_detail JSON,
            complaint_categories JSON,
            
            -- PFEO details
            pfeo_issued INTEGER DEFAULT 0,
            pfeo_proposed INTEGER DEFAULT 0,
            pfeo_evidence TEXT,
            pfeo_requirements TEXT,
            pfeo_complied INTEGER,
            pfeo_compliance_date TEXT,
            pfeo_referred_to_ministers INTEGER DEFAULT 0,
            compensation_awarded REAL DEFAULT 0,
            compensation_evidence TEXT,
            
            -- Factor behavior
            factor_attended TEXT DEFAULT 'unknown',
            factor_cooperated TEXT DEFAULT 'unknown',
            factor_admitted_breaches INTEGER DEFAULT 0,
            false_statement_to_tribunal INTEGER DEFAULT 0,
            factor_type TEXT DEFAULT 'unknown',
            attended_evidence TEXT,
            cooperated_evidence TEXT,
            
            -- Response times
            factor_sla_stated TEXT,
            response_time_failures JSON,
            unreturned_calls INTEGER,
            
            -- Authority/works
            authority_exceeded INTEGER DEFAULT 0,
            delegated_authority_limit REAL,
            works_value_per_owner REAL,
            consultation_method_invalid TEXT,
            independent_expert_ordered INTEGER DEFAULT 0,
            contractor_supervision_failure INTEGER DEFAULT 0,
            repair_attempts INTEGER,
            
            -- Financial
            overcharge_found REAL,
            disputed_fee_type TEXT,
            tribunal_fee_finding TEXT,
            
            -- Code breaches
            sections_breached JSON,
            breach_category TEXT,
            sections_evidence TEXT,
            
            -- Quotes
            tribunal_criticism JSON,
            criticism_quote TEXT,
            positive_quote TEXT,
            key_legal_principle TEXT,
            key_quote TEXT,
            summary TEXT,
            
            -- Property context
            property_unit_count INTEGER,
            property_floors INTEGER,
            property_has_lift TEXT DEFAULT 'unknown',
            property_facilities JSON,
            property_location TEXT,
            property_address TEXT,
            property_postcode TEXT,
            wss_response_time TEXT,
            
            -- Legacy
            previous_enforcement_mentioned INTEGER DEFAULT 0,
            previous_tribunal_mentioned INTEGER DEFAULT 0,
            severity_factors JSON,
            
            -- Severity
            severity_score REAL DEFAULT 0.0,
            severity_reasoning TEXT,
            
            -- Meta
            extraction_success INTEGER DEFAULT 0,
            extraction_error TEXT,
            extracted_at TEXT,
            pdf_url TEXT,
            pdf_count INTEGER DEFAULT 1
        )
    """)
    
    # Full text table with FTS5
    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_texts (
            case_reference TEXT PRIMARY KEY,
            full_text TEXT,
            pdf_url TEXT,
            extracted_at TEXT
        )
    """)
    
    # FTS5 virtual table for full-text search
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS case_texts_fts USING fts5(
            case_reference,
            full_text,
            content='case_texts',
            content_rowid='rowid'
        )
    """)
    
    # Triggers to keep FTS in sync
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS case_texts_ai AFTER INSERT ON case_texts BEGIN
            INSERT INTO case_texts_fts(rowid, case_reference, full_text)
            VALUES (new.rowid, new.case_reference, new.full_text);
        END;
        CREATE TRIGGER IF NOT EXISTS case_texts_ad AFTER DELETE ON case_texts BEGIN
            INSERT INTO case_texts_fts(case_texts_fts, rowid, case_reference, full_text)
            VALUES('delete', old.rowid, old.case_reference, old.full_text);
        END;
        CREATE TRIGGER IF NOT EXISTS case_texts_au AFTER UPDATE ON case_texts BEGIN
            INSERT INTO case_texts_fts(case_texts_fts, rowid, case_reference, full_text)
            VALUES('delete', old.rowid, old.case_reference, old.full_text);
            INSERT INTO case_texts_fts(rowid, case_reference, full_text)
            VALUES (new.rowid, new.case_reference, new.full_text);
        END;
    """)
    
    # Extraction log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extraction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_reference TEXT,
            status TEXT,
            error_message TEXT,
            duration_seconds REAL,
            extracted_at TEXT
        )
    """)
    
    # Fees table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_fees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_reference TEXT NOT NULL,
            fee_type TEXT NOT NULL,
            amount REAL,
            period TEXT,
            vat TEXT DEFAULT 'unknown',
            quote TEXT,
            notes TEXT,
            is_disputed INTEGER DEFAULT 0,
            tribunal_finding TEXT,
            FOREIGN KEY (case_reference) REFERENCES cases(case_reference)
        )
    """)
    
    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_factor ON cases(matched_registration_number)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_outcome ON cases(ai_outcome)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_severity ON cases(severity_score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_date ON cases(decision_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_pfeo ON cases(pfeo_issued)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_pfeo_complied ON cases(pfeo_complied)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_referred ON cases(referred_to_ministers)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_authority ON cases(authority_exceeded)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_false_stmt ON cases(false_statement_to_tribunal)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fees_case ON case_fees(case_reference)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fees_type ON case_fees(fee_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fees_period ON case_fees(period)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_postcode ON cases(property_postcode)")
    
    # v4.1 Migration: Add pfeo_proposed column if it doesn't exist
    try:
        conn.execute("ALTER TABLE cases ADD COLUMN pfeo_proposed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.commit()
    return conn


def save_case_to_db(conn: sqlite3.Connection, extraction: CaseExtraction):
    """Save extraction to database."""
    data = asdict(extraction)
    
    # Remove fees_extracted - stored in separate table
    data.pop('fees_extracted', None)
    
    # Convert booleans to integers for SQLite
    bool_fields = [
        'pfeo_issued', 'pfeo_proposed', 'pfeo_referred_to_ministers', 'referred_to_ministers',
        'factor_admitted_breaches', 'false_statement_to_tribunal',
        'authority_exceeded', 'independent_expert_ordered', 
        'contractor_supervision_failure', 'previous_enforcement_mentioned',
        'previous_tribunal_mentioned', 'extraction_success'
    ]
    for field in bool_fields:
        if field in data:
            data[field] = 1 if data[field] else 0
    
    # Handle pfeo_complied (tri-state: True=1, False=0, None=NULL)
    if 'pfeo_complied' in data:
        if data['pfeo_complied'] is None:
            data['pfeo_complied'] = None
        else:
            data['pfeo_complied'] = 1 if data['pfeo_complied'] else 0
    
    columns = list(data.keys())
    placeholders = ', '.join(['?' for _ in columns])
    column_names = ', '.join(columns)
    
    conn.execute(
        f"INSERT OR REPLACE INTO cases ({column_names}) VALUES ({placeholders})",
        [data[col] for col in columns]
    )
    conn.commit()


def save_full_text_to_db(conn: sqlite3.Connection, case_ref: str, text: str, pdf_url: str):
    """Store full text in database."""
    conn.execute(
        "INSERT OR REPLACE INTO case_texts (case_reference, full_text, pdf_url, extracted_at) VALUES (?, ?, ?, ?)",
        (case_ref, text, pdf_url, datetime.now().isoformat())
    )
    conn.commit()


def log_extraction(conn: sqlite3.Connection, case_ref: str, status: str, error: str = "", duration: float = 0):
    """Log extraction attempt."""
    conn.execute(
        "INSERT INTO extraction_log (case_reference, status, error_message, duration_seconds, extracted_at) VALUES (?, ?, ?, ?, ?)",
        (case_ref, status, error, duration, datetime.now().isoformat())
    )
    conn.commit()


def save_fees_to_db(conn: sqlite3.Connection, case_ref: str, fees: list):
    """Save extracted fees to case_fees table."""
    conn.execute("DELETE FROM case_fees WHERE case_reference = ?", (case_ref,))
    
    for fee in fees:
        conn.execute("""
            INSERT INTO case_fees 
            (case_reference, fee_type, amount, period, vat, quote, notes, is_disputed, tribunal_finding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_ref,
            fee.get("type", "other"),
            fee.get("amount"),
            fee.get("period"),
            fee.get("vat", "unknown"),
            fee.get("quote"),
            fee.get("notes"),
            1 if fee.get("is_disputed") else 0,
            fee.get("tribunal_finding")
        ))
    conn.commit()


def get_processed_cases(conn: sqlite3.Connection) -> set:
    """Get set of already-processed case references."""
    cursor = conn.execute("SELECT case_reference FROM cases WHERE extraction_success = 1")
    return {row[0] for row in cursor.fetchall()}


def get_stored_text(conn: sqlite3.Connection, case_ref: str) -> Optional[str]:
    """Get stored full text for a case."""
    cursor = conn.execute(
        "SELECT full_text FROM case_texts WHERE case_reference = ?",
        (case_ref,)
    )
    row = cursor.fetchone()
    return row[0] if row else None


def get_all_stored_cases(conn: sqlite3.Connection, pfeo_only: bool = False, dismissed_only: bool = False) -> List[Dict]:
    """Get all cases with stored full text for re-extraction.
    
    Args:
        conn: Database connection
        pfeo_only: If True, only return cases currently flagged pfeo_issued=1
        dismissed_only: If True, only return cases marked ai_outcome='Dismissed'
    """
    if pfeo_only:
        cursor = conn.execute("""
            SELECT ct.case_reference, ct.full_text, ct.pdf_url, c.matched_registration_number
            FROM case_texts ct
            JOIN cases c ON ct.case_reference = c.case_reference
            WHERE c.pfeo_issued = 1
        """)
    elif dismissed_only:
        cursor = conn.execute("""
            SELECT ct.case_reference, ct.full_text, ct.pdf_url, c.matched_registration_number
            FROM case_texts ct
            JOIN cases c ON ct.case_reference = c.case_reference
            WHERE c.ai_outcome = 'Dismissed'
        """)
    else:
        cursor = conn.execute("""
            SELECT ct.case_reference, ct.full_text, ct.pdf_url, c.matched_registration_number
            FROM case_texts ct
            LEFT JOIN cases c ON ct.case_reference = c.case_reference
        """)
    return [
        {
            'case_reference': row[0],
            'full_text': row[1],
            'pdf_url': row[2],
            'matched_registration_number': row[3] or ''
        }
        for row in cursor.fetchall()
    ]


def export_to_csv(conn: sqlite3.Connection, csv_path: Path):
    """Export cases table to CSV."""
    cursor = conn.execute("SELECT * FROM cases")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    
    print(f"📄 Exported {len(rows)} cases to {csv_path}")


def export_fees_to_csv(conn: sqlite3.Connection, csv_path: Path):
    """Export case_fees table to CSV with factor info."""
    cursor = conn.execute("""
        SELECT 
            cf.id,
            cf.case_reference,
            c.matched_registration_number as factor_id,
            cf.fee_type,
            cf.amount,
            cf.period,
            cf.vat,
            cf.quote,
            cf.notes,
            cf.is_disputed,
            cf.tribunal_finding
        FROM case_fees cf
        LEFT JOIN cases c ON cf.case_reference = c.case_reference
        ORDER BY c.matched_registration_number, cf.case_reference
    """)
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    
    print(f"📄 Exported {len(rows)} fees to {csv_path}")


# ============================================================================
# PDF HANDLING
# ============================================================================

def download_pdf(url: str, cache_dir: Path) -> Optional[Path]:
    """Download PDF to cache directory."""
    if not url:
        return None
    
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    filename = f"{url_hash}.pdf"
    cache_path = cache_dir / filename
    
    if cache_path.exists():
        return cache_path
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return cache_path
        
    except Exception as e:
        print(f"    ⚠️ Download failed: {e}")
        return None


def find_existing_pdf(pdf_dir: Path, pf_number: str, case_ref: str) -> Optional[Path]:
    """Find PDF in existing directory structure (legacy - single PDF)."""
    if not pdf_dir or not pf_number:
        return None
    
    case_ref_dir = case_ref.replace('/', '_').replace(' ', '_')
    case_path = pdf_dir / pf_number / case_ref_dir
    
    if not case_path.exists():
        case_path = pdf_dir / case_ref_dir
    
    if not case_path.exists():
        return None
    
    pdfs = list(case_path.glob("*.pdf"))
    if not pdfs:
        return None
    
    for pdf in pdfs:
        name_lower = pdf.name.lower()
        if 'decision' in name_lower and 'enforcement' not in name_lower:
            return pdf
    
    return pdfs[0]


def find_all_case_pdfs(pdf_dir: Path, pf_number: str, case_refs_string: str) -> List[Path]:
    """
    Find ALL PDFs in a case folder.
    
    Tries all case reference variants for cases with multiple refs.
    
    Returns list of PDF paths sorted by document priority:
    1. Decision documents (main ruling)
    2. PFEO documents (enforcement orders)
    3. Compliance documents (follow-up)
    4. CMD documents (procedural)
    5. Everything else
    """
    if not pdf_dir or not pf_number:
        return []
    
    # Get all possible folder names for this case
    folder_variants = get_all_case_ref_variants(case_refs_string)
    
    if not folder_variants:
        return []
    
    # Try each variant until we find PDFs
    pdfs = []
    for folder_name in folder_variants:
        case_path = pdf_dir / pf_number / folder_name
        
        if case_path.exists():
            pdfs = list(case_path.glob("*.pdf"))
            if pdfs:
                break
        
        # Also try without PF number (flat structure)
        case_path = pdf_dir / folder_name
        if case_path.exists():
            pdfs = list(case_path.glob("*.pdf"))
            if pdfs:
                break
    
    if not pdfs:
        return []
    
    # Sort by document type priority
    def sort_key(pdf_path: Path) -> tuple:
        name = pdf_path.name.lower()
        # Priority: lower number = processed first
        
        # LOWEST priority: Compliance certificates and "No_PFEO" documents
        # These are follow-up documents saying factor complied - NOT the main decision
        if 'certificate' in name or 'no_pfeo' in name or 'no pfeo' in name:
            return (5, name)  # Compliance certificates last
        
        # High priority: Main decisions (but not compliance certificates)
        if 'decision' in name and 'enforcement' not in name:
            # Prefer "Proposed_PFEO" decisions over others
            if 'proposed' in name:
                return (0, name)  # Proposed PFEO decision = highest
            return (1, name)  # Other decisions second
        
        # Medium priority: PFEO documents
        if 'enforcement' in name or 'pfeo' in name:
            return (2, name)  # PFEO documents
        
        # Lower priority: CMD and other procedural
        if 'cmd' in name or 'case management' in name:
            return (3, name)  # CMD fourth
        
        if 'compliance' in name:
            return (4, name)  # Compliance docs
        
        return (6, name)  # Everything else
    
    return sorted(pdfs, key=sort_key)


def extract_all_pdfs_text(pdf_paths: List[Path]) -> Tuple[str, Dict[str, str]]:
    """
    Extract and concatenate text from multiple PDFs.
    
    Returns:
        - combined_text: All text concatenated with document headers
        - doc_texts: Dict mapping filename to individual text (for debugging)
    """
    combined_text = ""
    doc_texts = {}
    
    for pdf_path in pdf_paths:
        doc_name = pdf_path.name
        text = extract_pdf_text(pdf_path)
        
        if text:
            doc_texts[doc_name] = text
            combined_text += f"\n\n{'='*70}\n"
            combined_text += f"DOCUMENT: {doc_name}\n"
            combined_text += f"{'='*70}\n\n"
            combined_text += text
    
    return combined_text.strip(), doc_texts


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pymupdf."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
        
        doc.close()
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
        
    except Exception as e:
        print(f"    ⚠️ PDF extraction failed: {e}")
        return ""


def is_scanned_pdf(pdf_path: Path) -> bool:
    """Check if PDF is a scanned image (needs OCR)."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text().strip()
            if len(text) > 100:
                doc.close()
                return False
        doc.close()
        return True
    except:
        return False


# ============================================================================
# VERTEX AI EXTRACTION
# ============================================================================

def init_vertex_ai():
    """Initialize Vertex AI client."""
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        return GenerativeModel(MODEL_ID)
    except Exception as e:
        print(f"❌ Failed to initialize Vertex AI: {e}")
        print("   Check your GOOGLE_APPLICATION_CREDENTIALS and project ID")
        sys.exit(1)


def extract_with_ai(model: GenerativeModel, text: str) -> Dict[str, Any]:
    """Send text to Gemini and parse response."""
    
    max_chars = 100000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    
    prompt = EXTRACTION_PROMPT + text
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {"error": str(e)}
    
    return {"error": "Max retries exceeded"}


def extract_with_ai_pdf(model: GenerativeModel, pdf_path: Path) -> Dict[str, Any]:
    """Send PDF directly to Gemini (for scanned documents)."""
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        pdf_part = Part.from_data(pdf_bytes, mime_type="application/pdf")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content([EXTRACTION_PROMPT, pdf_part])
                
                response_text = response.text
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")
                    
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return {"error": str(e)}
                    
    except Exception as e:
        return {"error": f"PDF processing failed: {e}"}
    
    return {"error": "Max retries exceeded"}


# ============================================================================
# RESULT MAPPING (v4)
# ============================================================================

def calculate_complaint_counts(complaints_detail: list) -> tuple:
    """
    Calculate complaint counts from the detailed breakdown.
    
    More reliable than AI counting separately - derive from the detail array.
    
    Args:
        complaints_detail: List of dicts with 'outcome' field
        
    Returns:
        (made, upheld, not_upheld, withdrawn)
    """
    if not complaints_detail or not isinstance(complaints_detail, list):
        return (0, 0, 0, 0)
    
    made = len(complaints_detail)
    upheld = sum(1 for c in complaints_detail if c.get('outcome', '').lower() == 'upheld')
    not_upheld = sum(1 for c in complaints_detail if c.get('outcome', '').lower() in ('dismissed', 'not upheld', 'rejected'))
    withdrawn = sum(1 for c in complaints_detail if c.get('outcome', '').lower() == 'withdrawn')
    
    return (made, upheld, not_upheld, withdrawn)


def map_ai_result_to_extraction(
    result: Dict[str, Any],
    case_ref: str,
    pf_number: str,
    pdf_url: str
) -> CaseExtraction:
    """Map AI JSON response to CaseExtraction dataclass (v4)."""
    
    extraction = CaseExtraction(
        case_reference=case_ref,
        matched_registration_number=pf_number,
        pdf_url=pdf_url,
        extracted_at=datetime.now().isoformat()
    )
    
    # Handle errors
    if "error" in result:
        extraction.extraction_error = result["error"]
        return extraction
    
    # Section A: Timeline
    extraction.decision_date = result.get("decision_date", "") or ""
    extraction.application_date = result.get("application_date", "") or ""
    extraction.first_complaint_date = result.get("first_complaint_date", "") or ""
    extraction.case_duration_months = result.get("case_duration_months")
    extraction.hearing_count = int(result.get("hearing_count", 0) or 0)
    extraction.cmds_held = int(result.get("cmds_held", 0) or 0)
    extraction.referred_to_ministers = bool(result.get("referred_to_ministers", False))
    
    # Section B: Complaints & Outcome
    # Get AI's counts as fallback
    ai_complaints_made = int(result.get("complaints_made", 0) or 0)
    ai_complaints_upheld = int(result.get("complaints_upheld", 0) or 0)
    ai_complaints_not_upheld = int(result.get("complaints_not_upheld", 0) or 0)
    ai_complaints_withdrawn = int(result.get("complaints_withdrawn", 0) or 0)
    
    # Parse complaints detail
    complaints_detail = result.get("complaints_detail", [])
    if not isinstance(complaints_detail, list):
        complaints_detail = []
    extraction.complaints_detail = json.dumps(complaints_detail) if complaints_detail else "[]"
    
    # Calculate counts from detail (more reliable than AI counting)
    if complaints_detail:
        made, upheld, not_upheld, withdrawn = calculate_complaint_counts(complaints_detail)
        extraction.complaints_made = made
        extraction.complaints_upheld = upheld
        extraction.complaints_not_upheld = not_upheld
        extraction.complaints_withdrawn = withdrawn
    else:
        # Fall back to AI counts if no detail provided
        extraction.complaints_made = ai_complaints_made
        extraction.complaints_upheld = ai_complaints_upheld
        extraction.complaints_not_upheld = ai_complaints_not_upheld
        extraction.complaints_withdrawn = ai_complaints_withdrawn
    
    # Sanity check
    if extraction.complaints_upheld > extraction.complaints_made:
        extraction.complaints_upheld = extraction.complaints_made
    
    extraction.ai_outcome = result.get("outcome", "") or ""
    extraction.outcome_evidence = result.get("outcome_evidence", "") or ""
    
    categories = result.get("complaint_categories", [])
    extraction.complaint_categories = json.dumps(categories) if isinstance(categories, list) else "[]"
    
    # Section C: PFEO & Enforcement
    extraction.pfeo_issued = bool(result.get("pfeo_issued", False))
    extraction.pfeo_proposed = bool(result.get("pfeo_proposed", False))
    extraction.pfeo_evidence = result.get("pfeo_evidence", "") or ""
    extraction.pfeo_requirements = result.get("pfeo_requirements", "") or ""
    extraction.pfeo_complied = result.get("pfeo_complied")  # Can be None
    extraction.pfeo_compliance_date = result.get("pfeo_compliance_date", "") or ""
    extraction.pfeo_referred_to_ministers = bool(result.get("pfeo_referred_to_ministers", False))
    extraction.compensation_awarded = float(result.get("compensation_awarded", 0) or 0)
    extraction.compensation_evidence = result.get("compensation_evidence", "") or ""
    
    # Section D: Factor Conduct
    extraction.factor_attended = result.get("factor_attended", "unknown") or "unknown"
    extraction.factor_cooperated = result.get("factor_cooperated", "unknown") or "unknown"
    extraction.factor_admitted_breaches = bool(result.get("factor_admitted_breaches", False))
    extraction.false_statement_to_tribunal = bool(result.get("false_statement_to_tribunal", False))
    extraction.factor_type = result.get("factor_type", "unknown") or "unknown"
    extraction.attended_evidence = result.get("attended_evidence", "") or ""
    extraction.cooperated_evidence = result.get("cooperated_evidence", "") or ""
    
    # Section E: Response Times
    extraction.factor_sla_stated = result.get("factor_sla_stated", "") or ""
    response_failures = result.get("response_time_failures", [])
    extraction.response_time_failures = json.dumps(response_failures) if isinstance(response_failures, list) else "[]"
    extraction.unreturned_calls = result.get("unreturned_calls")
    
    # Section F: Authority/Works
    extraction.authority_exceeded = bool(result.get("authority_exceeded", False))
    extraction.delegated_authority_limit = result.get("delegated_authority_limit")
    extraction.works_value_per_owner = result.get("works_value_per_owner")
    extraction.consultation_method_invalid = result.get("consultation_method_invalid", "") or ""
    extraction.independent_expert_ordered = bool(result.get("independent_expert_ordered", False))
    extraction.contractor_supervision_failure = bool(result.get("contractor_supervision_failure", False))
    extraction.repair_attempts = result.get("repair_attempts")
    
    # Section G: Financial
    extraction.overcharge_found = result.get("overcharge_found")
    fees = result.get("fees", []) or []
    extraction.fees_extracted = json.dumps(fees) if isinstance(fees, list) else "[]"
    extraction.disputed_fee_type = result.get("disputed_fee_type", "none") or "none"
    extraction.tribunal_fee_finding = result.get("tribunal_fee_finding", "") or ""
    
    # Section H: Code Breaches
    sections = result.get("sections_breached", [])
    extraction.sections_breached = json.dumps(sections) if isinstance(sections, list) else "[]"
    extraction.breach_category = result.get("breach_category", "none") or "none"
    extraction.sections_evidence = result.get("sections_evidence", "") or ""
    
    # Section I: Quotes
    criticism_arr = result.get("tribunal_criticism", [])
    extraction.tribunal_criticism = json.dumps(criticism_arr) if isinstance(criticism_arr, list) else "[]"
    extraction.criticism_quote = result.get("criticism_quote", "") or ""
    extraction.positive_quote = result.get("positive_quote", "") or ""
    extraction.key_legal_principle = result.get("key_legal_principle", "") or ""
    extraction.summary = result.get("summary", "") or ""
    
    # For backward compatibility, use criticism_quote as key_quote if not empty
    extraction.key_quote = extraction.criticism_quote or extraction.positive_quote
    
    # Section J: Property Context
    prop_ctx = result.get("property_context", {}) or {}
    extraction.property_unit_count = prop_ctx.get("unit_count")
    extraction.property_floors = prop_ctx.get("floors")
    extraction.property_has_lift = prop_ctx.get("has_lift", "unknown") or "unknown"
    facilities = prop_ctx.get("facilities", [])
    extraction.property_facilities = json.dumps(facilities) if isinstance(facilities, list) else "[]"
    extraction.property_location = prop_ctx.get("location", "") or ""
    extraction.property_address = prop_ctx.get("address", "") or ""
    extraction.property_postcode = prop_ctx.get("postcode", "") or ""
    extraction.wss_response_time = extraction.factor_sla_stated  # Copy for compatibility
    
    # Legacy fields
    extraction.previous_enforcement_mentioned = bool(result.get("previous_enforcement_mentioned", False))
    extraction.previous_tribunal_mentioned = bool(result.get("previous_tribunal_mentioned", False))
    sev_factors = result.get("severity_factors", [])
    extraction.severity_factors = json.dumps(sev_factors) if isinstance(sev_factors, list) else "[]"
    
    # Calculate severity
    extraction.severity_score, extraction.severity_reasoning = calculate_severity(extraction)
    
    extraction.extraction_success = True
    return extraction


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_case_from_text(
    model: GenerativeModel,
    case_ref: str,
    pf_number: str,
    text: str,
    pdf_url: str,
    conn: sqlite3.Connection
) -> CaseExtraction:
    """Process a case using already-stored full text (for re-extraction)."""
    
    start_time = time.time()
    
    if not text or text == "[SCANNED_PDF - OCR by AI]":
        extraction = CaseExtraction(
            case_reference=case_ref,
            matched_registration_number=pf_number,
            pdf_url=pdf_url,
            extracted_at=datetime.now().isoformat()
        )
        extraction.extraction_error = "No stored text available (was scanned PDF)"
        log_extraction(conn, case_ref, "error", extraction.extraction_error, time.time() - start_time)
        return extraction
    
    # Run AI extraction
    result = extract_with_ai(model, text)
    
    # Map to extraction
    extraction = map_ai_result_to_extraction(result, case_ref, pf_number, pdf_url)
    
    # Log
    if extraction.extraction_success:
        log_extraction(conn, case_ref, "success", "", time.time() - start_time)
    else:
        log_extraction(conn, case_ref, "error", extraction.extraction_error, time.time() - start_time)
    
    return extraction


def process_case(
    model: GenerativeModel,
    case: Dict,
    pdf_cache_dir: Path,
    conn: sqlite3.Connection,
    store_full_text_flag: bool = True,
    existing_pdf_dir: Optional[Path] = None
) -> CaseExtraction:
    """Process a single tribunal case - reads ALL PDFs in case folder."""
    
    start_time = time.time()
    
    # Handle multiple case references
    case_refs_raw = case.get('case_references', '')
    case_ref, _ = normalize_case_refs(case_refs_raw)  # Primary ref for display/storage
    
    pf_number = case.get('matched_registration_number', '')
    
    pdf_urls = case.get('pdf_urls', '')
    pdf_url_list = [url.strip() for url in pdf_urls.split(' | ') if url.strip()]
    
    pdf_names = case.get('pdf_files', '')
    pdf_name_list = [name.strip() for name in pdf_names.split(' | ') if name.strip()]
    
    # Find decision PDF URL (for reference/metadata)
    selected_url = None
    for i, name in enumerate(pdf_name_list):
        if 'decision' in name.lower() and i < len(pdf_url_list):
            selected_url = pdf_url_list[i]
            break
    
    if not selected_url and pdf_url_list:
        selected_url = pdf_url_list[0]
    
    # Try to find ALL existing PDFs first (using all ref variants)
    text = ""
    pdf_count = 0
    
    if existing_pdf_dir:
        pdf_paths = find_all_case_pdfs(existing_pdf_dir, pf_number, case_refs_raw)
        
        if pdf_paths:
            pdf_count = len(pdf_paths)
            text, doc_texts = extract_all_pdfs_text(pdf_paths)
        else:
            # Fall back to downloading single PDF
            if selected_url:
                pdf_path = download_pdf(selected_url, pdf_cache_dir)
                if pdf_path:
                    text = extract_pdf_text(pdf_path)
                    pdf_count = 1
    else:
        # Standard download mode - download all available PDFs
        downloaded_texts = []
        for i, url in enumerate(pdf_url_list):
            pdf_path = download_pdf(url, pdf_cache_dir)
            if pdf_path:
                doc_name = pdf_name_list[i] if i < len(pdf_name_list) else f"document_{i}.pdf"
                doc_text = extract_pdf_text(pdf_path)
                if doc_text:
                    downloaded_texts.append((doc_name, doc_text))
        
        if downloaded_texts:
            # Sort by document priority (same logic as find_all_case_pdfs)
            def doc_priority(item):
                name = item[0].lower()
                if 'certificate' in name or 'no_pfeo' in name or 'no pfeo' in name:
                    return (5, name)
                if 'decision' in name and 'enforcement' not in name:
                    if 'proposed' in name:
                        return (0, name)
                    return (1, name)
                if 'enforcement' in name or 'pfeo' in name:
                    return (2, name)
                if 'cmd' in name or 'case management' in name:
                    return (3, name)
                if 'compliance' in name:
                    return (4, name)
                return (6, name)
            
            downloaded_texts.sort(key=doc_priority)
            
            pdf_count = len(downloaded_texts)
            text = ""
            for doc_name, doc_text in downloaded_texts:
                text += f"\n\n{'='*70}\n"
                text += f"DOCUMENT: {doc_name}\n"
                text += f"{'='*70}\n\n"
                text += doc_text
            text = text.strip()
    
    if not text:
        _, refs_tuple = normalize_case_refs(case_refs_raw)
        extraction = CaseExtraction(
            case_reference=case_ref,
            all_case_references=" | ".join(refs_tuple),
            matched_registration_number=pf_number,
            pdf_url=selected_url or "",
            extracted_at=datetime.now().isoformat()
        )
        extraction.extraction_error = "No PDFs found or text extraction failed"
        log_extraction(conn, case_ref, "error", extraction.extraction_error, time.time() - start_time)
        return extraction
    
    # Store full text if requested
    if store_full_text_flag and text:
        save_full_text_to_db(conn, case_ref, text, selected_url or "")
    
    # Check if any PDF was scanned (needs direct PDF upload)
    # For multi-PDF, we use text extraction - scanned docs would have minimal text
    if len(text) < 500 and pdf_count == 1:
        # Likely scanned - try direct PDF upload for single doc cases
        if existing_pdf_dir:
            pdf_paths = find_all_case_pdfs(existing_pdf_dir, pf_number, case_refs_raw)
            if pdf_paths:
                result = extract_with_ai_pdf(model, pdf_paths[0])
            else:
                result = {"error": "Scanned PDF with no text"}
        else:
            result = {"error": "Scanned PDF - text extraction failed"}
    else:
        result = extract_with_ai(model, text)
    
    # Map to extraction
    extraction = map_ai_result_to_extraction(result, case_ref, pf_number, selected_url or "")
    
    # Set all case references (for multi-ref cases)
    _, refs_tuple = normalize_case_refs(case_refs_raw)
    extraction.all_case_references = " | ".join(refs_tuple)
    
    # Set PDF count
    extraction.pdf_count = pdf_count
    
    # Add PDF count to summary for transparency
    if extraction.extraction_success and pdf_count > 1:
        extraction.summary = f"[{pdf_count} documents analyzed] " + extraction.summary
    
    # Log
    if extraction.extraction_success:
        log_extraction(conn, case_ref, "success", "", time.time() - start_time)
    else:
        log_extraction(conn, case_ref, "error", extraction.extraction_error, time.time() - start_time)
    
    return extraction


def _print_final_stats(conn: sqlite3.Connection):
    """Print detailed extraction statistics."""
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(complaints_made) as total_made,
            SUM(complaints_upheld) as total_upheld,
            SUM(compensation_awarded) as total_comp,
            SUM(pfeo_issued) as pfeo_count,
            SUM(pfeo_referred_to_ministers) as pfeo_referred,
            SUM(referred_to_ministers) as case_referred,
            SUM(false_statement_to_tribunal) as false_stmt_count,
            SUM(authority_exceeded) as authority_count,
            SUM(CASE WHEN factor_attended = 'no' THEN 1 ELSE 0 END) as no_shows,
            SUM(CASE WHEN factor_cooperated = 'no' THEN 1 ELSE 0 END) as uncooperative,
            SUM(CASE WHEN criticism_quote != '' THEN 1 ELSE 0 END) as with_criticism,
            SUM(CASE WHEN key_legal_principle != '' THEN 1 ELSE 0 END) as with_legal_principle
        FROM cases
        WHERE extraction_success = 1
    """)
    stats = cursor.fetchone()
    
    if stats and stats[0] > 0:
        total = stats[0]
        total_made = stats[1] or 0
        total_upheld = stats[2] or 0
        total_comp = stats[3] or 0
        pfeo_count = stats[4] or 0
        pfeo_referred = stats[5] or 0
        case_referred = stats[6] or 0
        false_stmt_count = stats[7] or 0
        authority_count = stats[8] or 0
        no_shows = stats[9] or 0
        uncooperative = stats[10] or 0
        with_criticism = stats[11] or 0
        with_legal_principle = stats[12] or 0
        
        upheld_rate = (total_upheld / total_made * 100) if total_made > 0 else 0
        
        print(f"\n📊 Key stats ({total} cases):")
        print(f"   Complaints made: {total_made}")
        print(f"   Complaints upheld: {total_upheld} ({upheld_rate:.0f}%)")
        print(f"   Total compensation: £{total_comp:,.2f}")
        print(f"   PFEOs issued: {pfeo_count}")
        print(f"   PFEOs referred to Ministers: {pfeo_referred}")
        print(f"   Cases referred to Ministers: {case_referred}")
        
        print(f"\n📊 Factor conduct:")
        print(f"   Factor no-shows: {no_shows}")
        print(f"   Uncooperative factors: {uncooperative}")
        print(f"   False statements to tribunal: {false_stmt_count}")
        print(f"   Authority exceeded: {authority_count}")
        
        print(f"\n📊 Content quality:")
        print(f"   With criticism quotes: {with_criticism}")
        print(f"   With legal principles: {with_legal_principle}")
        
        # PDF count stats
        cursor = conn.execute("""
            SELECT 
                SUM(pdf_count) as total_pdfs,
                AVG(pdf_count) as avg_pdfs,
                MAX(pdf_count) as max_pdfs,
                SUM(CASE WHEN pdf_count > 1 THEN 1 ELSE 0 END) as multi_pdf_cases
            FROM cases WHERE extraction_success = 1
        """)
        pdf_stats = cursor.fetchone()
        if pdf_stats and pdf_stats[0]:
            print(f"\n📊 Document coverage:")
            print(f"   Total PDFs analyzed: {int(pdf_stats[0] or 0)}")
            print(f"   Average PDFs per case: {pdf_stats[1]:.1f}")
            print(f"   Max PDFs in one case: {int(pdf_stats[2] or 1)}")
            print(f"   Cases with multiple PDFs: {int(pdf_stats[3] or 0)}")
        
        # Multi-ref case stats
        cursor = conn.execute("""
            SELECT 
                SUM(CASE WHEN all_case_references LIKE '%|%' THEN 1 ELSE 0 END) as multi_ref_cases
            FROM cases WHERE extraction_success = 1
        """)
        ref_stats = cursor.fetchone()
        if ref_stats and ref_stats[0]:
            print(f"   Cases with multiple refs (conjoined): {int(ref_stats[0])}")
        
        # PFEO compliance stats
        cursor = conn.execute("""
            SELECT 
                SUM(CASE WHEN pfeo_complied = 1 THEN 1 ELSE 0 END) as complied,
                SUM(CASE WHEN pfeo_complied = 0 THEN 1 ELSE 0 END) as not_complied,
                SUM(CASE WHEN pfeo_complied IS NULL AND pfeo_issued = 1 THEN 1 ELSE 0 END) as pending
            FROM cases WHERE extraction_success = 1
        """)
        pfeo_stats = cursor.fetchone()
        if pfeo_stats[0] or pfeo_stats[1]:
            print(f"\n📊 PFEO compliance:")
            print(f"   Complied: {pfeo_stats[0] or 0}")
            print(f"   Not complied: {pfeo_stats[1] or 0}")
            print(f"   Pending/unknown: {pfeo_stats[2] or 0}")
        
        # Factor type breakdown
        cursor = conn.execute("""
            SELECT factor_type, COUNT(*) FROM cases 
            WHERE extraction_success = 1 AND factor_type != 'unknown'
            GROUP BY factor_type ORDER BY COUNT(*) DESC
        """)
        type_rows = cursor.fetchall()
        if type_rows:
            print(f"\n📊 Factor types:")
            for ftype, count in type_rows:
                print(f"   {count:4d} - {ftype}")
        
        # Severity distribution (grouped by integer bucket)
        print(f"\n📊 Severity distribution:")
        cursor = conn.execute("""
            SELECT CAST(severity_score AS INTEGER) as sev_bucket, COUNT(*) as count
            FROM cases
            WHERE extraction_success = 1
            GROUP BY sev_bucket
            ORDER BY sev_bucket
        """)
        for sev, count in cursor.fetchall():
            bar = "█" * (count // 2)
            print(f"   {sev or 0:2d}: {count:4d} {bar}")
        
        # Complaint category breakdown
        print(f"\n📊 Top complaint categories:")
        cursor = conn.execute("""
            SELECT complaint_categories FROM cases WHERE extraction_success = 1
        """)
        category_counts = {}
        for row in cursor.fetchall():
            try:
                cats = json.loads(row[0]) if row[0] else []
                for cat in cats:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            except:
                pass
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"   {count:4d} - {cat}")
        
        # Tribunal criticism terms
        print(f"\n📊 Tribunal criticism terms:")
        cursor = conn.execute("""
            SELECT tribunal_criticism FROM cases WHERE extraction_success = 1
        """)
        criticism_counts = {}
        for row in cursor.fetchall():
            try:
                terms = json.loads(row[0]) if row[0] else []
                for term in terms:
                    criticism_counts[term] = criticism_counts.get(term, 0) + 1
            except:
                pass
        
        for term, count in sorted(criticism_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"   {count:4d} - {term}")
        
        # Fee breakdown
        cursor = conn.execute("SELECT COUNT(*) FROM case_fees")
        total_fees = cursor.fetchone()[0]
        if total_fees > 0:
            print(f"\n📊 Fee data ({total_fees} fee mentions extracted):")
            cursor = conn.execute("""
                SELECT fee_type, COUNT(*), AVG(amount), SUM(is_disputed)
                FROM case_fees
                WHERE amount IS NOT NULL
                GROUP BY fee_type
                ORDER BY COUNT(*) DESC
            """)
            for row in cursor.fetchall():
                fee_type, count, avg_amt, disputed = row
                avg_str = f"avg £{avg_amt:,.0f}" if avg_amt else "no amounts"
                disp_str = f" ({int(disputed)} disputed)" if disputed else ""
                print(f"   {count:3d} - {fee_type}: {avg_str}{disp_str}")


def main():
    parser = argparse.ArgumentParser(description="Extract data from tribunal PDFs using AI (v4)")
    parser.add_argument('--limit', type=int, help='Limit number of cases to process')
    parser.add_argument('--force', action='store_true', help='Reprocess all cases')
    parser.add_argument('--year', type=int, help='Only process cases from this year')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed')
    parser.add_argument('--all', action='store_true', help='Include unmatched cases')
    parser.add_argument('--no-full-text', action='store_true', help='Skip storing full PDF text')
    parser.add_argument('--export-csv', action='store_true', help='Also export to CSV')
    parser.add_argument('--reextract', action='store_true',
                        help='Re-run AI extraction using stored full text')
    parser.add_argument('--pfeo-only', action='store_true',
                        help='With --reextract: only reprocess cases currently flagged pfeo_issued=1')
    parser.add_argument('--dismissed-only', action='store_true',
                        help='With --reextract: only reprocess cases marked ai_outcome=Dismissed')
    parser.add_argument('--pdf-dir', type=Path, default=DEFAULT_PDF_DIR,
                        help=f'PDF directory (default: {DEFAULT_PDF_DIR})')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRIBUNAL PDF ENRICHMENT v4 (Vertex AI)")
    print("=" * 70)
    
    # Initialize database
    conn = init_database(OUTPUT_DB)
    print(f"📦 Database: {OUTPUT_DB}")
    
    # REEXTRACT MODE
    if args.reextract:
        if args.pfeo_only:
            print("\n🔄 REEXTRACT MODE: PFEO cases only (using stored full text)")
        elif args.dismissed_only:
            print("\n🔄 REEXTRACT MODE: Dismissed cases only (using stored full text)")
        else:
            print("\n🔄 REEXTRACT MODE: Using stored full text")
        
        stored_cases = get_all_stored_cases(conn, pfeo_only=args.pfeo_only, dismissed_only=args.dismissed_only)
        
        if args.pfeo_only:
            print(f"   Found {len(stored_cases)} cases currently flagged pfeo_issued=1")
        elif args.dismissed_only:
            print(f"   Found {len(stored_cases)} cases currently marked ai_outcome='Dismissed'")
        else:
            print(f"   Found {len(stored_cases)} cases with stored text")
        
        if not stored_cases:
            print("❌ No stored text found. Run normal extraction first.")
            return
        
        if args.limit:
            stored_cases = stored_cases[:args.limit]
        
        print(f"   Will reextract {len(stored_cases)} cases")
        
        if args.dry_run:
            print("\n🔍 DRY RUN - would reextract:")
            for case in stored_cases[:10]:
                print(f"   {case['case_reference']}")
            if len(stored_cases) > 10:
                print(f"   ... and {len(stored_cases) - 10} more")
            return
        
        print(f"\n🤖 Initializing Vertex AI...")
        print(f"   Project: {PROJECT_ID}")
        print(f"   Model: {MODEL_ID}")
        model = init_vertex_ai()
        
        print(f"\n🔍 Reextracting {len(stored_cases)} cases...")
        print(f"   Rate: {REQUESTS_PER_MINUTE}/min (~{len(stored_cases) / REQUESTS_PER_MINUTE:.0f} minutes)")
        
        start_time = datetime.now()
        success_count = 0
        error_count = 0
        
        for i, case in enumerate(stored_cases):
            case_ref = case['case_reference']
            pf_number = case['matched_registration_number']
            
            print(f"[{i+1}/{len(stored_cases)}] {case_ref}...", end=" ", flush=True)
            
            try:
                extraction = process_case_from_text(
                    model, case_ref, pf_number,
                    case['full_text'], case['pdf_url'], conn
                )
                
                if extraction.extraction_success:
                    ratio = f"{extraction.complaints_upheld}/{extraction.complaints_made}"
                    sev = extraction.severity_score
                    print(f"✅ {extraction.ai_outcome} ({ratio}) sev={sev:.1f}")
                    success_count += 1
                else:
                    print(f"⚠️ {extraction.extraction_error[:40]}")
                    error_count += 1
                
                save_case_to_db(conn, extraction)
                
                if extraction.extraction_success:
                    try:
                        fees = json.loads(extraction.fees_extracted) if extraction.fees_extracted else []
                        if fees:
                            save_fees_to_db(conn, extraction.case_reference, fees)
                    except json.JSONDecodeError:
                        pass
                
                time.sleep(REQUEST_DELAY)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted! Progress saved to database.")
                break
            except Exception as e:
                print(f"❌ {e}")
                error_count += 1
        
        elapsed = datetime.now() - start_time
        print("\n" + "=" * 70)
        print("REEXTRACTION COMPLETE")
        print("=" * 70)
        print(f"✅ Successful: {success_count}")
        print(f"❌ Errors: {error_count}")
        print(f"⏱️  Time: {elapsed}")
        
        if args.export_csv:
            export_to_csv(conn, OUTPUT_CSV)
            export_fees_to_csv(conn, OUTPUT_FEES_CSV)
        
        _print_final_stats(conn)
        conn.close()
        return
    
    # NORMAL MODE
    if not INPUT_CSV.exists():
        print(f"❌ Input file not found: {INPUT_CSV}")
        print("   Run 03_tribunal_scrape.py first.")
        conn.close()
        return
    
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        all_cases = list(reader)
    
    print(f"📋 Loaded {len(all_cases)} cases from {INPUT_CSV}")
    
    if not args.all and MATCHED_ONLY:
        all_cases = [c for c in all_cases if c.get('matched_registration_number')]
        print(f"   Filtered to {len(all_cases)} matched cases")
    
    if args.year:
        all_cases = [c for c in all_cases if str(c.get('hearing_date', ''))[:4] == str(args.year)]
        print(f"   Filtered to {len(all_cases)} cases from {args.year}")
    
    cases_with_pdfs = [c for c in all_cases if c.get('pdf_urls')]
    print(f"   {len(cases_with_pdfs)} have PDF URLs")
    
    processed = set()
    if not args.force:
        processed = get_processed_cases(conn)
        if processed:
            print(f"   {len(processed)} already processed (skipping)")
    
    # Build pending list with deduplication
    pending = []
    seen_ref_sets = set()  # Track unique case ref combinations
    duplicates_skipped = 0
    
    for c in cases_with_pdfs:
        refs_raw = c.get('case_references', '')
        primary_ref, refs_tuple = normalize_case_refs(refs_raw)
        
        # Skip if we've seen this exact combination of refs (in any order)
        if refs_tuple in seen_ref_sets:
            duplicates_skipped += 1
            continue
        seen_ref_sets.add(refs_tuple)
        
        # Skip if primary ref already processed
        if primary_ref in processed:
            continue
        
        # Also check if any ref in the tuple was processed
        any_processed = any(ref in processed for ref in refs_tuple)
        if any_processed:
            continue
            
        pending.append(c)
    
    if duplicates_skipped:
        print(f"   {duplicates_skipped} duplicate cases skipped (same refs in different order)")
    
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} cases to process")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - would process:")
        for case in pending[:10]:
            refs_raw = case.get('case_references', '')
            primary_ref, refs_tuple = normalize_case_refs(refs_raw)
            factor = case.get('tribunal_property_factor', '')[:40]
            if len(refs_tuple) > 1:
                print(f"   {primary_ref} (+{len(refs_tuple)-1} more refs): {factor}")
            else:
                print(f"   {primary_ref}: {factor}")
        if len(pending) > 10:
            print(f"   ... and {len(pending) - 10} more")
        return
    
    if not pending:
        print("\n✅ All cases already processed!")
        if args.export_csv:
            export_to_csv(conn, OUTPUT_CSV)
            export_fees_to_csv(conn, OUTPUT_FEES_CSV)
        return
    
    print(f"\n🤖 Initializing Vertex AI...")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Model: {MODEL_ID}")
    model = init_vertex_ai()
    
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Processing {len(pending)} cases...")
    print(f"   Rate: {REQUESTS_PER_MINUTE}/min (~{len(pending) / REQUESTS_PER_MINUTE:.0f} minutes)")
    
    use_existing_pdfs = args.pdf_dir and args.pdf_dir.exists()
    if use_existing_pdfs:
        print(f"   📁 Using existing PDFs from: {args.pdf_dir}")
    else:
        print(f"   📥 Downloading PDFs (cache: {PDF_CACHE_DIR})")
    
    start_time = datetime.now()
    success_count = 0
    error_count = 0
    
    for i, case in enumerate(pending):
        refs_raw = case.get('case_references', '')
        case_ref, _ = normalize_case_refs(refs_raw)
        factor_name = case.get('tribunal_property_factor', '')[:30]
        
        print(f"[{i+1}/{len(pending)}] {case_ref} - {factor_name}...", end=" ", flush=True)
        
        try:
            extraction = process_case(
                model, case, PDF_CACHE_DIR, conn,
                store_full_text_flag=not args.no_full_text,
                existing_pdf_dir=args.pdf_dir if use_existing_pdfs else None
            )
            
            if extraction.extraction_success:
                ratio = f"{extraction.complaints_upheld}/{extraction.complaints_made}"
                sev = extraction.severity_score
                print(f"✅ {extraction.ai_outcome} ({ratio}) sev={sev:.1f}")
                success_count += 1
            else:
                print(f"⚠️ {extraction.extraction_error[:40]}")
                error_count += 1
            
            save_case_to_db(conn, extraction)
            
            if extraction.extraction_success:
                try:
                    fees = json.loads(extraction.fees_extracted) if extraction.fees_extracted else []
                    if fees:
                        save_fees_to_db(conn, extraction.case_reference, fees)
                except json.JSONDecodeError:
                    pass
            
            time.sleep(REQUEST_DELAY)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted! Progress saved to database.")
            break
        except Exception as e:
            print(f"❌ {e}")
            error_count += 1
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️  Time: {elapsed}")
    print(f"📦 Database: {OUTPUT_DB}")
    
    if args.export_csv:
        export_to_csv(conn, OUTPUT_CSV)
        export_fees_to_csv(conn, OUTPUT_FEES_CSV)
    
    _print_final_stats(conn)
    
    conn.close()


if __name__ == "__main__":
    main()