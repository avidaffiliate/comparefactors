#!/usr/bin/env python3
"""
============================================================================
SCRIPT 09: WSS EXTRACTION (Vertex AI)
============================================================================

PURPOSE: Extract structured data from Written Statement of Services (WSS)
         documents using Google Vertex AI (Gemini). Captures fee structures,
         service levels, insurance disclosure, and compliance information.

INPUT:  data/wss/wss_urls.csv
        Columns: registration_number, name, document_name, document_url, 
                 document_type, document_year, scraped_business_website

OUTPUT: data/wss/wss_extracted.db (SQLite - primary)
        data/wss/wss_extracted.csv (optional with --export-csv)
        data/wss/wss_extractions.json (optional with --export-json)

PDF SOURCE (checked in order):
  1. data/wss/pdfs/{url_hash}.pdf - Cached downloads
  2. Download from document_url (cached for reuse)

TABLES IN OUTPUT DB:
  - wss_documents: Document metadata and download status
  - wss_extractions: Full JSON extractions from Gemini
  - wss_factor_mapping: Links documents to factor registration numbers
  - wss_key_fields: Flattened key fields for easy querying

KEY EXTRACTED FIELDS:
  - Management fees (amount, frequency, VAT)
  - Billing (frequency, payment terms, float, sinking fund)
  - Debt recovery (late penalties, NOPLI usage, legal charges)
  - Insurance (provider, broker, commission disclosure)
  - Service levels (emergency/urgent/routine response times)
  - Communication (enquiry/complaint response, portal, app)
  - Termination (notice period, majority required)
  - Compliance (code of conduct version, professional memberships)

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
  python scripts/09_wss_extract.py                    # Process new documents only
  python scripts/09_wss_extract.py --force            # Reprocess all documents
  python scripts/09_wss_extract.py --limit 10         # Process first 10 only
  python scripts/09_wss_extract.py --export-csv       # Also export CSV
  python scripts/09_wss_extract.py --export-json      # Also export JSON
  python scripts/09_wss_extract.py --dry-run          # Show what would be processed
  python scripts/09_wss_extract.py --skip-download    # Use cached files only

DEPENDENCIES:
  pip install google-cloud-aiplatform pymupdf requests python-dotenv

COST: ~$0.01-0.02 per PDF (~$0.50-1.00 for 35 unique documents)
TIME: ~10-15 minutes for 35 documents (rate limited)
============================================================================
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
from urllib.parse import urlparse

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
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("⚠️  pymupdf not installed. Run: pip install pymupdf")

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

INPUT_CSV = Path("data/wss/wss_urls.csv")
OUTPUT_DB = Path("data/wss/wss_extracted.db")
OUTPUT_CSV = Path("data/wss/wss_extracted.csv")
OUTPUT_JSON = Path("data/wss/wss_extractions.json")
PDF_CACHE_DIR = Path("data/wss/pdfs")

# Vertex AI settings
PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "scotland-factors-places"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-lite-001"

# Rate limiting
REQUESTS_PER_MINUTE = 14
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Retry settings
MAX_RETRIES = 4
RETRY_DELAYS = [5, 15, 30, 60]

# Download settings
DOWNLOAD_TIMEOUT = 60
MAX_PDF_SIZE_MB = 50
MIN_TEXT_THRESHOLD = 500  # Min chars for successful text extraction

# Document types we can process
PROCESSABLE_TYPES = {"WSS", "SLA/WSS", "Terms/WSS-equivalent", "WSS Part 2", 
                     "Factoring Policy", "WSS v14", "WSS Jul22", "SLA v7",
                     "WSS (embedded)", "WSS Page", "WSS Page/Doc"}
SKIP_TYPES = {"Acquired by James Gibb (dissolved)", "Portal/request", 
              "Services overview", "Via Lowther Homes"}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WSSDocument:
    """Represents a unique WSS document URL and its associated factors."""
    url: str
    document_name: str
    document_type: str
    document_year: Optional[int]
    registration_numbers: List[str] = field(default_factory=list)
    factor_names: List[str] = field(default_factory=list)
    
    # Processing state
    downloaded: bool = False
    local_path: Optional[str] = None
    extracted: bool = False
    extraction_data: Optional[Dict] = None
    error: Optional[str] = None


# ============================================================================
# DATABASE SCHEMA
# ============================================================================

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS wss_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    document_name TEXT,
    document_type TEXT,
    document_year INTEGER,
    local_path TEXT,
    downloaded_at TEXT,
    extracted_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS wss_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    extraction_json TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    gemini_model TEXT,
    confidence_score REAL,
    FOREIGN KEY (document_id) REFERENCES wss_documents(id)
);

CREATE TABLE IF NOT EXISTS wss_factor_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    registration_number TEXT,
    factor_name TEXT,
    FOREIGN KEY (document_id) REFERENCES wss_documents(id)
);

CREATE TABLE IF NOT EXISTS wss_key_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    -- Organization
    legal_name TEXT,
    trading_name TEXT,
    registration_type TEXT,
    pf_registration_number TEXT,
    -- Contact
    phone TEXT,
    email TEXT,
    website TEXT,
    out_of_hours TEXT,
    -- Fees
    management_fee_amount TEXT,
    management_fee_frequency TEXT,
    management_fee_vat INTEGER,
    delegated_authority_limit TEXT,
    sale_transfer_charge TEXT,
    late_penalty TEXT,
    -- Service levels
    emergency_response TEXT,
    urgent_response TEXT,
    routine_response TEXT,
    enquiry_response TEXT,
    complaint_response TEXT,
    -- Insurance
    insurance_provider TEXT,
    insurance_broker TEXT,
    commission_disclosure TEXT,
    -- Billing
    billing_frequency TEXT,
    float_required INTEGER,
    sinking_fund INTEGER,
    nopli INTEGER,
    -- Digital
    portal TEXT,
    app TEXT,
    -- Compliance
    code_of_conduct_version TEXT,
    professional_memberships TEXT,
    -- Termination
    notice_period TEXT,
    majority_required TEXT,
    -- Extraction quality
    confidence_score REAL,
    fields_not_found TEXT,
    FOREIGN KEY (document_id) REFERENCES wss_documents(id)
);

CREATE INDEX IF NOT EXISTS idx_wss_docs_url ON wss_documents(url);
CREATE INDEX IF NOT EXISTS idx_wss_mapping_reg ON wss_factor_mapping(registration_number);
CREATE INDEX IF NOT EXISTS idx_wss_key_fields_doc ON wss_key_fields(document_id);
"""


# ============================================================================
# EXTRACTION SCHEMA AND PROMPT
# ============================================================================

WSS_SCHEMA = {
    "wss_schema_version": "1.0",
    "document_metadata": {
        "filename": "string",
        "file_id": "string",
        "factor_organization": "string",
        "pf_registration_number": "string|null",
        "document_title": "string",
        "document_version": "string|null",
        "issue_date": "string|null",
        "review_date": "string|null",
        "registration_date": "string|null",
        "document_type": "WSS|policy|terms|service_level_agreement"
    },
    "organization_details": {
        "legal_name": "string",
        "trading_name": "string|null",
        "registration_type": "Company|RSL|Local_Authority",
        "charity_number": "string|null",
        "ico_registration": "string|null",
        "vat_number": "string|null",
        "contact": {
            "phone": "string",
            "email": "string",
            "address": "string",
            "out_of_hours": "string|null",
            "website": "string|null"
        },
        "offices": [
            {
                "location": "string",
                "address": "string",
                "phone": "string",
                "email": "string"
            }
        ]
    },
    "authority_to_act": {
        "legal_basis": ["string"],
        "delegated_authority_limit": "string|null",
        "emergency_limit": "boolean",
        "tenement_management_scheme": "boolean",
        "custom_practice": "boolean"
    },
    "core_services": {
        "routine_maintenance": {
            "cleaning": {
                "frequency": "string",
                "scope": ["string"]
            },
            "grounds": {
                "frequency": "string",
                "scope": ["string"]
            },
            "inspections": {
                "frequency": "string",
                "types": ["string"]
            }
        },
        "repairs_response_times": {
            "emergency": {
                "definition": "string",
                "response": "string",
                "make_safe": "string"
            },
            "urgent": {
                "definition": "string",
                "response": "string"
            },
            "routine": {
                "definition": "string",
                "response": "string"
            }
        }
    },
    "financial_charging": {
        "management_fee": {
            "amount": "string",
            "frequency": "string",
            "review_date": "string",
            "vat_applicable": "boolean"
        },
        "billing": {
            "frequency": "string",
            "payment_terms": "string",
            "invoice_contents": ["string"],
            "float_required": "boolean",
            "sinking_fund": "boolean"
        },
        "debt_recovery": {
            "stages": ["string"],
            "late_penalty": "string|null",
            "legal_charges": "string|null",
            "nopli": "boolean"
        },
        "sale_transfer": {
            "admin_charge": "string|null",
            "notice_period": "string",
            "final_account": "string"
        }
    },
    "insurance": {
        "block_buildings": {
            "provider": "string",
            "broker": "string|null",
            "sum_insured": "string",
            "excess": "string",
            "renewal_frequency": "string",
            "valuation_frequency": "string"
        },
        "public_liability": "boolean",
        "commission_disclosure": "string|null"
    },
    "communication": {
        "office_hours": "string",
        "response_times": {
            "enquiries": "string",
            "complaints": "string"
        },
        "portal": "string|null",
        "app": "string|null"
    },
    "complaints": {
        "internal_stages": ["string"],
        "timescales": {
            "stage1": "string",
            "stage2": "string"
        },
        "ftt_reference": "string"
    },
    "termination": {
        "notice_period": "string",
        "majority_required": "string",
        "final_account": "string"
    },
    "declaration_interest": {
        "owns_properties": "boolean",
        "related_entities": ["string"],
        "contractor_relationships": ["string"]
    },
    "compliance": {
        "code_of_conduct_version": "string",
        "gdpr_policy": "boolean",
        "professional_memberships": ["string"]
    },
    "property_specific": {
        "shares": {
            "common_repairs": "string",
            "backcourt": "string",
            "amenity_areas": "string"
        },
        "threshold": "string",
        "additional_services": ["string"]
    },
    "_extraction_metadata": {
        "confidence_score": "float 0.0-1.0",
        "extraction_notes": "string",
        "fields_not_found": ["string"],
        "ambiguous_fields": ["string"]
    }
}

EXTRACTION_PROMPT = f"""You are extracting structured data from a Scottish Property Factor's Written Statement of Services (WSS) document.

Extract ALL available information into the following JSON schema. Use null for fields not found.
Be precise with fees - include VAT status, currency symbols, and conditions.
For time-based fields, preserve exact wording (e.g., "5 working days", "24 hours").

Return ONLY valid JSON matching this schema:

{json.dumps(WSS_SCHEMA, indent=2)}

SCOTTISH FACTORING CONTEXT:
- Property Factors (Scotland) Act 2011 requires WSS compliance
- Code of Conduct for Property Factors (version 2021 is current)
- First-tier Tribunal (Housing and Property Chamber) handles disputes
- Tenement Management Scheme (TMS) may apply under Tenements (Scotland) Act 2004
- NOPLI = Notice of Potential Liability on sale (registers debt against property)
- RSL = Registered Social Landlord (housing association)
- Management fees/floats often in separate "Development Schedule" - note if applicable

EXTRACTION GUIDANCE:
- authority_to_act.legal_basis: Look for deed of conditions, title deeds, TMS references
- authority_to_act.delegated_authority_limit: £ threshold for works without owner consent
- debt_recovery.nopli: Whether they register notices against properties for unpaid debts
- declaration_interest: Related companies, preferred contractors, property ownership
- property_specific.shares: How costs are split (equal, rateable value, floor area, title deed)
- compliance.professional_memberships: PMAS, RICS, Property Ombudsman, ARMA, FCA etc.

In _extraction_metadata:
- List fields you couldn't find in fields_not_found
- Note any ambiguous interpretations in ambiguous_fields
- Provide confidence_score (0.0-1.0) based on document completeness
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def init_vertex_ai() -> GenerativeModel:
    """Initialize Vertex AI and return the model."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and not Path(creds_path).exists():
        print(f"⚠️  Credentials file not found: {creds_path}")
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return GenerativeModel(MODEL_ID)


def get_db_connection() -> sqlite3.Connection:
    """Get database connection with schema initialized."""
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript(DB_SCHEMA)
    return conn


def get_processed_urls(conn: sqlite3.Connection) -> set:
    """Get set of already-processed document URLs."""
    cursor = conn.execute("""
        SELECT url FROM wss_documents 
        WHERE extracted_at IS NOT NULL AND error IS NULL
    """)
    return {row[0] for row in cursor.fetchall()}


def url_to_cache_path(url: str) -> Path:
    """Convert URL to cache file path."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    
    if path_lower.endswith('.pdf'):
        ext = '.pdf'
    elif any(path_lower.endswith(e) for e in ['.html', '.htm', '/']):
        ext = '.html'
    else:
        ext = '.pdf'  # Default to PDF
    
    return PDF_CACHE_DIR / f"{url_hash}{ext}"


def extract_text_locally(pdf_path: Path) -> Tuple[Optional[str], int]:
    """Extract text from PDF using pymupdf."""
    if not HAS_PYMUPDF:
        return None, 0
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        full_text = "\n".join(text_parts)
        
        # Check if we got meaningful text
        if len(full_text.strip()) < MIN_TEXT_THRESHOLD:
            return None, page_count  # Likely scanned PDF
        
        return full_text, page_count
        
    except Exception as e:
        print(f"  ⚠️  PDF read error: {e}")
        return None, 0


# ============================================================================
# LOADING
# ============================================================================

def load_wss_urls(csv_path: Path) -> Dict[str, WSSDocument]:
    """Load CSV and deduplicate by URL."""
    documents: Dict[str, WSSDocument] = {}
    stats = {"total": 0, "unique": 0, "skipped": 0}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats["total"] += 1
            
            url = row.get("document_url", "").strip()
            if not url:
                continue
            
            doc_type = row.get("document_type", "").strip()
            
            # Skip non-processable types
            if doc_type in SKIP_TYPES:
                stats["skipped"] += 1
                continue
            
            reg_num = row.get("registration_number", "").strip()
            factor_name = row.get("name", "").strip()
            
            # Parse year
            year_str = row.get("document_year", "")
            try:
                year = int(float(year_str)) if year_str else None
            except (ValueError, TypeError):
                year = None
            
            if url in documents:
                # Add this factor to existing document
                if reg_num and reg_num not in documents[url].registration_numbers:
                    documents[url].registration_numbers.append(reg_num)
                if factor_name and factor_name not in documents[url].factor_names:
                    documents[url].factor_names.append(factor_name)
            else:
                # New document
                documents[url] = WSSDocument(
                    url=url,
                    document_name=row.get("document_name", "").strip(),
                    document_type=doc_type,
                    document_year=year,
                    registration_numbers=[reg_num] if reg_num else [],
                    factor_names=[factor_name] if factor_name else [],
                )
    
    stats["unique"] = len(documents)
    return documents, stats


# ============================================================================
# DOWNLOADING
# ============================================================================

def download_document(doc: WSSDocument, session: requests.Session) -> bool:
    """Download a document to cache. Returns True if successful."""
    cache_path = url_to_cache_path(doc.url)
    
    # Check cache
    if cache_path.exists():
        doc.downloaded = True
        doc.local_path = str(cache_path)
        print(f"  ✓ Cached: {cache_path.name}")
        return True
    
    # Download
    try:
        print(f"  ↓ Downloading: {doc.url[:70]}...")
        response = session.get(
            doc.url,
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type:
            ext = '.pdf'
        elif 'html' in content_type:
            ext = '.html'
        else:
            ext = '.pdf'  # Default
        
        # Check size
        size_mb = len(response.content) / (1024 * 1024)
        if size_mb > MAX_PDF_SIZE_MB:
            doc.error = f"File too large: {size_mb:.1f}MB"
            return False
        
        # Update cache path with correct extension
        url_hash = hashlib.md5(doc.url.encode()).hexdigest()[:12]
        cache_path = PDF_CACHE_DIR / f"{url_hash}{ext}"
        
        # Save
        PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        
        doc.downloaded = True
        doc.local_path = str(cache_path)
        print(f"  ✓ Saved: {cache_path.name} ({size_mb:.2f}MB)")
        
        time.sleep(1.0)  # Be polite
        return True
        
    except requests.exceptions.RequestException as e:
        doc.error = f"Download failed: {str(e)}"
        print(f"  ✗ Error: {e}")
        return False


# ============================================================================
# EXTRACTION
# ============================================================================

def extract_with_vertex(doc: WSSDocument, model: GenerativeModel) -> Optional[Dict]:
    """Extract data from document using Vertex AI Gemini."""
    
    if not doc.local_path or not Path(doc.local_path).exists():
        doc.error = "Document not downloaded"
        return None
    
    local_path = Path(doc.local_path)
    is_pdf = local_path.suffix.lower() == '.pdf'
    
    print(f"  🤖 Extracting with Vertex AI...")
    
    # Hybrid approach: try local text extraction first for PDFs
    local_text = None
    if is_pdf and HAS_PYMUPDF:
        local_text, page_count = extract_text_locally(local_path)
        if local_text:
            print(f"     📝 Using local text extraction ({len(local_text):,} chars)")
    
    # Prepare content for Gemini
    if local_text:
        # Send as text (cheaper, faster)
        gemini_input = [EXTRACTION_PROMPT, f"WSS DOCUMENT:\n\n{local_text}"]
    else:
        # Send as bytes (for scanned PDFs or HTML)
        print(f"     📄 Using direct file upload")
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        
        if is_pdf:
            file_part = Part.from_data(file_bytes, mime_type="application/pdf")
        else:
            file_part = Part.from_data(file_bytes, mime_type="text/html")
        
        gemini_input = [EXTRACTION_PROMPT, file_part]
    
    # Call Gemini with retries
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            response = model.generate_content(
                gemini_input,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                    "max_output_tokens": 8192,
                }
            )
            
            result = json.loads(response.text)
            
            # Handle if model returns a list instead of dict
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            if not isinstance(result, dict):
                print(f"  ⚠️  Unexpected response type: {type(result)}")
                return None
            
            # Add our metadata
            result["_source_url"] = doc.url
            result["_document_name"] = doc.document_name
            result["_extraction_timestamp"] = datetime.now().isoformat()
            result["_gemini_model"] = MODEL_ID
            result["_registration_numbers"] = doc.registration_numbers
            result["_factor_names"] = doc.factor_names
            
            doc.extracted = True
            doc.extraction_data = result
            
            # Show extracted name
            org = result.get("organization_details", {})
            factor_name = org.get("legal_name") or org.get("trading_name") or "Unknown"
            confidence = result.get("_extraction_metadata", {}).get("confidence_score", "?")
            print(f"  ✓ Extracted: {factor_name} (confidence: {confidence})")
            
            return result
            
        except json.JSONDecodeError as e:
            # Try to recover from markdown code blocks
            try:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text)
            except:
                doc.error = f"JSON parse error: {e}"
                print(f"  ⚠️  JSON parse error")
                return None
                
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "quota" in error_str or "resource exhausted" in error_str:
                print(f"  ⏳ Rate limited, waiting {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ⚠️  Error: {e}")
                if attempt == len(RETRY_DELAYS) - 1:
                    doc.error = f"Extraction error: {str(e)}"
                    return None
                time.sleep(delay)
    
    return None


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def save_document_to_db(conn: sqlite3.Connection, doc: WSSDocument):
    """Save a single document and its extraction to the database."""
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    # Insert document
    cursor.execute("""
        INSERT OR REPLACE INTO wss_documents 
        (url, document_name, document_type, document_year, local_path, downloaded_at, extracted_at, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc.url,
        doc.document_name,
        doc.document_type,
        doc.document_year,
        doc.local_path,
        now if doc.downloaded else None,
        now if doc.extracted else None,
        doc.error
    ))
    
    doc_id = cursor.lastrowid
    
    # Insert extraction if available
    if doc.extraction_data:
        meta = doc.extraction_data.get("_extraction_metadata", {})
        confidence = meta.get("confidence_score")
        
        cursor.execute("""
            INSERT INTO wss_extractions (document_id, extraction_json, extracted_at, gemini_model, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc_id,
            json.dumps(doc.extraction_data, indent=2),
            now,
            MODEL_ID,
            confidence
        ))
        
        # Insert flattened key fields
        insert_key_fields(cursor, doc_id, doc.extraction_data)
    
    # Insert factor mappings
    for i, reg_num in enumerate(doc.registration_numbers):
        factor_name = doc.factor_names[i] if i < len(doc.factor_names) else None
        cursor.execute("""
            INSERT INTO wss_factor_mapping (document_id, registration_number, factor_name)
            VALUES (?, ?, ?)
        """, (doc_id, reg_num, factor_name))
    
    conn.commit()


def insert_key_fields(cursor, doc_id: int, data: Dict):
    """Extract key fields from nested JSON and insert into flattened table."""
    
    def safe_get(d, *keys, default=None):
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, default)
            else:
                return default
        return d if d is not None else default
    
    def bool_to_int(val):
        if val is None:
            return None
        return 1 if val else 0
    
    # Extract fields from nested structure
    org = data.get("organization_details", {})
    contact = org.get("contact", {})
    auth = data.get("authority_to_act", {})
    fin = data.get("financial_charging", {})
    mgmt = fin.get("management_fee", {})
    billing = fin.get("billing", {})
    debt = fin.get("debt_recovery", {})
    sale = fin.get("sale_transfer", {})
    ins = data.get("insurance", {})
    block = ins.get("block_buildings", {})
    comm = data.get("communication", {})
    resp = comm.get("response_times", {})
    svc = data.get("core_services", {})
    repairs = svc.get("repairs_response_times", {})
    comp = data.get("compliance", {})
    term = data.get("termination", {})
    meta = data.get("_extraction_metadata", {})
    doc_meta = data.get("document_metadata", {})
    
    # Professional memberships as comma-separated string
    memberships = comp.get("professional_memberships", [])
    memberships_str = ", ".join(memberships) if memberships else None
    
    # Fields not found as comma-separated string
    not_found = meta.get("fields_not_found", [])
    not_found_str = ", ".join(not_found) if not_found else None
    
    cursor.execute("""
        INSERT INTO wss_key_fields (
            document_id,
            legal_name, trading_name, registration_type, pf_registration_number,
            phone, email, website, out_of_hours,
            management_fee_amount, management_fee_frequency, management_fee_vat,
            delegated_authority_limit, sale_transfer_charge, late_penalty,
            emergency_response, urgent_response, routine_response,
            enquiry_response, complaint_response,
            insurance_provider, insurance_broker, commission_disclosure,
            billing_frequency, float_required, sinking_fund, nopli,
            portal, app,
            code_of_conduct_version, professional_memberships,
            notice_period, majority_required,
            confidence_score, fields_not_found
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc_id,
        org.get("legal_name"),
        org.get("trading_name"),
        org.get("registration_type"),
        doc_meta.get("pf_registration_number"),
        contact.get("phone"),
        contact.get("email"),
        contact.get("website"),
        contact.get("out_of_hours"),
        mgmt.get("amount"),
        mgmt.get("frequency"),
        bool_to_int(mgmt.get("vat_applicable")),
        auth.get("delegated_authority_limit"),
        sale.get("admin_charge"),
        debt.get("late_penalty"),
        safe_get(repairs, "emergency", "response"),
        safe_get(repairs, "urgent", "response"),
        safe_get(repairs, "routine", "response"),
        resp.get("enquiries"),
        resp.get("complaints"),
        block.get("provider"),
        block.get("broker"),
        ins.get("commission_disclosure"),
        billing.get("frequency"),
        bool_to_int(billing.get("float_required")),
        bool_to_int(billing.get("sinking_fund")),
        bool_to_int(debt.get("nopli")),
        comm.get("portal"),
        comm.get("app"),
        comp.get("code_of_conduct_version"),
        memberships_str,
        term.get("notice_period"),
        term.get("majority_required"),
        meta.get("confidence_score"),
        not_found_str
    ))


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_csv(conn: sqlite3.Connection, output_path: Path):
    """Export key fields to CSV."""
    cursor = conn.execute("""
        SELECT 
            m.registration_number,
            m.factor_name,
            k.*
        FROM wss_key_fields k
        JOIN wss_factor_mapping m ON k.document_id = m.document_id
        ORDER BY m.registration_number
    """)
    
    rows = cursor.fetchall()
    if not rows:
        print("  ⚠️  No data to export")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([desc[0] for desc in cursor.description])
        writer.writerows(rows)
    
    print(f"  ✅ Exported {len(rows)} rows to {output_path}")


def export_to_json(conn: sqlite3.Connection, output_path: Path):
    """Export full extractions to JSON."""
    cursor = conn.execute("""
        SELECT extraction_json FROM wss_extractions
    """)
    
    extractions = []
    for row in cursor:
        try:
            extractions.append(json.loads(row[0]))
        except json.JSONDecodeError:
            pass
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extractions, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Exported {len(extractions)} extractions to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract data from WSS documents using Vertex AI")
    parser.add_argument('--force', action='store_true', help='Reprocess all documents')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    parser.add_argument('--export-csv', action='store_true', help='Export to CSV after processing')
    parser.add_argument('--export-json', action='store_true', help='Export to JSON after processing')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed')
    parser.add_argument('--skip-download', action='store_true', help='Use cached files only')
    parser.add_argument('--input', type=str, help='Input CSV path (default: data/wss/wss_urls.csv)')
    args = parser.parse_args()
    
    input_csv = Path(args.input) if args.input else INPUT_CSV
    
    print("=" * 70)
    print("WSS EXTRACTION (Vertex AI)")
    print("=" * 70)
    print(f"Input: {input_csv}")
    print(f"Output: {OUTPUT_DB}")
    print(f"Model: {MODEL_ID}")
    print()
    
    # Check input file
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        print("   Place wss_urls.csv in data/wss/ directory")
        return
    
    # Load documents
    documents, load_stats = load_wss_urls(input_csv)
    print(f"📋 Loaded {load_stats['total']} rows, {load_stats['unique']} unique URLs")
    print(f"   Skipped {load_stats['skipped']} (non-processable types)")
    
    # Initialize database
    conn = get_db_connection()
    
    # Filter to unprocessed
    processed_urls = set()
    if not args.force:
        processed_urls = get_processed_urls(conn)
        if processed_urls:
            print(f"   {len(processed_urls)} already processed (skipping)")
    
    # Build pending list
    pending = [(url, doc) for url, doc in documents.items() if url not in processed_urls]
    
    if args.limit:
        pending = pending[:args.limit]
    
    print(f"   {len(pending)} documents to process")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - would process:")
        for url, doc in pending[:10]:
            print(f"   [{doc.document_type}] {doc.document_name}")
            print(f"     URL: {url[:60]}...")
            print(f"     Factors: {', '.join(doc.registration_numbers) or 'none'}")
            print()
        if len(pending) > 10:
            print(f"   ... and {len(pending) - 10} more")
        conn.close()
        return
    
    if not pending:
        print("\n✅ All documents already processed!")
        if args.export_csv:
            export_to_csv(conn, OUTPUT_CSV)
        if args.export_json:
            export_to_json(conn, OUTPUT_JSON)
        conn.close()
        return
    
    # Initialize Vertex AI
    print(f"\n🤖 Initializing Vertex AI...")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Model: {MODEL_ID}")
    model = init_vertex_ai()
    print("   ✅ Connected")
    
    # Estimate cost and time
    est_cost = len(pending) * 0.015
    est_time = len(pending) * (60 / REQUESTS_PER_MINUTE) / 60
    print(f"\n💰 Estimated cost: ~${est_cost:.2f}")
    print(f"⏱️  Estimated time: ~{est_time:.1f} minutes")
    
    # Create session for downloads
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    # Process documents
    print(f"\n🔍 Processing {len(pending)} documents...")
    
    success_count = 0
    error_count = 0
    start_time = datetime.now()
    
    for i, (url, doc) in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {doc.document_name}")
        print(f"  Type: {doc.document_type} | Year: {doc.document_year or 'unknown'}")
        print(f"  Factors: {', '.join(doc.registration_numbers) or 'none'}")
        
        # Download
        if not args.skip_download:
            download_document(doc, session)
        else:
            # Check cache
            cache_path = url_to_cache_path(url)
            if cache_path.exists():
                doc.downloaded = True
                doc.local_path = str(cache_path)
                print(f"  ✓ Cached: {cache_path.name}")
        
        # Extract
        if doc.downloaded:
            extract_with_vertex(doc, model)
            
            if doc.extracted:
                success_count += 1
            else:
                error_count += 1
            
            # Save to database
            save_document_to_db(conn, doc)
            
            # Rate limiting
            if i < len(pending) - 1:
                time.sleep(REQUEST_DELAY)
        else:
            error_count += 1
            save_document_to_db(conn, doc)
    
    elapsed = datetime.now() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️  Time: {elapsed}")
    print(f"📦 Database: {OUTPUT_DB}")
    
    # Export if requested
    if args.export_csv:
        export_to_csv(conn, OUTPUT_CSV)
    if args.export_json:
        export_to_json(conn, OUTPUT_JSON)
    
    # Final stats
    total_docs = conn.execute("SELECT COUNT(*) FROM wss_documents").fetchone()[0]
    total_mappings = conn.execute("SELECT COUNT(*) FROM wss_factor_mapping").fetchone()[0]
    total_fields = conn.execute("SELECT COUNT(*) FROM wss_key_fields").fetchone()[0]
    
    print(f"\n📊 Database totals:")
    print(f"   Documents: {total_docs}")
    print(f"   Factor mappings: {total_mappings}")
    print(f"   Key field records: {total_fields}")
    
    conn.close()


if __name__ == "__main__":
    main()
