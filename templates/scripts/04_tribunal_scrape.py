#!/usr/bin/env python3
"""
============================================================================
SCRIPT 04: TRIBUNAL DECISIONS SCRAPER & FACTOR ENRICHMENT (FIXED)
============================================================================

PURPOSE: Scrape all Property Factor tribunal decisions from the Housing and
         Property Chamber website, match them to your factors database, and
         generate enriched statistics.

FIXES IN THIS VERSION:
  - Blocks generic names like "Property Management" from fuzzy matching
  - Incremental scraping: only downloads new cases not previously grabbed
  - Higher fuzzy match threshold (90% instead of 85%)
  - Weighted average scoring instead of max (more conservative)

SOURCE:  https://housingandpropertychamber.scot/apply-tribunal/property-factors/property-factors-decisions

INPUT:   data/csv/factors_register.csv (from script 02)
         data/manual/tribunal_name_mappings.csv (optional, for tricky matches)

OUTPUTS:
    data/tribunal/tribunal_cases.csv        - All tribunal cases with matches
    data/tribunal/tribunal_unmatched.csv    - Cases that couldn't be matched
    data/tribunal/factors_enriched.csv      - Factors CSV with tribunal stats
    data/tribunal/pdfs/{PF_NUMBER}/         - Downloaded PDFs (optional)

USAGE:
    # Basic scrape (incremental - only new cases)
    python scripts/04_tribunal_scrape.py

    # Force full rescrape
    python scripts/04_tribunal_scrape.py --force

    # With PDF downloads
    python scripts/04_tribunal_scrape.py --download-pdfs

    # Limit pages for testing
    python scripts/04_tribunal_scrape.py --max-pages 5

DEPENDENCIES:
    pip install requests beautifulsoup4 rapidfuzz
============================================================================
"""

import requests
from bs4 import BeautifulSoup
import csv
import os
import re
import time
from collections import defaultdict
from urllib.parse import urljoin
from dataclasses import dataclass, field
from typing import Optional, Set
from pathlib import Path

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️ rapidfuzz not installed, using slower difflib. Install with: pip install rapidfuzz")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_FACTORS_CSV = Path("data/csv/factors_register.csv")
DEFAULT_OUTPUT_DIR = Path("data/tribunal")
MANUAL_MAPPINGS_CSV = Path("data/manual/tribunal_name_mappings.csv")

BASE_URL = "https://housingandpropertychamber.scot"
DECISIONS_URL = f"{BASE_URL}/apply-tribunal/property-factors/property-factors-decisions"

# Matching threshold for fuzzy name matching (raised from 85)
FUZZY_MATCH_THRESHOLD = 90  # 0-100 scale for rapidfuzz

# Names that are too generic to match on - these will be skipped
BLOCKED_NORMALIZED_NAMES = {
    "property management", "property factors", "factoring", "factors",
    "management", "services", "property", "residential", "commercial",
    "management services", "property services", "residential factors",
    ""  # Empty string after normalization
}

# Minimum length for a normalized name to be matchable
MIN_NAME_LENGTH = 4

# Outcome classification
NEGATIVE_OUTCOMES = {
    "PFEO Issued", "PFEO Issued - Non-Compliant", "PFEO Proposed",
    "Failure to Comply", "Decision Issued"
}
POSITIVE_OUTCOMES = {
    "Complied", "Dismissed", "Rejected", "Withdrawn"
}
# Everything else is "neutral" (Unknown, etc.)


# ============================================================================
# DATA STRUCTURE
# ============================================================================

@dataclass
class Decision:
    case_references: list = field(default_factory=list)
    case_urls: list = field(default_factory=list)
    property_factor: str = ""
    registration_number: str = ""
    application_complaints: str = ""
    hearing_date: str = ""
    pdf_files: list = field(default_factory=list)
    upper_tribunal_decisions: list = field(default_factory=list)
    outcome: str = ""
    outcome_type: str = ""  # "negative", "positive", "neutral"
    pfeo_resolved: str = ""  # "resolved", "unresolved", "pending", "" (no PFEO)
    # Matched factor from database
    matched_registration_number: str = ""
    match_method: str = ""  # "registration_number", "exact_name", "fuzzy_name", "manual_mapping", "unmatched"


# ============================================================================
# SCRAPER CLASS
# ============================================================================

class PropertyFactorsScraper:
    
    def __init__(self, factors_csv: Path, output_dir: Path, mappings_csv: Path = None):
        self.output_dir = output_dir
        self.factors_csv = factors_csv
        self.mappings_csv = mappings_csv or MANUAL_MAPPINGS_CSV
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pdfs"), exist_ok=True)
        
        # Load factors database and build lookup indexes
        self.factors_data = {}  # registration_number -> full row dict
        self.factors_by_name = {}  # normalized name -> registration_number
        self.manual_mappings = {}  # normalized tribunal name -> registration_number
        self.factors_fieldnames = []
        self._load_factors_database()
        self._load_manual_mappings()
        
        # Track existing cases for incremental scraping
        self.existing_case_refs: Set[str] = set()
    
    def _is_matchable_name(self, normalized_name: str) -> bool:
        """Check if a normalized name is specific enough to match on."""
        if not normalized_name:
            return False
        if normalized_name in BLOCKED_NORMALIZED_NAMES:
            return False
        if len(normalized_name) < MIN_NAME_LENGTH:
            return False
        return True
    
    def _load_factors_database(self):
        """Load factors CSV and build lookup indexes."""
        print(f"Loading factors database from {self.factors_csv}...")
        
        if not self.factors_csv.exists():
            print(f"❌ Factors file not found: {self.factors_csv}")
            print("   Run 02_registry_enrich.py first.")
            raise FileNotFoundError(f"Factors file not found: {self.factors_csv}")
        
        skipped_generic = 0
        
        with open(self.factors_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.factors_fieldnames = reader.fieldnames or []
            
            for row in reader:
                reg_num = row.get("registration_number", "").strip()
                if reg_num:
                    self.factors_data[reg_num] = dict(row)
                    
                    # Build name indexes for fuzzy matching
                    for name_field in ["name", "trading_name", "register_company_name", "register_trading_name"]:
                        name = row.get(name_field, "").strip()
                        if name:
                            normalized = self._normalize_name(name)
                            # Only index names that are specific enough
                            if self._is_matchable_name(normalized):
                                self.factors_by_name[normalized] = reg_num
                            else:
                                skipped_generic += 1
        
        print(f"  ✅ Loaded {len(self.factors_data)} factors")
        print(f"  ✅ Built {len(self.factors_by_name)} name lookup entries")
        if skipped_generic > 0:
            print(f"  ⚠️ Skipped {skipped_generic} generic names (too broad for matching)")
    
    def _load_manual_mappings(self):
        """Load manual tribunal name to registration number mappings."""
        if not self.mappings_csv.exists():
            print(f"  ℹ️ No manual mappings file at {self.mappings_csv}")
            return
        
        count = 0
        with open(self.mappings_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle None values from CSV
                tribunal_name = (row.get("tribunal_name") or "").strip()
                reg_num = (row.get("registration_number") or "").strip().upper()
                
                # Skip comment lines and empty rows
                if not tribunal_name or tribunal_name.startswith("#"):
                    continue
                if not reg_num or reg_num.startswith("#"):
                    continue
                    
                normalized = self._normalize_name(tribunal_name)
                if normalized:
                    self.manual_mappings[normalized] = reg_num
                    count += 1
        
        if count > 0:
            print(f"  ✅ Loaded {count} manual mappings")
    
    def _load_existing_cases(self) -> Set[str]:
        """Load existing case references from CSV to enable incremental scraping."""
        cases_csv = os.path.join(self.output_dir, "tribunal_cases.csv")
        existing_refs = set()
        
        if not os.path.exists(cases_csv):
            return existing_refs
        
        try:
            with open(cases_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    refs_str = row.get("case_references", "")
                    # Split on pipe separator
                    for ref in refs_str.split(" | "):
                        ref = ref.strip()
                        if ref:
                            existing_refs.add(ref)
        except Exception as e:
            print(f"  ⚠️ Error loading existing cases: {e}")
        
        return existing_refs
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a company name for matching."""
        name = name.lower().strip()
        # Remove common suffixes
        for suffix in [" limited", " ltd", " llp", " plc", " property management", 
                       " property factors", " factoring", " factors", " management services",
                       " management", " services"]:
            name = name.replace(suffix, "")
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def _match_factor(self, decision: Decision) -> tuple:
        """
        Match a tribunal decision to a factor in our database.
        Returns (registration_number, match_method) or ("", "unmatched")
        
        Priority order:
        1. Exact registration number match
        2. Manual mapping (for known difficult names)
        3. Exact normalized name match
        4. Fuzzy name match (using rapidfuzz if available)
        """
        # 1. Try exact registration number match
        if decision.registration_number:
            reg_num = decision.registration_number.strip().upper()
            if reg_num in self.factors_data:
                return reg_num, "registration_number"
        
        # 2. Try manual mapping
        if decision.property_factor:
            normalized = self._normalize_name(decision.property_factor)
            
            # Check manual mappings first (these can override generic name blocking)
            if normalized in self.manual_mappings:
                return self.manual_mappings[normalized], "manual_mapping"
            
            # Skip generic names that match too broadly
            if not self._is_matchable_name(normalized):
                return "", "unmatched_generic"
            
            # 3. Try exact name match
            if normalized in self.factors_by_name:
                return self.factors_by_name[normalized], "exact_name"
            
            # 4. Try fuzzy name match
            best_match = None
            best_score = 0
            
            if RAPIDFUZZ_AVAILABLE:
                # Use rapidfuzz with multiple algorithms for better matching
                for db_name, reg_num in self.factors_by_name.items():
                    # Skip generic database names too
                    if not self._is_matchable_name(db_name):
                        continue
                    
                    # Combine multiple fuzzy matching strategies
                    score1 = fuzz.ratio(normalized, db_name)
                    score2 = fuzz.token_sort_ratio(normalized, db_name)
                    score3 = fuzz.token_set_ratio(normalized, db_name)
                    
                    # Use weighted average instead of max (more conservative)
                    score = (score1 * 0.4 + score2 * 0.3 + score3 * 0.3)
                    
                    if score > best_score and score >= FUZZY_MATCH_THRESHOLD:
                        best_score = score
                        best_match = reg_num
            else:
                # Fallback to difflib
                for db_name, reg_num in self.factors_by_name.items():
                    if not self._is_matchable_name(db_name):
                        continue
                    score = SequenceMatcher(None, normalized, db_name).ratio() * 100
                    if score > best_score and score >= FUZZY_MATCH_THRESHOLD:
                        best_score = score
                        best_match = reg_num
            
            if best_match:
                return best_match, f"fuzzy_name ({best_score:.0f}%)"
        
        return "", "unmatched"
    
    def get_total_pages(self) -> int:
        """Get the total number of pages from pagination."""
        response = self.session.get(DECISIONS_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        last_link = soup.select_one('a.page-link[title="Go to last page"]')
        if last_link:
            href = last_link.get("href", "")
            if "page=" in href:
                page_num = href.split("page=")[-1]
                return int(page_num) + 1
        return 1
    
    def scrape_page(self, page_num: int, skip_existing: bool = True) -> tuple:
        """
        Scrape a single page of decisions.
        
        Returns:
            (decisions, new_count, skipped_count)
        """
        url = f"{DECISIONS_URL}?search_api_fulltext=&page={page_num}"
        print(f"  Scraping page {page_num + 1}...")
        
        response = self.session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        decisions = []
        skipped = 0
        table = soup.select_one("table.tablesaw")
        if not table:
            print(f"    ⚠️ No table found on page {page_num}")
            return decisions, 0, 0
        
        rows = table.select("tbody tr")
        for row in rows:
            decision = self._parse_row(row)
            if decision:
                # Check if we already have this case
                if skip_existing and self.existing_case_refs:
                    is_existing = any(ref in self.existing_case_refs for ref in decision.case_references)
                    if is_existing:
                        skipped += 1
                        continue
                
                # Match to our database
                reg_num, method = self._match_factor(decision)
                decision.matched_registration_number = reg_num
                decision.match_method = method
                decisions.append(decision)
        
        matched = sum(1 for d in decisions if d.matched_registration_number)
        if skipped > 0:
            print(f"    Found {len(decisions)} new decisions ({matched} matched), skipped {skipped} existing")
        else:
            print(f"    Found {len(decisions)} decisions ({matched} matched)")
        
        return decisions, len(decisions), skipped
    
    def _parse_row(self, row) -> Optional[Decision]:
        """Parse a single table row into a Decision object."""
        decision = Decision()
        cells = row.select("td")
        
        if len(cells) < 6:
            return None
        
        # Case Reference Numbers (column 0)
        case_links = cells[0].select("a")
        for link in case_links:
            ref = link.get_text(strip=True)
            url = urljoin(BASE_URL, link.get("href", ""))
            if ref:
                decision.case_references.append(ref)
                decision.case_urls.append(url)
        
        # Property Factor (column 1)
        decision.property_factor = cells[1].get_text(strip=True)
        
        # Registration Number (column 2)
        decision.registration_number = cells[2].get_text(strip=True)
        
        # Application Complaints (column 3)
        decision.application_complaints = cells[3].get_text(strip=True)
        
        # Hearing Date (column 4)
        time_elem = cells[4].select_one("time")
        if time_elem:
            decision.hearing_date = time_elem.get("datetime", "")
        else:
            decision.hearing_date = cells[4].get_text(strip=True)
        
        # PDF Files (column 5)
        file_divs = cells[5].select("div")
        for div in file_divs:
            link = div.select_one("a")
            if link:
                pdf_info = {
                    "name": link.get_text(strip=True),
                    "url": urljoin(BASE_URL, link.get("href", "")),
                }
                size_span = div.select_one("span:not(.file)")
                if size_span:
                    pdf_info["size"] = size_span.get_text(strip=True).strip("()")
                decision.pdf_files.append(pdf_info)
        
        # Upper Tribunal Decisions (column 6)
        if len(cells) > 6:
            ut_links = cells[6].select("a")
            for link in ut_links:
                decision.upper_tribunal_decisions.append({
                    "name": link.get_text(strip=True),
                    "url": link.get("href", "")
                })
        
        # Determine outcome, PFEO resolution status, and outcome type
        decision.outcome, decision.pfeo_resolved, decision.outcome_type = self._determine_outcome(decision.pdf_files)
        
        return decision
    
    def _determine_outcome(self, pdf_files: list) -> tuple:
        """
        Determine the case outcome, PFEO resolution status, and outcome type.
        """
        filenames = " ".join(p.get("name", "").lower() for p in pdf_files)
        
        # Determine outcome
        if "certificate of compliance" in filenames:
            outcome = "Complied"
            pfeo_resolved = "resolved"
        elif "failure to comply with order" in filenames or "failure to comply" in filenames:
            if "pfeo" in filenames and "proposed" not in filenames:
                outcome = "PFEO Issued - Non-Compliant"
            else:
                outcome = "Failure to Comply"
            pfeo_resolved = "unresolved"
        elif "pfeo" in filenames:
            if "proposed" in filenames:
                outcome = "PFEO Proposed"
                pfeo_resolved = "pending"
            else:
                outcome = "PFEO Issued"
                pfeo_resolved = "pending"
        elif "dismissal" in filenames or "dismissed" in filenames:
            outcome = "Dismissed"
            pfeo_resolved = ""
        elif "reject" in filenames:
            outcome = "Rejected"
            pfeo_resolved = ""
        elif "withdrawn" in filenames:
            outcome = "Withdrawn"
            pfeo_resolved = ""
        elif "decision" in filenames:
            outcome = "Decision Issued"
            pfeo_resolved = ""
        else:
            outcome = "Unknown"
            pfeo_resolved = ""
        
        # Determine outcome type
        if outcome in NEGATIVE_OUTCOMES:
            outcome_type = "negative"
        elif outcome in POSITIVE_OUTCOMES:
            outcome_type = "positive"
        else:
            outcome_type = "neutral"
        
        return outcome, pfeo_resolved, outcome_type
    
    def scrape_all(self, max_pages: Optional[int] = None, delay: float = 1.0, 
                    batch_size: int = 5, download_pdfs: bool = False, pdf_delay: float = 0.5,
                    force: bool = False) -> list:
        """
        Scrape all pages of decisions in batches, saving progress after each batch.
        
        Args:
            force: If True, rescrape all cases. If False, only scrape new cases.
        """
        # Load existing cases for incremental scraping
        if not force:
            self.existing_case_refs = self._load_existing_cases()
            if self.existing_case_refs:
                print(f"📋 Found {len(self.existing_case_refs)} existing case references")
                print(f"   Will skip already-scraped cases (use --force to rescrape all)")
        else:
            self.existing_case_refs = set()
            print("🔄 Force mode: will rescrape all cases")
        
        total_pages = self.get_total_pages()
        print(f"Found {total_pages} pages to scrape")
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
            print(f"Limiting to {total_pages} pages")
        
        all_decisions = []
        total_skipped = 0
        page = 0
        batch_num = 0
        consecutive_all_existing = 0  # Track pages with all existing cases
        
        while page < total_pages:
            batch_num += 1
            batch_end = min(page + batch_size, total_pages)
            batch_decisions = []
            batch_skipped = 0
            
            print(f"\n{'='*60}")
            print(f"BATCH {batch_num}: Pages {page + 1} to {batch_end}")
            print(f"{'='*60}")
            
            # Scrape this batch
            for p in range(page, batch_end):
                try:
                    decisions, new_count, skipped_count = self.scrape_page(p, skip_existing=not force)
                    batch_decisions.extend(decisions)
                    batch_skipped += skipped_count
                    
                    # Track consecutive pages with all existing cases
                    if new_count == 0 and skipped_count > 0:
                        consecutive_all_existing += 1
                    else:
                        consecutive_all_existing = 0
                    
                    if p < batch_end - 1:
                        time.sleep(delay)
                        
                except requests.RequestException as e:
                    print(f"  ⚠️ Error on page {p}: {e}")
                    continue
            
            all_decisions.extend(batch_decisions)
            total_skipped += batch_skipped
            
            # Download PDFs for this batch if requested
            if download_pdfs and batch_decisions:
                print(f"\n📥 Downloading PDFs for batch {batch_num}...")
                self.download_pdfs(batch_decisions, delay=pdf_delay)
            
            # Save progress after each batch (if we have new decisions)
            if batch_decisions:
                print(f"\n💾 Saving progress after batch {batch_num}...")
                # Load existing decisions and merge with new ones
                existing_decisions = self._load_existing_decisions() if not force else []
                combined_decisions = existing_decisions + all_decisions
                self.save_cases_csv(combined_decisions)
                self.save_unmatched_csv(combined_decisions)
                self.generate_enriched_factors(combined_decisions)
            
            matched_batch = sum(1 for d in batch_decisions if d.matched_registration_number)
            print(f"✅ Batch {batch_num} complete: {len(batch_decisions)} new decisions ({matched_batch} matched)")
            if batch_skipped > 0:
                print(f"   Skipped {batch_skipped} existing cases")
            print(f"📊 Running total: {len(all_decisions)} new decisions")
            
            # Early exit if we've hit 3 consecutive pages of all existing cases
            # (we've likely caught up to where we left off)
            if not force and consecutive_all_existing >= 3:
                print(f"\n⏹️ Stopping early: found {consecutive_all_existing} consecutive pages with no new cases")
                break
            
            page = batch_end
            
            # Pause between batches (except after the last one)
            if page < total_pages:
                print(f"\n⏳ Pausing before next batch...")
                time.sleep(delay * 2)
        
        matched = sum(1 for d in all_decisions if d.matched_registration_number)
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"New decisions scraped: {len(all_decisions)}")
        print(f"Matched to database: {matched}")
        print(f"Unmatched: {len(all_decisions) - matched}")
        if total_skipped > 0:
            print(f"Skipped (already existed): {total_skipped}")
        
        return all_decisions
    
    def _load_existing_decisions(self) -> list:
        """Load existing decisions from CSV for merging."""
        cases_csv = os.path.join(self.output_dir, "tribunal_cases.csv")
        decisions = []
        
        if not os.path.exists(cases_csv):
            return decisions
        
        try:
            with open(cases_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = Decision()
                    d.matched_registration_number = row.get("matched_registration_number", "")
                    d.match_method = row.get("match_method", "")
                    d.property_factor = row.get("tribunal_property_factor", "")
                    d.registration_number = row.get("tribunal_registration_number", "")
                    d.case_references = [r.strip() for r in row.get("case_references", "").split(" | ") if r.strip()]
                    d.case_urls = [r.strip() for r in row.get("case_urls", "").split(" | ") if r.strip()]
                    d.application_complaints = row.get("application_complaints", "")
                    d.hearing_date = row.get("hearing_date", "")
                    d.outcome = row.get("outcome", "")
                    d.outcome_type = row.get("outcome_type", "")
                    d.pfeo_resolved = row.get("pfeo_resolved", "")
                    
                    # Parse PDF files
                    pdf_names = [n.strip() for n in row.get("pdf_files", "").split(" | ") if n.strip()]
                    pdf_urls = [u.strip() for u in row.get("pdf_urls", "").split(" | ") if u.strip()]
                    d.pdf_files = [{"name": n, "url": u} for n, u in zip(pdf_names, pdf_urls)]
                    
                    # Parse upper tribunal decisions
                    ut_names = [n.strip() for n in row.get("upper_tribunal_decisions", "").split(" | ") if n.strip()]
                    ut_urls = [u.strip() for u in row.get("upper_tribunal_urls", "").split(" | ") if u.strip()]
                    d.upper_tribunal_decisions = [{"name": n, "url": u} for n, u in zip(ut_names, ut_urls)]
                    
                    decisions.append(d)
        except Exception as e:
            print(f"  ⚠️ Error loading existing decisions: {e}")
        
        return decisions
    
    def _safe_filename(self, name: str) -> str:
        """Convert a string to a safe filename."""
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = re.sub(r'\s+', '_', safe)
        safe = safe.strip('._')
        if len(safe) > 100:
            safe = safe[:100]
        return safe or "unnamed"
    
    def _parse_complaint_types(self, complaints_str: str) -> dict:
        """Parse complaint string into standardized categories."""
        complaints_lower = complaints_str.lower()
        return {
            "section_1": "section 1" in complaints_lower or "written statement" in complaints_lower,
            "section_2": "section 2" in complaints_lower or "communication" in complaints_lower,
            "section_3": "section 3" in complaints_lower or "financial" in complaints_lower,
            "section_4": "section 4" in complaints_lower or "debt" in complaints_lower,
            "section_5": "section 5" in complaints_lower or "insurance" in complaints_lower,
            "section_6": "section 6" in complaints_lower or "repair" in complaints_lower or "maintenance" in complaints_lower,
            "section_7": "section 7" in complaints_lower or "complaints resolution" in complaints_lower,
            "pf_duties": "property factor duties" in complaints_lower,
        }
    
    def save_cases_csv(self, decisions: list, filename: str = "tribunal_cases.csv"):
        """Save all tribunal cases to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            writer.writerow([
                "matched_registration_number",
                "match_method",
                "tribunal_property_factor",
                "tribunal_registration_number", 
                "case_references",
                "case_urls",
                "application_complaints",
                "hearing_date",
                "outcome",
                "outcome_type",
                "pfeo_resolved",
                "section_1_written_statement",
                "section_2_communication",
                "section_3_financial",
                "section_4_debt",
                "section_5_insurance",
                "section_6_repairs",
                "section_7_complaints",
                "property_factor_duties",
                "pdf_files",
                "pdf_urls",
                "upper_tribunal_decisions",
                "upper_tribunal_urls"
            ])
            
            for d in decisions:
                complaints = self._parse_complaint_types(d.application_complaints)
                writer.writerow([
                    d.matched_registration_number,
                    d.match_method,
                    d.property_factor,
                    d.registration_number,
                    " | ".join(d.case_references),
                    " | ".join(d.case_urls),
                    d.application_complaints,
                    d.hearing_date,
                    d.outcome,
                    d.outcome_type,
                    d.pfeo_resolved,
                    complaints["section_1"],
                    complaints["section_2"],
                    complaints["section_3"],
                    complaints["section_4"],
                    complaints["section_5"],
                    complaints["section_6"],
                    complaints["section_7"],
                    complaints["pf_duties"],
                    " | ".join(p["name"] for p in d.pdf_files),
                    " | ".join(p["url"] for p in d.pdf_files),
                    " | ".join(u["name"] for u in d.upper_tribunal_decisions),
                    " | ".join(u["url"] for u in d.upper_tribunal_decisions)
                ])
        
        print(f"  📄 Saved {len(decisions)} cases to {filepath}")
        return filepath
    
    def save_unmatched_csv(self, decisions: list, filename: str = "tribunal_unmatched.csv"):
        """Save unmatched tribunal cases for review."""
        unmatched = [d for d in decisions if not d.matched_registration_number]
        if not unmatched:
            return None
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tribunal_property_factor",
                "tribunal_registration_number",
                "case_references",
                "hearing_date",
                "outcome",
                "outcome_type",
                "pfeo_resolved",
                "match_method"  # Added to help diagnose why it didn't match
            ])
            
            for d in unmatched:
                writer.writerow([
                    d.property_factor,
                    d.registration_number,
                    " | ".join(d.case_references),
                    d.hearing_date,
                    d.outcome,
                    d.outcome_type,
                    d.pfeo_resolved,
                    d.match_method
                ])
        
        print(f"  📄 Saved {len(unmatched)} unmatched cases to {filepath}")
        return filepath
    
    def download_pdfs(self, decisions: list, delay: float = 0.5):
        """Download PDFs only for factors in our database, organized by registration number."""
        pdf_dir = os.path.join(self.output_dir, "pdfs")
        
        # Filter to only matched decisions
        matched_decisions = [d for d in decisions if d.matched_registration_number]
        total_pdfs = sum(len(d.pdf_files) for d in matched_decisions)
        
        print(f"  Downloading {total_pdfs} PDFs for {len(matched_decisions)} matched cases...")
        print(f"  (Skipping {len(decisions) - len(matched_decisions)} unmatched cases)")
        
        downloaded = 0
        skipped = 0
        
        for decision in matched_decisions:
            if not decision.pdf_files:
                continue
            
            # Organize by registration number / case reference
            reg_num = decision.matched_registration_number
            case_ref = self._safe_filename(decision.case_references[0] if decision.case_references else "unknown_case")
            
            case_dir = os.path.join(pdf_dir, reg_num, case_ref)
            os.makedirs(case_dir, exist_ok=True)
            
            for pdf in decision.pdf_files:
                try:
                    safe_name = self._safe_filename(pdf["name"])
                    if not safe_name.endswith('.pdf'):
                        safe_name += '.pdf'
                    filepath = os.path.join(case_dir, safe_name)
                    
                    if os.path.exists(filepath):
                        skipped += 1
                        continue
                    
                    response = self.session.get(pdf["url"], stream=True)
                    response.raise_for_status()
                    
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"    Downloaded {downloaded} PDFs...")
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"    ⚠️ Error downloading {pdf['name']}: {e}")
        
        print(f"  ✅ Downloaded {downloaded} PDFs, skipped {skipped} existing")
    
    def generate_enriched_factors(self, decisions: list):
        """Generate enriched factors CSV with tribunal statistics."""
        # Build stats per factor
        factor_stats = defaultdict(lambda: {
            "total_cases": 0,
            "negative_outcomes": 0,
            "positive_outcomes": 0,
            "neutral_outcomes": 0,
            "pfeo_issued": 0,
            "pfeo_proposed": 0,
            "pfeo_resolved": 0,
            "pfeo_unresolved": 0,
            "pfeo_pending": 0,
            "complied": 0,
            "failure_to_comply": 0,
            "dismissed": 0,
            "rejected": 0,
            "withdrawn": 0,
            "section_1": 0,
            "section_2": 0,
            "section_3": 0,
            "section_4": 0,
            "section_5": 0,
            "section_6": 0,
            "section_7": 0,
            "pf_duties": 0,
            "earliest_case": "",
            "latest_case": "",
        })
        
        for d in decisions:
            if not d.matched_registration_number:
                continue
            
            stats = factor_stats[d.matched_registration_number]
            stats["total_cases"] += 1
            
            # Outcome types
            if d.outcome_type == "negative":
                stats["negative_outcomes"] += 1
            elif d.outcome_type == "positive":
                stats["positive_outcomes"] += 1
            else:
                stats["neutral_outcomes"] += 1
            
            # Specific outcomes
            outcome_lower = d.outcome.lower()
            if "pfeo issued" in outcome_lower and "non-compliant" not in outcome_lower:
                stats["pfeo_issued"] += 1
            if "pfeo proposed" in outcome_lower:
                stats["pfeo_proposed"] += 1
            if d.outcome == "Complied":
                stats["complied"] += 1
            if "failure" in outcome_lower:
                stats["failure_to_comply"] += 1
            if d.outcome == "Dismissed":
                stats["dismissed"] += 1
            if d.outcome == "Rejected":
                stats["rejected"] += 1
            if d.outcome == "Withdrawn":
                stats["withdrawn"] += 1
            
            # PFEO resolution
            if d.pfeo_resolved == "resolved":
                stats["pfeo_resolved"] += 1
            elif d.pfeo_resolved == "unresolved":
                stats["pfeo_unresolved"] += 1
            elif d.pfeo_resolved == "pending":
                stats["pfeo_pending"] += 1
            
            # Complaint sections
            complaints = self._parse_complaint_types(d.application_complaints)
            for section in ["section_1", "section_2", "section_3", "section_4", 
                           "section_5", "section_6", "section_7", "pf_duties"]:
                if complaints.get(section):
                    stats[section] += 1
            
            # Date tracking
            if d.hearing_date:
                if not stats["earliest_case"] or d.hearing_date < stats["earliest_case"]:
                    stats["earliest_case"] = d.hearing_date
                if not stats["latest_case"] or d.hearing_date > stats["latest_case"]:
                    stats["latest_case"] = d.hearing_date
        
        # Write enriched factors CSV
        filepath = os.path.join(self.output_dir, "factors_enriched.csv")
        
        # Build output fieldnames
        tribunal_fields = [
            "tribunal_total_cases",
            "tribunal_negative_outcomes",
            "tribunal_positive_outcomes", 
            "tribunal_neutral_outcomes",
            "tribunal_pfeo_issued",
            "tribunal_pfeo_proposed",
            "tribunal_pfeo_resolved",
            "tribunal_pfeo_unresolved",
            "tribunal_pfeo_pending",
            "tribunal_complied",
            "tribunal_failure_to_comply",
            "tribunal_dismissed",
            "tribunal_rejected",
            "tribunal_withdrawn",
            "tribunal_section_1",
            "tribunal_section_2",
            "tribunal_section_3",
            "tribunal_section_4",
            "tribunal_section_5",
            "tribunal_section_6",
            "tribunal_section_7",
            "tribunal_pf_duties",
            "tribunal_earliest_case",
            "tribunal_latest_case",
        ]
        
        output_fields = self.factors_fieldnames + tribunal_fields
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields)
            writer.writeheader()
            
            for reg_num, factor_row in self.factors_data.items():
                output_row = dict(factor_row)
                stats = factor_stats.get(reg_num, {})
                
                output_row["tribunal_total_cases"] = stats.get("total_cases", 0)
                output_row["tribunal_negative_outcomes"] = stats.get("negative_outcomes", 0)
                output_row["tribunal_positive_outcomes"] = stats.get("positive_outcomes", 0)
                output_row["tribunal_neutral_outcomes"] = stats.get("neutral_outcomes", 0)
                output_row["tribunal_pfeo_issued"] = stats.get("pfeo_issued", 0)
                output_row["tribunal_pfeo_proposed"] = stats.get("pfeo_proposed", 0)
                output_row["tribunal_pfeo_resolved"] = stats.get("pfeo_resolved", 0)
                output_row["tribunal_pfeo_unresolved"] = stats.get("pfeo_unresolved", 0)
                output_row["tribunal_pfeo_pending"] = stats.get("pfeo_pending", 0)
                output_row["tribunal_complied"] = stats.get("complied", 0)
                output_row["tribunal_failure_to_comply"] = stats.get("failure_to_comply", 0)
                output_row["tribunal_dismissed"] = stats.get("dismissed", 0)
                output_row["tribunal_rejected"] = stats.get("rejected", 0)
                output_row["tribunal_withdrawn"] = stats.get("withdrawn", 0)
                output_row["tribunal_section_1"] = stats.get("section_1", 0)
                output_row["tribunal_section_2"] = stats.get("section_2", 0)
                output_row["tribunal_section_3"] = stats.get("section_3", 0)
                output_row["tribunal_section_4"] = stats.get("section_4", 0)
                output_row["tribunal_section_5"] = stats.get("section_5", 0)
                output_row["tribunal_section_6"] = stats.get("section_6", 0)
                output_row["tribunal_section_7"] = stats.get("section_7", 0)
                output_row["tribunal_pf_duties"] = stats.get("pf_duties", 0)
                output_row["tribunal_earliest_case"] = stats.get("earliest_case", "")
                output_row["tribunal_latest_case"] = stats.get("latest_case", "")
                
                writer.writerow(output_row)
        
        factors_with_cases = sum(1 for s in factor_stats.values() if s["total_cases"] > 0)
        print(f"  📄 Saved enriched factors to {filepath}")
        print(f"     {factors_with_cases} factors have tribunal cases")
    
    def print_summary(self, decisions: list):
        """Print summary statistics."""
        matched = [d for d in decisions if d.matched_registration_number]
        unmatched = [d for d in decisions if not d.matched_registration_number]
        
        # Count outcomes
        all_outcomes = defaultdict(int)
        for d in decisions:
            all_outcomes[d.outcome] += 1
        
        # Count outcome types
        outcome_types = defaultdict(int)
        for d in decisions:
            outcome_types[d.outcome_type] += 1
        
        # Count PFEO resolution status
        pfeo_resolution = defaultdict(int)
        for d in decisions:
            if d.pfeo_resolved:
                pfeo_resolution[d.pfeo_resolved] += 1
        
        # Count unique factors with cases
        factors_with_cases = len(set(d.matched_registration_number for d in matched))
        
        # Match method breakdown
        match_methods = defaultdict(int)
        for d in matched:
            method = d.match_method.split(" (")[0]  # Strip fuzzy score
            match_methods[method] += 1
        
        # Unmatched reasons
        unmatched_reasons = defaultdict(int)
        for d in unmatched:
            unmatched_reasons[d.match_method] += 1
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total tribunal cases: {len(decisions)}")
        print(f"Matched to database: {len(matched)} ({len(matched)/len(decisions)*100:.1f}%)")
        print(f"Unmatched: {len(unmatched)} ({len(unmatched)/len(decisions)*100:.1f}%)")
        print(f"Unique factors with cases: {factors_with_cases}")
        
        print(f"\n📊 Match method breakdown:")
        for method, count in sorted(match_methods.items(), key=lambda x: -x[1]):
            print(f"   {method}: {count}")
        
        if unmatched_reasons:
            print(f"\n📊 Unmatched reasons:")
            for reason, count in sorted(unmatched_reasons.items(), key=lambda x: -x[1]):
                print(f"   {reason}: {count}")
        
        print(f"\n📊 Outcome type breakdown:")
        for otype in ["negative", "positive", "neutral"]:
            count = outcome_types.get(otype, 0)
            pct = (count / len(decisions)) * 100 if decisions else 0
            emoji = {"negative": "🔴", "positive": "🟢", "neutral": "⚪"}.get(otype, "")
            print(f"   {emoji} {otype}: {count} ({pct:.1f}%)")
        
        print(f"\n📊 Outcome breakdown:")
        for outcome, count in sorted(all_outcomes.items(), key=lambda x: -x[1]):
            pct = (count / len(decisions)) * 100
            print(f"   {outcome}: {count} ({pct:.1f}%)")
        
        if pfeo_resolution:
            total_pfeo = sum(pfeo_resolution.values())
            print(f"\n⚖️ PFEO Resolution Status ({total_pfeo} cases with PFEOs):")
            for status, count in sorted(pfeo_resolution.items(), key=lambda x: -x[1]):
                pct = (count / total_pfeo) * 100
                print(f"   {status}: {count} ({pct:.1f}%)")
        
        # Top factors by case count
        factor_counts = defaultdict(int)
        for d in matched:
            factor_counts[d.matched_registration_number] += 1
        
        print(f"\n🏆 Top 10 factors by tribunal case count:")
        for reg_num, count in sorted(factor_counts.items(), key=lambda x: -x[1])[:10]:
            name = self.factors_data.get(reg_num, {}).get("name", "Unknown")[:35]
            print(f"   {reg_num}: {count} cases - {name}")
        
        print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scrape Property Factor Tribunal Decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/04_tribunal_scrape.py                  # Incremental (new cases only)
  python scripts/04_tribunal_scrape.py --force          # Full rescrape
  python scripts/04_tribunal_scrape.py --download-pdfs  # With PDF downloads
  python scripts/04_tribunal_scrape.py --max-pages 5    # Limited test run
        """
    )
    parser.add_argument("--factors", "-f", type=Path, default=DEFAULT_FACTORS_CSV,
                        help=f"Path to factors CSV (default: {DEFAULT_FACTORS_CSV})")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--max-pages", "-m", type=int, 
                        help="Maximum pages to scrape (for testing)")
    parser.add_argument("--download-pdfs", "-d", action="store_true", 
                        help="Download PDF decision files")
    parser.add_argument("--delay", type=float, default=1.0, 
                        help="Delay between page requests in seconds (default: 1.0)")
    parser.add_argument("--pdf-delay", type=float, default=0.5, 
                        help="Delay between PDF downloads in seconds (default: 0.5)")
    parser.add_argument("--batch-size", "-b", type=int, default=5, 
                        help="Pages per batch before saving (default: 5)")
    parser.add_argument("--force", action="store_true",
                        help="Force full rescrape (ignore existing cases)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRIBUNAL DECISIONS SCRAPER (FIXED)")
    print("=" * 60)
    print(f"Source: {DECISIONS_URL}")
    print(f"Factors: {args.factors}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Full rescrape' if args.force else 'Incremental (new cases only)'}")
    print()
    
    try:
        scraper = PropertyFactorsScraper(
            factors_csv=args.factors,
            output_dir=args.output
        )
    except FileNotFoundError:
        return
    
    # Scrape all decisions
    decisions = scraper.scrape_all(
        max_pages=args.max_pages, 
        delay=args.delay,
        batch_size=args.batch_size,
        download_pdfs=args.download_pdfs,
        pdf_delay=args.pdf_delay,
        force=args.force
    )
    
    # Print final summary (load all decisions for complete stats)
    if not args.force:
        all_decisions = scraper._load_existing_decisions()
        print(f"\n📊 Complete dataset statistics:")
        scraper.print_summary(all_decisions)
    else:
        scraper.print_summary(decisions)
    
    print("📁 Output files:")
    print(f"   {args.output}/tribunal_cases.csv")
    print(f"   {args.output}/tribunal_unmatched.csv")
    print(f"   {args.output}/factors_enriched.csv")
    if args.download_pdfs:
        print(f"   {args.output}/pdfs/")


if __name__ == "__main__":
    main()