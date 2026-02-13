#!/usr/bin/env python3
"""Regenerate factor profile pages with fixed complaint_categories parsing."""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

DB_PATH = Path("data/database/comparefactors.db")
TEMPLATE_DIR = Path("templates")
FACTORS_DIR = Path("site/factors")


def is_adverse_outcome(outcome):
    """Determine if a tribunal case outcome is adverse for the factor."""
    if not outcome:
        return False
    outcome_lower = outcome.lower()
    not_adverse = ['dismissed', 'rejected', 'withdrawn', 'unknown', 'complied', 'procedural', 'settled', 'ambiguous']
    for term in not_adverse:
        if term in outcome_lower:
            return False
    return True


def main():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template("factor_profile.html")

    # Get all factors with profiles
    cursor = conn.execute("SELECT * FROM v_factor_profiles ORDER BY name")
    factors = [dict(row) for row in cursor]
    print(f"Found {len(factors)} factors")

    # Calculate sector averages
    sector_stats = conn.execute("""
        SELECT
            AVG(tribunal_rate_per_10k) AS avg_rate,
            AVG(CAST(tribunal_cases_upheld AS FLOAT) / NULLIF(tribunal_case_count, 0) * 100) AS avg_upheld
        FROM factors
        WHERE tribunal_case_count > 0
    """).fetchone()
    sector_avg_rate = round(sector_stats['avg_rate'] or 2.1, 1)
    sector_avg_upheld = round(sector_stats['avg_upheld'] or 45, 0)

    current_year = datetime.now().year
    cutoff_5y = current_year - 5

    generated = 0
    for i, profile in enumerate(factors):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(factors)}...")

        pf = profile['registration_number']
        output_dir = FACTORS_DIR / pf.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get tribunal cases
            cases_raw = conn.execute("""
                SELECT * FROM tribunal_cases
                WHERE factor_registration_number = ?
                ORDER BY decision_date DESC
            """, [pf]).fetchall()
            cases = [dict(c) for c in cases_raw]

            # Parse complaint_categories JSON for each case
            for case in cases:
                if case.get('complaint_categories'):
                    try:
                        parsed = json.loads(case['complaint_categories'])
                        case['complaint_categories'] = parsed if isinstance(parsed, list) else []
                    except:
                        case['complaint_categories'] = []
                else:
                    case['complaint_categories'] = []

            # Calculate tribunal stats
            cases_5y = [c for c in cases if c['decision_date'] and c['decision_date'][:4].isdigit() and int(c['decision_date'][:4]) >= cutoff_5y]

            enriched_case_count = profile.get('tribunal_case_count') or 0
            enriched_upheld = profile.get('tribunal_cases_upheld') or 0

            if enriched_case_count > 0:
                case_count_5y = enriched_case_count
                adverse_5y = enriched_upheld
            else:
                case_count_5y = len(cases_5y)
                adverse_5y = sum(1 for c in cases_5y if is_adverse_outcome(c.get('outcome')))

            tribunal_last_5_years = {
                'case_count': case_count_5y,
                'pfeo_count': sum(1 for c in cases_5y if c['pfeo_issued']) if cases_5y else (profile.get('tribunal_pfeo_count') or 0),
                'compensation': sum(c['compensation_awarded'] or 0 for c in cases_5y) if cases_5y else (profile.get('tribunal_total_compensation') or 0),
                'rate_per_10k': profile['tribunal_rate_per_10k'],
                'upheld': adverse_5y if case_count_5y > 0 else None,
                'upheld_count': adverse_5y if case_count_5y > 0 else None,
                'adverse_outcome_pct': (adverse_5y / case_count_5y * 100) if case_count_5y > 0 else None,
                'complaints_upheld_pct': (adverse_5y / case_count_5y * 100) if case_count_5y > 0 else None,
            }

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

            # Cases by year for trend chart
            year_data = {}
            for c in cases:
                if c['decision_date'] and len(c['decision_date']) >= 4:
                    year = c['decision_date'][:4]
                    if year.isdigit():
                        year_int = int(year)
                        if year_int not in year_data:
                            year_data[year_int] = {'year': year_int, 'count': 0, 'upheld': 0}
                        year_data[year_int]['count'] += 1
                        if is_adverse_outcome(c.get('outcome')):
                            year_data[year_int]['upheld'] += 1
            cases_by_year = [year_data[y] for y in sorted(year_data.keys())]

            # Complaint categories
            complaint_categories_dict = {}
            for c in cases:
                if c.get('complaint_categories'):
                    for cat in c['complaint_categories']:
                        complaint_categories_dict[cat] = complaint_categories_dict.get(cat, 0) + 1

            complaint_categories = sorted(
                [{'name': k, 'count': v} for k, v in complaint_categories_dict.items()],
                key=lambda x: x['count'],
                reverse=True
            )

            # Build profile dict with required fields
            profile_data = dict(profile)
            profile_data['slug'] = pf.lower()

            # At a glance and band
            band = profile.get('risk_band') or 'UNKNOWN'
            at_a_glance = {
                'risk_band': band,
                'case_count': case_count_5y,
                'pfeo_count': tribunal_last_5_years['pfeo_count'],
            }

            # Google Places data (table may not exist)
            google_places = None
            google_locations = []
            try:
                gp_row = conn.execute("SELECT * FROM google_places WHERE factor_registration_number = ?", [pf]).fetchone()
                if gp_row:
                    google_places = dict(gp_row)
                gl_rows = conn.execute("SELECT * FROM google_locations WHERE factor_registration_number = ?", [pf]).fetchall()
                for gl in gl_rows:
                    google_locations.append(dict(gl))
            except sqlite3.OperationalError:
                pass

            # Trustpilot (table may not exist)
            trustpilot = None
            try:
                tp_row = conn.execute("SELECT * FROM trustpilot_reviews WHERE registration_number = ?", [pf]).fetchone()
                if tp_row:
                    trustpilot = dict(tp_row)
            except sqlite3.OperationalError:
                pass

            # WSS data
            wss_data = None
            wss_url = None
            try:
                wss_row = conn.execute("""
                    SELECT document_url, management_fee_amount, management_fee_frequency,
                           delegated_authority_limit, emergency_response, urgent_response,
                           routine_response, enquiry_response, complaint_response,
                           billing_frequency, float_required, notice_period,
                           code_of_conduct_version, professional_memberships, portal, app,
                           confidence_score
                    FROM wss WHERE registration_number = ?
                """, [pf]).fetchone()
                if wss_row:
                    wss_data = dict(wss_row)
                    wss_url = wss_row['document_url']
            except sqlite3.OperationalError:
                pass

            # Recent reviews (with text for snippets)
            recent_reviews = []
            try:
                review_rows = conn.execute("""
                    SELECT platform, rating, review_text, review_date, author_name
                    FROM reviews
                    WHERE factor_registration_number = ? AND review_text IS NOT NULL AND review_text != ''
                    ORDER BY review_date DESC
                    LIMIT 5
                """, [pf]).fetchall()
                recent_reviews = [dict(r) for r in review_rows]
            except sqlite3.OperationalError:
                pass

            # Deep review page (check if one exists)
            deep_review_url = None
            deep_review_path = FACTORS_DIR / pf.lower() / "review" / "index.html"
            if deep_review_path.exists():
                deep_review_url = f"/factors/{pf.lower()}/review/"

            # Registry URL
            registry_url = f"https://www.gov.scot/publications/property-factor-register/pages/{pf.lower()}/"

            # Coverage areas
            coverage_areas_list = []
            if profile.get('coverage'):
                coverage_areas_list = [a.strip() for a in profile['coverage'].split(',')]

            html = template.render(
                profile=profile_data,
                band=band,
                at_a_glance=at_a_glance,
                google_places=google_places,
                google_locations=google_locations,
                trustpilot=trustpilot,
                wss=wss_data,
                wss_url=wss_url,
                fee_examples=[],
                fee_summary=None,
                case_fees=[],
                registry_url=registry_url,
                coverage_areas_list=coverage_areas_list,
                tribunal_last_5_years=tribunal_last_5_years,
                tribunal_full_history=tribunal_full_history,
                cases_by_year=cases_by_year,
                recent_cases=cases[:5],
                complaint_categories=complaint_categories,
                recent_reviews=recent_reviews,
                deep_review_url=deep_review_url,
                similar_factors=[],
                timeline_events=[],
                generated_date=datetime.now().strftime('%Y-%m-%d'),
                reviews_updated=datetime.now().strftime('%Y-%m-%d'),
                tribunal_updated=datetime.now().strftime('%Y-%m-%d'),
                sector_avg_upheld=sector_avg_upheld,
                sector_avg_rate=sector_avg_rate,
                current_year=current_year,
            )

            with open(output_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html)

            generated += 1

        except Exception as e:
            print(f"  Failed to generate profile {pf}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nGenerated {generated} profile pages")
    conn.close()


if __name__ == '__main__':
    main()
