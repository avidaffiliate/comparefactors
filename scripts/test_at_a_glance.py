#!/usr/bin/env python3
"""
Test At a Glance AI generation on specific factors.

Usage:
    python test_at_a_glance.py PF000123 PF000456
    python test_at_a_glance.py --list 5           # Show 5 factors without summaries
    python test_at_a_glance.py --top 3            # Generate for top 3 by case count
"""

import sqlite3
import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

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
            print(f"📁 Loaded .env from: {env_path}")
            break
    else:
        load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables directly")

# Configuration - adjust path as needed
DB_PATH = Path("data/database/comparefactors.db")

# Try importing Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False
    print("⚠️  Vertex AI not available. Install: pip install google-cloud-aiplatform")

# GCP config - match the enrich script
GCP_PROJECT = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "compare-factors"
GCP_LOCATION = "us-central1"  # Match enrich script
GEMINI_MODEL = "gemini-2.0-flash-lite-001"  # Match enrich script

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


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
    
    # Tribunal data - wrap in try/except
    try:
        cases = conn.execute("""
            SELECT 
                case_reference, decision_date, outcome, pfeo_issued,
                compensation_awarded, summary, complaint_categories, severity_score
            FROM tribunal_cases
            WHERE factor_registration_number = ?
            ORDER BY decision_date DESC
            LIMIT 10
        """, [pf]).fetchall()
        
        if cases:
            total_cases = conn.execute(
                "SELECT COUNT(*) FROM tribunal_cases WHERE factor_registration_number = ?", [pf]
            ).fetchone()[0]
            
            pfeo_count = sum(1 for c in cases if c['pfeo_issued'])
            upheld_count = sum(1 for c in cases if c['outcome'] and 'upheld' in c['outcome'].lower() and 'not' not in c['outcome'].lower())
            
            all_categories = []
            for case in cases:
                if case['complaint_categories']:
                    try:
                        cats = json.loads(case['complaint_categories'])
                        if isinstance(cats, list):
                            all_categories.extend(cats)
                    except:
                        pass
            
            category_counts = Counter(all_categories)
            top_categories = [cat for cat, _ in category_counts.most_common(5)]
            
            context["tribunal"] = {
                "total_cases": total_cases,
                "recent_cases": len(cases),
                "pfeo_count": pfeo_count,
                "top_complaint_categories": top_categories,
                "recent_case_summaries": [
                    {
                        "date": c['decision_date'],
                        "outcome": c['outcome'],
                        "summary": c['summary'][:200] if c['summary'] else None,
                        "pfeo": c['pfeo_issued'],
                        "compensation": c['compensation_awarded']
                    }
                    for c in cases[:5] if c['summary']
                ]
            }
    except Exception as e:
        # Fall back to factor-level tribunal data
        if factor.get('tribunal_case_count'):
            context["tribunal"] = {
                "total_cases": factor.get('tribunal_case_count'),
                "note": "Detailed case data unavailable"
            }
    
    # Reviews - wrap in try/except as schema may vary
    reviews_data = {}
    
    try:
        google_reviews = conn.execute("""
            SELECT rating, review_count, review_text
            FROM reviews
            WHERE factor_registration_number = ? AND source = 'google'
            ORDER BY review_count DESC NULLS LAST
            LIMIT 5
        """, [pf]).fetchall()
        
        if google_reviews:
            aggregate = next((r for r in google_reviews if r['review_count']), None)
            samples = [r['review_text'] for r in google_reviews if r['review_text']][:3]
            
            reviews_data["google"] = {
                "rating": aggregate['rating'] if aggregate else factor.get('google_rating'),
                "count": aggregate['review_count'] if aggregate else factor.get('google_review_count'),
                "sample_reviews": samples if samples else None
            }
    except Exception as e:
        # Fall back to factor-level data
        if factor.get('google_rating'):
            reviews_data["google"] = {
                "rating": factor.get('google_rating'),
                "count": factor.get('google_review_count'),
                "sample_reviews": None
            }
    
    try:
        tp_reviews = conn.execute("""
            SELECT rating, review_count, review_text
            FROM reviews
            WHERE factor_registration_number = ? AND source = 'trustpilot'
            ORDER BY review_count DESC NULLS LAST
            LIMIT 5
        """, [pf]).fetchall()
        
        if tp_reviews:
            aggregate = next((r for r in tp_reviews if r['review_count']), None)
            samples = [r['review_text'] for r in tp_reviews if r['review_text']][:3]
            
            reviews_data["trustpilot"] = {
                "rating": aggregate['rating'] if aggregate else factor.get('trustpilot_rating'),
                "count": aggregate['review_count'] if aggregate else factor.get('trustpilot_review_count'),
                "sample_reviews": samples if samples else None
            }
    except Exception as e:
        # Fall back to factor-level data
        if factor.get('trustpilot_rating'):
            reviews_data["trustpilot"] = {
                "rating": factor.get('trustpilot_rating'),
                "count": factor.get('trustpilot_review_count'),
                "sample_reviews": None
            }
    
    if reviews_data:
        context["reviews"] = reviews_data
    
    # WSS fees - wrap in try/except
    try:
        wss = conn.execute("""
            SELECT management_fee_amount, insurance_admin_fee, late_penalty
            FROM wss WHERE registration_number = ?
        """, [pf]).fetchone()
        
        if wss:
            context["fees_wss"] = {
                "management_fee": wss['management_fee_amount'],
                "insurance_admin": wss['insurance_admin_fee'],
                "late_penalty": wss['late_penalty']
            }
    except Exception:
        pass
    
    # Tribunal fees - wrap in try/except
    try:
        case_fees = conn.execute("""
            SELECT fee_type, amount, frequency
            FROM case_fees
            WHERE factor_registration_number = ?
              AND amount > 0 AND amount < 1000
            ORDER BY fee_type LIMIT 6
        """, [pf]).fetchall()
        
        if case_fees:
            context["fees_tribunal"] = [
                {"type": f['fee_type'], "amount": f['amount'], "frequency": f['frequency']}
                for f in case_fees
            ]
    except Exception:
        pass
    
    # Company info - wrap in try/except
    try:
        company = conn.execute("""
            SELECT company_name, company_number, incorporation_date, company_status
            FROM companies_house WHERE registration_number = ?
        """, [pf]).fetchone()
        
        if company:
            context["company"] = {
                "name": company['company_name'],
                "number": company['company_number'],
                "incorporated": company['incorporation_date'],
                "status": company['company_status']
            }
    except Exception:
        pass
    
    return context


def generate_summary(model, conn, factor: dict, save: bool = False) -> dict:
    """Generate At a Glance summary for a factor."""
    context = build_factor_context(conn, factor)
    
    print(f"\n{'='*60}")
    print(f"Factor: {factor['name']}")
    print(f"PF: {factor['registration_number']}")
    print(f"Risk Band: {factor.get('risk_band', 'N/A')}")
    print(f"Cases: {factor.get('tribunal_case_count', 0)}")
    print(f"{'='*60}")
    
    print("\n📊 Context being sent to AI:")
    print(json.dumps(context, indent=2, default=str)[:2000] + "..." if len(json.dumps(context)) > 2000 else json.dumps(context, indent=2, default=str))
    
    prompt = AT_A_GLANCE_PROMPT + json.dumps(context, indent=2, default=str)
    response = model.generate_content(prompt)
    
    text = response.text.strip()
    if text.startswith('```'):
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
    
    result = json.loads(text)
    result['generated_at'] = datetime.now().isoformat()
    result['data_sources'] = list(context.keys())
    
    print("\n✨ Generated At a Glance:")
    print("-" * 40)
    for bullet in result.get('bullets', []):
        print(f"  • {bullet}")
    print(f"\nSentiment: {result.get('overall_sentiment')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Data sources: {', '.join(result.get('data_sources', []))}")
    
    if save:
        conn.execute("""
            UPDATE factors SET
                at_a_glance = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE registration_number = ?
        """, [json.dumps(result), factor['registration_number']])
        conn.commit()
        print(f"\n💾 Saved to database")
    
    return result


def list_factors_without_summary(conn, limit: int = 10):
    """List factors that don't have summaries yet."""
    cursor = conn.execute("""
        SELECT registration_number, name, tribunal_case_count, risk_band
        FROM factors
        WHERE at_a_glance IS NULL AND status = 'registered'
        ORDER BY tribunal_case_count DESC
        LIMIT ?
    """, [limit])
    
    factors = list(cursor)
    
    print(f"\n📋 Top {len(factors)} factors without At a Glance summaries:")
    print("-" * 70)
    for f in factors:
        cases = f['tribunal_case_count'] or 0
        risk = f['risk_band'] or 'N/A'
        print(f"  {f['registration_number']:12} | {cases:3} cases | {risk:6} | {f['name'][:40]}")
    
    return factors


def main():
    parser = argparse.ArgumentParser(description="Test At a Glance AI generation")
    parser.add_argument('pf_numbers', nargs='*', help='PF registration numbers to process')
    parser.add_argument('--list', type=int, metavar='N', help='List N factors without summaries')
    parser.add_argument('--top', type=int, metavar='N', help='Generate for top N by case count')
    parser.add_argument('--save', action='store_true', help='Save results to database')
    parser.add_argument('--db', type=str, default="data/database/comparefactors.db", help='Database path')
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    if args.list:
        list_factors_without_summary(conn, args.list)
        return
    
    if not HAS_VERTEX:
        print("❌ Cannot generate summaries without Vertex AI")
        sys.exit(1)
    
    # Initialize Vertex AI
    print("🔌 Initializing Vertex AI...")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = GenerativeModel(GEMINI_MODEL)
    
    # Get factors to process
    if args.top:
        cursor = conn.execute("""
            SELECT registration_number, name, status, factor_type,
                   tribunal_case_count, tribunal_composite_score, risk_band,
                   google_rating, trustpilot_rating, google_review_count, trustpilot_review_count
            FROM factors
            WHERE at_a_glance IS NULL AND status = 'registered'
            ORDER BY tribunal_case_count DESC
            LIMIT ?
        """, [args.top])
        factors = [dict(row) for row in cursor]
    elif args.pf_numbers:
        factors = []
        for pf in args.pf_numbers:
            pf_upper = pf.upper()
            cursor = conn.execute("""
                SELECT registration_number, name, status, factor_type,
                       tribunal_case_count, tribunal_composite_score, risk_band,
                       google_rating, trustpilot_rating, google_review_count, trustpilot_review_count
                FROM factors
                WHERE registration_number = ?
            """, [pf_upper])
            row = cursor.fetchone()
            if row:
                factors.append(dict(row))
            else:
                print(f"⚠️  Factor not found: {pf}")
    else:
        print("Usage: python test_at_a_glance.py PF000123 [PF000456 ...]")
        print("       python test_at_a_glance.py --list 10")
        print("       python test_at_a_glance.py --top 3 --save")
        return
    
    if not factors:
        print("No factors to process")
        return
    
    print(f"\n🚀 Processing {len(factors)} factor(s)...")
    
    for factor in factors:
        try:
            generate_summary(model, conn, factor, save=args.save)
        except Exception as e:
            print(f"❌ Error processing {factor['name']}: {e}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()