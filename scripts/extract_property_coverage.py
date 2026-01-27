#!/usr/bin/env python3
"""
Extract factor coverage summary from pfr_properties.db.

Replaces FOI-based coverage data with actual scraped property data.
Generates:
- factors_property_coverage.csv for pipeline import (national counts)
- factors_area_counts.csv for area-specific property counts
"""

import sqlite3
import csv
import re
from pathlib import Path
from collections import defaultdict

# Paths
PROPERTIES_DB = Path("property detail/pfr_properties.db")
OUTPUT_CSV = Path("data/csv/factors_property_coverage.csv")
AREA_COUNTS_CSV = Path("data/csv/factors_area_counts.csv")
MAIN_DB = Path("data/database/comparefactors.db")

# Area definitions (must match pipeline)
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


def extract_postcode_area(outward_code):
    """Extract 1-2 letter postcode area from outward code (e.g., 'EH1' -> 'EH', 'G41' -> 'G')."""
    if not outward_code:
        return None
    # Match 1-2 letters at start
    match = re.match(r'^([A-Z]{1,2})', outward_code.upper())
    return match.group(1) if match else None


def get_geographic_reach(district_count, area_count):
    """Classify geographic reach based on postcode coverage.

    Uses combination of district count (granular) and area count (broad).
    """
    # National: covers many different postcode areas (EH, G, AB, etc.)
    if area_count >= 10:
        return 'national'
    # Regional: covers several areas or many districts in fewer areas
    elif area_count >= 4 or district_count >= 30:
        return 'regional'
    else:
        return 'local'


def main():
    print(f"Reading from {PROPERTIES_DB}...")

    if not PROPERTIES_DB.exists():
        print(f"ERROR: Database not found: {PROPERTIES_DB}")
        return

    conn = sqlite3.connect(PROPERTIES_DB)
    conn.row_factory = sqlite3.Row

    # Get all properties grouped by factor
    print("Aggregating property data by factor...")
    cursor = conn.execute("""
        SELECT
            pf_number,
            COUNT(*) as property_count,
            GROUP_CONCAT(DISTINCT outward_code) as outward_codes
        FROM properties
        WHERE pf_number IS NOT NULL AND pf_number != ''
        GROUP BY pf_number
        ORDER BY property_count DESC
    """)

    results = []
    for row in cursor:
        pf_number = row['pf_number']
        property_count = row['property_count']
        outward_codes_str = row['outward_codes'] or ''

        # Parse outward codes
        outward_codes = [c.strip() for c in outward_codes_str.split(',') if c.strip()]
        outward_codes = [c for c in outward_codes if c]  # Remove empty

        # Extract unique postcode areas (2-letter prefix)
        areas = set()
        for code in outward_codes:
            area = extract_postcode_area(code)
            if area:
                areas.add(area)

        # Sort areas alphabetically
        areas_sorted = sorted(areas)
        outward_codes_sorted = sorted(set(outward_codes))

        geographic_reach = get_geographic_reach(len(outward_codes_sorted), len(areas_sorted))

        results.append({
            'registration_number': pf_number,
            'property_count': property_count,
            'postcode_districts': ','.join(outward_codes_sorted),  # Full districts (EH1, EH2, G41, etc.)
            'postcode_district_count': len(outward_codes_sorted),
            'postcode_areas': ','.join(areas_sorted),  # 2-letter areas (EH, G, etc.)
            'postcode_area_count': len(areas_sorted),
            'geographic_reach': geographic_reach,
        })

    conn.close()

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'registration_number',
        'property_count',
        'postcode_districts',
        'postcode_district_count',
        'postcode_areas',
        'postcode_area_count',
        'geographic_reach'
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} factors to {OUTPUT_CSV}")

    # Generate area-specific property counts
    print("\n=== Generating area-specific property counts ===")
    conn = sqlite3.connect(PROPERTIES_DB)
    conn.row_factory = sqlite3.Row

    area_counts = []
    for area_slug, area_info in AREA_DEFINITIONS.items():
        area_postcodes = area_info['postcodes']
        # Build SQL with placeholders for each postcode
        placeholders = ','.join('?' * len(area_postcodes))
        cursor = conn.execute(f"""
            SELECT pf_number, COUNT(*) as property_count
            FROM properties
            WHERE pf_number IS NOT NULL AND pf_number != ''
              AND outward_code IN ({placeholders})
            GROUP BY pf_number
            ORDER BY property_count DESC
        """, area_postcodes)

        area_total = 0
        for row in cursor:
            area_counts.append({
                'area_slug': area_slug,
                'registration_number': row['pf_number'],
                'property_count': row['property_count']
            })
            area_total += row['property_count']
        print(f"  {area_info['name']}: {area_total:,} properties")

    conn.close()

    # Write area counts CSV
    with open(AREA_COUNTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['area_slug', 'registration_number', 'property_count'])
        writer.writeheader()
        writer.writerows(area_counts)

    print(f"\nWrote {len(area_counts)} area-factor combinations to {AREA_COUNTS_CSV}")

    # Print summary stats
    print("\n=== Summary Statistics ===")
    total_properties = sum(r['property_count'] for r in results)
    print(f"Total factors: {len(results)}")
    print(f"Total properties: {total_properties:,}")

    reach_counts = defaultdict(int)
    for r in results:
        reach_counts[r['geographic_reach']] += 1
    print(f"\nGeographic reach distribution:")
    for reach, count in sorted(reach_counts.items()):
        print(f"  {reach}: {count} factors")

    # Top 10 by property count
    print("\nTop 10 factors by property count:")
    for r in results[:10]:
        print(f"  {r['registration_number']}: {r['property_count']:,} properties, {r['postcode_area_count']} areas ({r['geographic_reach']})")

    # Also update main database if it exists
    if MAIN_DB.exists():
        print(f"\n=== Updating {MAIN_DB} ===")
        main_conn = sqlite3.connect(MAIN_DB)

        # Check if columns exist, add if not
        cursor = main_conn.execute("PRAGMA table_info(factors)")
        existing_cols = {row[1] for row in cursor}

        new_cols = [
            ('scraped_property_count', 'INTEGER DEFAULT 0'),
            ('scraped_postcode_districts', 'TEXT'),
            ('scraped_postcode_areas', 'TEXT'),
            ('scraped_geographic_reach', 'TEXT'),
        ]

        for col_name, col_type in new_cols:
            if col_name not in existing_cols:
                main_conn.execute(f"ALTER TABLE factors ADD COLUMN {col_name} {col_type}")
                print(f"  Added column: {col_name}")

        # Update factors with scraped data
        updated = 0
        for r in results:
            main_conn.execute("""
                UPDATE factors SET
                    scraped_property_count = ?,
                    scraped_postcode_districts = ?,
                    scraped_postcode_areas = ?,
                    scraped_geographic_reach = ?
                WHERE registration_number = ?
            """, [
                r['property_count'],
                r['postcode_districts'],
                r['postcode_areas'],
                r['geographic_reach'],
                r['registration_number']
            ])
            if main_conn.total_changes > updated:
                updated = main_conn.total_changes

        main_conn.commit()
        main_conn.close()
        print(f"  Updated {updated} factors in database")

    print("\nDone!")


if __name__ == '__main__':
    main()
