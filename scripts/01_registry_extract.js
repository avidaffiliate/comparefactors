// ============================================================================
// SCOTTISH PROPERTY FACTOR REGISTER EXTRACTOR - FINAL VERSION
// ============================================================================
// 
// URL: https://www.propertyfactorregister.gov.scot/search
// Last tested: January 2026
// Expected output: ~675 factors (active + expired)
//
// HOW TO USE:
//   1. Go to: https://www.propertyfactorregister.gov.scot/search
//   2. Make sure "Property factor" radio button is selected (default)
//   3. Open DevTools (F12 or Cmd+Option+I)
//   4. Go to Console tab
//   5. Click inside the console area (important for clipboard!)
//   6. Paste this entire script and press Enter
//   7. Wait ~1 minute for extraction to complete
//   8. CSV will be copied to clipboard and displayed in console
//   9. Save as: data/csv/factors_registry_raw.csv
//
// OUTPUT FORMAT:
//   registration_number,name,status
//   PF000103,"James Gibb Residential Factors Limited",Active
//   PF000567,"4C Scotland Limited",Expired
//   ...
//
// ============================================================================

(async function extractPropertyFactors() {
    console.log('🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Property Factor Register Extractor');
    console.log('='.repeat(55));
    console.log('Target: propertyfactorregister.gov.scot/search');
    console.log('');
    
    // ========================================================================
    // SETUP
    // ========================================================================
    
    const results = [];
    const seen = new Set();
    
    // Find the search input
    const searchInput = document.getElementById('property-factor-name');
    
    if (!searchInput) {
        console.error('❌ Could not find input #property-factor-name');
        console.log('');
        console.log('Make sure you are on: https://www.propertyfactorregister.gov.scot/search');
        console.log('And that "Property factor" search type is selected.');
        return;
    }
    
    console.log('✅ Found search input: #property-factor-name');
    console.log('');
    console.log('🔍 Starting extraction (this takes about 1 minute)...');
    console.log('');
    
    // ========================================================================
    // SEARCH FUNCTION
    // ========================================================================
    
    async function search(query) {
        // Clear the input
        searchInput.value = '';
        searchInput.focus();
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
        
        await new Promise(r => setTimeout(r, 100));
        
        // Type the query character by character (simulates real typing)
        for (const char of query) {
            searchInput.value += char;
            searchInput.dispatchEvent(new Event('input', { bubbles: true }));
            searchInput.dispatchEvent(new KeyboardEvent('keydown', { key: char, bubbles: true }));
            searchInput.dispatchEvent(new KeyboardEvent('keyup', { key: char, bubbles: true }));
            await new Promise(r => setTimeout(r, 40));
        }
        
        // Wait for autocomplete dropdown to populate
        await new Promise(r => setTimeout(r, 1000));
        
        // Extract from the autocomplete dropdown
        // Class: .rosgovuk-autocomplete__option
        // Format: "Company Name (Status) - PF000XXX" or "Company Name - PF000XXX"
        
        document.querySelectorAll('.rosgovuk-autocomplete__option').forEach(el => {
            const text = el.textContent.trim();
            
            // Skip empty or "no results" entries
            if (!text || text.toLowerCase().includes('no search results')) return;
            
            // Parse the text
            // Examples:
            //   "James Gibb Residential Factors Limited - PF000103"
            //   "4C Scotland Limited (Expired) - PF000567"
            //   "Company Name (Trading Name) - PF000XXX"
            
            // Regex to extract: name, optional parenthetical, PF number
            const match = text.match(/^(.+?)\s*(?:\(([^)]+)\))?\s*-\s*(PF\d+)$/);
            
            if (match) {
                const rawName = match[1].trim();
                const parenthetical = match[2]?.trim() || '';
                const pfNumber = match[3].toUpperCase();
                
                // Determine if parenthetical is a status or trading name
                const isStatus = ['Active', 'Expired'].includes(parenthetical);
                const status = isStatus ? parenthetical : 'Active';
                const name = rawName;
                
                // Deduplicate by PF number
                if (!seen.has(pfNumber)) {
                    seen.add(pfNumber);
                    results.push({
                        registration_number: pfNumber,
                        name: name,
                        status: status
                    });
                }
            }
        });
    }
    
    // ========================================================================
    // MAIN EXTRACTION LOOP
    // ========================================================================
    
    // Search A-Z (covers most factors)
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    
    for (let i = 0; i < letters.length; i++) {
        const letter = letters[i];
        const beforeCount = results.length;
        
        await search(letter);
        
        const newCount = results.length - beforeCount;
        const status = newCount > 0 ? `+${newCount} new` : 'no new';
        console.log(`  [${(i + 1).toString().padStart(2)}/26] "${letter}" → ${results.length} total (${status})`);
    }
    
    // Search numbers (some factors start with numbers)
    console.log('');
    console.log('  Checking numeric prefixes...');
    
    for (const num of ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']) {
        const beforeCount = results.length;
        await search(num);
        if (results.length > beforeCount) {
            console.log(`  "${num}" → +${results.length - beforeCount} new`);
        }
    }
    
    // Clear the search box
    searchInput.value = '';
    searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    
    // ========================================================================
    // OUTPUT RESULTS
    // ========================================================================
    
    console.log('');
    console.log('='.repeat(55));
    console.log(`✅ EXTRACTION COMPLETE: ${results.length} factors found`);
    console.log('='.repeat(55));
    
    // Sort alphabetically by name
    results.sort((a, b) => a.name.localeCompare(b.name));
    
    // Count active vs expired
    const activeCount = results.filter(r => r.status === 'Active').length;
    const expiredCount = results.filter(r => r.status === 'Expired').length;
    
    console.log('');
    console.log('📊 Summary:');
    console.log(`   Total factors: ${results.length}`);
    console.log(`   Active: ${activeCount}`);
    console.log(`   Expired: ${expiredCount}`);
    
    // Generate CSV
    const csvHeader = 'registration_number,name,status';
    const csvRows = results.map(r => {
        // Escape quotes in name
        const escapedName = r.name.replace(/"/g, '""');
        return `${r.registration_number},"${escapedName}",${r.status}`;
    });
    const csv = [csvHeader, ...csvRows].join('\n');
    
    // Display CSV
    console.log('');
    console.log('📋 CSV OUTPUT:');
    console.log('─'.repeat(55));
    console.log(csv);
    console.log('─'.repeat(55));
    
    // Try to copy to clipboard
    try {
        await navigator.clipboard.writeText(csv);
        console.log('');
        console.log('✅ CSV copied to clipboard!');
    } catch (e) {
        console.log('');
        console.log('⚠️  Could not copy to clipboard (window not focused).');
        console.log('    Run this to copy: copy(window._factorsCsv)');
    }
    
    // Store in window for easy access
    window._factors = results;
    window._factorsCsv = csv;
    
    console.log('');
    console.log('💾 To save:');
    console.log('   1. Copy the CSV above (or run: copy(window._factorsCsv))');
    console.log('   2. Paste into: data/csv/factors_registry_raw.csv');
    console.log('');
    console.log('➡️  Next step: python 02_registry_enrich.py');
    
    return results;
})();
