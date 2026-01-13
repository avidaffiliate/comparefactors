#!/usr/bin/env python3
"""
Cookie Consent Injection Script
Injects UK GDPR compliant cookie consent into all HTML files missing it.
Run after the main pipeline build.

Usage:
    python inject_cookie_consent.py site/
"""

import argparse
from pathlib import Path
import re

# Cookie consent CSS (minified)
COOKIE_CSS = '''<style>
.cf-cookie-overlay{position:fixed;inset:0;background:rgba(15,23,42,0.4);backdrop-filter:blur(2px);z-index:9998;opacity:0;visibility:hidden;transition:opacity 0.3s,visibility 0.3s}.cf-cookie-overlay.active{opacity:1;visibility:visible}
.cf-cookie-banner{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #e2e8f0;box-shadow:0 -4px 20px rgba(0,0,0,0.1);z-index:9999;transform:translateY(100%);transition:transform 0.4s cubic-bezier(0.16,1,0.3,1)}.cf-cookie-banner.active{transform:translateY(0)}
.cf-cookie-inner{max-width:1200px;margin:0 auto;padding:1.5rem}
.cf-cookie-title{display:flex;align-items:center;gap:0.625rem;margin:0 0 1rem;font-size:1.125rem;font-weight:600;color:#0f172a}.cf-cookie-icon{width:24px;height:24px;flex-shrink:0}
.cf-cookie-text{font-size:0.9375rem;line-height:1.6;color:#1e293b;margin:0 0 1.25rem}.cf-cookie-text a{color:#2563eb;text-decoration:underline}
.cf-cookie-actions{display:flex;flex-wrap:wrap;gap:0.75rem;align-items:center}
.cf-btn{padding:0.75rem 1.5rem;font-size:0.9375rem;font-weight:500;border-radius:8px;cursor:pointer;border:none;font-family:inherit;transition:all 0.2s}
.cf-btn-accept{background:#16a34a;color:#fff}.cf-btn-accept:hover{background:#15803d}
.cf-btn-reject{background:#fff;color:#1e293b;border:1.5px solid #e2e8f0}.cf-btn-reject:hover{background:#f8fafc}
.cf-btn-settings{background:transparent;color:#64748b;text-decoration:underline}.cf-btn-settings:hover{color:#1e293b}
.cf-cookie-settings{display:none;margin-top:1.5rem;padding-top:1.5rem;border-top:1px solid #e2e8f0}.cf-cookie-settings.active{display:block}
.cf-settings-title{font-size:1rem;font-weight:600;color:#0f172a;margin:0 0 1rem}
.cf-cookie-category{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;padding:1rem;background:#f8fafc;border-radius:8px;margin-bottom:0.75rem}
.cf-category-info{flex:1}.cf-category-name{font-weight:600;color:#1e293b;margin:0 0 0.25rem;font-size:0.9375rem}.cf-category-desc{font-size:0.8125rem;color:#64748b;margin:0;line-height:1.5}
.cf-toggle{position:relative;flex-shrink:0}.cf-toggle input{position:absolute;opacity:0;width:0;height:0}
.cf-toggle-slider{display:block;width:48px;height:26px;background:#cbd5e1;border-radius:13px;cursor:pointer;transition:background 0.2s;position:relative}
.cf-toggle-slider::after{content:'';position:absolute;top:3px;left:3px;width:20px;height:20px;background:#fff;border-radius:50%;transition:transform 0.2s;box-shadow:0 1px 3px rgba(0,0,0,0.15)}
.cf-toggle input:checked+.cf-toggle-slider{background:#16a34a}.cf-toggle input:checked+.cf-toggle-slider::after{transform:translateX(22px)}
.cf-toggle input:disabled+.cf-toggle-slider{background:#16a34a;opacity:0.6;cursor:not-allowed}
.cf-required-badge{display:inline-block;font-size:0.6875rem;font-weight:500;text-transform:uppercase;color:#64748b;background:#e2e8f0;padding:0.125rem 0.5rem;border-radius:4px;margin-left:0.5rem}
.cf-btn-save{background:#0f172a;color:#fff}.cf-btn-save:hover{background:#1e293b}
.cf-cookie-settings-btn{position:fixed;bottom:20px;left:20px;width:44px;height:44px;background:#0f172a;border:none;border-radius:50%;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,0.15);z-index:9990;display:none;align-items:center;justify-content:center;transition:transform 0.2s,background 0.2s}.cf-cookie-settings-btn:hover{background:#1e293b;transform:scale(1.05)}.cf-cookie-settings-btn svg{width:22px;height:22px;color:#fff}.cf-cookie-settings-btn.active{display:flex}
@media(max-width:640px){.cf-cookie-inner{padding:1.25rem}.cf-cookie-actions{flex-direction:column;width:100%}.cf-btn{width:100%;text-align:center}.cf-btn-settings{order:3}.cf-cookie-category{flex-direction:column;gap:0.75rem}.cf-toggle{align-self:flex-start}}
</style>'''

# Cookie consent HTML
COOKIE_HTML = '''
<div class="cf-cookie-overlay" id="cfCookieOverlay"></div>
<div class="cf-cookie-banner" id="cfCookieBanner" role="dialog" aria-modal="true" aria-labelledby="cfCookieTitle">
    <div class="cf-cookie-inner">
        <h2 class="cf-cookie-title" id="cfCookieTitle">
            <svg class="cf-cookie-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="8" cy="9" r="1" fill="currentColor"/><circle cx="15" cy="8" r="1" fill="currentColor"/><circle cx="10" cy="14" r="1" fill="currentColor"/><circle cx="16" cy="13" r="1" fill="currentColor"/><circle cx="13" cy="17" r="1" fill="currentColor"/></svg>
            We value your privacy
        </h2>
        <p class="cf-cookie-text">We use cookies to improve your experience and analyse site traffic. You can choose which cookies you're happy with. Read our <a href="/privacy/">Privacy Policy</a> and <a href="/cookie-policy/">Cookie Policy</a> to learn more.</p>
        <div class="cf-cookie-actions">
            <button type="button" class="cf-btn cf-btn-accept" id="cfAcceptAll">Accept all cookies</button>
            <button type="button" class="cf-btn cf-btn-reject" id="cfRejectAll">Reject non-essential</button>
            <button type="button" class="cf-btn cf-btn-settings" id="cfShowSettings">Manage preferences</button>
        </div>
        <div class="cf-cookie-settings" id="cfCookieSettings">
            <h3 class="cf-settings-title">Cookie preferences</h3>
            <div class="cf-cookie-category">
                <div class="cf-category-info"><p class="cf-category-name">Essential cookies<span class="cf-required-badge">Always on</span></p><p class="cf-category-desc">Required for the website to function properly.</p></div>
                <label class="cf-toggle"><input type="checkbox" checked disabled><span class="cf-toggle-slider"></span></label>
            </div>
            <div class="cf-cookie-category">
                <div class="cf-category-info"><p class="cf-category-name">Analytics cookies</p><p class="cf-category-desc">Help us understand how visitors use our site. Data is anonymised.</p></div>
                <label class="cf-toggle"><input type="checkbox" id="cfAnalyticsCookies"><span class="cf-toggle-slider"></span></label>
            </div>
            <div class="cf-cookie-category">
                <div class="cf-category-info"><p class="cf-category-name">Marketing cookies</p><p class="cf-category-desc">Used to show relevant content and measure advertising effectiveness.</p></div>
                <label class="cf-toggle"><input type="checkbox" id="cfMarketingCookies"><span class="cf-toggle-slider"></span></label>
            </div>
            <div class="cf-cookie-actions"><button type="button" class="cf-btn cf-btn-save" id="cfSavePreferences">Save preferences</button></div>
        </div>
    </div>
</div>
<button type="button" class="cf-cookie-settings-btn" id="cfFloatingSettings" aria-label="Cookie settings"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="8" cy="9" r="1" fill="currentColor"/><circle cx="15" cy="8" r="1" fill="currentColor"/><circle cx="10" cy="14" r="1" fill="currentColor"/></svg></button>
'''

# Cookie consent JavaScript (minified)
COOKIE_JS = '''<script>
(function(){var COOKIE_NAME='cf_cookie_consent',VERSION='1.0',EXPIRY=365;var banner=document.getElementById('cfCookieBanner'),overlay=document.getElementById('cfCookieOverlay'),settings=document.getElementById('cfCookieSettings'),floatBtn=document.getElementById('cfFloatingSettings'),acceptBtn=document.getElementById('cfAcceptAll'),rejectBtn=document.getElementById('cfRejectAll'),showSettingsBtn=document.getElementById('cfShowSettings'),saveBtn=document.getElementById('cfSavePreferences'),analyticsToggle=document.getElementById('cfAnalyticsCookies'),marketingToggle=document.getElementById('cfMarketingCookies');
function getConsent(){try{var c=document.cookie.split('; ').find(function(r){return r.startsWith(COOKIE_NAME+'=')});if(c){var v=JSON.parse(decodeURIComponent(c.split('=')[1]));if(v.version===VERSION)return v}}catch(e){}return null}
function setConsent(p){var v={version:VERSION,timestamp:new Date().toISOString(),essential:true,analytics:p.analytics||false,marketing:p.marketing||false};var exp=new Date();exp.setDate(exp.getDate()+EXPIRY);document.cookie=COOKIE_NAME+'='+encodeURIComponent(JSON.stringify(v))+'; expires='+exp.toUTCString()+'; path=/; SameSite=Lax; Secure';window.dispatchEvent(new CustomEvent('cfConsentUpdated',{detail:v}));return v}
function showBanner(){banner.classList.add('active');overlay.classList.add('active');floatBtn.classList.remove('active');document.body.style.overflow='hidden'}
function hideBanner(){banner.classList.remove('active');overlay.classList.remove('active');settings.classList.remove('active');floatBtn.classList.add('active');document.body.style.overflow=''}
function loadScripts(c){if(c.analytics&&!window.cfGA4Loaded){var s=document.createElement('script');s.async=true;s.src='https://www.googletagmanager.com/gtag/js?id=G-P9QSNCJEBQ';s.onload=function(){window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}window.gtag=gtag;gtag('js',new Date());gtag('config','G-P9QSNCJEBQ',{anonymize_ip:true})};document.head.appendChild(s);window.cfGA4Loaded=true}}
function removeNonEssential(c){c=c||{analytics:false,marketing:false};var cookies=['_ga','_gid','_gat','_fbp','_fbc','_gcl_au'];document.cookie.split(';').forEach(function(ck){var n=ck.split('=')[0].trim();if(cookies.some(function(p){return n===p||n.startsWith(p+'_')})){document.cookie=n+'=;expires=Thu,01 Jan 1970 00:00:00 GMT;path=/';document.cookie=n+'=;expires=Thu,01 Jan 1970 00:00:00 GMT;path=/;domain='+location.hostname}})}
function init(){var c=getConsent();if(c){floatBtn.classList.add('active');analyticsToggle.checked=c.analytics;marketingToggle.checked=c.marketing;loadScripts(c)}else{showBanner()}
acceptBtn.addEventListener('click',function(){var c=setConsent({analytics:true,marketing:true});hideBanner();loadScripts(c)});
rejectBtn.addEventListener('click',function(){setConsent({analytics:false,marketing:false});hideBanner();removeNonEssential()});
showSettingsBtn.addEventListener('click',function(){settings.classList.toggle('active');showSettingsBtn.textContent=settings.classList.contains('active')?'Hide preferences':'Manage preferences'});
saveBtn.addEventListener('click',function(){var c=setConsent({analytics:analyticsToggle.checked,marketing:marketingToggle.checked});hideBanner();loadScripts(c);if(!c.analytics||!c.marketing)removeNonEssential(c)});
floatBtn.addEventListener('click',function(){var c=getConsent();if(c){analyticsToggle.checked=c.analytics;marketingToggle.checked=c.marketing}showBanner()})}
window.cfCookieConsent={getConsent:getConsent,hasAnalyticsConsent:function(){var c=getConsent();return c?c.analytics:false},hasMarketingConsent:function(){var c=getConsent();return c?c.marketing:false},showBanner:showBanner};
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init)}else{init()}})();
</script>'''

# Marker to detect if cookie consent already exists
COOKIE_MARKER = 'cfCookieBanner'


def has_cookie_consent(content: str) -> bool:
    """Check if file already has cookie consent."""
    return COOKIE_MARKER in content


def inject_cookie_consent(content: str) -> str:
    """Inject cookie consent before </body>."""
    # Check if already has consent
    if has_cookie_consent(content):
        return content
    
    # Find </body> tag
    body_match = re.search(r'</body>', content, re.IGNORECASE)
    if not body_match:
        return content
    
    # Check if CSS already exists in head, if not inject before </head>
    if '.cf-cookie-banner' not in content:
        head_match = re.search(r'</head>', content, re.IGNORECASE)
        if head_match:
            content = content[:head_match.start()] + COOKIE_CSS + '\n' + content[head_match.start():]
            # Re-find body tag after CSS injection
            body_match = re.search(r'</body>', content, re.IGNORECASE)
    
    # Inject HTML and JS before </body>
    injection = COOKIE_HTML + COOKIE_JS + '\n'
    content = content[:body_match.start()] + injection + content[body_match.start():]
    
    return content


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single HTML file. Returns True if modified."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = filepath.read_text(encoding='latin-1')
    
    if has_cookie_consent(content):
        return False
    
    new_content = inject_cookie_consent(content)
    
    if new_content == content:
        return False
    
    if not dry_run:
        filepath.write_text(new_content, encoding='utf-8')
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Inject cookie consent into HTML files'
    )
    parser.add_argument(
        'directory',
        type=Path,
        help='Directory to process (e.g., site/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all files processed'
    )
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' not found")
        return 1
    
    # Find all HTML files
    html_files = list(args.directory.rglob('*.html'))
    
    modified = 0
    skipped = 0
    
    for filepath in html_files:
        # Skip data files
        if 'data' in filepath.parts or 'wss' in filepath.parts:
            continue
            
        if process_file(filepath, dry_run=args.dry_run):
            modified += 1
            print(f"{'[DRY RUN] Would inject' if args.dry_run else 'Injected'}: {filepath}")
        else:
            skipped += 1
            if args.verbose:
                print(f"Skipped (already has consent): {filepath}")
    
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Modified: {modified}")
    print(f"  Skipped:  {skipped}")
    print(f"  Total:    {modified + skipped}")
    
    return 0


if __name__ == '__main__':
    exit(main())
