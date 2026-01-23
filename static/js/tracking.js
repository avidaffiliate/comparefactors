/**
 * Compare Factors - Google Analytics Event Tracking
 * Tracks user interactions across the site
 */

(function() {
    'use strict';

    // Check if gtag is available
    function track(eventName, params) {
        if (typeof gtag === 'function') {
            gtag('event', eventName, params);
        }
    }

    // Get current page type from URL
    function getPageType() {
        const path = window.location.pathname;
        if (path === '/') return 'homepage';
        if (path.startsWith('/factors/') && path.split('/').length > 3) return 'factor_profile';
        if (path.startsWith('/factors/')) return 'factors_listing';
        if (path.startsWith('/areas/') && path.split('/').length > 3) return 'area_page';
        if (path.startsWith('/areas/')) return 'areas_index';
        if (path.startsWith('/guides/')) return 'guide';
        if (path.startsWith('/compare/')) return 'compare';
        if (path.startsWith('/get-quotes/')) return 'get_quotes';
        if (path.startsWith('/contribute/')) return 'contribute';
        if (path.startsWith('/contact/')) return 'contact';
        return 'other';
    }

    // ========================================
    // SEARCH TRACKING
    // ========================================
    function initSearchTracking() {
        // Homepage search
        const heroSearchInput = document.querySelector('.hero .search-input');
        const heroSearchBtn = document.querySelector('.hero .search-btn');

        if (heroSearchBtn && heroSearchInput) {
            heroSearchBtn.addEventListener('click', function() {
                const query = heroSearchInput.value.trim();
                if (query) {
                    track('search', {
                        search_term: query,
                        search_location: 'homepage_hero'
                    });
                }
            });

            heroSearchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    const query = heroSearchInput.value.trim();
                    if (query) {
                        track('search', {
                            search_term: query,
                            search_location: 'homepage_hero'
                        });
                    }
                }
            });
        }

        // Listing page search (factors, areas)
        const listingSearchInput = document.getElementById('searchInput');
        if (listingSearchInput) {
            let searchTimeout;
            listingSearchInput.addEventListener('input', function() {
                clearTimeout(searchTimeout);
                const query = this.value.trim();
                if (query.length >= 2) {
                    searchTimeout = setTimeout(function() {
                        track('search', {
                            search_term: query,
                            search_location: getPageType() + '_filter'
                        });
                    }, 1000); // Debounce 1 second
                }
            });
        }
    }

    // ========================================
    // FIND YOUR FACTOR CLICKS
    // ========================================
    function initFindFactorTracking() {
        // Track clicks on "Find Your Factor" links to official register
        document.addEventListener('click', function(e) {
            const link = e.target.closest('a[href*="propertyfactorregister.gov.scot"]');
            if (link) {
                track('click', {
                    event_category: 'outbound',
                    event_label: 'find_your_factor',
                    link_url: link.href,
                    page_location: getPageType()
                });
            }
        });
    }

    // ========================================
    // PDF & DOCUMENT CLICKS
    // ========================================
    function initDocumentTracking() {
        document.addEventListener('click', function(e) {
            const link = e.target.closest('a');
            if (!link) return;

            const href = link.href || '';

            // WSS PDF clicks
            if (href.includes('written-statement') || href.includes('wss') ||
                (href.endsWith('.pdf') && link.closest('#wss, .wss-section'))) {
                track('file_download', {
                    file_name: 'written_statement_of_services',
                    file_extension: 'pdf',
                    link_url: href,
                    factor_id: getFactorId()
                });
            }
            // Other PDF clicks
            else if (href.endsWith('.pdf')) {
                track('file_download', {
                    file_name: extractFileName(href),
                    file_extension: 'pdf',
                    link_url: href,
                    page_location: getPageType()
                });
            }
            // Tribunal decision clicks (Housing and Property Chamber)
            else if (href.includes('housingandpropertychamber.scot') &&
                     (href.includes('decision') || href.includes('FTS/HPC'))) {
                const caseRef = extractCaseReference(href);
                track('click', {
                    event_category: 'tribunal',
                    event_label: 'view_decision',
                    case_reference: caseRef,
                    link_url: href,
                    factor_id: getFactorId()
                });
            }
            // Civil and Justice Tribunal PDF clicks
            else if (href.includes('civilandjusticetribunal.scot')) {
                const caseRef = extractCaseReference(href);
                track('file_download', {
                    file_name: 'tribunal_decision',
                    file_extension: 'pdf',
                    case_reference: caseRef,
                    link_url: href,
                    factor_id: getFactorId()
                });
            }
        });
    }

    // ========================================
    // ACCORDION / EXPAND TRACKING
    // ========================================
    function initExpandTracking() {
        // Intercept toggleCases function
        const originalToggleCases = window.toggleCases;
        if (typeof originalToggleCases === 'function') {
            window.toggleCases = function(btn) {
                const container = btn.parentElement;
                const hiddenCards = container.querySelectorAll('.case-card.hidden');
                const isExpanding = hiddenCards.length > 0;

                track('expand_content', {
                    content_type: 'tribunal_cases',
                    action: isExpanding ? 'expand' : 'collapse',
                    items_count: hiddenCards.length,
                    factor_id: getFactorId()
                });

                originalToggleCases.call(this, btn);
            };
        }

        // Intercept toggleLocations function
        const originalToggleLocations = window.toggleLocations;
        if (typeof originalToggleLocations === 'function') {
            window.toggleLocations = function(btn) {
                const container = btn.closest('.review-locations-list');
                const hiddenLocations = container.querySelector('.hidden-locations');
                const isHidden = hiddenLocations && hiddenLocations.style.display === 'none';

                track('expand_content', {
                    content_type: 'review_locations',
                    action: isHidden ? 'expand' : 'collapse',
                    factor_id: getFactorId()
                });

                originalToggleLocations.call(this, btn);
            };
        }

        // Track "Show more" button clicks generically
        document.addEventListener('click', function(e) {
            const btn = e.target.closest('.show-more-btn, [class*="toggle"], [class*="expand"]');
            if (btn && !btn.hasAttribute('data-tracking-handled')) {
                const btnText = btn.textContent.toLowerCase();
                if (btnText.includes('show') || btnText.includes('more') ||
                    btnText.includes('expand') || btnText.includes('view all')) {
                    track('expand_content', {
                        content_type: 'generic',
                        button_text: btn.textContent.trim().substring(0, 50),
                        page_location: getPageType()
                    });
                }
            }
        });
    }

    // ========================================
    // WSS SECTION INTERACTION
    // ========================================
    function initWssTracking() {
        // Track when WSS section comes into view
        const wssSection = document.getElementById('wss');
        if (wssSection && 'IntersectionObserver' in window) {
            let wssViewed = false;
            const observer = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting && !wssViewed) {
                        wssViewed = true;
                        track('view_item', {
                            item_name: 'written_statement_of_services',
                            factor_id: getFactorId()
                        });
                    }
                });
            }, { threshold: 0.5 });
            observer.observe(wssSection);
        }

        // Track clicks on WSS-related elements
        document.addEventListener('click', function(e) {
            const wssLink = e.target.closest('#wss a, .wss-section a, a[href*="written-statement"]');
            if (wssLink) {
                track('click', {
                    event_category: 'wss',
                    event_label: wssLink.textContent.trim().substring(0, 50),
                    link_url: wssLink.href,
                    factor_id: getFactorId()
                });
            }
        });
    }

    // ========================================
    // FILTER TRACKING
    // ========================================
    function initFilterTracking() {
        // Risk filter
        const riskFilter = document.getElementById('riskFilter');
        if (riskFilter) {
            riskFilter.addEventListener('change', function() {
                track('filter', {
                    filter_type: 'risk_band',
                    filter_value: this.value || 'all',
                    page_location: getPageType()
                });
            });
        }

        // Sort filter
        const sortFilter = document.getElementById('sortFilter');
        if (sortFilter) {
            sortFilter.addEventListener('change', function() {
                track('sort', {
                    sort_by: this.value,
                    page_location: getPageType()
                });
            });
        }

        // Checkbox filters (expired, RSL)
        const expiredCheckbox = document.getElementById('includeExpired');
        if (expiredCheckbox) {
            expiredCheckbox.addEventListener('change', function() {
                track('filter', {
                    filter_type: 'include_expired',
                    filter_value: this.checked ? 'yes' : 'no',
                    page_location: getPageType()
                });
            });
        }

        const rslCheckbox = document.getElementById('includeRsl');
        if (rslCheckbox) {
            rslCheckbox.addEventListener('change', function() {
                track('filter', {
                    filter_type: 'include_rsl',
                    filter_value: this.checked ? 'yes' : 'no',
                    page_location: getPageType()
                });
            });
        }
    }

    // ========================================
    // EXTERNAL LINK TRACKING
    // ========================================
    function initExternalLinkTracking() {
        document.addEventListener('click', function(e) {
            const link = e.target.closest('a[target="_blank"]');
            if (!link) return;

            const href = link.href || '';
            const hostname = new URL(href, window.location.origin).hostname;

            // Skip if we're already tracking this link type
            if (href.includes('propertyfactorregister.gov.scot') ||
                href.includes('housingandpropertychamber.scot') ||
                href.includes('civilandjusticetribunal.scot') ||
                href.endsWith('.pdf')) {
                return;
            }

            // Factor website clicks
            if (link.closest('.sidebar-section, .contact-item, .quick-links')) {
                track('click', {
                    event_category: 'outbound',
                    event_label: 'factor_website',
                    link_url: href,
                    link_domain: hostname,
                    factor_id: getFactorId()
                });
            }
            // Other external links
            else if (!hostname.includes('comparefactors.co.uk')) {
                track('click', {
                    event_category: 'outbound',
                    event_label: 'external_link',
                    link_url: href,
                    link_domain: hostname,
                    page_location: getPageType()
                });
            }
        });
    }

    // ========================================
    // CTA BUTTON TRACKING
    // ========================================
    function initCtaTracking() {
        document.addEventListener('click', function(e) {
            // Get Quotes CTA
            const getQuotesBtn = e.target.closest('a[href*="/get-quotes/"]');
            if (getQuotesBtn) {
                track('click', {
                    event_category: 'cta',
                    event_label: 'get_quotes',
                    page_location: getPageType(),
                    factor_id: getFactorId()
                });
            }

            // Compare button
            const compareBtn = e.target.closest('a[href*="/compare/"]');
            if (compareBtn) {
                track('click', {
                    event_category: 'cta',
                    event_label: 'compare',
                    page_location: getPageType()
                });
            }

            // View Profile clicks
            const viewProfileBtn = e.target.closest('.factor-link, .view-btn, a[href^="/factors/pf"]');
            if (viewProfileBtn && viewProfileBtn.href && viewProfileBtn.href.includes('/factors/pf')) {
                const factorId = extractFactorIdFromUrl(viewProfileBtn.href);
                track('click', {
                    event_category: 'navigation',
                    event_label: 'view_factor_profile',
                    factor_id: factorId,
                    page_location: getPageType()
                });
            }
        });
    }

    // ========================================
    // FORM TRACKING (Get Quotes & Contribute)
    // ========================================
    function initFormTracking() {
        // Get Quotes form (quote-interest)
        const quoteForm = document.querySelector('form[name="quote-interest"]');
        if (quoteForm) {
            let formStarted = false;

            // Track form start (first interaction)
            quoteForm.addEventListener('focusin', function(e) {
                if (!formStarted && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
                    formStarted = true;
                    track('form_start', {
                        form_name: 'quote_interest',
                        form_location: 'get_quotes_page'
                    });
                }
            });

            // Track form submission
            quoteForm.addEventListener('submit', function() {
                track('generate_lead', {
                    form_name: 'quote_interest',
                    form_location: 'get_quotes_page',
                    has_name: !!quoteForm.querySelector('#name').value.trim(),
                    has_message: !!quoteForm.querySelector('#message').value.trim()
                });
            });
        }

        // Contribute / Pricing Data form
        const contributeForm = document.querySelector('form[name="pricing-data"]');
        if (contributeForm) {
            let formStarted = false;
            let fieldsCompleted = [];

            // Track form start
            contributeForm.addEventListener('focusin', function(e) {
                if (!formStarted && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT')) {
                    formStarted = true;
                    track('form_start', {
                        form_name: 'contribute_pricing',
                        form_location: 'contribute_page'
                    });
                }
            });

            // Track file upload
            const fileUpload = contributeForm.querySelector('#file-upload');
            if (fileUpload) {
                fileUpload.addEventListener('change', function() {
                    if (this.files && this.files.length > 0) {
                        const fileTypes = Array.from(this.files).map(function(f) {
                            return f.name.split('.').pop().toLowerCase();
                        });
                        track('file_upload', {
                            form_name: 'contribute_pricing',
                            file_count: this.files.length,
                            file_types: fileTypes.join(',')
                        });
                    }
                });
            }

            // Track building info completion (key fields)
            var buildingFields = ['postcode', 'units', 'building-age'];
            buildingFields.forEach(function(fieldName) {
                var field = contributeForm.querySelector('[name="' + fieldName + '"]');
                if (field) {
                    field.addEventListener('change', function() {
                        if (this.value && fieldsCompleted.indexOf(fieldName) === -1) {
                            fieldsCompleted.push(fieldName);
                            if (fieldsCompleted.length === buildingFields.length) {
                                track('form_progress', {
                                    form_name: 'contribute_pricing',
                                    milestone: 'building_info_complete'
                                });
                            }
                        }
                    });
                }
            });

            // Track form submission
            contributeForm.addEventListener('submit', function() {
                var hasFile = fileUpload && fileUpload.files && fileUpload.files.length > 0;
                var hasFeeNotes = !!contributeForm.querySelector('#fee-notes').value.trim();
                var hasEmail = !!contributeForm.querySelector('#email').value.trim();
                var wantsNotify = contributeForm.querySelector('[name="notify-benchmarks"]').checked;

                track('generate_lead', {
                    form_name: 'contribute_pricing',
                    form_location: 'contribute_page',
                    has_file_upload: hasFile,
                    has_fee_notes: hasFeeNotes,
                    has_email: hasEmail,
                    wants_notification: wantsNotify,
                    building_age: contributeForm.querySelector('#building-age').value || 'not_specified'
                });
            });
        }

        // Contact form (if exists)
        const contactForm = document.querySelector('form[name="contact"]');
        if (contactForm) {
            let formStarted = false;

            contactForm.addEventListener('focusin', function(e) {
                if (!formStarted && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
                    formStarted = true;
                    track('form_start', {
                        form_name: 'contact',
                        form_location: 'contact_page'
                    });
                }
            });

            contactForm.addEventListener('submit', function() {
                track('generate_lead', {
                    form_name: 'contact',
                    form_location: 'contact_page'
                });
            });
        }
    }

    // ========================================
    // HELPER FUNCTIONS
    // ========================================
    function getFactorId() {
        const path = window.location.pathname;
        const match = path.match(/\/factors\/(pf\d+)/i);
        return match ? match[1].toUpperCase() : null;
    }

    function extractFactorIdFromUrl(url) {
        const match = url.match(/\/factors\/(pf\d+)/i);
        return match ? match[1].toUpperCase() : null;
    }

    function extractFileName(url) {
        try {
            const parts = url.split('/');
            return parts[parts.length - 1].replace('.pdf', '');
        } catch (e) {
            return 'unknown';
        }
    }

    function extractCaseReference(url) {
        const match = url.match(/FTS\/HPC\/PF\/\d+\/\d+/i);
        return match ? match[0] : null;
    }

    // ========================================
    // INITIALIZATION
    // ========================================
    function init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initAll);
        } else {
            initAll();
        }
    }

    function initAll() {
        initSearchTracking();
        initFindFactorTracking();
        initDocumentTracking();
        initExpandTracking();
        initWssTracking();
        initFilterTracking();
        initExternalLinkTracking();
        initCtaTracking();
        initFormTracking();

        // Log initialization in development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            console.log('[CF Tracking] Initialized on', getPageType());
        }
    }

    init();
})();
