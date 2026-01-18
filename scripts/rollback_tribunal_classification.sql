-- Rollback SQL for tribunal case reclassification
-- Run against tribunal_enriched_v2.db to restore original state

-- Restore outcome_detailed to original values
UPDATE cases SET
    outcome_detailed = outcome_original,
    outcome_category = NULL,
    is_substantive = NULL,
    classification_confidence = NULL,
    classification_method = NULL,
    classification_timestamp = NULL;

-- Verify rollback
SELECT
    'Rows with outcome_detailed != outcome_original: ' || COUNT(*) as verification
FROM cases
WHERE outcome_detailed != outcome_original;
