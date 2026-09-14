-- The retired 2026_03_24 bootstrap created trg_touch_study_updated_at_insert on
-- reports. The current bootstrap creates trg_touch_study_updated_at for the same
-- AFTER INSERT OR UPDATE OR DELETE event, and each file only dropped its own
-- names, so a database that ran both fires touch_study_updated_at twice per row.
DROP TRIGGER IF EXISTS trg_touch_study_updated_at_insert ON reports;
DROP TRIGGER IF EXISTS trg_touch_study_updated_at_update ON reports;
DROP TRIGGER IF EXISTS trg_touch_study_updated_at_delete ON reports;
