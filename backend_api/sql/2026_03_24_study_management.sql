-- Study Management schema (profiles, studies, reports)
-- Safe additive migration: preserves existing users and report_analyses tables.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Guardrail: this codebase currently defines users.id as INTEGER in SQLAlchemy.
-- If your deployed users.id is UUID, update account_owner_id accordingly before applying.
DO $$
DECLARE
  users_id_type TEXT;
BEGIN
  SELECT data_type
  INTO users_id_type
  FROM information_schema.columns
  WHERE table_name = 'users' AND column_name = 'id';

  IF users_id_type IS NULL THEN
    RAISE EXCEPTION 'users.id not found; cannot create profiles foreign key';
  END IF;

  IF users_id_type NOT IN ('smallint', 'integer', 'bigint') THEN
    RAISE EXCEPTION
      'users.id type is %, but migration expects an integer type. If users.id is UUID, change profiles.account_owner_id to UUID first.',
      users_id_type;
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  full_name VARCHAR(256) NOT NULL,
  relationship VARCHAR(64) NOT NULL,
  date_of_birth DATE NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_profiles_account_owner_id ON profiles(account_owner_id);

CREATE TABLE IF NOT EXISTS studies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  name VARCHAR(256) NOT NULL,
  description TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'uq_studies_profile_name'
  ) THEN
    ALTER TABLE studies
    ADD CONSTRAINT uq_studies_profile_name UNIQUE (profile_id, name);
  END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_studies_profile_id ON studies(profile_id);
CREATE INDEX IF NOT EXISTS idx_studies_updated_at ON studies(updated_at DESC);

CREATE TABLE IF NOT EXISTS reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  study_id UUID NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
  file_name VARCHAR(512) NOT NULL,
  file_url TEXT NOT NULL,
  report_date DATE NOT NULL,
  lab_name VARCHAR(512) NULL,
  analysis_data JSONB NOT NULL,
  uploaded_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_study_id ON reports(study_id);
CREATE INDEX IF NOT EXISTS idx_reports_report_date ON reports(report_date);

-- Keep study.updated_at current when reports are inserted or changed.
CREATE OR REPLACE FUNCTION touch_study_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    UPDATE studies SET updated_at = NOW() WHERE id = OLD.study_id;
    RETURN OLD;
  END IF;

  UPDATE studies SET updated_at = NOW() WHERE id = NEW.study_id;

  -- If a report is reassigned to a different study, touch the previous one too.
  IF TG_OP = 'UPDATE' AND NEW.study_id IS DISTINCT FROM OLD.study_id THEN
    UPDATE studies SET updated_at = NOW() WHERE id = OLD.study_id;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_touch_study_updated_at_insert ON reports;
DROP TRIGGER IF EXISTS trg_touch_study_updated_at_update ON reports;
DROP TRIGGER IF EXISTS trg_touch_study_updated_at_delete ON reports;

CREATE TRIGGER trg_touch_study_updated_at_insert
AFTER INSERT OR UPDATE OR DELETE ON reports
FOR EACH ROW
EXECUTE FUNCTION touch_study_updated_at();
