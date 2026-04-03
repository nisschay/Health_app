-- Verifies PostgreSQL schema integrity for the Medical Project.
-- Fails with an exception when required objects are missing.

DO $$
DECLARE
  missing text[] := ARRAY[]::text[];
BEGIN
  -- Required tables
  IF to_regclass('public.users') IS NULL THEN
    missing := array_append(missing, 'table public.users');
  END IF;

  IF to_regclass('public.report_analyses') IS NULL THEN
    missing := array_append(missing, 'table public.report_analyses');
  END IF;

  IF to_regclass('public.profiles') IS NULL THEN
    missing := array_append(missing, 'table public.profiles');
  END IF;

  IF to_regclass('public.studies') IS NULL THEN
    missing := array_append(missing, 'table public.studies');
  END IF;

  IF to_regclass('public.reports') IS NULL THEN
    missing := array_append(missing, 'table public.reports');
  END IF;

  -- Required columns
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'users' AND column_name = 'firebase_uid'
  ) THEN
    missing := array_append(missing, 'column users.firebase_uid');
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'users' AND column_name = 'is_admin'
  ) THEN
    missing := array_append(missing, 'column users.is_admin');
  END IF;

  -- Required indexes
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = 'profiles' AND indexname = 'idx_profiles_account_owner_id'
  ) THEN
    missing := array_append(missing, 'index idx_profiles_account_owner_id');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = 'studies' AND indexname = 'idx_studies_profile_id'
  ) THEN
    missing := array_append(missing, 'index idx_studies_profile_id');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = 'reports' AND indexname = 'idx_reports_study_id'
  ) THEN
    missing := array_append(missing, 'index idx_reports_study_id');
  END IF;

  -- Required helper function
  IF to_regprocedure('touch_study_updated_at()') IS NULL THEN
    missing := array_append(missing, 'function touch_study_updated_at()');
  END IF;

  -- At least one trigger on reports should call touch_study_updated_at.
  IF NOT EXISTS (
    SELECT 1
    FROM pg_trigger t
    JOIN pg_class c ON c.oid = t.tgrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_proc p ON p.oid = t.tgfoid
    WHERE n.nspname = 'public'
      AND c.relname = 'reports'
      AND NOT t.tgisinternal
      AND p.proname = 'touch_study_updated_at'
  ) THEN
    missing := array_append(missing, 'reports trigger using touch_study_updated_at');
  END IF;

  IF array_length(missing, 1) IS NOT NULL THEN
    RAISE EXCEPTION 'PostgreSQL schema verification failed. Missing: %', array_to_string(missing, ', ');
  END IF;

  RAISE NOTICE 'PostgreSQL schema verification passed.';
END
$$;
