-- Verifies Firebase + PostgreSQL bootstrap schema integrity.
-- Fails with an exception when required objects are missing.

DO $$
DECLARE
  missing text[] := ARRAY[]::text[];
  metrics_rls_enabled boolean := false;
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
  IF to_regclass('public.metrics') IS NULL THEN
    missing := array_append(missing, 'table public.metrics');
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

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'metrics' AND column_name = 'metric_name'
  ) THEN
    missing := array_append(missing, 'column metrics.metric_name');
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'metrics' AND column_name = 'metadata'
  ) THEN
    missing := array_append(missing, 'column metrics.metadata');
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'metrics' AND column_name = 'user_id'
  ) THEN
    missing := array_append(missing, 'column metrics.user_id');
  END IF;

  -- Required indexes
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = 'metrics' AND indexname = 'idx_metrics_metric_name_created_at'
  ) THEN
    missing := array_append(missing, 'index idx_metrics_metric_name_created_at');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = 'metrics' AND indexname = 'idx_metrics_user_id_created_at'
  ) THEN
    missing := array_append(missing, 'index idx_metrics_user_id_created_at');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND tablename = 'users' AND indexname = 'idx_users_is_admin'
  ) THEN
    missing := array_append(missing, 'index idx_users_is_admin');
  END IF;

  -- Required policy + RLS
  IF to_regclass('public.metrics') IS NOT NULL THEN
    SELECT c.relrowsecurity
    INTO metrics_rls_enabled
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relname = 'metrics';

    IF NOT COALESCE(metrics_rls_enabled, false) THEN
      missing := array_append(missing, 'RLS enabled on metrics');
    END IF;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'metrics' AND policyname = 'metrics_admin_read'
  ) THEN
    missing := array_append(missing, 'policy metrics_admin_read');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'metrics' AND policyname = 'metrics_service_insert'
  ) THEN
    missing := array_append(missing, 'policy metrics_service_insert');
  END IF;

  -- Required triggers
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'trg_touch_study_updated_at' AND NOT tgisinternal
  ) THEN
    missing := array_append(missing, 'trigger trg_touch_study_updated_at');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'trg_profiles_set_updated_at' AND NOT tgisinternal
  ) THEN
    missing := array_append(missing, 'trigger trg_profiles_set_updated_at');
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'trg_studies_set_updated_at' AND NOT tgisinternal
  ) THEN
    missing := array_append(missing, 'trigger trg_studies_set_updated_at');
  END IF;

  IF array_length(missing, 1) IS NOT NULL THEN
    RAISE EXCEPTION 'Firebase PostgreSQL schema verification failed. Missing: %', array_to_string(missing, ', ');
  END IF;

  RAISE NOTICE 'Firebase PostgreSQL schema verification passed.';
END
$$;
