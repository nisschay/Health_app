"""One row per measured value; one row per upload in flight; one self profile per account."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"

SELF_PROFILE = sa.text("lower(relationship) = 'self'")


def upgrade() -> None:
    op.create_table(
        "report_findings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("report_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("profile_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("canonical_test", sa.String(256), nullable=False),
        sa.Column("original_test_name", sa.String(256)),
        sa.Column("category", sa.String(128)),
        sa.Column("test_date", sa.Date()),
        sa.Column("result_text", sa.String(256)),
        sa.Column("value_numeric", sa.Float()),
        sa.Column("comparator", sa.String(4)),
        sa.Column("unit", sa.String(64)),
        sa.Column("reference_range", sa.String(256)),
        sa.Column("ref_low", sa.Float()),
        sa.Column("ref_high", sa.Float()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("source_filename", sa.String(512)),
        sa.Column("aliases", postgresql.JSONB()),
    )
    op.create_index("ix_report_findings_report_id", "report_findings", ["report_id"])
    op.create_index(
        "ix_report_findings_profile_test_date", "report_findings", ["profile_id", "canonical_test", "test_date"]
    )

    op.create_table(
        "report_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("study_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("studies.id", ondelete="SET NULL")),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("report_analyses.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("source_filenames", postgresql.JSONB(), nullable=False),
        sa.Column("progress", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("finished_at", sa.DateTime()),
    )
    op.create_index("ix_report_jobs_owner_id", "report_jobs", ["owner_id"])

    op.create_index(
        "uq_profiles_owner_self",
        "profiles",
        ["account_owner_id"],
        unique=True,
        postgresql_where=SELF_PROFILE,
        sqlite_where=SELF_PROFILE,
    )


def downgrade() -> None:
    op.drop_index("uq_profiles_owner_self", table_name="profiles")
    op.drop_table("report_jobs")
    op.drop_table("report_findings")
