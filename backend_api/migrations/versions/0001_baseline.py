"""Baseline: the schema as deployed before Alembic."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("firebase_uid", sa.String(128), nullable=False),
        sa.Column("email", sa.String(256)),
        sa.Column("display_name", sa.String(256)),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("last_login", sa.DateTime()),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_firebase_uid", "users", ["firebase_uid"], unique=True)

    op.create_table(
        "report_analyses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("firebase_uid", sa.String(128), nullable=False),
        sa.Column("patient_name", sa.String(256)),
        sa.Column("patient_age", sa.String(64)),
        sa.Column("patient_gender", sa.String(64)),
        sa.Column("patient_id", sa.String(128)),
        sa.Column("lab_name", sa.String(512)),
        sa.Column("report_date", sa.String(64)),
        sa.Column("total_records", sa.Integer()),
        sa.Column("analysis_json", sa.Text(), nullable=False),
        sa.Column("source_filenames", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_report_analyses_id", "report_analyses", ["id"])
    op.create_index("ix_report_analyses_firebase_uid", "report_analyses", ["firebase_uid"])

    op.create_table(
        "profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("account_owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("full_name", sa.String(256), nullable=False),
        sa.Column("relationship", sa.String(64), nullable=False),
        sa.Column("date_of_birth", sa.Date()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_profiles_account_owner_id", "profiles", ["account_owner_id"])

    op.create_table(
        "studies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("profile_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("profiles.id"), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("profile_id", "name", name="uq_studies_profile_name"),
    )
    op.create_index("ix_studies_profile_id", "studies", ["profile_id"])

    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("study_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("studies.id"), nullable=False),
        sa.Column("file_name", sa.String(512), nullable=False),
        sa.Column("file_url", sa.Text(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("lab_name", sa.String(512)),
        sa.Column("analysis_data", postgresql.JSONB(), nullable=False),
        sa.Column("normalized_records", postgresql.JSONB()),
        sa.Column("is_normalized", sa.Boolean(), nullable=False),
        sa.Column("normalization_version", sa.Integer()),
        sa.Column("uploaded_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_reports_study_id", "reports", ["study_id"])

    op.create_table(
        "extraction_cache",
        sa.Column("text_hash", sa.String(64), primary_key=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    for table in ("extraction_cache", "reports", "studies", "profiles", "report_analyses", "users"):
        op.drop_table(table)
