"""Uploads are kept until their job finishes, and running jobs send a heartbeat, so a restart resumes them."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    op.add_column("report_jobs", sa.Column("heartbeat_at", sa.DateTime(), nullable=True))
    op.add_column("report_jobs", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.create_table(
        "report_job_files",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("report_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
    )
    op.create_index("ix_report_job_files_job_id", "report_job_files", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_report_job_files_job_id", table_name="report_job_files")
    op.drop_table("report_job_files")
    op.drop_column("report_jobs", "attempts")
    op.drop_column("report_jobs", "heartbeat_at")
