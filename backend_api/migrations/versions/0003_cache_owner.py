"""Each cached extraction belongs to one user, so deleting an account can delete it."""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.add_column("extraction_cache", sa.Column("owner_uid", sa.String(128), nullable=True))
    op.create_index("ix_extraction_cache_owner_uid", "extraction_cache", ["owner_uid"])
    # Rows from before this had no owner and could never be deleted for a user. It is only a cache.
    op.execute(sa.text("DELETE FROM extraction_cache WHERE owner_uid IS NULL"))


def downgrade() -> None:
    op.drop_index("ix_extraction_cache_owner_uid", table_name="extraction_cache")
    op.drop_column("extraction_cache", "owner_uid")
