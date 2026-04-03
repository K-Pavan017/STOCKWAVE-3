"""initial_schema_creation_and_hash_fix

Revision ID: b15e98614bc7
Revises: 
Create Date: 2025-11-11 14:51:22.147399

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b15e98614bc7'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create the user_data table with the CORRECT password_hash size (256)
    op.create_table('user_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(80), nullable=False),
        sa.Column('email', sa.String(120), nullable=False),
        sa.Column('password_hash', sa.String(256), nullable=False), # <<-- THE FIX IS HERE
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )

    # 2. Create the stock_data table
    op.create_table('stock_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('company_symbol', sa.String(10), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open_price', sa.Float(), nullable=True),
        sa.Column('high_price', sa.Float(), nullable=True),
        sa.Column('low_price', sa.Float(), nullable=True),
        sa.Column('close_price', sa.Float(), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    # Add index and unique constraint for StockData
    op.create_index('idx_symbol_date', 'stock_data', ['company_symbol', 'date'], unique=False)
    op.create_unique_constraint('uq_symbol_date', 'stock_data', ['company_symbol', 'date'])


def downgrade() -> None:
    # Reverse order for clean rollback
    op.drop_table('stock_data')
    op.drop_table('user_data')