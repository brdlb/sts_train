"""
Database connection and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base
from ..config import web_config


# Create engine with SQLite
engine = create_engine(
    f"sqlite:///{web_config.database_url}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    _ensure_game_player_columns()


def _ensure_game_player_columns():
    """Add nullable multiplayer columns to existing SQLite databases."""
    with engine.begin() as connection:
        existing_columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(game_players)")
        }
        column_defs = {
            "display_name": "VARCHAR",
            "seat_type": "VARCHAR",
            "join_token": "VARCHAR",
        }
        for column_name, column_type in column_defs.items():
            if column_name not in existing_columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE game_players ADD COLUMN {column_name} {column_type}"
                )


def get_db() -> Session:
    """
    Get database session.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

