"""
SQLAlchemy models for game history and statistics.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Game(Base):
    """Game model."""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    winner = Column(Integer, nullable=True)  # Player ID who won
    num_players = Column(Integer, default=4, nullable=False)
    is_finished = Column(Boolean, default=False, nullable=False)

    # Relationships
    players = relationship("GamePlayer", back_populates="game", cascade="all, delete-orphan")
    actions = relationship("GameAction", back_populates="game", cascade="all, delete-orphan")
    states = relationship("GameState", back_populates="game", cascade="all, delete-orphan")


class GamePlayer(Base):
    """Game player model."""

    __tablename__ = "game_players"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    player_id = Column(Integer, nullable=False)  # 0-3 for 4 players
    player_type = Column(String, nullable=False)  # 'human' or 'ai'
    model_path = Column(String, nullable=True)  # Path to model if AI player

    # Relationship
    game = relationship("Game", back_populates="players")


class GameAction(Base):
    """Game action model."""

    __tablename__ = "game_actions"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    player_id = Column(Integer, nullable=False)
    action_type = Column(String, nullable=False)  # 'bid', 'challenge', 'believe'
    action_data = Column(JSON, nullable=True)  # Additional action data (e.g., quantity, value for bid)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    turn_number = Column(Integer, nullable=False)  # Turn number in the game

    # Relationship
    game = relationship("Game", back_populates="actions")


class GameState(Base):
    """Game state snapshot model."""

    __tablename__ = "game_states"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    turn_number = Column(Integer, nullable=False)
    state_json = Column(JSON, nullable=False)  # Serialized game state
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    game = relationship("Game", back_populates="states")

