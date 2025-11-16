"""
Opponent pool for rule-based bots.
"""

import random
from typing import Dict, List, Optional
from collections import defaultdict

from ..agents.rule_based_agent import RuleBasedAgent
from ..agents.bot_personalities import BOT_PERSONALITIES
from ..agents.bot_types import BotPersonality


class RuleBasedOpponentPool:
    """
    Manages a pool of rule-based bot opponents.
    
    Features:
    - Create bots of all personalities
    - Select bots by difficulty level
    - Track statistics for each bot type
    - Weighted sampling based on difficulty distribution
    """

    def __init__(
        self,
        max_quantity: int = 30,
        max_players: int = 8,
        max_history_length: int = 20,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize rule-based opponent pool.
        
        Args:
            max_quantity: Maximum dice quantity in bids
            max_players: Maximum number of players
            max_history_length: Maximum bid history length
            difficulty_distribution: Distribution of difficulty levels
                Format: {"EASY": 0.33, "MEDIUM": 0.33, "HARD": 0.34}
                If None, uses equal distribution
        """
        self.max_quantity = max_quantity
        self.max_players = max_players
        self.max_history_length = max_history_length
        
        # Default difficulty distribution
        if difficulty_distribution is None:
            difficulty_distribution = {"EASY": 1, "MEDIUM": 0, "HARD": 0}
        self.difficulty_distribution = difficulty_distribution
        
        # Create bots for each personality
        self.bots: Dict[str, RuleBasedAgent] = {}
        self.bots_by_difficulty: Dict[str, List[str]] = defaultdict(list)
        
        for personality_key, personality in BOT_PERSONALITIES.items():
            # Create agent with unique ID (use index)
            agent_id = len(self.bots)
            bot = RuleBasedAgent(
                agent_id=agent_id,
                personality=personality,
                max_quantity=max_quantity,
                max_players=max_players,
                max_history_length=max_history_length,
            )
            self.bots[personality_key] = bot
            self.bots_by_difficulty[personality.skill_level].append(personality_key)
        
        # Statistics tracking
        self.stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"games": 0, "wins": 0, "losses": 0}
        )

    def sample_bot(
        self,
        exclude_personalities: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ) -> RuleBasedAgent:
        """
        Sample a bot from the pool.
        
        Args:
            exclude_personalities: List of personality keys to exclude
            difficulty: Specific difficulty level to sample from ("EASY", "MEDIUM", "HARD")
                       If None, samples according to difficulty_distribution
        
        Returns:
            RuleBasedAgent instance
        """
        if exclude_personalities is None:
            exclude_personalities = []
        
        # Filter available personalities
        available = [
            key
            for key in self.bots.keys()
            if key not in exclude_personalities
        ]
        
        if not available:
            # Fallback: use all if exclude list is too restrictive
            available = list(self.bots.keys())
        
        # Filter by difficulty if specified
        if difficulty:
            available = [
                key
                for key in available
                if BOT_PERSONALITIES[key].skill_level == difficulty
            ]
            if not available:
                # Fallback: use all available
                available = list(self.bots.keys())
        
        # Sample according to difficulty_distribution if not specified
        if not difficulty:
            # Weight by difficulty distribution
            weighted_choices = []
            for key in available:
                personality = BOT_PERSONALITIES[key]
                weight = self.difficulty_distribution.get(personality.skill_level, 0.33)
                weighted_choices.append((key, weight))
            
            # Normalize weights
            total_weight = sum(w for _, w in weighted_choices)
            if total_weight > 0:
                # Sample with weights
                keys, weights = zip(*weighted_choices)
                selected_key = random.choices(keys, weights=weights, k=1)[0]
            else:
                selected_key = random.choice(available)
        else:
            selected_key = random.choice(available)
        
        return self.bots[selected_key]

    def get_bot_by_personality(self, personality_key: str) -> Optional[RuleBasedAgent]:
        """
        Get a bot by personality key.
        
        Args:
            personality_key: Key from BOT_PERSONALITIES
        
        Returns:
            RuleBasedAgent or None if not found
        """
        return self.bots.get(personality_key)

    def get_bots_by_difficulty(self, difficulty: str) -> List[RuleBasedAgent]:
        """
        Get all bots of a specific difficulty level.
        
        Args:
            difficulty: Difficulty level ("EASY", "MEDIUM", "HARD")
        
        Returns:
            List of RuleBasedAgent instances
        """
        personality_keys = self.bots_by_difficulty.get(difficulty, [])
        return [self.bots[key] for key in personality_keys]

    def update_stats(self, personality_key: str, won: bool):
        """
        Update statistics for a bot type.
        
        Args:
            personality_key: Personality key
            won: Whether the bot won
        """
        stats = self.stats[personality_key]
        stats["games"] += 1
        if won:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    def sample(self, exclude_personalities: Optional[List[str]] = None) -> str:
        """
        Sample a personality key from the pool.
        
        This is a convenience method that returns just the personality key,
        which can be used to create new RuleBasedAgent instances.
        
        Args:
            exclude_personalities: List of personality keys to exclude from sampling
        
        Returns:
            Personality key (string)
        """
        bot = self.sample_bot(exclude_personalities=exclude_personalities)
        # Find personality key for this bot
        for key, b in self.bots.items():
            if b is bot:
                return key
        # Fallback: return first available
        return list(self.bots.keys())[0]
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all bot types.
        
        Returns:
            Dictionary mapping personality keys to stats
        """
        return dict(self.stats)

    def reset_stats(self):
        """Reset all statistics."""
        self.stats.clear()

