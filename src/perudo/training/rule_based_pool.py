"""
Opponent pool for rule-based bots.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from ..agents.rule_based_agent import RuleBasedAgent
from ..agents.bot_personalities import BOT_PERSONALITIES
from ..agents.bot_types import BotPersonality


@dataclass
class BotStatistics:
    """Statistics for a rule-based bot personality."""
    
    personality_key: str
    name: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0.0
    elo: float = 1500.0
    total_steps: int = 0
    avg_steps_per_game: float = 0.0


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
        statistics_dir: Optional[str] = None,
        elo_k: int = 32,
        allowed_bot_personalities: Optional[List[str]] = None,
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
            statistics_dir: Directory to store statistics (default: models/rule_based_pool)
            elo_k: ELO K-factor for rating updates
            allowed_bot_personalities: List of allowed bot personality keys (e.g., ["CONSERVATIVE"]).
                                      If None, all bots are allowed.
        """
        self.max_quantity = max_quantity
        self.max_players = max_players
        self.max_history_length = max_history_length
        self.elo_k = elo_k
        
        # Default difficulty distribution
        if difficulty_distribution is None:
            difficulty_distribution =  {"EASY": 0.33, "MEDIUM": 0.34, "HARD": 0.33}
            
        self.difficulty_distribution = difficulty_distribution
        
        # Set statistics directory
        if statistics_dir is None:
            statistics_dir = os.path.join("models", "rule_based_pool")
        self.statistics_dir = statistics_dir
        os.makedirs(statistics_dir, exist_ok=True)
        
        # Statistics file path
        self.statistics_file = os.path.join(statistics_dir, "bot_statistics.json")
        
        # Filter personalities if allowed_bot_personalities is specified
        if allowed_bot_personalities is not None:
            # Validate that all requested personalities exist
            valid_personalities = set(BOT_PERSONALITIES.keys())
            requested_personalities = set(allowed_bot_personalities)
            invalid_personalities = requested_personalities - valid_personalities
            if invalid_personalities:
                raise ValueError(
                    f"Invalid bot personality keys: {invalid_personalities}. "
                    f"Valid keys are: {sorted(valid_personalities)}"
                )
            # Filter to only allowed personalities
            filtered_personalities = {
                k: v for k, v in BOT_PERSONALITIES.items()
                if k in allowed_bot_personalities
            }
        else:
            filtered_personalities = BOT_PERSONALITIES
        
        # Create bots for each allowed personality
        self.bots: Dict[str, RuleBasedAgent] = {}
        self.bots_by_difficulty: Dict[str, List[str]] = defaultdict(list)
        
        for personality_key, personality in filtered_personalities.items():
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
        
        # Statistics tracking (new format with BotStatistics)
        self.bot_statistics: Dict[str, BotStatistics] = {}
        self._load_statistics()
        
        # Legacy statistics tracking (for backward compatibility)
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
        # Always return SOME bot
        #if "SOME" in self.bots:
        #    return self.bots["SOME"]
        
        # Fallback to original logic if LATE_BLOOMER not found (should not happen)
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
                                  (ignored when returning LATE_BLOOMER to allow multiple instances)
        
        Returns:
            Personality key (string)
        """
        # Always return LATE_BLOOMER (ignore exclude_personalities to allow multiple instances)
        if "LATE_BLOOMER" in self.bots:
            return "LATE_BLOOMER"
        
        # Fallback to original logic if LATE_BLOOMER not found (should not happen)
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
    
    def _load_statistics(self):
        """Load statistics from JSON file."""
        if os.path.exists(self.statistics_file):
            try:
                with open(self.statistics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    bots_data = data.get("bots", {})
                    for personality_key, bot_data in bots_data.items():
                        # Create BotStatistics from loaded data
                        stats = BotStatistics(
                            personality_key=bot_data.get("personality_key", personality_key),
                            name=bot_data.get("name", ""),
                            games_played=bot_data.get("games_played", 0),
                            wins=bot_data.get("wins", 0),
                            losses=bot_data.get("losses", 0),
                            winrate=bot_data.get("winrate", 0.0),
                            elo=bot_data.get("elo", 1500.0),
                            total_steps=bot_data.get("total_steps", 0),
                            avg_steps_per_game=bot_data.get("avg_steps_per_game", 0.0),
                        )
                        # Clamp ELO to reasonable bounds
                        stats.elo = max(0.0, min(3000.0, stats.elo))
                        self.bot_statistics[personality_key] = stats
            except Exception as e:
                print(f"Warning: Could not load bot statistics: {e}")
        
        # Initialize statistics for personalities that don't have them yet
        # Only initialize for allowed personalities (those in self.bots)
        for personality_key in self.bots.keys():
            if personality_key not in self.bot_statistics:
                personality = BOT_PERSONALITIES[personality_key]
                self.bot_statistics[personality_key] = BotStatistics(
                    personality_key=personality_key,
                    name=personality.name,
                    elo=1500.0,
                )
    
    def _save_statistics(self):
        """Save statistics to JSON file."""
        # Statistics collection disabled
        if False:
            data = {
                "bots": {
                    personality_key: asdict(stats)
                    for personality_key, stats in self.bot_statistics.items()
                }
            }
            try:
                with open(self.statistics_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Could not save bot statistics: {e}")
    
    def update_game_result(self, personality_key: str, won: bool, steps: int = 0):
        """
        Update game result statistics for a bot.
        
        Args:
            personality_key: Personality key
            won: Whether the bot won
            steps: Number of steps taken in the game
        """
        if personality_key not in self.bot_statistics:
            # Initialize if not exists
            personality = BOT_PERSONALITIES.get(personality_key)
            if personality is None:
                return
            self.bot_statistics[personality_key] = BotStatistics(
                personality_key=personality_key,
                name=personality.name,
                elo=1500.0,
            )
        
        stats = self.bot_statistics[personality_key]
        stats.games_played += 1
        if won:
            stats.wins += 1
        else:
            stats.losses += 1
        
        # Update winrate
        if stats.games_played > 0:
            stats.winrate = stats.wins / stats.games_played
        
        # Update steps
        stats.total_steps += steps
        if stats.games_played > 0:
            stats.avg_steps_per_game = stats.total_steps / stats.games_played
        
        # Save statistics
        self._save_statistics()
    
    def update_elo_pair(self, personality_key1: str, personality_key2: str, won1: bool):
        """
        Update ELO ratings for a pair of bots.
        
        Args:
            personality_key1: First bot's personality key
            personality_key2: Second bot's personality key
            won1: Whether first bot won
        """
        # Ensure both bots have statistics
        for key in [personality_key1, personality_key2]:
            if key not in self.bot_statistics:
                personality = BOT_PERSONALITIES.get(key)
                if personality is None:
                    return
                self.bot_statistics[key] = BotStatistics(
                    personality_key=key,
                    name=personality.name,
                    elo=1500.0,
                )
        
        stats1 = self.bot_statistics[personality_key1]
        stats2 = self.bot_statistics[personality_key2]
        
        # Calculate ELO change
        elo_diff = (stats2.elo - stats1.elo) / 400.0
        
        # Clamp elo_diff to prevent overflow
        if elo_diff > 10.0:
            expected_score1 = 0.0
        elif elo_diff < -10.0:
            expected_score1 = 1.0
        else:
            expected_score1 = 1.0 / (1.0 + 10 ** elo_diff)
        
        expected_score2 = 1.0 - expected_score1
        
        # Actual scores
        actual_score1 = 1.0 if won1 else 0.0
        actual_score2 = 1.0 - actual_score1
        
        # Calculate ELO changes
        elo_change1 = self.elo_k * (actual_score1 - expected_score1)
        elo_change2 = self.elo_k * (actual_score2 - expected_score2)
        
        # Update ELO ratings
        stats1.elo += elo_change1
        stats2.elo += elo_change2
        
        # Clamp ELO to reasonable bounds
        stats1.elo = max(0.0, min(3000.0, stats1.elo))
        stats2.elo = max(0.0, min(3000.0, stats2.elo))
        
        # Save statistics
        self._save_statistics()
    
    def update_elo(self, personality_key: str, opponent_elo: float, won: bool):
        """
        Update ELO rating for a bot playing against an opponent with known ELO.
        
        Args:
            personality_key: Bot's personality key
            opponent_elo: Opponent's ELO rating
            won: Whether the bot won
        """
        if personality_key not in self.bot_statistics:
            personality = BOT_PERSONALITIES.get(personality_key)
            if personality is None:
                return
            self.bot_statistics[personality_key] = BotStatistics(
                personality_key=personality_key,
                name=personality.name,
                elo=1500.0,
            )
        
        stats = self.bot_statistics[personality_key]
        
        # Calculate ELO change
        elo_diff = (opponent_elo - stats.elo) / 400.0
        
        # Clamp elo_diff to prevent overflow
        if elo_diff > 10.0:
            expected_score = 0.0
        elif elo_diff < -10.0:
            expected_score = 1.0
        else:
            expected_score = 1.0 / (1.0 + 10 ** elo_diff)
        
        # Actual score
        actual_score = 1.0 if won else 0.0
        
        # Calculate ELO change
        elo_change = self.elo_k * (actual_score - expected_score)
        
        # Update ELO rating
        stats.elo += elo_change
        
        # Clamp ELO to reasonable bounds
        stats.elo = max(0.0, min(3000.0, stats.elo))
        
        # Save statistics
        self._save_statistics()
    
    def update_elos_for_game(
        self,
        participants: List[Tuple[str, int, int]],
        winner_id: int,
    ):
        """
        Update ELO ratings for all participating rule-based bots.
        
        Args:
            participants: List of tuples (personality_key, player_id, steps) for each bot
            winner_id: Player ID of the winner
        """
        # Filter to only rule-based bots (exclude player_id 0 which is RL agent)
        bot_participants = [
            (personality_key, player_id, steps)
            for personality_key, player_id, steps in participants
            if player_id != 0 and personality_key is not None
        ]
        
        if len(bot_participants) < 2:
            # Need at least 2 bots to update ELO
            return
        
        # Update ELO for all pairs
        for i in range(len(bot_participants)):
            key1, player_id1, _ = bot_participants[i]
            
            for j in range(i + 1, len(bot_participants)):
                key2, player_id2, _ = bot_participants[j]
                # Determine winner for this pair
                if player_id1 == winner_id:
                    pair_won1 = True
                elif player_id2 == winner_id:
                    pair_won1 = False
                else:
                    # Neither won, skip this pair (shouldn't happen in normal game)
                    continue
                
                self.update_elo_pair(key1, key2, pair_won1)
    
    def get_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all bots.
        
        Returns:
            Dictionary mapping personality keys to statistics
        """
        return {
            personality_key: asdict(stats)
            for personality_key, stats in self.bot_statistics.items()
        }

