"""
Bot personality statistics tracking for training.
Similar to opponent pool, but tracks statistics of different bot personalities.
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class BotPersonalityStats:
    """Statistics for a single bot personality."""
    
    personality_name: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    total_rounds: int = 0
    total_dice_lost: int = 0
    total_dice_won: int = 0
    successful_challenges: int = 0
    failed_challenges: int = 0
    successful_calzas: int = 0
    failed_calzas: int = 0
    times_challenged: int = 0
    times_challenged_correctly: int = 0
    avg_survival_rounds: float = 0.0
    elo: float = 1500.0
    
    def update_averages(self):
        """Update calculated averages."""
        if self.games_played > 0:
            self.avg_survival_rounds = self.total_rounds / self.games_played
    
    @property
    def winrate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    @property
    def challenge_accuracy(self) -> float:
        """Calculate challenge accuracy."""
        total_challenges = self.successful_challenges + self.failed_challenges
        if total_challenges == 0:
            return 0.0
        return self.successful_challenges / total_challenges
    
    @property
    def calza_accuracy(self) -> float:
        """Calculate calza accuracy."""
        total_calzas = self.successful_calzas + self.failed_calzas
        if total_calzas == 0:
            return 0.0
        return self.successful_calzas / total_calzas


class BotPersonalityTracker:
    """
    Tracks statistics for different bot personalities during training.
    
    Features:
    - Track win rates, ELO, and detailed statistics for each personality
    - Save/load statistics to/from JSON file
    - Update ELO ratings after games
    - Provide summary statistics
    """
    
    def __init__(
        self,
        stats_file: str = "models/bot_personality_stats.json",
        elo_k: int = 32,
    ):
        """
        Initialize bot personality tracker.
        
        Args:
            stats_file: Path to JSON file for saving statistics
            elo_k: ELO K-factor for rating updates
        """
        self.stats_file = stats_file
        self.elo_k = elo_k
        
        # Statistics for each personality
        self.personality_stats: Dict[str, BotPersonalityStats] = {}
        
        # Track RL agent ELO separately
        self.rl_agent_elo: float = 1500.0
        
        # Create directory if needed
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        
        # Load existing statistics
        self._load_stats()
    
    def _load_stats(self):
        """Load statistics from JSON file."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Load RL agent ELO
                    if "rl_agent_elo" in data:
                        self.rl_agent_elo = float(data["rl_agent_elo"])
                    
                    # Load personality statistics
                    for personality_name, stats_data in data.get("personalities", {}).items():
                        stats = BotPersonalityStats(**stats_data)
                        # Fix invalid ELO values
                        if stats.elo < 0 or stats.elo > 3000:
                            print(f"Warning: Fixing invalid ELO {stats.elo} for {personality_name}")
                            stats.elo = max(0.0, min(3000.0, stats.elo))
                        self.personality_stats[personality_name] = stats
                        
                print(f"Loaded bot personality statistics from {self.stats_file}")
            except Exception as e:
                print(f"Warning: Could not load bot personality statistics: {e}")
    
    def _save_stats(self):
        """Save statistics to JSON file."""
        data = {
            "rl_agent_elo": self.rl_agent_elo,
            "personalities": {
                name: asdict(stats)
                for name, stats in self.personality_stats.items()
            }
        }
        
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save bot personality statistics: {e}")
    
    def register_personality(self, personality_name: str):
        """
        Register a new personality for tracking.
        
        Args:
            personality_name: Name of the personality
        """
        if personality_name not in self.personality_stats:
            self.personality_stats[personality_name] = BotPersonalityStats(
                personality_name=personality_name
            )
    
    def update_game_result(
        self,
        personality_name: str,
        won: bool,
        rounds_survived: int = 0,
        dice_lost: int = 0,
        dice_won: int = 0,
        successful_challenges: int = 0,
        failed_challenges: int = 0,
        successful_calzas: int = 0,
        failed_calzas: int = 0,
        times_challenged: int = 0,
        times_challenged_correctly: int = 0,
        update_elo: bool = True,
        opponent_elo: Optional[float] = None,
    ):
        """
        Update statistics after a game.
        
        Args:
            personality_name: Name of the personality
            won: Whether the bot won
            rounds_survived: Number of rounds the bot survived
            dice_lost: Number of dice lost during the game
            dice_won: Number of dice won during the game
            successful_challenges: Number of successful challenges
            failed_challenges: Number of failed challenges
            successful_calzas: Number of successful calzas
            failed_calzas: Number of failed calzas
            times_challenged: Number of times the bot was challenged
            times_challenged_correctly: Number of times the bot was correctly challenged
            update_elo: Whether to update ELO ratings
            opponent_elo: ELO of opponent (uses rl_agent_elo if None)
        """
        # Register personality if not exists
        self.register_personality(personality_name)
        
        stats = self.personality_stats[personality_name]
        
        # Update basic statistics
        stats.games_played += 1
        if won:
            stats.wins += 1
        else:
            stats.losses += 1
        
        stats.total_rounds += rounds_survived
        stats.total_dice_lost += dice_lost
        stats.total_dice_won += dice_won
        stats.successful_challenges += successful_challenges
        stats.failed_challenges += failed_challenges
        stats.successful_calzas += successful_calzas
        stats.failed_calzas += failed_calzas
        stats.times_challenged += times_challenged
        stats.times_challenged_correctly += times_challenged_correctly
        
        # Update averages
        stats.update_averages()
        
        # Update ELO ratings
        if update_elo:
            if opponent_elo is None:
                opponent_elo = self.rl_agent_elo
            
            # Calculate expected scores
            elo_diff = (opponent_elo - stats.elo) / 400.0
            
            # Clamp to prevent overflow
            if elo_diff > 10.0:
                expected_score_bot = 0.0
            elif elo_diff < -10.0:
                expected_score_bot = 1.0
            else:
                expected_score_bot = 1.0 / (1.0 + 10 ** elo_diff)
            
            expected_score_opponent = 1.0 - expected_score_bot
            
            # Actual scores
            actual_score_bot = 1.0 if won else 0.0
            actual_score_opponent = 1.0 - actual_score_bot
            
            # Calculate ELO changes
            elo_change_bot = self.elo_k * (actual_score_bot - expected_score_bot)
            elo_change_opponent = self.elo_k * (actual_score_opponent - expected_score_opponent)
            
            # Update ELO ratings
            stats.elo += elo_change_bot
            self.rl_agent_elo += elo_change_opponent
            
            # Clamp ELO to reasonable bounds
            stats.elo = max(0.0, min(3000.0, stats.elo))
            self.rl_agent_elo = max(0.0, min(3000.0, self.rl_agent_elo))
        
        # Save updated statistics
        self._save_stats()
    
    def get_stats(self, personality_name: str) -> Optional[BotPersonalityStats]:
        """
        Get statistics for a specific personality.
        
        Args:
            personality_name: Name of the personality
            
        Returns:
            BotPersonalityStats or None if not found
        """
        return self.personality_stats.get(personality_name)
    
    def get_all_stats(self) -> Dict[str, BotPersonalityStats]:
        """
        Get statistics for all personalities.
        
        Returns:
            Dictionary mapping personality names to their statistics
        """
        return self.personality_stats.copy()
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics for all personalities.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.personality_stats:
            return {
                "total_personalities": 0,
                "total_games": 0,
                "rl_agent_elo": self.rl_agent_elo,
            }
        
        total_games = sum(stats.games_played for stats in self.personality_stats.values())
        avg_elo = sum(stats.elo for stats in self.personality_stats.values()) / len(self.personality_stats)
        
        # Find best and worst performers
        best_winrate_bot = max(
            self.personality_stats.items(),
            key=lambda x: (x[1].winrate, x[1].games_played)
        )
        worst_winrate_bot = min(
            self.personality_stats.items(),
            key=lambda x: (x[1].winrate, -x[1].games_played)
        )
        highest_elo_bot = max(
            self.personality_stats.items(),
            key=lambda x: x[1].elo
        )
        
        return {
            "total_personalities": len(self.personality_stats),
            "total_games": total_games,
            "avg_elo": avg_elo,
            "rl_agent_elo": self.rl_agent_elo,
            "best_winrate": {
                "name": best_winrate_bot[0],
                "winrate": best_winrate_bot[1].winrate,
                "games": best_winrate_bot[1].games_played,
            },
            "worst_winrate": {
                "name": worst_winrate_bot[0],
                "winrate": worst_winrate_bot[1].winrate,
                "games": worst_winrate_bot[1].games_played,
            },
            "highest_elo": {
                "name": highest_elo_bot[0],
                "elo": highest_elo_bot[1].elo,
                "games": highest_elo_bot[1].games_played,
            },
        }
    
    def print_summary(self):
        """Print formatted summary of all personality statistics."""
        if not self.personality_stats:
            print("\nNo bot personality statistics available yet.")
            return
        
        print("\n" + "="*80)
        print("BOT PERSONALITY STATISTICS")
        print("="*80)
        
        summary = self.get_summary()
        print(f"\nRL Agent ELO: {summary['rl_agent_elo']:.1f}")
        print(f"Total personalities tracked: {summary['total_personalities']}")
        print(f"Total games played: {summary['total_games']}")
        print(f"Average bot ELO: {summary['avg_elo']:.1f}")
        
        print(f"\nBest win rate: {summary['best_winrate']['name']}")
        print(f"  Win rate: {summary['best_winrate']['winrate']*100:.1f}% ({summary['best_winrate']['games']} games)")
        
        print(f"\nWorst win rate: {summary['worst_winrate']['name']}")
        print(f"  Win rate: {summary['worst_winrate']['winrate']*100:.1f}% ({summary['worst_winrate']['games']} games)")
        
        print(f"\nHighest ELO: {summary['highest_elo']['name']}")
        print(f"  ELO: {summary['highest_elo']['elo']:.1f} ({summary['highest_elo']['games']} games)")
        
        # Print detailed statistics for each personality
        print("\n" + "-"*80)
        print("DETAILED STATISTICS BY PERSONALITY")
        print("-"*80)
        
        # Sort by ELO descending
        sorted_stats = sorted(
            self.personality_stats.items(),
            key=lambda x: x[1].elo,
            reverse=True
        )
        
        for name, stats in sorted_stats:
            print(f"\n{name}:")
            print(f"  Games: {stats.games_played} | Wins: {stats.wins} | Losses: {stats.losses}")
            print(f"  Win rate: {stats.winrate*100:.1f}%")
            print(f"  ELO: {stats.elo:.1f}")
            print(f"  Avg survival rounds: {stats.avg_survival_rounds:.1f}")
            
            if stats.successful_challenges + stats.failed_challenges > 0:
                print(f"  Challenge accuracy: {stats.challenge_accuracy*100:.1f}% ({stats.successful_challenges}/{stats.successful_challenges + stats.failed_challenges})")
            
            if stats.successful_calzas + stats.failed_calzas > 0:
                print(f"  Calza accuracy: {stats.calza_accuracy*100:.1f}% ({stats.successful_calzas}/{stats.successful_calzas + stats.failed_calzas})")
            
            if stats.times_challenged > 0:
                print(f"  Times challenged: {stats.times_challenged} | Correctly: {stats.times_challenged_correctly} ({stats.times_challenged_correctly/stats.times_challenged*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def reset_stats(self, personality_name: Optional[str] = None):
        """
        Reset statistics for a personality or all personalities.
        
        Args:
            personality_name: Name of personality to reset, or None to reset all
        """
        if personality_name is None:
            self.personality_stats.clear()
            self.rl_agent_elo = 1500.0
            print("Reset all bot personality statistics")
        elif personality_name in self.personality_stats:
            del self.personality_stats[personality_name]
            print(f"Reset statistics for {personality_name}")
        
        self._save_stats()

