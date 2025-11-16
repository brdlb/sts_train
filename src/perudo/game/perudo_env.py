"""
Gymnasium environment for Perudo game.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

from .base_perudo_env import BasePerudoEnv
from .rules import PerudoRules
from ..utils.helpers import action_to_bid
from ..training.config import RewardConfig, DEFAULT_CONFIG, EnvironmentConfig

class PerudoEnv(BasePerudoEnv):
    """Gymnasium environment for Perudo game."""

    def __init__(
        self,
        num_players: int = 4,
        dice_per_player: int = 5,
        total_dice_values: int = 6,
        max_quantity: int = 30,
        history_length: int = 10,
        max_history_length: Optional[int] = None,
        render_mode: Optional[str] = None,
        random_num_players: bool = True,
        min_players: int = 3,
        max_players: int = 8,
        reward_config: Optional[RewardConfig] = None,
        environment_config: Optional[EnvironmentConfig] = None,
    ):
        """
        Initialize Perudo environment.

        Args:
            num_players: Number of players (used if random_num_players=False, or as max for observation space)
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
            max_quantity: Maximum dice quantity in bid
            history_length: Bid history length in observation (deprecated, use max_history_length)
            max_history_length: Maximum length of bid history sequence (defaults to history_length)
            render_mode: Render mode
            random_num_players: If True, randomly select num_players in each episode
            min_players: Minimum number of players (used when random_num_players=True)
            max_players: Maximum number of players (used when random_num_players=True)
            reward_config: Reward configuration (uses DEFAULT_CONFIG.reward if not provided)
            environment_config: Full environment configuration (uses DEFAULT_CONFIG.environment if not provided)
        """
        super().__init__(
            num_players=num_players,
            dice_per_player=dice_per_player,
            total_dice_values=total_dice_values,
            max_quantity=max_quantity,
            history_length=history_length,
            max_history_length=max_history_length,
            render_mode=render_mode,
            random_num_players=random_num_players,
            min_players=min_players,
            max_players=max_players,
            reward_config=reward_config,
            environment_config=environment_config,
        )
        
        # Episode statistics for monitoring (use episode_tracker from base class)
        self.episode_bid_count = 0
        self.episode_challenge_count = 0
        self.episode_believe_count = 0
        self.episode_invalid_action_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.invalid_action_attempts = 0
        self.invalid_action_penalty_accumulated = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment to initial state.
        Randomly selects number of players (3-8) at the start of each episode.

        Args:
            seed: Random seed
            options: Optional parameters

        Returns:
            Tuple (observation, info)
        """
        # Call parent reset for seed handling
        if seed is not None:
            super().reset(seed=seed)

        # Select number of players for this episode
        if self.random_num_players:
            self.num_players = np.random.randint(self.min_players, self.max_players + 1)
        else:
            self.num_players = self.max_num_players

        # Recreate game state with new number of players
        from .game_state import GameState
        self.game_state = GameState(
            num_players=self.num_players,
            dice_per_player=self.dice_per_player,
            total_dice_values=self.total_dice_values,
        )

        # Set active player
        self.active_player_id = self.game_state.current_player

        # Get observation for active player
        observation = self.get_observation_for_player(self.active_player_id)

        info = {
            "player_id": self.active_player_id,
            "num_players": self.num_players,
            "game_state": self.game_state.get_public_info(),
        }

        # Reset episode statistics
        self.episode_tracker.reset()
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_bid_count = 0
        self.episode_challenge_count = 0
        self.episode_believe_count = 0
        self.episode_invalid_action_count = 0
        self.invalid_action_attempts = 0
        self.invalid_action_penalty_accumulated = 0.0

        # Initialize round tracking
        self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
        self.agent_bid_this_round = False
        self.agent_deferred_reward = 0.0

        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action from action_space

        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        # Check that game is not over
        if self.game_state.game_over:
            observation = self.get_observation_for_player(self.active_player_id)
            winner = self.game_state.winner if hasattr(self.game_state, "winner") and self.game_state.winner is not None else -1
            info = {
                "game_over": True,
                "winner": self.game_state.winner,
                "_episode_bid_count": self.episode_bid_count,
                "_episode_challenge_count": self.episode_challenge_count,
                "_episode_believe_count": self.episode_believe_count,
                "_episode_invalid_action_count": self.episode_invalid_action_count,
                "_episode_reward_raw": self.episode_reward,
                "_episode_length_raw": self.episode_length,
            }
            return observation, 0.0, True, False, info

        # Check that active player has dice - skip if not
        if not self.game_state.is_player_active(self.active_player_id):
            try:
                next_player = self.game_controller.skip_to_next_active_player(
                    self.game_state, self.active_player_id
                )
                self.active_player_id = next_player
                self.game_state.current_player = next_player
            except ValueError:
                # No active players - game should be over
                self.game_state._check_game_over()
            observation = self.get_observation_for_player(self.active_player_id)
            return observation, 0.0, False, False, {"error": "Player has no dice"}

        # Check that it's active player's turn
        if self.game_state.current_player != self.active_player_id:
            observation = self.get_observation_for_player(self.active_player_id)
            return observation, 0.0, False, False, {"error": "Not player's turn"}

        # Force initial bid if needed (using GameController)
        action, was_forced = self.game_controller.force_initial_bid_if_needed(
            self.game_state, self.active_player_id, action, self.max_quantity
        )

        # Convert action to game format
        action_type, param1, param2 = action_to_bid(action, self.max_quantity)

        # Check action masking for learning agent (player_id=0)
        if self.active_player_id == 0:
            action_mask = self.observation_builder.get_action_mask(
                self.game_state, self.active_player_id
            )
            if not action_mask[action]:
                self.episode_invalid_action_count += 1

        # Execute action
        reward = 0.0
        dice_lost = 0
        challenge_success = None
        believe_success = None
        action_valid = False
        retry_needed = False  # Flag to indicate if action needs to be retried
        error_msg = None  # Reason for rejection if action is invalid
        actual_count_info = None  # Initialize for info dict (actual dice count for challenge/believe)
        loser_id_info = None  # Initialize for info dict (player who lost dice, if any)

        if action_type == "bid":
            # Make a bid
            quantity, value = param1, param2
            is_valid, error_msg = PerudoRules.is_valid_bid(
                self.game_state, self.active_player_id, quantity, value
            )
            if is_valid:
                action_valid = self.game_state.set_bid(
                    self.active_player_id, quantity, value
                )
                if action_valid:
                    # Track that agent made a bid this round
                    if self.active_player_id == 0:
                        self.agent_bid_this_round = True
                    # Count bid action (for all players, not just learning agent)
                    self.episode_bid_count += 1
                    self.game_state.next_player()
                    # Calculate reward using RewardCalculator
                    action_result = {
                        "player_id": self.active_player_id,
                        "game_over": False,
                        "winner": -1,
                        "dice_lost": 0,
                    }
                    reward = self.reward_calculator.calculate_step_reward(
                        "bid", action_result, self.game_state, 0
                    )
            else:
                # Invalid action - give penalty and pass turn
                reward = self.reward_config.invalid_action_penalty
                action_valid = False
                retry_needed = False
                self.game_state.next_player()
                # Reset invalid action attempts counter
                self.invalid_action_attempts = 0
                self.invalid_action_penalty_accumulated = 0.0
                
                # Check if game ended after passing turn
                if self.game_state.game_over:
                    winner = self.game_state.winner
                    winner_dice_count = None
                    if winner is not None and winner >= 0 and winner == self.active_player_id:
                        # Only pass dice count if the active player won
                        if winner < len(self.game_state.player_dice_count):
                            winner_dice_count = self.game_state.player_dice_count[winner]
                    action_result = {
                        "player_id": self.active_player_id,
                        "game_over": self.game_state.game_over,
                        "winner": winner or -1,
                        "dice_lost": 0,
                        "winner_dice_count": winner_dice_count,
                    }
                    reward = self.reward_calculator.calculate_step_reward(
                        "bid", action_result, self.game_state, 0
                    )

        elif action_type == "challenge":
            # Challenge previous player
            can_challenge, error_msg = PerudoRules.can_challenge(
                self.game_state, self.active_player_id
            )
            if can_challenge:
                challenge_success, actual_count, bid_quantity = (
                    self.game_state.challenge_bid(self.active_player_id)
                )
                loser_id, dice_lost = PerudoRules.process_challenge_result(
                    self.game_state,
                    self.active_player_id,
                    challenge_success,
                    actual_count,
                    bid_quantity,
                )
                # Store for info
                actual_count_info = actual_count
                loser_id_info = loser_id
                
                # Check if agent (player 0) made the bid and successfully defended it
                # (opponent challenged and failed)
                agent_id = 0
                agent_defended_bid = False
                if self.game_state.bid_history:
                    bid_maker_id = self.game_state.bid_history[-1][0]
                    # Agent made the bid, challenge failed (opponent was wrong), agent didn't lose dice
                    if (bid_maker_id == agent_id and 
                        not challenge_success and 
                        loser_id != agent_id):
                        agent_defended_bid = True
                
                # Check round reward BEFORE losing dice
                agent_lost_dice = (loser_id == 0 and dice_lost > 0)
                round_reward = self.reward_calculator.calculate_round_end_reward(
                    agent_lost_dice, self.agent_bid_this_round, 0
                )
                self.game_state.lose_dice(loser_id, dice_lost)
                
                # CRITICAL: Check if game ended after losing dice
                # If game is over, don't start new round
                if not self.game_state.game_over:
                    # Use GameController to handle round end
                    self.game_controller.handle_round_end(self.game_state, loser_id)
                    # Reset round tracking after dice roll
                    self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
                    self.agent_bid_this_round = False
                else:
                    # Game ended, clear bid state
                    self.game_state.current_bid = None
                    self.game_state.believe_called = False

                winner = self.game_state.winner
                winner_dice_count = None
                if self.game_state.game_over and winner is not None and winner >= 0 and winner == self.active_player_id:
                    # Only pass dice count if the active player won
                    if winner < len(self.game_state.player_dice_count):
                        winner_dice_count = self.game_state.player_dice_count[winner]
                action_result = {
                    "player_id": self.active_player_id,
                    "game_over": self.game_state.game_over,
                    "winner": winner or -1,
                    "challenge_success": challenge_success,
                    "dice_lost": dice_lost if loser_id == self.active_player_id else 0,
                    "winner_dice_count": winner_dice_count,
                }
                reward = self.reward_calculator.calculate_step_reward(
                    "challenge", action_result, self.game_state, 0
                )
                # Count challenge action (for all players, not just learning agent)
                self.episode_challenge_count += 1
                # Add round reward
                if self.active_player_id == 0:
                    reward += round_reward
                # If agent successfully defended their bid (opponent challenged and failed),
                # store deferred reward for agent (will be given on agent's next turn)
                elif agent_defended_bid:
                    # Agent gets reward for successfully defending their risky bid
                    # This reward will be added when agent makes their next action
                    self.agent_deferred_reward += self.reward_config.defend_bid_reward_challenge
                action_valid = True
            else:
                # Invalid challenge - give penalty and pass turn
                reward = self.reward_config.invalid_action_penalty
                action_valid = False
                retry_needed = False
                self.game_state.next_player()
                # Reset invalid action attempts counter
                self.invalid_action_attempts = 0
                self.invalid_action_penalty_accumulated = 0.0
                
                # Check if game ended after passing turn
                if self.game_state.game_over:
                    winner = self.game_state.winner
                    winner_dice_count = None
                    if winner is not None and winner >= 0 and winner == self.active_player_id:
                        # Only pass dice count if the active player won
                        if winner < len(self.game_state.player_dice_count):
                            winner_dice_count = self.game_state.player_dice_count[winner]
                    action_result = {
                        "player_id": self.active_player_id,
                        "game_over": self.game_state.game_over,
                        "winner": winner or -1,
                        "challenge_success": False,
                        "dice_lost": 0,
                        "winner_dice_count": winner_dice_count,
                    }
                    reward = self.reward_calculator.calculate_step_reward(
                        "challenge", action_result, self.game_state, 0
                    )

        elif action_type == "believe":
            # Call believe
            can_believe, error_msg = PerudoRules.can_call_believe(
                self.game_state, self.active_player_id
            )
            if can_believe:
                believe_success, actual_count = self.game_state.call_believe(
                    self.active_player_id
                )
                bid_quantity = self.game_state.current_bid[0] if self.game_state.current_bid else 0
                loser_id, dice_lost, next_round_starter = PerudoRules.process_believe_result(
                    self.game_state,
                    self.active_player_id,
                    believe_success,
                    actual_count,
                    bid_quantity,
                )
                # Store for info
                actual_count_info = actual_count
                loser_id_info = loser_id
                
                # Handle believe result according to new rules
                player_who_changed_dice = None  # Track who gained or lost a die
                if believe_success:
                    # Dice exactly equals bid: believer benefits
                    if self.game_state.player_dice_count[self.active_player_id] < self.game_state.dice_per_player:
                        # Gain a die
                        self.game_state.gain_dice(self.active_player_id, 1)
                        dice_lost = 0
                        player_who_changed_dice = self.active_player_id
                    elif next_round_starter is not None:
                        # Believer has 5 dice and starts next round
                        dice_lost = 0
                        player_who_changed_dice = next_round_starter
                else:
                    # Dice doesn't equal bid: believer loses die
                    # Don't lose dice here - it will be handled by process_believe_result
                    dice_lost = 1
                    player_who_changed_dice = self.active_player_id
                
                # Check if agent (player 0) made the bid and successfully defended it
                # (opponent called believe and failed)
                agent_id = 0
                agent_defended_bid = False
                if self.game_state.bid_history:
                    bid_maker_id = self.game_state.bid_history[-1][0]
                    # Agent made the bid, believe failed (opponent was wrong), agent didn't lose dice
                    if (bid_maker_id == agent_id and 
                        not believe_success and 
                        loser_id == self.active_player_id):
                        agent_defended_bid = True
                
                # Check round reward BEFORE losing dice
                agent_lost_dice = (loser_id is not None and loser_id == 0 and dice_lost > 0)
                round_reward = self.reward_calculator.calculate_round_end_reward(
                    agent_lost_dice, self.agent_bid_this_round, 0
                )
                if loser_id is not None:
                    self.game_state.lose_dice(loser_id, dice_lost)
                
                # CRITICAL: Check if game ended after losing dice
                # If game is over, don't start new round
                if not self.game_state.game_over:
                    # Determine who starts next round
                    round_starter = None
                    if player_who_changed_dice is not None:
                        round_starter = player_who_changed_dice
                    elif next_round_starter is not None:
                        round_starter = next_round_starter
                    elif loser_id is not None:
                        round_starter = loser_id
                    
                    # Use GameController to handle round end
                    self.game_controller.handle_round_end(
                        self.game_state, 
                        loser_id if loser_id is not None else self.active_player_id,
                        winner_id=round_starter
                    )
                    # Reset round tracking after dice roll
                    self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
                    self.agent_bid_this_round = False
                else:
                    # Game ended, clear bid state
                    self.game_state.current_bid = None
                    self.game_state.believe_called = False

                winner = self.game_state.winner
                winner_dice_count = None
                if self.game_state.game_over and winner is not None and winner >= 0 and winner == self.active_player_id:
                    # Only pass dice count if the active player won
                    if winner < len(self.game_state.player_dice_count):
                        winner_dice_count = self.game_state.player_dice_count[winner]
                action_result = {
                    "player_id": self.active_player_id,
                    "game_over": self.game_state.game_over,
                    "winner": winner or -1,
                    "believe_success": believe_success,
                    "dice_lost": dice_lost if (loser_id is not None and loser_id == self.active_player_id) else 0,
                    "winner_dice_count": winner_dice_count,
                }
                reward = self.reward_calculator.calculate_step_reward(
                    "believe", action_result, self.game_state, 0
                )
                # Count believe action (for all players, not just learning agent)
                self.episode_believe_count += 1
                # Add round reward
                if self.active_player_id == 0:
                    reward += round_reward
                # If agent successfully defended their bid (opponent called believe and failed),
                # store deferred reward for agent (will be given on agent's next turn)
                elif agent_defended_bid:
                    # Agent gets reward for successfully defending their risky bid
                    # This reward will be added when agent makes their next action
                    self.agent_deferred_reward += self.reward_config.defend_bid_reward_believe
                action_valid = True
            else:
                # Invalid believe - give penalty and pass turn
                reward = self.reward_config.invalid_action_penalty
                action_valid = False
                retry_needed = False
                self.game_state.next_player()
                # Reset invalid action attempts counter
                self.invalid_action_attempts = 0
                self.invalid_action_penalty_accumulated = 0.0
                
                # Check if game ended after passing turn
                if self.game_state.game_over:
                    winner = self.game_state.winner
                    winner_dice_count = None
                    if winner is not None and winner >= 0 and winner == self.active_player_id:
                        # Only pass dice count if the active player won
                        if winner < len(self.game_state.player_dice_count):
                            winner_dice_count = self.game_state.player_dice_count[winner]
                    action_result = {
                        "player_id": self.active_player_id,
                        "game_over": self.game_state.game_over,
                        "winner": winner or -1,
                        "believe_success": False,
                        "dice_lost": 0,
                        "winner_dice_count": winner_dice_count,
                    }
                    reward = self.reward_calculator.calculate_step_reward(
                        "believe", action_result, self.game_state, 0
                    )

        # Save action information
        self.last_action_info = {
            "action_type": action_type,
            "action_valid": action_valid,
            "reward": reward,
            "dice_lost": dice_lost,
            "challenge_success": challenge_success,
            "believe_success": believe_success,
            "error_msg": error_msg,  # Reason for rejection if action is invalid
            "actual_count": actual_count_info,  # Actual dice count for challenge/believe
            "loser_id": loser_id_info,  # Player who lost dice (if any)
        }

        # Get new observation
        observation = self.get_observation_for_player(self.active_player_id)

        # накопление суммарной награды и длины эпизода
        # ВАЖНО: накапливаем статистику только для learning agent (player_id=0)
        # В vec_env только learning agent делает ходы через этот метод для сбора статистики
        # Оппоненты играют отдельно и их награды не должны учитываться в статистике learning agent
        # Invalid actions now give -1 reward and pass turn (no retry mechanism)
        if self.active_player_id == 0:
            # Add any deferred reward for agent (e.g., from successfully defending bid)
            if self.agent_deferred_reward != 0.0:
                reward += self.agent_deferred_reward
                self.agent_deferred_reward = 0.0
            
            # Add dice advantage reward
            agent_dice = self.game_state.player_dice_count[0]
            opponent_dice_counts = [
                count for i, count in enumerate(self.game_state.player_dice_count)
                if i != 0 and i < self.num_players
            ]
            dice_advantage_reward = self.reward_calculator.calculate_dice_advantage_reward(
                agent_dice, opponent_dice_counts, 0
            )
            reward += dice_advantage_reward
            
            self.episode_reward += reward
            # Only increment episode_length for valid actions
            # This ensures episode_length matches the sum of valid actions (bid_count + challenge_count + believe_count)
            if action_valid:
                self.episode_length += 1
            else:
                # Track invalid actions for debugging
                self.episode_invalid_action_count += 1

        # Check game over
        terminated = self.game_state.game_over
        truncated = False  # Not implemented yet
        done = terminated or truncated
        
        # If episode ended and there's deferred reward for agent, add it to episode reward
        # (This handles the case where episode ends before agent's next turn)
        if done and self.agent_deferred_reward != 0.0 and self.active_player_id != 0:
            # Episode ended on opponent's turn, but agent had deferred reward
            # Add it to episode reward directly
            self.episode_reward += self.agent_deferred_reward
            self.agent_deferred_reward = 0.0
        
        info = {
            "player_id": self.active_player_id,
            "game_over": terminated,
            "winner": self.game_state.winner,
            "action_info": self.last_action_info,
            "game_state": self.game_state.get_public_info(),
            "retry": retry_needed,  # Flag indicating if action needs to be retried
            "invalid_action_attempts": self.invalid_action_attempts,  # Number of invalid attempts
        }

        # если эпизод завершился, сохраняем статистику для realtime
        # CRITICAL: When used in VecEnv, episode info should be set by VecEnv wrapper, not here
        # This prevents incorrect rewards from being recorded when game ends on opponent's turn
        # VecEnv wrapper (perudo_vec_env.py) will recalculate and set correct reward for learning agent
        if done:
            # Only set episode info if it's not already set by VecEnv wrapper
            # VecEnv wrapper sets episode info with correct learning agent reward
            if "episode" not in info:
                # VecMonitor expects "episode" key with "r" (reward) and "l" (length) subkeys
                # Additional statistics: bid_count, challenge_count, believe_count, winner
                # NOTE: This is only used when env is used standalone (not in VecEnv)
                # When used in VecEnv, perudo_vec_env.py will set correct episode info
                info["episode"] = {
                    "r": self.episode_reward,
                    "l": self.episode_length,
                    "bid_count": self.episode_bid_count,
                    "challenge_count": self.episode_challenge_count,
                    "believe_count": self.episode_believe_count,
                    "invalid_action_count": self.episode_invalid_action_count,
                    "winner": self.game_state.winner if hasattr(self.game_state, "winner") and self.game_state.winner is not None else -1,
                }
                # Also keep old format for backward compatibility with custom callbacks
                info["episode_reward"] = self.episode_reward
                info["episode_length"] = self.episode_length
            # Always set winner and other game state info (VecEnv may need it)
            if hasattr(self.game_state, "winner"):
                info["winner"] = self.game_state.winner
            # Always set episode statistics (bid_count, etc.) for VecEnv to use
            info["_episode_bid_count"] = self.episode_bid_count
            info["_episode_challenge_count"] = self.episode_challenge_count
            info["_episode_believe_count"] = self.episode_believe_count
            info["_episode_invalid_action_count"] = self.episode_invalid_action_count

        return observation, reward, terminated, truncated, info

