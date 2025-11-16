"""
Gymnasium environment for Perudo game.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any

from .game_state import GameState
from .rules import PerudoRules
from ..utils.helpers import (
    create_observation_vector,
    create_observation_dict,
    calculate_reward,
    action_to_bid,
    decode_bid,
    create_action_mask,
)
from ..training.config import RewardConfig, DEFAULT_CONFIG

class PerudoEnv(gym.Env):
    """Gymnasium environment for Perudo game."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

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
        """
        super().__init__()

        # Store reward configuration
        self.reward_config = reward_config if reward_config is not None else DEFAULT_CONFIG.reward

        # Store parameters for random player selection
        self.random_num_players = random_num_players
        self.min_players = min_players
        self.max_players = max_players
        
        # Use maximum number of players for observation space
        if random_num_players:
            self.max_num_players = max(max_players, num_players)  # At least max_players for random selection
        else:
            self.max_num_players = num_players
        
        self.num_players = num_players  # Will be updated in reset() if random_num_players=True
        self.dice_per_player = dice_per_player
        self.total_dice_values = total_dice_values
        self.max_quantity = max_quantity
        self.history_length = history_length  # Keep for backward compatibility
        self.max_history_length = max_history_length if max_history_length is not None else history_length
        self.render_mode = render_mode

        # Create game state with initial num_players (will be recreated in reset())
        self.game_state = GameState(
            num_players=num_players,
            dice_per_player=dice_per_player,
            total_dice_values=total_dice_values,
        )

        # Define observation space as Dict for transformer architecture
        # bid_history: sequence of bids (max_history_length, 3) - (player_id, quantity, value)
        # This preserves turn order context, allowing agents to understand who made each bid
        # static_info: static information (agent_id, current_bid, dice_count, current_player, palifico, believe, player_dice)
        
        # Calculate static_info size
        static_info_size = (
            self.max_num_players  # agent_id one-hot
            + 2  # current_bid (quantity, value)
            + self.max_num_players  # dice_count
            + 1  # current_player
            + self.max_num_players  # palifico
            + 1  # believe
            + 1  # special_round_active
            + 1  # round_number
            + 5  # player_dice
        )
        
        action_size = 2 + max_quantity * 6
        self.observation_space = spaces.Dict({
            "bid_history": spaces.Box(
                low=0, high=100, shape=(self.max_history_length, 3), dtype=np.int32
            ),
            "static_info": spaces.Box(
                low=0, high=100, shape=(static_info_size,), dtype=np.float32
            ),
            "action_mask": spaces.Box(low=0, high=1, shape=(action_size,), dtype=np.bool_),
        })

        # Define action space
        # Actions: 0=challenge, 1=believe, 2+=bid(encoded)
        self.action_space = spaces.Discrete(action_size)

        # Current active player (for whom observation is returned)
        self.active_player_id = 0

        # Information about last action (for debugging)
        self.last_action_info = {}

        self.episode_reward = 0.0
        self.episode_length = 0

        # Track round information for intermediate rewards
        # Agent dice count at the start of current round
        self.agent_dice_at_round_start = dice_per_player
        # Track if agent made a bid this round (for successful bluff detection)
        self.agent_bid_this_round = False
        # Deferred reward for agent (e.g., when opponent challenges agent's bid and fails)
        self.agent_deferred_reward = 0.0
        
        # Episode statistics for monitoring
        self.episode_bid_count = 0  # Total number of bids made in episode
        self.episode_challenge_count = 0  # Total number of challenges made in episode
        self.episode_believe_count = 0  # Total number of believe calls made in episode
        self.episode_invalid_action_count = 0  # Total number of invalid actions made by learning agent in episode
        
        # Track invalid action attempts (for compatibility, not used for retry anymore)
        self.invalid_action_attempts = 0
        self.invalid_action_penalty_accumulated = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        Randomly selects number of players (3-8) at the start of each episode.

        Args:
            seed: Random seed
            options: Optional parameters

        Returns:
            Tuple (observation, info)
        """
        super().reset(seed=seed)

        # Select number of players for this episode
        if self.random_num_players:
            # Randomly select number of players within min-max range
            self.num_players = np.random.randint(self.min_players, self.max_players + 1)  # min to max inclusive
        else:
            # Use fixed number of players from config
            self.num_players = self.max_num_players

        # Recreate game state with new number of players
        self.game_state = GameState(
            num_players=self.num_players,
            dice_per_player=self.dice_per_player,
            total_dice_values=self.total_dice_values,
        )
        
        # Reset game state with seed to ensure proper randomization of starting player
        # This is necessary because GameState.__init__() calls reset() without seed
        self.game_state.reset(seed=seed)

        # Set active player
        self.active_player_id = self.game_state.current_player

        # Get observation for active player
        observation = self._get_observation(self.active_player_id)

        info = {
            "player_id": self.active_player_id,
            "num_players": self.num_players,
            "game_state": self.game_state.get_public_info(),
        }

        self.episode_reward = 0.0
        self.episode_length = 0

        # Initialize round tracking
        self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
        self.agent_bid_this_round = False
        self.agent_deferred_reward = 0.0
        
        # Reset episode statistics
        self.episode_bid_count = 0
        self.episode_challenge_count = 0
        self.episode_believe_count = 0
        self.episode_invalid_action_count = 0
        
        # Reset invalid action tracking
        self.invalid_action_attempts = 0
        self.invalid_action_penalty_accumulated = 0.0

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action from action_space

        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        # Check that game is not over
        if self.game_state.game_over:
            observation = self._get_observation(self.active_player_id)
            
            winner = self.game_state.winner if hasattr(self.game_state, "winner") and self.game_state.winner is not None else -1
            # CRITICAL: When used in VecEnv, episode info should be set by VecEnv wrapper, not here
            # This prevents incorrect rewards from being recorded when game ends on opponent's turn
            # VecEnv wrapper (perudo_vec_env.py) will recalculate and set correct reward for learning agent
            info = {
                "game_over": True,
                "winner": self.game_state.winner,
                # Do NOT set episode info here - let VecEnv wrapper set it with correct learning agent reward
                # VecEnv wrapper will recalculate reward based on final game state
                # Episode info is only set here if env is used standalone (not in VecEnv)
                "_episode_bid_count": self.episode_bid_count,
                "_episode_challenge_count": self.episode_challenge_count,
                "_episode_believe_count": self.episode_believe_count,
                "_episode_invalid_action_count": self.episode_invalid_action_count,
                "_episode_reward_raw": self.episode_reward,  # Raw reward for debugging
                "_episode_length_raw": self.episode_length,  # Raw length for debugging
            }
            return observation, 0.0, True, False, info

        # CRITICAL: Check that active player has dice - players with 0 dice cannot make moves
        # Players without dice have already lost and should be automatically skipped
        if self.game_state.player_dice_count[self.active_player_id] == 0:
            # Player has no dice, skip to next player with dice
            # Use next_player() which automatically skips players with 0 dice
            self.game_state.next_player()
            # Update active_player_id to match current_player (which now points to next player with dice)
            self.active_player_id = self.game_state.current_player
            observation = self._get_observation(self.active_player_id)
            return observation, 0.0, False, False, {"error": "Player has no dice"}

        # Convert action to game format
        action_type, param1, param2 = action_to_bid(action, self.max_quantity)

        # Check that it's active player's turn
        if self.game_state.current_player != self.active_player_id:
            # If it's not agent's turn, move to next player
            observation = self._get_observation(self.active_player_id)
            return observation, 0.0, False, False, {"error": "Not player's turn"}

        # Check action masking for learning agent (player_id=0)
        # This helps diagnose if MaskablePPO is properly using action masks
        # NOTE: This check is useful for debugging, but invalid actions are handled
        # by the environment (penalty and turn pass), so training can continue
        if self.active_player_id == 0:
            # Get available actions and create mask
            available_actions = PerudoRules.get_available_actions(self.game_state, self.active_player_id)
            action_mask = create_action_mask(
                available_actions, self.action_space.n, self.max_quantity
            )
            # Check if the selected action is allowed by the mask
            if not action_mask[action]:
                # Count invalid actions for statistics
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
                # Save game state BEFORE making the bid (for minimal bid calculation in reward)
                saved_current_bid = self.game_state.current_bid
                saved_bid_history = self.game_state.bid_history.copy()
                saved_current_player = self.game_state.current_player
                
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
                    
                    # calculate_reward now handles bid_small_penalty and minimal bid incentives
                    # Temporarily restore state BEFORE the bid was made for reward calculation
                    # (so we can compute what was minimal at the time)
                    current_bid_after = self.game_state.current_bid
                    current_history_after = self.game_state.bid_history.copy()
                    current_player_after = self.game_state.current_player
                    
                    # Restore pre-bid state for calculation
                    self.game_state.current_bid = saved_current_bid
                    self.game_state.bid_history = saved_bid_history
                    self.game_state.current_player = saved_current_player
                    
                    reward += calculate_reward(
                        "bid", False, -1, self.active_player_id, dice_lost=0,
                        reward_config=self.reward_config,
                        game_state=self.game_state,
                        bid_quantity=quantity,
                        bid_value=value
                    )
                    
                    # Restore post-bid state
                    self.game_state.current_bid = current_bid_after
                    self.game_state.bid_history = current_history_after
                    self.game_state.current_player = current_player_after
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
                    reward = calculate_reward(
                        "bid",
                        self.game_state.game_over,
                        winner or -1,
                        self.active_player_id,
                        dice_lost=0,
                        reward_config=self.reward_config,
                        winner_dice_count=winner_dice_count,
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
                # Pass action_type to avoid double penalty (challenge already penalized in calculate_reward)
                round_reward = self._check_round_end_reward(loser_id, dice_lost, action_type="challenge")
                self.game_state.lose_dice(loser_id, dice_lost)
                
                # CRITICAL: Check if game ended after losing dice
                # If game is over, don't start new round
                if not self.game_state.game_over:
                    self.game_state.current_bid = None
                    self.game_state.believe_called = False

                    # Restart round: next round starts with the player who lost the die
                    # BUT: if that player now has 0 dice (eliminated), skip to next player with dice
                    # Players with 0 dice have already lost and cannot make moves
                    self.game_state.current_player = loser_id
                    # CRITICAL: If loser now has 0 dice (eliminated), skip to next player with dice
                    if self.game_state.player_dice_count[loser_id] == 0:
                        self.game_state.next_player()

                    # Roll dice again - round ends
                    # Reset special round at end of round
                    self.game_state.special_round_active = False
                    self.game_state.special_round_declared_by = None
                    # Increment round number for new round
                    self.game_state.round_number += 1
                    self.game_state.roll_dice()
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
                reward = calculate_reward(
                    "challenge",
                    self.game_state.game_over,
                    winner or -1,
                    self.active_player_id,
                    challenge_success=challenge_success,
                    dice_lost=dice_lost if loser_id == self.active_player_id else 0,
                    reward_config=self.reward_config,
                    winner_dice_count=winner_dice_count,
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
                    reward = calculate_reward(
                        "challenge",
                        self.game_state.game_over,
                        winner or -1,
                        self.active_player_id,
                        challenge_success=False,
                        dice_lost=0,
                        reward_config=self.reward_config,
                        winner_dice_count=winner_dice_count,
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
                # Pass action_type to avoid double penalty (believe already penalized in calculate_reward)
                round_reward = self._check_round_end_reward(loser_id if loser_id is not None else -1, dice_lost, action_type="believe")
                if loser_id is not None:
                    self.game_state.lose_dice(loser_id, dice_lost)
                
                # CRITICAL: Check if game ended after losing dice
                # If game is over, don't start new round
                if not self.game_state.game_over:
                    self.game_state.current_bid = None
                    self.game_state.believe_called = False

                    # Restart round: next round starts with the player who gained or lost a die
                    # BUT: if that player now has 0 dice (eliminated), skip to next player with dice
                    # Players with 0 dice have already lost and cannot make moves
                    if player_who_changed_dice is not None:
                        self.game_state.current_player = player_who_changed_dice
                        # CRITICAL: If player now has 0 dice (eliminated), skip to next player with dice
                        if self.game_state.player_dice_count[player_who_changed_dice] == 0:
                            self.game_state.next_player()
                    elif next_round_starter is not None:
                        # Believer with 5 dice starts next round
                        self.game_state.current_player = next_round_starter
                        # CRITICAL: If player now has 0 dice (eliminated), skip to next player with dice
                        if self.game_state.player_dice_count[next_round_starter] == 0:
                            self.game_state.next_player()
                    elif loser_id is not None:
                        # Fallback: if no one gained dice but someone lost, that player starts
                        self.game_state.current_player = loser_id
                        # CRITICAL: If loser now has 0 dice (eliminated), skip to next player with dice
                        if self.game_state.player_dice_count[loser_id] == 0:
                            self.game_state.next_player()
                    else:
                        # Fallback: continue normally (next_player() will skip players with 0 dice)
                        self.game_state.next_player()

                    # Roll dice again - round ends
                    # Reset special round at end of round
                    self.game_state.special_round_active = False
                    self.game_state.special_round_declared_by = None
                    # Increment round number for new round
                    self.game_state.round_number += 1
                    self.game_state.roll_dice()
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
                reward = calculate_reward(
                    "believe",
                    self.game_state.game_over,
                    winner or -1,
                    self.active_player_id,
                    believe_success=believe_success,
                    dice_lost=dice_lost if (loser_id is not None and loser_id == self.active_player_id) else 0,
                    reward_config=self.reward_config,
                    winner_dice_count=winner_dice_count,
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
                    reward = calculate_reward(
                        "believe",
                        self.game_state.game_over,
                        winner or -1,
                        self.active_player_id,
                        believe_success=False,
                        dice_lost=0,
                        reward_config=self.reward_config,
                        winner_dice_count=winner_dice_count,
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
        observation = self._get_observation(self.active_player_id)

        # Accumulate total reward and episode length for learning agent
        # CRITICAL: Only accumulate statistics for learning agent (player_id=0)
        # In vec_env only learning agent makes moves through this method for statistics collection
        # Opponents play separately and their rewards should not be counted in learning agent statistics
        if self.active_player_id == 0:
            # NOTE: Deferred rewards are NOT added to reward here to avoid double counting.
            # They are tracked separately in info["deferred_reward"] and accumulated in RewardManager
            # via accumulate_deferred_reward() in perudo_vec_env.py.
            # This ensures deferred rewards are only counted once in the final reward calculation.
            
            # NOTE: dice_advantage_reward is removed to avoid reward inflation
            # Dice advantage already affects win probability, so explicit reward is not needed
            
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
        
        # Handle deferred reward for learning agent
        # Deferred reward is accumulated during opponent turns (e.g., when agent successfully defends bid)
        # Store it in info dict for RewardManager to accumulate separately and include in final reward calculation
        # NOTE: Deferred rewards are NOT added to reward here to avoid double counting.
        # They are tracked separately in RewardManager and added only once in final calculation.
        deferred_reward_for_info = 0.0
        if self.active_player_id == 0:
            # Learning agent's turn - store deferred reward in info for RewardManager to accumulate
            # RewardManager will accumulate it separately from step rewards via accumulate_deferred_reward()
            if self.agent_deferred_reward != 0.0:
                deferred_reward_for_info = self.agent_deferred_reward
                self.agent_deferred_reward = 0.0  # Reset after storing
        elif done and self.agent_deferred_reward != 0.0:
            # Episode ended on opponent's turn, but agent had deferred reward
            # Store it in info dict for RewardManager to accumulate (will be added to final reward calculation)
            deferred_reward_for_info = self.agent_deferred_reward
            self.agent_deferred_reward = 0.0  # Reset after storing
        
        info = {
            "player_id": self.active_player_id,
            "game_over": terminated,
            "winner": self.game_state.winner,
            "action_info": self.last_action_info,
            "game_state": self.game_state.get_public_info(),
            "retry": retry_needed,  # Flag indicating if action needs to be retried
            "invalid_action_attempts": self.invalid_action_attempts,  # Number of invalid attempts
            "deferred_reward": deferred_reward_for_info,  # Deferred reward for RewardManager
        }

        # If episode ended, save statistics for monitoring
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

    def _get_observation(self, player_id: int) -> Dict[str, np.ndarray]:
        """
        Get observation for specific player.

        Args:
            player_id: Player ID

        Returns:
            Observation dictionary with 'bid_history' and 'static_info' keys
        """
        player_dice = self.game_state.get_player_dice(player_id)

        observation = create_observation_dict(
            current_bid=self.game_state.current_bid,
            bid_history=self.game_state.bid_history,
            player_dice_count=self.game_state.player_dice_count,
            current_player=self.game_state.current_player,
            palifico_active=self.game_state.palifico_active,
            believe_called=self.game_state.believe_called,
            player_dice=player_dice,
            max_history_length=self.max_history_length,
            max_players=self.max_num_players,  # Use max for observation space
            agent_id=player_id,
            num_agents=self.max_num_players,  # Use max for observation space
            special_round_active=self.game_state.special_round_active,
            round_number=self.game_state.round_number,
        )

        available_actions = PerudoRules.get_available_actions(self.game_state, player_id)
        action_mask = create_action_mask(
            available_actions, self.action_space.n, self.max_quantity
        )
        observation["action_mask"] = action_mask

        return observation

    def render(self):
        """Render current game state."""
        if self.render_mode == "human":
            print(f"\n=== Perudo Game State ===")
            print(f"Players: {self.num_players}")
            print(f"Current player: {self.game_state.current_player}")
            print(f"Active player (for observation): {self.active_player_id}")
            if self.game_state.current_bid:
                q, v = self.game_state.current_bid
                print(f"Current bid: {q}x{v}")
            else:
                print("Current bid: none")
            print(f"Player dice counts: {self.game_state.player_dice_count}")
            print(f"Palifico active: {self.game_state.palifico_active}")
            print(f"Believe called: {self.game_state.believe_called}")
            print(f"Game over: {self.game_state.game_over}")
            if self.game_state.winner is not None:
                print(f"Winner: Player {self.game_state.winner}")
            print("=" * 30)

    def set_active_player(self, player_id: int) -> None:
        """
        Set active player (for whom observation is returned).

        Args:
            player_id: Player ID
        """
        # Reset invalid action attempts when switching to a different player
        if self.active_player_id != player_id:
            self.invalid_action_attempts = 0
            self.invalid_action_penalty_accumulated = 0.0
        self.active_player_id = player_id

    def get_observation_for_player(self, player_id: int) -> Dict[str, np.ndarray]:
        """
        Get observation for specific player.

        Args:
            player_id: Player ID

        Returns:
            Observation dictionary with 'bid_history' and 'static_info' keys
        """
        return self._get_observation(player_id)

    def _check_round_end_reward(self, loser_id: int, dice_lost: int, action_type: Optional[str] = None) -> float:
        """
        Check and return reward for round end.

        Args:
            loser_id: ID of player who lost dice
            dice_lost: Number of dice lost
            action_type: Type of action that ended the round ('challenge', 'believe', or None)
                        Used to avoid double penalty when agent's challenge/believe fails

        Returns:
            Round reward (positive if agent didn't lose dice, negative if unsuccessful bid)
        """
        LEARNING_AGENT_ID = 0
        
        # Check if agent (player 0) lost dice in this round
        agent_lost_dice_this_round = (loser_id == LEARNING_AGENT_ID and dice_lost > 0)
        
        # Early return if agent didn't lose dice and didn't make a bid
        if not agent_lost_dice_this_round and not self.agent_bid_this_round:
            # Agent didn't lose dice and didn't make a bid - standard round reward
            return self.reward_config.round_no_dice_lost_reward
        
        reward = 0.0
        
        # Penalty for unsuccessful bid that led to dice loss
        # (agent made a bid that was successfully challenged by opponent)
        # Only apply if the current action is NOT agent's own challenge/believe
        # (because those are already penalized in calculate_reward)
        if agent_lost_dice_this_round and self.agent_bid_this_round:
            # Check if this is agent's own challenge/believe (already penalized in calculate_reward)
            is_agent_own_action = (
                action_type in ["challenge", "believe"] and 
                self.active_player_id == LEARNING_AGENT_ID
            )
            if not is_agent_own_action:
                # Agent's bid was successfully challenged by opponent
                reward += self.reward_config.unsuccessful_bid_penalty
        
        # Reward for each round in which the agent did not lose a die
        if not agent_lost_dice_this_round:
            reward += self.reward_config.round_no_dice_lost_reward
            
            if self.agent_bid_this_round:
                # Agent successfully bluffed (bid was correct or never challenged)
                reward += self.reward_config.successful_bluff_reward

        return reward
