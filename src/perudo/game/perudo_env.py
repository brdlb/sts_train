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
        """
        super().__init__()

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
        
        # Track invalid action attempts for retry mechanism
        self.invalid_action_attempts = 0  # Count of consecutive invalid actions for current player
        self.max_invalid_attempts = 1000  # Safety limit to prevent infinite loops
        self.invalid_action_penalty_accumulated = 0.0  # Accumulated penalty from invalid actions

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
            info = {
                "game_over": True,
                "winner": self.game_state.winner,
                "episode": {
                    "r": self.episode_reward,
                    "l": self.episode_length,
                    "bid_count": self.episode_bid_count,
                    "challenge_count": self.episode_challenge_count,
                    "believe_count": self.episode_believe_count,
                    "winner": winner,
                },
                "episode_reward": self.episode_reward,
                "episode_length": self.episode_length,
            }
            return observation, 0.0, True, False, info

        # Convert action to game format
        action_type, param1, param2 = action_to_bid(action, self.max_quantity)

        # Check that it's active player's turn
        if self.game_state.current_player != self.active_player_id:
            # If it's not agent's turn, move to next player
            observation = self._get_observation(self.active_player_id)
            return observation, 0.0, False, False, {"error": "Not player's turn"}

        # Execute action
        reward = 0.0
        dice_lost = 0
        challenge_success = None
        believe_success = None
        action_valid = False
        retry_needed = False  # Flag to indicate if action needs to be retried
        error_msg = None  # Reason for rejection if action is invalid

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
                    # Valid action - compensate for previous invalid attempts
                    if self.invalid_action_attempts > 0:
                        # Calculate compensation: sum of all penalties, slightly less than the total penalty
                        compensation = self.invalid_action_penalty_accumulated - 0.1
                        reward += compensation
                        self.invalid_action_attempts = 0  # Reset counter
                        self.invalid_action_penalty_accumulated = 0.0
                    
                    # Track that agent made a bid this round
                    if self.active_player_id == 0:
                        self.agent_bid_this_round = True
                    # Count bid action (for all players, not just learning agent)
                    self.episode_bid_count += 1
                    self.game_state.next_player()
                    # Small negative reward for bidding to encourage finishing the round
                    reward += calculate_reward(
                        "bid", False, -1, self.active_player_id, dice_lost=0
                    ) - 0.01
            else:
                # Invalid action - accumulate penalty and retry
                self.invalid_action_attempts += 1
                penalty = -self.invalid_action_attempts  # Progressive penalty: -1, -2, -3, ...
                self.invalid_action_penalty_accumulated += abs(penalty)
                reward = penalty
                
                # Safety check: prevent infinite loops
                if self.invalid_action_attempts >= self.max_invalid_attempts:
                    # Force move to next player after max attempts
                    self.game_state.next_player()
                    self.invalid_action_attempts = 0
                    self.invalid_action_penalty_accumulated = 0.0
                    if self.game_state.game_over:
                        reward = calculate_reward(
                            "bid",
                            self.game_state.game_over,
                            self.game_state.winner or -1,
                            self.active_player_id,
                            dice_lost=0,
                        )
                    action_valid = False
                else:
                    # Don't advance game state - will retry same player
                    action_valid = False
                    retry_needed = True

        elif action_type == "challenge":
            # Challenge previous player
            can_challenge, error_msg = PerudoRules.can_challenge(
                self.game_state, self.active_player_id
            )
            if can_challenge:
                # Valid action - compensate for previous invalid attempts
                if self.invalid_action_attempts > 0:
                    # Calculate compensation: sum of all penalties, slightly less than the total penalty
                    compensation = self.invalid_action_penalty_accumulated - 0.1
                    reward += compensation
                    self.invalid_action_attempts = 0  # Reset counter
                    self.invalid_action_penalty_accumulated = 0.0
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
                self.game_state.current_bid = None
                self.game_state.believe_called = False

                # Restart round: next round starts with the player who lost the die
                self.game_state.current_player = loser_id

                # Roll dice again - round ends
                # Reset special round at end of round
                self.game_state.special_round_active = False
                self.game_state.special_round_declared_by = None
                self.game_state.roll_dice()
                # Reset round tracking after dice roll
                self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
                self.agent_bid_this_round = False

                reward = calculate_reward(
                    "challenge",
                    self.game_state.game_over,
                    self.game_state.winner or -1,
                    self.active_player_id,
                    challenge_success=challenge_success,
                    dice_lost=dice_lost if loser_id == self.active_player_id else 0,
                )
                # Count challenge action (for all players, not just learning agent)
                self.episode_challenge_count += 1
                # Add round reward
                if self.active_player_id == 0:
                    reward += round_reward
                # If agent successfully defended their bid (opponent challenged and failed),
                # store deferred reward for agent (will be given on agent's next turn)
                elif agent_defended_bid:
                    # Agent gets +2.5 reward for successfully defending their risky bid (reduced from +5.0)
                    # This reward will be added when agent makes their next action
                    self.agent_deferred_reward += 2.5
                action_valid = True
            else:
                # Invalid challenge - accumulate penalty and retry
                self.invalid_action_attempts += 1
                penalty = -self.invalid_action_attempts  # Progressive penalty: -1, -2, -3, ...
                self.invalid_action_penalty_accumulated += abs(penalty)
                reward = penalty
                
                # Safety check: prevent infinite loops
                if self.invalid_action_attempts >= self.max_invalid_attempts:
                    # Force move to next player after max attempts
                    self.game_state.next_player()
                    self.invalid_action_attempts = 0
                    self.invalid_action_penalty_accumulated = 0.0
                    if self.game_state.game_over:
                        reward = calculate_reward(
                            "challenge",
                            self.game_state.game_over,
                            self.game_state.winner or -1,
                            self.active_player_id,
                            challenge_success=False,
                            dice_lost=0,
                        )
                    action_valid = False
                else:
                    # Don't advance game state - will retry same player
                    action_valid = False
                    retry_needed = True

        elif action_type == "believe":
            # Call believe
            can_believe, error_msg = PerudoRules.can_call_believe(
                self.game_state, self.active_player_id
            )
            if can_believe:
                # Valid action - compensate for previous invalid attempts
                if self.invalid_action_attempts > 0:
                    # Calculate compensation: sum of all penalties, slightly less than the total penalty
                    compensation = self.invalid_action_penalty_accumulated - 0.1
                    reward += compensation
                    self.invalid_action_attempts = 0  # Reset counter
                    self.invalid_action_penalty_accumulated = 0.0
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
                self.game_state.current_bid = None
                self.game_state.believe_called = False

                # Restart round: next round starts with the player who gained or lost a die
                if player_who_changed_dice is not None:
                    self.game_state.current_player = player_who_changed_dice
                elif next_round_starter is not None:
                    # Believer with 5 dice starts next round
                    self.game_state.current_player = next_round_starter
                elif loser_id is not None:
                    # Fallback: if no one gained dice but someone lost, that player starts
                    self.game_state.current_player = loser_id
                else:
                    # Fallback: continue normally
                    self.game_state.next_player()

                # Roll dice again - round ends
                # Reset special round at end of round
                self.game_state.special_round_active = False
                self.game_state.special_round_declared_by = None
                self.game_state.roll_dice()
                # Reset round tracking after dice roll
                self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
                self.agent_bid_this_round = False

                reward = calculate_reward(
                    "believe",
                    self.game_state.game_over,
                    self.game_state.winner or -1,
                    self.active_player_id,
                    believe_success=believe_success,
                    dice_lost=dice_lost if (loser_id is not None and loser_id == self.active_player_id) else 0,
                )
                # Count believe action (for all players, not just learning agent)
                self.episode_believe_count += 1
                # Add round reward
                if self.active_player_id == 0:
                    reward += round_reward
                # If agent successfully defended their bid (opponent called believe and failed),
                # store deferred reward for agent (will be given on agent's next turn)
                elif agent_defended_bid:
                    # Agent gets +5.0 reward for successfully defending their risky bid
                    # This reward will be added when agent makes their next action
                    self.agent_deferred_reward += 5.0
                action_valid = True
            else:
                # Invalid believe - accumulate penalty and retry
                self.invalid_action_attempts += 1
                penalty = -self.invalid_action_attempts  # Progressive penalty: -1, -2, -3, ...
                self.invalid_action_penalty_accumulated += abs(penalty)
                reward = penalty
                
                # Safety check: prevent infinite loops
                if self.invalid_action_attempts >= self.max_invalid_attempts:
                    # Force move to next player after max attempts
                    self.game_state.next_player()
                    self.invalid_action_attempts = 0
                    self.invalid_action_penalty_accumulated = 0.0
                    if self.game_state.game_over:
                        reward = calculate_reward(
                            "believe",
                            self.game_state.game_over,
                            self.game_state.winner or -1,
                            self.active_player_id,
                            believe_success=False,
                            dice_lost=0,
                        )
                    action_valid = False
                else:
                    # Don't advance game state - will retry same player
                    action_valid = False
                    retry_needed = True

        # Save action information
        self.last_action_info = {
            "action_type": action_type,
            "action_valid": action_valid,
            "reward": reward,
            "dice_lost": dice_lost,
            "challenge_success": challenge_success,
            "believe_success": believe_success,
            "error_msg": error_msg,  # Reason for rejection if action is invalid
        }
        
        # Print step information for training room
        # Format action description
        # action_desc = ""
        # if action_type == "bid":
        #     action_desc = f"ставка {param1}x{param2}"
        # elif action_type == "challenge":
        #     action_desc = "вызов (challenge)"
        # elif action_type == "believe":
        #     action_desc = "верить (believe)"
        
        # Print step information
        # status = "ПРИНЯТ" if action_valid else "ОТКЛОНЕН"
        # print(f"[Шаг] Игрок {self.active_player_id}: {action_desc} - {status}", end="")
        
        # if not action_valid:
        #     attempt_info = f", попытка #{self.invalid_action_attempts}" if self.invalid_action_attempts > 0 else ""
        #     if error_msg:
        #         print(f" (причина: {error_msg}{attempt_info})")
        #     else:
        #         print(f" (попытка #{self.invalid_action_attempts})" if self.invalid_action_attempts > 0 else "")
        # else:
        #     print()
        
        # Additional info for valid actions
        #if action_valid:
        #    if action_type == "challenge" and challenge_success is not None:
        #        challenge_result = "успешен" if challenge_success else "неудачен"
        #        print(f"  -> Вызов {challenge_result}, потеряно костей: {dice_lost}")
        #    elif action_type == "believe" and believe_success is not None:
        #        believe_result = "успешен" if believe_success else "неудачен"
        #        print(f"  -> Believe {believe_result}, потеряно костей: {dice_lost}")

        # Get new observation
        observation = self._get_observation(self.active_player_id)

        # накопление суммарной награды и длины эпизода
        # ВАЖНО: накапливаем статистику только для learning agent (player_id=0)
        # В vec_env только learning agent делает ходы через этот метод для сбора статистики
        # Оппоненты играют отдельно и их награды не должны учитываться в статистике learning agent
        # НЕ накапливаем статистику при retry (retry - это не настоящий шаг, это повторная попытка)
        if self.active_player_id == 0 and not retry_needed:
            # Add any deferred reward for agent (e.g., from successfully defending bid)
            if self.agent_deferred_reward != 0.0:
                reward += self.agent_deferred_reward
                self.agent_deferred_reward = 0.0
            
            # Add dice advantage reward (includes reward for having more dice than average and being leader)
            dice_advantage_reward = self._calculate_dice_advantage_reward()
            reward += dice_advantage_reward
            
            self.episode_reward += reward
            self.episode_length += 1

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
        if done:
            # VecMonitor expects "episode" key with "r" (reward) and "l" (length) subkeys
            # Additional statistics: bid_count, challenge_count, believe_count, winner
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
                "bid_count": self.episode_bid_count,
                "challenge_count": self.episode_challenge_count,
                "believe_count": self.episode_believe_count,
                "winner": self.game_state.winner if hasattr(self.game_state, "winner") and self.game_state.winner is not None else -1,
            }
            # Also keep old format for backward compatibility with custom callbacks
            info["episode_reward"] = self.episode_reward
            info["episode_length"] = self.episode_length
            if hasattr(self.game_state, "winner"):
                info["winner"] = self.game_state.winner

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
            Round reward (+2.0 if agent didn't lose dice, 0 otherwise)
            Also handles -1.5 for unsuccessful bid that led to dice loss
        """
        reward = 0.0
        agent_id = 0

        # Check if agent (player 0) lost dice in this round
        agent_lost_dice = (loser_id == agent_id and dice_lost > 0)

        # -1.5 for unsuccessful bid that led to dice loss
        # (agent made a bid that was successfully challenged by opponent)
        # Only apply if the current action is NOT agent's challenge/believe
        # (because those are already penalized in calculate_reward)
        if agent_lost_dice and self.agent_bid_this_round:
            # Check if this is agent's own challenge/believe (already penalized in calculate_reward)
            is_agent_action = (action_type in ["challenge", "believe"] and self.active_player_id == agent_id)
            if not is_agent_action:
                # Agent's bid was successfully challenged by opponent
                reward -= 1.5

        # +2.0 reward for each round in which the agent did not lose a die (increased from +0.5)
        if not agent_lost_dice:
            reward += 2.0
            
            if self.agent_bid_this_round:
                # Agent successfully bluffed (bid was correct or never challenged)
                # Increased from +1.5 to +3.0
                reward += 3.0

        return reward

    def _calculate_dice_advantage_reward(self) -> float:
        """
        Calculate reward for having more dice than average opponents and being leader.
        
        Returns:
            Reward based on dice advantage (+0.5 per die advantage) plus bonus for being leader
        """
        agent_id = 0
        if agent_id >= len(self.game_state.player_dice_count):
            return 0.0
        
        agent_dice = self.game_state.player_dice_count[agent_id]
        
        # Calculate average dice count for opponents (excluding agent)
        opponent_dice_counts = [
            count for i, count in enumerate(self.game_state.player_dice_count)
            if i != agent_id and i < self.num_players
        ]
        
        if not opponent_dice_counts:
            return 0.0
        
        avg_opponent_dice = sum(opponent_dice_counts) / len(opponent_dice_counts)
        dice_advantage = agent_dice - avg_opponent_dice
        
        reward = 0.0
        
        # Reward: +0.5 per die advantage (capped at reasonable maximum)
        if dice_advantage > 0:
            reward += 0.5 * min(dice_advantage, 5.0)  # Cap at 5 dice advantage
        
        # Bonus reward for being leader (having more dice than all opponents)
        if opponent_dice_counts:
            max_opponent_dice = max(opponent_dice_counts)
            if agent_dice > max_opponent_dice:
                # Additional +2.0 reward for being the leader
                reward += 2.0
        
        return reward
