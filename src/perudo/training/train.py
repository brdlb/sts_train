"""
Main script for training Perudo agents.
"""

import os
import numpy as np
from typing import List, Optional
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from ..game.perudo_env import PerudoEnv
from ..agents.rl_agent import RLAgent
from .config import Config, DEFAULT_CONFIG


class SelfPlayTraining:
    """Class for self-play training of agents."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        """
        Initialize training.

        Args:
            config: Training configuration
        """
        self.config = config
        self.num_players = config.game.num_players

        # Create directories
        os.makedirs(config.training.log_dir, exist_ok=True)
        os.makedirs(config.training.model_dir, exist_ok=True)

        # Create environment
        self.env = PerudoEnv(
            num_players=config.game.num_players,
            dice_per_player=config.game.dice_per_player,
            total_dice_values=config.game.total_dice_values,
            max_quantity=config.game.max_quantity,
            history_length=config.game.history_length,
        )

        # Create agents
        self.agents: List[RLAgent] = []
        self._create_agents()

        # Statistics
        self.episode_rewards = [[] for _ in range(self.num_players)]
        self.episode_lengths = []
        self.wins = [0] * self.num_players

    def _create_agents(self):
        """Create agents for all players."""
        for agent_id in range(self.num_players):
            agent = RLAgent(
                agent_id=agent_id,
                env=self.env,
                policy=self.config.training.policy,
                learning_rate=self.config.training.learning_rate,
                n_steps=self.config.training.n_steps,
                batch_size=self.config.training.batch_size,
                n_epochs=self.config.training.n_epochs,
                gamma=self.config.training.gamma,
                gae_lambda=self.config.training.gae_lambda,
                clip_range=self.config.training.clip_range,
                ent_coef=self.config.training.ent_coef,
                vf_coef=self.config.training.vf_coef,
                max_grad_norm=self.config.training.max_grad_norm,
                verbose=self.config.training.verbose,
            )
            self.agents.append(agent)

    def train(self):
        """Main training loop."""
        print(f"Starting training with {self.num_players} agents")
        print(f"Total timesteps: {self.config.training.total_timesteps}")

        total_steps = 0
        episode = 0

        # Create callback for saving models
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.training.save_freq,
            save_path=self.config.training.model_dir,
            name_prefix="perudo_model",
        )

        while total_steps < self.config.training.total_timesteps:
            episode += 1
            episode_reward = [0.0] * self.num_players
            episode_length = 0

            # Reset environment
            obs, info = self.env.reset(seed=self.config.training.seed)
            self.env.set_active_player(0)

            done = False
            while not done:
                # Get current player
                current_player = self.env.game_state.current_player
                active_player = self.env.active_player_id

                # If it's current player's turn, get observation for them
                if current_player == active_player:
                    obs = self.env.get_observation_for_player(current_player)
                    self.env.set_active_player(current_player)

                    # Agent chooses action
                    agent = self.agents[current_player]
                    action = agent.act(obs, deterministic=False)

                    # Execute action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)

                    # Update statistics
                    episode_reward[current_player] += reward
                    episode_length += 1

                    # Check episode end
                    done = terminated or truncated

                    # If game is over, update win statistics
                    if terminated and info.get("winner") is not None:
                        winner = info["winner"]
                        if 0 <= winner < self.num_players:
                            self.wins[winner] += 1

                    # Update observation for next step
                    obs = next_obs

                    # Update active player
                    if not done:
                        next_player = self.env.game_state.current_player
                        self.env.set_active_player(next_player)
                else:
                    # If it's not active player's turn, move to next
                    self.env.set_active_player(self.env.game_state.current_player)

                total_steps += 1

                # Periodic training (simplified version)
                # In reality, need to collect experience and train in batches
                if total_steps % self.config.training.n_steps == 0:
                    # Train all agents on collected experience
                    # Note: this is a simplified version, in reality need to
                    # collect experience and train through vectorized environment
                    pass

            # Save episode statistics
            for i in range(self.num_players):
                self.episode_rewards[i].append(episode_reward[i])
            self.episode_lengths.append(episode_length)

            # Print statistics periodically
            if episode % 100 == 0:
                avg_rewards = [
                    np.mean(self.episode_rewards[i][-100:]) for i in range(self.num_players)
                ]
                avg_length = np.mean(self.episode_lengths[-100:])
                print(
                    f"Episode {episode} | Steps: {total_steps} | "
                    f"Avg length: {avg_length:.1f} | "
                    f"Avg rewards: {[f'{r:.2f}' for r in avg_rewards]} | "
                    f"Wins: {self.wins}"
                )

            # Save models periodically
            if episode % (self.config.training.save_freq // 100) == 0:
                for agent in self.agents:
                    model_path = os.path.join(
                        self.config.training.model_dir,
                        f"agent_{agent.agent_id}_episode_{episode}.zip",
                    )
                    agent.save(model_path)

        print("Training completed!")

        # Save final models
        for agent in self.agents:
            model_path = os.path.join(
                self.config.training.model_dir, f"agent_{agent.agent_id}_final.zip"
            )
            agent.save(model_path)

        print(f"Models saved to {self.config.training.model_dir}")


def train_single_agent_loop(config: Config = DEFAULT_CONFIG):
    """
    Training through separate loops for each agent.

    This is an alternative approach where each agent trains separately,
    using vectorized environment.
    """
    print(f"Training through separate loops for {config.game.num_players} agents")

    # Create directories
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.model_dir, exist_ok=True)

    # Train each agent
    for agent_id in range(config.game.num_players):
        print(f"\nTraining agent {agent_id}...")

        # Create environment for agent
        env = PerudoEnv(
            num_players=config.game.num_players,
            dice_per_player=config.game.dice_per_player,
            total_dice_values=config.game.total_dice_values,
            max_quantity=config.game.max_quantity,
            history_length=config.game.history_length,
        )

        # Create agent
        agent = RLAgent(
            agent_id=agent_id,
            env=env,
            policy=config.training.policy,
            learning_rate=config.training.learning_rate,
            n_steps=config.training.n_steps,
            batch_size=config.training.batch_size,
            n_epochs=config.training.n_epochs,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_range=config.training.clip_range,
            ent_coef=config.training.ent_coef,
            vf_coef=config.training.vf_coef,
            max_grad_norm=config.training.max_grad_norm,
            verbose=config.training.verbose,
        )

        # Train agent
        tb_log_name = config.training.tb_log_name or "perudo_training"
        agent.learn(
            total_timesteps=config.training.total_timesteps,
            tb_log_name=f"{tb_log_name}_agent_{agent_id}",
        )

        # Save model
        model_path = os.path.join(
            config.training.model_dir, f"agent_{agent_id}_final.zip"
        )
        agent.save(model_path)
        print(f"Agent {agent_id} model saved: {model_path}")

    print("\nTraining of all agents completed!")


def main():
    """Main function to run training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Perudo agents")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        help="Number of players",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "selfplay"],
        help="Training mode: single (separate) or selfplay (self-play)",
    )

    args = parser.parse_args()

    # Create configuration
    config = DEFAULT_CONFIG
    config.game.num_players = args.num_players
    config.training.total_timesteps = args.total_timesteps

    if args.mode == "selfplay":
        # Self-play (all agents play against each other)
        trainer = SelfPlayTraining(config)
        trainer.train()
    else:
        # Separate training for each agent
        train_single_agent_loop(config)


if __name__ == "__main__":
    main()
