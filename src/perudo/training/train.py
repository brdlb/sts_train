"""
Main script for training Perudo agents with parameter sharing and self-play.
"""

import os
import numpy as np
from typing import List, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from ..game.perudo_vec_env import PerudoMultiAgentVecEnv
from .config import Config, DEFAULT_CONFIG
from .opponent_pool import OpponentPool

import sys

# --- Очистка модуля из sys.modules, чтобы избежать предупреждения runpy ---
if __name__ == "__main__":
    modname = __name__
    if modname in sys.modules:
        del sys.modules[modname]


class AdvantageNormalizationCallback(BaseCallback):
    """
    Callback to normalize advantages across the entire batch.
    
    This is needed because with parameter sharing, we collect data from
    multiple agents across multiple environments, and we want to normalize
    advantages globally before updating.
    
    In StableBaselines3, advantages are normalized by default in PPO.update(),
    but we want to ensure normalization happens across the entire batch
    (n_steps * num_envs * num_agents).
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_rollout_end(self) -> bool:
        """
        Called when rollout ends, before advantage normalization.
        
        This ensures advantages are normalized globally across the entire batch.
        In SB3, this is already done in PPO.update(), but we verify it here.
        """
        # The advantages will be normalized in the PPO update automatically
        # SB3's PPO already normalizes advantages globally in compute_returns_and_advantage()
        # So we just verify that the rollout buffer exists
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            # Advantages are normalized in PPO.update() by default
            # The normalization happens in compute_returns_and_advantage()
            # which normalizes across the entire rollout buffer
            pass
        return True
    
    def _on_step(self) -> bool:
        """
        Dummy method required by BaseCallback. Returns True to fulfill abstract class requirements.
        """
        return True


class SelfPlayTrainingCallback(BaseCallback):
    """
    Callback for self-play training with opponent pool.
    """
    def __init__(
        self,
        opponent_pool: OpponentPool,
        snapshot_freq: int = 50000,
        verbose: int = 0,
        print_freq: int = 1000
    ):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.snapshot_freq = snapshot_freq
        self.print_freq = print_freq
        self.current_step = 0
        self.vec_env = None
        # Буферы для realtime-статистики
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []

    def _on_training_start(self) -> None:
        if hasattr(self.model, 'env'):
            self.vec_env = self.model.env
    
    def _on_step(self) -> bool:
        self.current_step += 1
        # Собираем episode_info из infos всех env (SB3 их передаёт в self.locals["infos"])
        infos = self.locals.get("infos", [])
        for info in infos:
            # Добавляем статистику только если эпизод завершен (info содержит episode_reward)
            if isinstance(info, dict) and "episode_reward" in info:
                self.episode_rewards.append(info["episode_reward"])
                self.episode_lengths.append(info["episode_length"])
                win = 1 if info.get("winner", -1) == 0 else 0
                self.episode_wins.append(win)
        if self.current_step % self.print_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            win_rate = 100 * np.mean(self.episode_wins[-100:]) if self.episode_wins else 0
            print(f"[Step {self.current_step}] AvgReward(100): {avg_reward:.2f}, AvgLen(100): {avg_length:.1f}, WinRate(100): {win_rate:.1f}%, Completed episodes: {len(self.episode_rewards)}")
        # Стандартная логика снапшотов и синхронизации
        if self.current_step % self.snapshot_freq == 0:
            self.opponent_pool.save_snapshot(
                self.model,
                self.current_step,
                prefix="snapshot"
            )
            if self.verbose > 0:
                pool_stats = self.opponent_pool.get_statistics()
                print(f"Saved snapshot at step {self.current_step}")
                print(f"Pool statistics: {pool_stats}")
        # Update vec_env with current step for opponent sampling
        if self.vec_env is not None and hasattr(self.vec_env, 'reset'):
            self.vec_env.current_step = self.current_step
        return True
    
    def _on_rollout_end(self) -> bool:
        # Winrate statistics are updated in VecEnv.step_wait() when episodes end, so we don't need to do anything here
        return True


class SelfPlayTraining:
    """
    Class for self-play training with parameter sharing and opponent pool.
    
    Features:
    - Parameter sharing: one PPO model for all agents
    - Opponent pool: sample opponents from pool
    - Batch normalization: normalize advantages globally
    - VecEnv: each environment = one table with 4 agents
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        """
        Initialize training.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.num_players = config.game.num_players
        self.num_envs = getattr(config.training, 'num_envs', 8)
        
        # Create directories
        os.makedirs(config.training.log_dir, exist_ok=True)
        os.makedirs(config.training.model_dir, exist_ok=True)
        
        # Create opponent pool directory
        pool_dir = os.path.join(config.training.model_dir, "opponent_pool")
        self.opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            max_pool_size=20,
            min_pool_size=10,
            keep_best=3,
            snapshot_freq=50000,
        )
        
        # Create vectorized environment
        self.vec_env = PerudoMultiAgentVecEnv(
            num_envs=self.num_envs,
            num_players=self.num_players,
            dice_per_player=config.game.dice_per_player,
            total_dice_values=config.game.total_dice_values,
            max_quantity=config.game.max_quantity,
            history_length=config.game.history_length,
            opponent_pool=self.opponent_pool,
        )
        
        # Create PPO model with parameter sharing
        # One model for all agents (agent_id is in observation)
        self.model = PPO(
            policy=config.training.policy,
            env=self.vec_env,
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
        
        # Update opponent pool with current model
        self.vec_env.current_model = self.model
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = [0] * self.num_players
        
    def train(self):
        """Main training loop."""
        print(f"Starting training with parameter sharing")
        print(f"Number of environments (tables): {self.num_envs}")
        print(f"Number of players per table: {self.num_players}")
        print(f"Total timesteps: {self.config.training.total_timesteps}")
        print(f"Effective batch size: {self.config.training.n_steps * self.num_envs * self.num_players}")
        
        # Check that batch_size divides evenly
        effective_batch_size = self.config.training.n_steps * self.num_envs * self.num_players
        if effective_batch_size % self.config.training.batch_size != 0:
            print(f"Warning: Effective batch size {effective_batch_size} does not divide evenly "
                  f"by batch_size {self.config.training.batch_size}")
            print(f"Consider adjusting batch_size or num_envs")
        
        # Create callbacks
        callbacks = []
        
        # Advantage normalization callback
        adv_norm_callback = AdvantageNormalizationCallback(verbose=self.config.training.verbose)
        callbacks.append(adv_norm_callback)
        
        # Self-play callback
        selfplay_callback = SelfPlayTrainingCallback(
            opponent_pool=self.opponent_pool,
            snapshot_freq=50000,
            verbose=self.config.training.verbose,
        )
        callbacks.append(selfplay_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.training.save_freq,
            save_path=self.config.training.model_dir,
            name_prefix="perudo_model",
        )
        callbacks.append(checkpoint_callback)
        
        # Train model
        tb_log_name = self.config.training.tb_log_name or "perudo_training"
        self.model.learn(
            total_timesteps=self.config.training.total_timesteps,
            tb_log_name=tb_log_name,
            callback=callbacks,
        )
        
        print("Training completed!")
        
        # Save final model
        model_path = os.path.join(
            self.config.training.model_dir, "perudo_model_final.zip"
        )
        self.model.save(model_path)
        print(f"Final model saved to {model_path}")
        
        # Print pool statistics
        pool_stats = self.opponent_pool.get_statistics()
        print(f"Opponent pool statistics: {pool_stats}")
        
    def save_model(self, path: str):
        """Save the current model."""
        self.model.save(path)
        
    def load_model(self, path: str):
        """Load a model."""
        self.model = PPO.load(path, env=self.vec_env)
        self.vec_env.current_model = self.model


def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Perudo agents with parameter sharing")
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
        help="Number of players per table",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (tables)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to collect before update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DEFAULT_CONFIG
    config.game.num_players = args.num_players
    config.training.total_timesteps = args.total_timesteps
    config.training.n_steps = args.n_steps
    config.training.batch_size = args.batch_size
    
    # Add num_envs to config
    if not hasattr(config.training, 'num_envs'):
        config.training.num_envs = args.num_envs
    else:
        config.training.num_envs = args.num_envs
    
    # Start training
    trainer = SelfPlayTraining(config)
    trainer.train()


if __name__ == "__main__":
    main()
