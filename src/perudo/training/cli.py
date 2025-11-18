"""
Command-line interface for training Perudo agents.

Provides CLI for configuring and starting training.
"""

import argparse
from .config import DEFAULT_CONFIG
from .train import SelfPlayTraining


def main():
    """
    Main function to run training from command line.
    
    Parses command-line arguments and starts training with the specified configuration.
    """
    parser = argparse.ArgumentParser(description="Train Perudo agents with parameter sharing")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (tables). If not specified, uses value from config.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=8192,
        help="Number of steps to collect before update (recommended: 8192 for 16 envs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (recommended: 256 for n_steps=8192)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cpu, cuda, cuda:0, etc.). If not specified, auto-detects GPU with CPU fallback.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0 = no output, 1 = progress bars and logs, 2 = debug). Default: 1",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DEFAULT_CONFIG
    # total_timesteps is set only in config.py, not overridden here
    config.training.n_steps = args.n_steps
    config.training.batch_size = args.batch_size
    config.training.device = args.device
    config.training.verbose = args.verbose  # Set verbose from command line
    
    # Override num_envs from command line if provided
    if args.num_envs is not None:
        config.training.num_envs = args.num_envs
    
    # Start training
    trainer = SelfPlayTraining(config)
    trainer.train()


if __name__ == "__main__":
    main()

