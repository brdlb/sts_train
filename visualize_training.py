"""
Script to visualize training data from VecMonitor CSV files.

Usage:
    python visualize_training.py [--csv-path logs/monitor/monitor.csv] [--window 100] [--save-dir plots/]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Try to import seaborn for better styling (optional)
try:
    import seaborn as sns

    sns.set_style("darkgrid")
except ImportError:
    pass

# Set style for better-looking plots
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


def read_monitor_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Read VecMonitor CSV file and extract metadata.

    Args:
        csv_path: Path to the monitor CSV file

    Returns:
        Tuple of (dataframe, metadata_dict)
    """
    with open(csv_path, "r") as f:
        # Read first line for metadata
        first_line = f.readline().strip()
        if first_line.startswith("#"):
            metadata_str = first_line[1:]  # Remove '#' prefix
            metadata = json.loads(metadata_str)
        else:
            metadata = {}
            # If first line is not metadata, reset file pointer
            f.seek(0)

    # Read CSV data (skip first line if it was metadata)
    df = pd.read_csv(csv_path, skiprows=1 if metadata else 0)

    return df, metadata


def plot_rewards(
    df: pd.DataFrame,
    window: int = 100,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot reward progression over episodes.

    Args:
        df: DataFrame with training data
        window: Window size for rolling average
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Raw rewards
    axes[0].plot(df.index, df["r"], alpha=0.3, color="blue", label="Raw rewards")
    if len(df) > window:
        df["r_smooth"] = df["r"].rolling(window=window, center=True).mean()
        axes[0].plot(
            df.index,
            df["r_smooth"],
            color="red",
            linewidth=2,
            label=f"Rolling average (window={window})",
        )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Rewards Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Statistics subplot
    stats_text = f"""
    Total Episodes: {len(df)}
    Mean Reward: {df['r'].mean():.2f}
    Std Reward: {df['r'].std():.2f}
    Min Reward: {df['r'].min():.2f}
    Max Reward: {df['r'].max():.2f}
    """
    if len(df) > window and "r_smooth" in df.columns:
        stats_text += f"\n    Latest Smooth Avg: {df['r_smooth'].iloc[-window:].mean():.2f}"

    axes[1].axis("off")
    axes[1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_lengths(
    df: pd.DataFrame,
    window: int = 100,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot episode length progression over episodes.

    Args:
        df: DataFrame with training data
        window: Window size for rolling average
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Raw episode lengths
    axes[0].plot(df.index, df["l"], alpha=0.3, color="green", label="Raw lengths")
    if len(df) > window:
        df["l_smooth"] = df["l"].rolling(window=window, center=True).mean()
        axes[0].plot(
            df.index,
            df["l_smooth"],
            color="orange",
            linewidth=2,
            label=f"Rolling average (window={window})",
        )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Length (steps)")
    axes[0].set_title("Episode Lengths Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Statistics subplot
    stats_text = f"""
    Total Episodes: {len(df)}
    Mean Length: {df['l'].mean():.2f}
    Std Length: {df['l'].std():.2f}
    Min Length: {df['l'].min():.0f}
    Max Length: {df['l'].max():.0f}
    """
    if len(df) > window and "l_smooth" in df.columns:
        stats_text += f"\n    Latest Smooth Avg: {df['l_smooth'].iloc[-window:].mean():.2f}"

    axes[1].axis("off")
    axes[1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_combined(
    df: pd.DataFrame,
    window: int = 100,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot combined rewards and episode lengths in a single figure.

    Args:
        df: DataFrame with training data
        window: Window size for rolling average
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Rewards
    axes[0].plot(df.index, df["r"], alpha=0.2, color="blue", label="Raw rewards")
    if len(df) > window:
        r_smooth = df["r"].rolling(window=window, center=True).mean()
        axes[0].plot(
            df.index,
            r_smooth,
            color="red",
            linewidth=2,
            label=f"Rolling avg (w={window})",
        )
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Metrics")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Episode lengths
    axes[1].plot(df.index, df["l"], alpha=0.2, color="green", label="Raw lengths")
    if len(df) > window:
        l_smooth = df["l"].rolling(window=window, center=True).mean()
        axes[1].plot(
            df.index,
            l_smooth,
            color="orange",
            linewidth=2,
            label=f"Rolling avg (w={window})",
        )
    axes[1].set_ylabel("Episode Length")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Reward distribution histogram
    axes[2].hist(df["r"], bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[2].axvline(df["r"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['r'].mean():.2f}")
    axes[2].set_xlabel("Reward")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Reward Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_time_progression(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot time progression over episodes.

    Args:
        df: DataFrame with training data
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate time differences
    if "t" in df.columns:
        ax.plot(df.index, df["t"], color="purple", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Time (seconds)")
        ax.set_title("Training Time Progression")
        ax.grid(True, alpha=0.3)

        # Add total time annotation
        total_time = df["t"].iloc[-1]
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"Total: {hours}h {minutes}m {seconds}s"
        ax.text(
            0.02,
            0.98,
            time_str,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize training data from VecMonitor CSV files"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="logs/monitor/monitor.csv",
        help="Path to the monitor CSV file (default: logs/monitor/monitor.csv)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Window size for rolling average (default: 100)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: None, plots are only displayed)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful when saving to file)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["rewards", "lengths", "combined", "time", "all"],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    # Read data
    print(f"Reading data from {args.csv_path}...")
    df, metadata = read_monitor_csv(args.csv_path)
    print(f"Loaded {len(df)} episodes")
    if metadata:
        print(f"Metadata: {metadata}")

    # Create save directory if specified
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    show = not args.no_show

    # Generate plots
    if args.plot_type in ["rewards", "all"]:
        save_path = (
            str(save_dir / "rewards.png") if save_dir else None
        )
        plot_rewards(df, window=args.window, save_path=save_path, show=show)

    if args.plot_type in ["lengths", "all"]:
        save_path = (
            str(save_dir / "episode_lengths.png") if save_dir else None
        )
        plot_episode_lengths(
            df, window=args.window, save_path=save_path, show=show
        )

    if args.plot_type in ["combined", "all"]:
        save_path = (
            str(save_dir / "combined.png") if save_dir else None
        )
        plot_combined(df, window=args.window, save_path=save_path, show=show)

    if args.plot_type in ["time", "all"]:
        save_path = (
            str(save_dir / "time_progression.png") if save_dir else None
        )
        plot_time_progression(df, save_path=save_path, show=show)

    print("Visualization complete!")


if __name__ == "__main__":
    main()

