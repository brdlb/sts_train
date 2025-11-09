"""
Script to visualize training data from VecMonitor CSV files.

Usage:
    python visualize_training.py [--csv-path logs/monitor/monitor.csv] [--window 100] [--save-dir plots/]
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme style
plt.style.use('dark_background')

# Try to import seaborn for better styling (optional)
try:
    import seaborn as sns
    # Override seaborn style to work with dark background
    sns.set_style("darkgrid", {'axes.facecolor': '#1e1e1e', 'figure.facecolor': '#1e1e1e'})
except ImportError:
    pass

# Set style for better-looking plots with dark theme
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["figure.facecolor"] = '#1e1e1e'
plt.rcParams["axes.facecolor"] = '#1e1e1e'
plt.rcParams["savefig.facecolor"] = '#1e1e1e'
plt.rcParams["savefig.edgecolor"] = 'none'


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
    axes[0].plot(df.index, df["r"], alpha=0.3, color="#4A90E2", label="Raw rewards")
    if len(df) > window:
        df["r_smooth"] = df["r"].rolling(window=window, center=True).mean()
        axes[0].plot(
            df.index,
            df["r_smooth"],
            color="#E24A4A",
            linewidth=2,
            label=f"Rolling average (window={window})",
        )
    axes[0].set_xlabel("Episode", color="white")
    axes[0].set_ylabel("Reward", color="white")
    axes[0].set_title("Training Rewards Over Time", color="white")
    axes[0].legend(facecolor='#2d2d2d', edgecolor='#555555')
    axes[0].grid(True, alpha=0.2, color='gray')
    axes[0].tick_params(colors='white')

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
        color="white",
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
    axes[0].plot(df.index, df["l"], alpha=0.3, color="#4AE24A", label="Raw lengths")
    if len(df) > window:
        df["l_smooth"] = df["l"].rolling(window=window, center=True).mean()
        axes[0].plot(
            df.index,
            df["l_smooth"],
            color="#E2A84A",
            linewidth=2,
            label=f"Rolling average (window={window})",
        )
    axes[0].set_xlabel("Episode", color="white")
    axes[0].set_ylabel("Episode Length (steps)", color="white")
    axes[0].set_title("Episode Lengths Over Time", color="white")
    axes[0].legend(facecolor='#2d2d2d', edgecolor='#555555')
    axes[0].grid(True, alpha=0.2, color='gray')
    axes[0].tick_params(colors='white')

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
        color="white",
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
    fig: Optional[plt.Figure] = None,
    axes: Optional[list] = None,
):
    """
    Plot combined rewards and episode lengths in a single figure.

    Args:
        df: DataFrame with training data
        window: Window size for rolling average
        save_path: Optional path to save the plot
        show: Whether to display the plot
        fig: Optional existing figure to update
        axes: Optional existing axes to update
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    else:
        # Clear existing plots for update
        for ax in axes:
            ax.clear()

    # Rewards
    axes[0].plot(df.index, df["r"], alpha=0.3, color="#4A90E2", label="Raw rewards")
    if len(df) > window:
        r_smooth = df["r"].rolling(window=window, center=True).mean()
        axes[0].plot(
            df.index,
            r_smooth,
            color="#E24A4A",
            linewidth=2,
            label=f"Rolling avg (w={window})",
        )
    axes[0].set_ylabel("Reward", color="white")
    axes[0].set_title("Training Metrics", color="white")
    axes[0].legend(loc="upper left", facecolor='#2d2d2d', edgecolor='#555555')
    axes[0].grid(True, alpha=0.2, color='gray')
    axes[0].tick_params(colors='white')

    # Episode lengths
    axes[1].plot(df.index, df["l"], alpha=0.3, color="#4AE24A", label="Raw lengths")
    if len(df) > window:
        l_smooth = df["l"].rolling(window=window, center=True).mean()
        axes[1].plot(
            df.index,
            l_smooth,
            color="#E2A84A",
            linewidth=2,
            label=f"Rolling avg (w={window})",
        )
    axes[1].set_ylabel("Episode Length", color="white")
    axes[1].legend(loc="upper left", facecolor='#2d2d2d', edgecolor='#555555')
    axes[1].grid(True, alpha=0.2, color='gray')
    axes[1].tick_params(colors='white')

    # Time progression
    if "t" in df.columns:
        axes[2].plot(df.index, df["t"], color="#A84AE2", linewidth=2, label="Cumulative time")
        axes[2].set_ylabel("Time (seconds)", color="white")
        axes[2].set_title("Training Time Progression", color="white")
        axes[2].legend(loc="upper left", facecolor='#2d2d2d', edgecolor='#555555')
        axes[2].grid(True, alpha=0.2, color='gray')
        axes[2].tick_params(colors='white')
        
        # Add total time annotation
        total_time = df["t"].iloc[-1]
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"Total: {hours}h {minutes}m {seconds}s"
        axes[2].text(
            0.98,
            0.98,
            time_str,
            transform=axes[2].transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            color="white",
            bbox=dict(boxstyle="round", facecolor="#2d2d2d", edgecolor="#555555", alpha=0.8),
        )
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Time data not available", 
                    ha="center", va="center", color="white", fontsize=12)

    # Reward distribution histogram
    axes[3].hist(df["r"], bins=50, alpha=0.7, color="#4A90E2", edgecolor="#6BA3E6")
    axes[3].axvline(df["r"].mean(), color="#E24A4A", linestyle="--", linewidth=2, label=f"Mean: {df['r'].mean():.2f}")
    axes[3].set_xlabel("Reward", color="white")
    axes[3].set_ylabel("Frequency", color="white")
    axes[3].set_title("Reward Distribution", color="white")
    axes[3].legend(facecolor='#2d2d2d', edgecolor='#555555')
    axes[3].grid(True, alpha=0.2, color='gray', axis="y")
    axes[3].tick_params(colors='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.close(fig)

    return fig, axes


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
        ax.plot(df.index, df["t"], color="#A84AE2", linewidth=2)
        ax.set_xlabel("Episode", color="white")
        ax.set_ylabel("Cumulative Time (seconds)", color="white")
        ax.set_title("Training Time Progression", color="white")
        ax.grid(True, alpha=0.2, color='gray')
        ax.tick_params(colors='white')

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
            color="white",
            bbox=dict(boxstyle="round", facecolor="#2d2d2d", edgecolor="#555555", alpha=0.8),
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
        default="combined",
        help="Type of plot to generate (default: combined)",
    )
    parser.add_argument(
        "--auto-refresh",
        action="store_true",
        help="Automatically refresh the plot every 5 seconds (enabled by default for combined plot)",
    )
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Disable automatic refresh",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds (default: 5.0)",
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

    # Enable auto-refresh by default for combined plot, unless explicitly disabled
    auto_refresh = args.auto_refresh or (args.plot_type == "combined" and not args.no_auto_refresh)

    # Enable interactive mode for auto-refresh
    if auto_refresh and args.plot_type == "combined":
        plt.ion()

    # Generate plots
    fig = None
    axes = None
    
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
        if auto_refresh:
            # Initial plot
            fig, axes = plot_combined(
                df, window=args.window, save_path=save_path, show=show
            )
            print(f"Auto-refresh enabled. Updating every {args.refresh_interval} seconds. Press Ctrl+C to stop.")
            
            # Auto-refresh loop with non-blocking updates
            try:
                last_update_time = time.time()
                while True:
                    # Process matplotlib events to keep UI responsive
                    plt.pause(0.1)
                    
                    # Check if it's time to update
                    current_time = time.time()
                    if current_time - last_update_time >= args.refresh_interval:
                        # Re-read data
                        df_new, _ = read_monitor_csv(args.csv_path)
                        old_len = len(df)
                        df = df_new
                        if len(df) != old_len:
                            print(f"Updated: {len(df)} episodes (was {old_len})")
                        fig, axes = plot_combined(
                            df, window=args.window, save_path=save_path, 
                            show=show, fig=fig, axes=axes
                        )
                        last_update_time = current_time
                    
                    # Check if window is closed
                    if not plt.fignum_exists(fig.number):
                        print("\nWindow closed. Stopping auto-refresh...")
                        break
            except KeyboardInterrupt:
                print("\nStopping auto-refresh...")
                plt.ioff()
                if fig:
                    plt.close(fig)
        else:
            plot_combined(df, window=args.window, save_path=save_path, show=show)

    if args.plot_type in ["time", "all"]:
        save_path = (
            str(save_dir / "time_progression.png") if save_dir else None
        )
        plot_time_progression(df, save_path=save_path, show=show)

    if not auto_refresh:
        print("Visualization complete!")


if __name__ == "__main__":
    main()

