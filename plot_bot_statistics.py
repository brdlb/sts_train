"""
Script to visualize bot statistics from the rule-based pool.

This script loads the bot_statistics.json file and creates graphs showing
various metrics for all bots: ELO ratings, winrates, games played, etc.
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10
plt.rcParams["figure.facecolor"] = '#1e1e1e'
plt.rcParams["axes.facecolor"] = '#1e1e1e'
plt.rcParams["savefig.facecolor"] = '#1e1e1e'
plt.rcParams["savefig.edgecolor"] = 'none'


def load_bot_statistics(statistics_path: str) -> Dict:
    """Load bot statistics from JSON file.
    
    Args:
        statistics_path: Path to the bot_statistics.json file
        
    Returns:
        Dictionary containing bot statistics
    """
    with open(statistics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_bot_data(statistics: Dict) -> Tuple[List[str], List[float], List[float], List[int], List[float]]:
    """Extract bot data from statistics.
    
    Args:
        statistics: Dictionary containing bot statistics
        
    Returns:
        Tuple of (names, elos, winrates, games_played, avg_steps)
    """
    bots = statistics.get('bots', {})
    
    names = []
    elos = []
    winrates = []
    games_played = []
    avg_steps = []
    
    for bot_key, bot_data in bots.items():
        names.append(bot_data.get('name', bot_key))
        elos.append(bot_data.get('elo', 0))
        winrates.append(bot_data.get('winrate', 0))
        games_played.append(bot_data.get('games_played', 0))
        avg_steps.append(bot_data.get('avg_steps_per_game', 0))
    
    # Sort by ELO (descending)
    sorted_pairs = sorted(zip(names, elos, winrates, games_played, avg_steps), 
                         key=lambda x: x[1], reverse=True)
    if sorted_pairs:
        names, elos, winrates, games_played, avg_steps = zip(*sorted_pairs)
    
    return list(names), list(elos), list(winrates), list(games_played), list(avg_steps)


def plot_bot_statistics(
    names: List[str],
    elos: List[float],
    winrates: List[float],
    games_played: List[int],
    avg_steps: List[float],
    output_path: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[List[plt.Axes]] = None,
    show: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot bot statistics.
    
    Args:
        names: List of bot names
        elos: List of ELO ratings
        winrates: List of winrates
        games_played: List of games played counts
        avg_steps: List of average steps per game
        output_path: Optional path to save the figure
        fig: Optional existing figure to update
        axes: Optional existing axes to update
        show: Whether to display the plot
        
    Returns:
        Tuple of (figure, axes list)
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
    else:
        # Clear existing plots for update
        for ax in axes:
            ax.clear()
    
    # Prepare data
    x_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    # Plot 1: ELO Ratings
    ax1 = axes[0]
    bars1 = ax1.barh(x_pos, elos, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('ELO Rating', fontsize=11, fontweight='bold', color='white')
    ax1.set_title('Bot ELO Ratings', fontsize=12, fontweight='bold', color='white')
    ax1.grid(True, alpha=0.2, color='gray', linestyle='--', axis='x')
    ax1.tick_params(colors='white')
    ax1.invert_yaxis()  # Top bot at top
    
    # Add value labels on bars
    for i, (bar, elo) in enumerate(zip(bars1, elos)):
        width = bar.get_width()
        ax1.text(width + 10, bar.get_y() + bar.get_height()/2, 
                f'{elo:.1f}', ha='left', va='center', fontsize=8, color='white')
    
    # Plot 2: Winrates
    ax2 = axes[1]
    bars2 = ax2.barh(x_pos, winrates, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Winrate', fontsize=11, fontweight='bold', color='white')
    ax2.set_title('Bot Winrates', fontsize=12, fontweight='bold', color='white')
    ax2.grid(True, alpha=0.2, color='gray', linestyle='--', axis='x')
    ax2.tick_params(colors='white')
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(winrates) * 1.1 if winrates else 1.0)
    
    # Add value labels on bars
    for i, (bar, wr) in enumerate(zip(bars2, winrates)):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{wr:.2%}', ha='left', va='center', fontsize=8, color='white')
    
    # Plot 3: Games Played
    ax3 = axes[2]
    bars3 = ax3.barh(x_pos, games_played, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel('Games Played', fontsize=11, fontweight='bold', color='white')
    ax3.set_title('Games Played by Bot', fontsize=12, fontweight='bold', color='white')
    ax3.grid(True, alpha=0.2, color='gray', linestyle='--', axis='x')
    ax3.tick_params(colors='white')
    ax3.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, gp) in enumerate(zip(bars3, games_played)):
        width = bar.get_width()
        ax3.text(width + max(games_played) * 0.02, bar.get_y() + bar.get_height()/2, 
                f'{gp}', ha='left', va='center', fontsize=8, color='white')
    
    # Plot 4: Average Steps per Game
    ax4 = axes[3]
    bars4 = ax4.barh(x_pos, avg_steps, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(names, fontsize=9)
    ax4.set_xlabel('Average Steps per Game', fontsize=11, fontweight='bold', color='white')
    ax4.set_title('Average Steps per Game', fontsize=12, fontweight='bold', color='white')
    ax4.grid(True, alpha=0.2, color='gray', linestyle='--', axis='x')
    ax4.tick_params(colors='white')
    ax4.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, steps) in enumerate(zip(bars4, avg_steps)):
        width = bar.get_width()
        ax4.text(width + max(avg_steps) * 0.02, bar.get_y() + bar.get_height()/2, 
                f'{steps:.1f}', ha='left', va='center', fontsize=8, color='white')
    
    # Add overall statistics as text
    if elos:
        avg_elo = sum(elos) / len(elos)
        max_elo = max(elos)
        min_elo = min(elos)
        avg_wr = sum(winrates) / len(winrates)
        total_games = sum(games_played)
        
        stats_text = (f'Total Bots: {len(names)} | Total Games: {total_games}\n'
                     f'ELO: Avg={avg_elo:.1f}, Min={min_elo:.1f}, Max={max_elo:.1f}\n'
                     f'Avg Winrate: {avg_wr:.2%}')
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, color='white',
                bbox=dict(boxstyle='round', facecolor='#2d2d2d', edgecolor='#555555', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    
    if show:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.close(fig)
    
    return fig, axes


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Visualize bot statistics from the rule-based pool"
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default=None,
        help="Path to the bot_statistics.json file (default: models/rule_based_pool/bot_statistics.json)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the plot (default: None, plot is only displayed)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=3.0,
        help="Refresh interval in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Disable automatic refresh",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful when saving to file)",
    )
    
    args = parser.parse_args()
    
    # Determine statistics path
    if args.statistics_path:
        statistics_path = Path(args.statistics_path)
    else:
        statistics_path = Path(__file__).parent / 'models' / 'rule_based_pool' / 'bot_statistics.json'
    
    if not statistics_path.exists():
        print(f"Error: Statistics file not found at {statistics_path}")
        return
    
    show = not args.no_show
    auto_refresh = not args.no_auto_refresh
    
    # Enable interactive mode for auto-refresh
    if auto_refresh:
        plt.ion()
    
    # Initial load and plot
    print(f"Loading statistics from {statistics_path}")
    statistics = load_bot_statistics(str(statistics_path))
    
    names, elos, winrates, games_played, avg_steps = extract_bot_data(statistics)
    
    if not names:
        print("Error: No bot data found in statistics")
        return
    
    print(f"Found {len(names)} bots")
    print(f"ELO range: {min(elos):.2f} - {max(elos):.2f}")
    print(f"Winrate range: {min(winrates):.2%} - {max(winrates):.2%}")
    print(f"Total games: {sum(games_played)}")
    
    # Initial plot
    fig, axes = plot_bot_statistics(
        names, elos, winrates, games_played, avg_steps,
        output_path=args.output_path,
        show=show
    )
    
    if auto_refresh:
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
                    # Re-read statistics
                    try:
                        statistics_new = load_bot_statistics(str(statistics_path))
                        names_new, elos_new, winrates_new, games_played_new, avg_steps_new = extract_bot_data(statistics_new)
                        
                        old_len = len(names)
                        names, elos, winrates, games_played, avg_steps = (
                            names_new, elos_new, winrates_new, games_played_new, avg_steps_new
                        )
                        
                        if len(names) != old_len:
                            print(f"Updated: {len(names)} bots (was {old_len})")
                            print(f"ELO range: {min(elos):.2f} - {max(elos):.2f}")
                            print(f"Total games: {sum(games_played)}")
                        
                        fig, axes = plot_bot_statistics(
                            names, elos, winrates, games_played, avg_steps,
                            output_path=args.output_path,
                            fig=fig, axes=axes, show=show
                        )
                        last_update_time = current_time
                    except Exception as e:
                        print(f"Error updating plot: {e}")
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
        if show:
            plt.show()


if __name__ == '__main__':
    main()



