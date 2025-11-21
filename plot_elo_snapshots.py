"""
Script to visualize ELO ratings of snapshots from the opponent pool.

This script loads the pool_metadata.json file and creates a graph showing
ELO ratings of all snapshots over training steps.
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
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
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["figure.facecolor"] = '#1e1e1e'
plt.rcParams["axes.facecolor"] = '#1e1e1e'
plt.rcParams["savefig.facecolor"] = '#1e1e1e'
plt.rcParams["savefig.edgecolor"] = 'none'


def load_pool_metadata(metadata_path: str) -> Dict:
    """Load pool metadata from JSON file.
    
    Args:
        metadata_path: Path to the pool_metadata.json file
        
    Returns:
        Dictionary containing pool metadata
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_snapshot_data(metadata: Dict) -> Tuple[List[int], List[float], float]:
    """Extract step and ELO data from metadata.
    
    Args:
        metadata: Dictionary containing pool metadata
        
    Returns:
        Tuple of (steps, elos, current_policy_elo)
    """
    snapshots = metadata.get('snapshots', {})
    current_policy_elo = metadata.get('current_policy_elo', 0)
    
    steps = []
    elos = []
    
    for snapshot_name, snapshot_data in snapshots.items():
        step = snapshot_data.get('step')
        elo = snapshot_data.get('elo')
        if step is not None and elo is not None:
            steps.append(step)
            elos.append(elo)
    
    # Sort by step
    sorted_pairs = sorted(zip(steps, elos))
    steps, elos = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(steps), list(elos), current_policy_elo


def plot_elo_snapshots(
    steps: List[int],
    elos: List[float],
    current_policy_elo: float,
    output_path: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ELO ratings of snapshots.
    
    Args:
        steps: List of training steps
        elos: List of ELO ratings corresponding to steps
        current_policy_elo: Current policy ELO rating
        output_path: Optional path to save the figure
        fig: Optional existing figure to update
        ax: Optional existing axes to update
        show: Whether to display the plot
        
    Returns:
        Tuple of (figure, axes)
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        # Clear existing plot for update
        ax.clear()
    
    # Plot snapshot ELOs
    ax.plot(steps, elos, 'o-', linewidth=2, markersize=8, 
             label='Snapshot ELO', color='#4A90E2', alpha=0.8)

    
    # Calculate and plot average ELO
    if elos:
        avg_elo = sum(elos) / len(elos)
        ax.axhline(y=avg_elo, color='#4AE24A', linestyle='--', 
                   linewidth=1.5, label=f'Average ELO: {avg_elo:.1f}', 
                   alpha=0.5)
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold', color='white')
    ax.set_title('ELO Ratings of Opponent Pool Snapshots', fontsize=14, fontweight='bold', color='white')
    ax.grid(True, alpha=0.2, color='gray', linestyle='--')
    ax.legend(loc='best', fontsize=10, facecolor='#2d2d2d', edgecolor='#555555')
    ax.tick_params(colors='white')
    
    # Add some statistics as text
    if elos:
        min_elo = min(elos)
        max_elo = max(elos)
        stats_text = f'Min: {min_elo:.1f} | Max: {max_elo:.1f} | Range: {max_elo - min_elo:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', color='white',
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
    
    return fig, ax


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Visualize ELO ratings of snapshots from the opponent pool"
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to the pool_metadata.json file (default: models/opponent_pool/pool_metadata.json)",
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
    
    # Determine metadata path
    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    else:
        metadata_path = Path(__file__).parent / 'models' / 'opponent_pool' / 'pool_metadata.json'
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    show = not args.no_show
    auto_refresh = not args.no_auto_refresh
    
    # Enable interactive mode for auto-refresh
    if auto_refresh:
        plt.ion()
    
    # Initial load and plot
    print(f"Loading metadata from {metadata_path}")
    metadata = load_pool_metadata(str(metadata_path))
    
    steps, elos, current_policy_elo = extract_snapshot_data(metadata)
    
    if not steps:
        print("Error: No snapshot data found in metadata")
        return
    
    print(f"Found {len(steps)} snapshots")
    print(f"Current policy ELO: {current_policy_elo:.2f}")
    print(f"Snapshot ELO range: {min(elos):.2f} - {max(elos):.2f}")
    
    # Initial plot
    fig, ax = plot_elo_snapshots(
        steps, elos, current_policy_elo, 
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
                    # Re-read metadata
                    try:
                        metadata_new = load_pool_metadata(str(metadata_path))
                        steps_new, elos_new, current_policy_elo_new = extract_snapshot_data(metadata_new)
                        
                        old_len = len(steps)
                        steps, elos, current_policy_elo = steps_new, elos_new, current_policy_elo_new
                        
                        if len(steps) != old_len:
                            print(f"Updated: {len(steps)} snapshots (was {old_len})")
                            print(f"Current policy ELO: {current_policy_elo:.2f}")
                        
                        fig, ax = plot_elo_snapshots(
                            steps, elos, current_policy_elo,
                            output_path=args.output_path,
                            fig=fig, ax=ax, show=show
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

