"""
Script to visualize ELO ratings of snapshots from the opponent pool.

This script loads the pool_metadata.json file and creates a graph showing
ELO ratings of all snapshots over training steps.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


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
    output_path: str = None
) -> None:
    """Plot ELO ratings of snapshots.
    
    Args:
        steps: List of training steps
        elos: List of ELO ratings corresponding to steps
        current_policy_elo: Current policy ELO rating
        output_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot snapshot ELOs
    plt.plot(steps, elos, 'o-', linewidth=2, markersize=8, 
             label='Snapshot ELO', color='#2E86AB', alpha=0.8)
    
    # Plot current policy ELO as horizontal line
    if current_policy_elo > 0:
        plt.axhline(y=current_policy_elo, color='r', linestyle='--', 
                   linewidth=2, label=f'Current Policy ELO: {current_policy_elo:.1f}', 
                   alpha=0.7)
    
    # Calculate and plot average ELO
    if elos:
        avg_elo = sum(elos) / len(elos)
        plt.axhline(y=avg_elo, color='g', linestyle='--', 
                   linewidth=1.5, label=f'Average ELO: {avg_elo:.1f}', 
                   alpha=0.5)
    
    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.ylabel('ELO Rating', fontsize=12, fontweight='bold')
    plt.title('ELO Ratings of Opponent Pool Snapshots', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Add some statistics as text
    if elos:
        min_elo = min(elos)
        max_elo = max(elos)
        stats_text = f'Min: {min_elo:.1f} | Max: {max_elo:.1f} | Range: {max_elo - min_elo:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    
    plt.show()


def main():
    """Main function to run the script."""
    # Path to pool metadata file
    metadata_path = Path(__file__).parent / 'models' / 'opponent_pool' / 'pool_metadata.json'
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    print(f"Loading metadata from {metadata_path}")
    metadata = load_pool_metadata(str(metadata_path))
    
    steps, elos, current_policy_elo = extract_snapshot_data(metadata)
    
    if not steps:
        print("Error: No snapshot data found in metadata")
        return
    
    print(f"Found {len(steps)} snapshots")
    print(f"Current policy ELO: {current_policy_elo:.2f}")
    print(f"Snapshot ELO range: {min(elos):.2f} - {max(elos):.2f}")
    
    # Plot the graph
    plot_elo_snapshots(steps, elos, current_policy_elo)


if __name__ == '__main__':
    main()

