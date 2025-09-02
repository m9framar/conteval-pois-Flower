#!/usr/bin/env python3
"""
Test script for selective round attack visualization.
This script loads selective round attack data and generates boxplots.
"""

from pathlib import Path
import sys

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from data_reader import load_results, create_dataframe_from_results
from plotting import plot_final_contribution_boxplot, _y_axis_limits_cache
from visualization_utils import get_max_clients, get_plot_subtitle

def calculate_all_y_limits(df):
    """Pre-calculate y-axis limits for all scenarios to ensure consistent scaling across methods."""
    print("Pre-calculating y-axis limits for all scenarios...")
    
    # Clear any existing cache
    _y_axis_limits_cache.clear()
    
    # Group by scenario (dataset, distribution, attack_status) and calculate limits
    for (dataset, distribution, attack_status), group in df.groupby(['dataset', 'distribution', 'attack_status']):
        # For selective rounds, get all unique attack rounds
        if 'attack_round' in df.columns:
            attack_rounds = sorted([r for r in group['attack_round'].unique() if r is not None])
            if not attack_rounds and attack_status == 'no_attack':
                attack_rounds = [None]  # Handle no_attack case
        else:
            attack_rounds = [df['round'].max()]  # Use final round for non-selective cases
        
        for attack_round in attack_rounds:
            if attack_round is not None:
                round_data = group[group['attack_round'] == attack_round] if 'attack_round' in df.columns else group[group['round'] == attack_round]
                target_round = attack_round
            else:
                round_data = group[group['attack_status'] == 'no_attack']  # For no_attack baseline
                target_round = group['round'].max() if not round_data.empty else 1
            
            if round_data.empty:
                continue
                
            # Get max clients for this scenario
            max_clients = max([int(col.split('_')[1]) for col in round_data.columns 
                             if col.startswith('client_') and col.endswith('_contribution')] + [0])
            
            if max_clients == 0:
                continue
                
            # Melt the data to get all contribution values
            if target_round is not None:
                value_vars = [f"client_{i}_contribution" for i in range(1, max_clients + 1) 
                             if f"client_{i}_contribution" in round_data.columns]
            else:
                value_vars = [f"client_{i}_avg_contribution" for i in range(1, max_clients + 1) 
                             if f"client_{i}_avg_contribution" in round_data.columns]
            
            if not value_vars:
                continue
                
            melted_data = round_data.melt(
                id_vars=["dataset", "method", "distribution", "attack_status"],
                value_vars=value_vars,
                value_name="contribution"
            )
            
            # Calculate limits for this scenario
            all_contributions = melted_data['contribution'].dropna()
            if not all_contributions.empty:
                ymin_raw = all_contributions.min()
                ymax_raw = all_contributions.max()
                max_abs = max(abs(ymin_raw), abs(ymax_raw))
                max_abs *= 1.2  # Add padding
                
                cache_key = f"{dataset}_{distribution}_{attack_status}_{target_round}"
                _y_axis_limits_cache[cache_key] = (-max_abs, max_abs)
                print(f"  {cache_key}: limits [{-max_abs:.6f}, {max_abs:.6f}] from range [{ymin_raw:.6f}, {ymax_raw:.6f}]")

def main():
    # Paths
    RESULTS_SELECTIVE_DIR = Path("results_selective")
    PLOTS_DIR = Path("plots")
    
    if not RESULTS_SELECTIVE_DIR.exists():
        print(f"Error: {RESULTS_SELECTIVE_DIR} does not exist")
        return
    
    print("Loading selective round attack results...")
    
    # Load selective round data
    results_data = load_results(RESULTS_SELECTIVE_DIR, selective_rounds=True)
    if not results_data:
        print("No selective round data found")
        return
    
    # Create DataFrame
    df = create_dataframe_from_results(results_data)
    if df.empty:
        print("Empty DataFrame created")
        return
    
    print(f"Loaded {len(df)} records")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Distributions: {df['distribution'].unique()}")
    print(f"Attack statuses: {df['attack_status'].unique()}")
    print(f"Attack rounds: {sorted([r for r in df['attack_round'].unique() if r is not None])}")
    
    # Pre-calculate all y-axis limits for consistent scaling across methods
    calculate_all_y_limits(df)
    
    # Process each dataset and method combination
    for (dataset, method), group_df in df.groupby(['dataset', 'method']):

            
        print(f"\nProcessing {dataset} - {method}")
        
        max_clients = get_max_clients(group_df)
        subtitle = get_plot_subtitle(group_df)
        
        # Create output path
        plots_dir = PLOTS_DIR / dataset / method
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = plots_dir / "selective_round_boxplots.png"
        
        # Generate boxplots using extended function with selective round filtering
        plot_final_contribution_boxplot(
            group_df, max_clients, subtitle, output_path, 
            use_attack_round_filtering=True, center_y_axis=True,
            apply_y_limits=False
        )
    
    print("\nSelective round visualization complete!")

if __name__ == "__main__":
    main()
