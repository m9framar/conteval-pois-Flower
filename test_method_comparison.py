#!/usr/bin/env python3
"""
Test script for method comparison contribution changes visualization.
This script loads results data and generates boxplots comparing loo vs shapley methods.
Also exports CSV data for verification of boxplot data points.
"""

from pathlib import Path
import sys
import pandas as pd

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from data_reader import load_results, create_dataframe_from_results
from plotting import plot_method_comparison_contribution_changes
from visualization_utils import get_max_clients, get_plot_subtitle

def main():
    # Paths
    RESULTS_DIR = Path("results")
    PLOTS_DIR = Path("plots")
    
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist")
        return
    
    print("Loading results data...")
    
    # Load results data
    results_data = load_results(RESULTS_DIR)
    if not results_data:
        print("No results data found")
        return
    
    # Create DataFrame
    df = create_dataframe_from_results(results_data)
    print(f"Loaded {len(df)} records")
    
    # Show data overview
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Distributions: {df['distribution'].unique()}")
    print(f"Attack statuses: {df['attack_status'].unique()}")
    
    # Process each dataset (plots go directly to dataset subfolders)
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        
        if len(dataset_df) == 0:
            continue
        
        print(f"\nProcessing {dataset} dataset")
        
        max_clients = get_max_clients(dataset_df)
        subtitle = get_plot_subtitle(dataset_df)
        
        # Create output directory for this dataset
        dataset_plots_dir = PLOTS_DIR / dataset
        dataset_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate method comparison plots and get data for CSV export
        output_path = dataset_plots_dir / "method_comparison_contribution_changes.png"
        
        plotting_data = plot_method_comparison_contribution_changes(
            dataset_df, max_clients, subtitle, output_path, use_suptitle=False, return_data=True
        )
        
        # Export data to CSV for verification
        if plotting_data is not None and not plotting_data.empty:
            csv_path = dataset_plots_dir / f"method_comparison_data_{dataset}.csv"
            plotting_data.to_csv(csv_path, index=False)
            print(f"Exported plotting data to {csv_path}")
            
            # Print summary statistics for verification
            print(f"Data summary for {dataset}:")
            print(f"  Total data points: {len(plotting_data)}")
            print(f"  Methods: {plotting_data['method'].unique()}")
            print(f"  Distributions: {plotting_data['distribution'].unique()}")
            print(f"  Attack types: {plotting_data['attack_type'].unique()}")
            print(f"  Client IDs: {sorted(plotting_data['client_id'].unique())}")
            
            # Count data points per client/method/distribution combination
            summary = plotting_data.groupby(['client_id', 'method', 'distribution', 'attack_type']).size().reset_index(name='count')
            print(f"  Data points per combination:")
            for _, row in summary.head(10).iterrows():  # Show first 10 combinations
                print(f"    Client {row['client_id']}, {row['method']}, {row['distribution']}, {row['attack_type']}: {row['count']} points")
            if len(summary) > 10:
                print(f"    ... and {len(summary) - 10} more combinations")
        else:
            print(f"No plotting data returned for {dataset}")
    
    print("\nMethod comparison visualization complete!")

if __name__ == "__main__":
    main()
