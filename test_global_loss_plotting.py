#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_reader import load_results, create_dataframe_from_results
from plotting import plot_method_comparison_global_loss_changes
from visualization_utils import setup_plot_style, POSTER_FONT_SIZE

def main():
    print("=== Testing plot_method_comparison_global_loss_changes function ===\n")
    
    # Load both regular and selective results
    print("Loading regular results...")
    results_dir = Path("results")
    regular_data = load_results(results_dir, selective_rounds=False)
    regular_df = create_dataframe_from_results(regular_data)
    
    print("Loading selective results...")
    selective_results_dir = Path("results_selective")
    if selective_results_dir.exists():
        selective_data = load_results(selective_results_dir, selective_rounds=True)
        selective_df = create_dataframe_from_results(selective_data)
    else:
        print("No selective results directory found")
        selective_df = pd.DataFrame()
    
    # Create output directory for test plots
    test_output_dir = Path("test_plots")
    test_output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    setup_plot_style(POSTER_FONT_SIZE+10)
    
    # Test configurations: datasets and distributions to test
    datasets = ['tabular', 'fashion']
    distributions = ['iid', 'dirichlet_1.0']
    
    # Test 1: Regular results (full attacks) for all dataset/distribution combinations
    print("\n=== Regular Results (Full Attacks) ===")
    for dataset in datasets:
        for distribution in distributions:
            print(f"\n--- Testing {dataset.upper()} {distribution.upper()} ---")
            test_regular_results(regular_df, test_output_dir, dataset, distribution)
    
    # Test 2: Selective results if available
    if not selective_df.empty:
        print("\n=== Selective Results (Round-Specific Attacks) ===")
        for dataset in datasets:
            for distribution in distributions:
                print(f"\n--- Testing {dataset.upper()} {distribution.upper()} Selective ---")
                test_selective_results(selective_df, test_output_dir, dataset, distribution)
    
    print(f"\n=== All test plots saved to {test_output_dir} ===")

def test_regular_results(df, output_dir, dataset, distribution):
    """Test with regular results for a specific dataset and distribution"""
    # Filter for specific dataset and distribution, both methods
    test_df = df[
        (df['dataset'] == dataset) & 
        (df['distribution'] == distribution) &
        (df['method'].isin(['loo', 'shapley']))
    ].copy()
    
    print(f"Test data shape: {test_df.shape}")
    print("Available methods:", test_df['method'].unique())
    print("Available attack_status:", test_df['attack_status'].unique())
    
    if test_df.empty:
        print(f"No data available for {dataset} {distribution} test")
        return
    
    # Quick data sanity check for self_promotion if available
    if 'self_promotion' in test_df['attack_status'].unique():
        for method in ['loo', 'shapley']:
            method_data = test_df[test_df['method'] == method]
            if method_data.empty:
                continue
                
            no_attack = method_data[method_data['attack_status'] == 'no_attack']
            self_promotion = method_data[method_data['attack_status'] == 'self_promotion']
            
            if not no_attack.empty and not self_promotion.empty:
                no_attack_round2 = no_attack[no_attack['round'] == 2]['global_loss'].mean()
                self_promo_round2 = self_promotion[self_promotion['round'] == 2]['global_loss'].mean()
                print(f"  {method.upper()}: Round 2 - no_attack={no_attack_round2:.4f}, self_promotion={self_promo_round2:.4f}")
    
    # Test the function with clean plots (no titles, labels, legends)
    subtitle = f"Full Attack: {dataset.title()} {distribution.title()}"
    output_path = output_dir / f"FULL_ATTACK_{dataset}_{distribution}_global_loss.png"
    
    try:
        result_data = plot_method_comparison_global_loss_changes(
            test_df, 
            subtitle, 
            output_path, 
            distributions=[distribution],
            use_attack_round_filtering=False,
            use_suptitle=False,  # Disable all titles and labels
            show_variance=False,
            return_data=True
        )
        
        if result_data is not None and not result_data.empty:
            print(f"Function executed successfully, returned data shape: {result_data.shape}")
            # Save the returned data for inspection
            result_data.to_csv(output_dir / f"FULL_ATTACK_{dataset}_{distribution}_plot_data.csv", index=False)
            print(f"Plot data saved to FULL_ATTACK_{dataset}_{distribution}_plot_data.csv")
        else:
            print("Function executed but returned no data")
            
    except Exception as e:
        print(f"Error executing function: {e}")
        import traceback
        traceback.print_exc()

def test_selective_results(df, output_dir, dataset, distribution):
    """Test with selective results for a specific dataset and distribution"""
    # Filter for specific dataset and distribution, both methods
    test_df = df[
        (df['dataset'] == dataset) & 
        (df['distribution'] == distribution) &
        (df['method'].isin(['loo', 'shapley']))
    ].copy()
    
    print(f"Test data shape: {test_df.shape}")
    print("Available methods:", test_df['method'].unique())
    print("Available attack_status:", test_df['attack_status'].unique())
    if 'attack_round' in test_df.columns:
        print("Available attack_round:", sorted([r for r in test_df['attack_round'].unique() if pd.notna(r)]))
    
    if test_df.empty:
        print(f"No data available for selective {dataset} {distribution} test")
        return
    
    # Test the function with selective round filtering and clean plots
    subtitle = f"Selective Rounds: {dataset.title()} {distribution.title()}"
    output_path = output_dir / f"SELECTIVE_ROUNDS_{dataset}_{distribution}_global_loss.png"
    
    try:
        result_data = plot_method_comparison_global_loss_changes(
            test_df, 
            subtitle, 
            output_path, 
            distributions=[distribution],
            use_attack_round_filtering=True,
            use_suptitle=False,  # Disable all titles and labels
            show_variance=True,
            return_data=True
        )
        
        if result_data is not None and not result_data.empty:
            print(f"Function executed successfully, returned data shape: {result_data.shape}")
            result_data.to_csv(output_dir / f"SELECTIVE_ROUNDS_{dataset}_{distribution}_plot_data.csv", index=False)
            print(f"Plot data saved to SELECTIVE_ROUNDS_{dataset}_{distribution}_plot_data.csv")
        else:
            print("Function executed but returned no data")
            
    except Exception as e:
        print(f"Error executing function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
