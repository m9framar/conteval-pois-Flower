#!/usr/bin/env python3
"""
Test script to verify backward compatibility of the extended plot_final_contribution_boxplot function
"""

from pathlib import Path
from data_reader import load_results, create_dataframe_from_results
from plotting import plot_final_contribution_boxplot
from visualization_utils import get_max_clients, get_plot_subtitle

def test_regular_final_rounds():
    print("Testing regular final round plotting (backward compatibility)...")
    
    # Load regular results (not selective)
    results_data = load_results(Path('results'))
    results_df = create_dataframe_from_results(results_data)
    print(f"Loaded {len(results_df)} records")
    
    # Test regular final round plotting
    for dataset in ['tabular']:  # Just test one for now
        for method in ['loo']:
            dataset_method_df = results_df[
                (results_df['dataset'] == dataset) & 
                (results_df['method'] == method)
            ]
            
            if len(dataset_method_df) > 0:
                print(f"\nProcessing {dataset} - {method} (regular final rounds)")
                
                max_clients = get_max_clients(dataset_method_df)
                subtitle = get_plot_subtitle(dataset_method_df)
                
                # Create output path  
                plots_dir = Path('plots') / dataset / method
                plots_dir.mkdir(parents=True, exist_ok=True)
                output_path = plots_dir / "regular_final_round_boxplots.png"
                
                # Generate boxplots using the extended function but without selective round filtering (backward compatibility)
                plot_final_contribution_boxplot(
                    dataset_method_df, max_clients, subtitle, output_path
                    # Note: use_attack_round_filtering=False is the default, so we don't need to specify it
                )
    
    print("Regular final round visualization complete!")

if __name__ == "__main__":
    test_regular_final_rounds()
