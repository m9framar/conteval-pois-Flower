#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from data_reader import load_results, create_dataframe_from_results

def main():
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
    
    # Focus on tabular dataset with iid distribution
    print("\n=== Regular Results Analysis ===")
    tabular_iid = regular_df[
        (regular_df['dataset'] == 'tabular') & 
        (regular_df['distribution'] == 'iid')
    ].copy()
    
    print(f"Regular data shape: {tabular_iid.shape}")
    print("Available attack_status values:", tabular_iid['attack_status'].unique())
    print("Available methods:", tabular_iid['method'].unique())
    
    # Check global_loss values for self_promotion vs no_attack
    for method in ['loo', 'shapley']:
        method_data = tabular_iid[tabular_iid['method'] == method].copy()
        
        print(f"\n--- Method: {method} ---")
        
        no_attack = method_data[method_data['attack_status'] == 'no_attack']
        self_promotion = method_data[method_data['attack_status'] == 'self_promotion']
        
        if not no_attack.empty and not self_promotion.empty:
            print(f"No attack - global_loss range: {no_attack['global_loss'].min():.4f} to {no_attack['global_loss'].max():.4f}")
            print(f"Self promotion - global_loss range: {self_promotion['global_loss'].min():.4f} to {self_promotion['global_loss'].max():.4f}")
            
            # Check round-by-round comparison
            no_attack_by_round = no_attack.groupby('round')['global_loss'].mean()
            self_promo_by_round = self_promotion.groupby('round')['global_loss'].mean()
            
            print("Round-by-round comparison (avg global_loss):")
            for round_num in sorted(no_attack_by_round.index):
                if round_num in self_promo_by_round.index:
                    no_att_loss = no_attack_by_round[round_num]
                    self_promo_loss = self_promo_by_round[round_num]
                    diff = self_promo_loss - no_att_loss
                    print(f"  Round {round_num}: no_attack={no_att_loss:.4f}, self_promotion={self_promo_loss:.4f}, diff={diff:.4f}")
    
    # Save detailed CSV for manual inspection
    tabular_iid_summary = tabular_iid.groupby(['method', 'attack_status', 'round']).agg({
        'global_loss': ['mean', 'std', 'count'],
        'global_accuracy': ['mean', 'std'],
        'run_id': 'nunique'
    }).round(6)
    
    tabular_iid_summary.to_csv('debug_tabular_iid_summary.csv')
    print(f"\nSaved summary to debug_tabular_iid_summary.csv")
    
    # Save full detailed data for tabular iid
    tabular_iid_detailed = tabular_iid[['method', 'attack_status', 'round', 'run_id', 'global_loss', 'global_accuracy']].copy()
    tabular_iid_detailed.to_csv('debug_tabular_iid_detailed.csv', index=False)
    print(f"Saved detailed data to debug_tabular_iid_detailed.csv")
    
    # Check for selective results too
    if not selective_df.empty:
        print("\n=== Selective Results Analysis ===")
        selective_tabular_iid = selective_df[
            (selective_df['dataset'] == 'tabular') & 
            (selective_df['distribution'] == 'iid')
        ].copy()
        
        print(f"Selective data shape: {selective_tabular_iid.shape}")
        if not selective_tabular_iid.empty:
            print("Available attack_status values:", selective_tabular_iid['attack_status'].unique())
            print("Available attack_round values:", selective_tabular_iid['attack_round'].unique())
            
            selective_tabular_iid_detailed = selective_tabular_iid[['method', 'attack_status', 'attack_round', 'round', 'run_id', 'global_loss', 'global_accuracy']].copy()
            selective_tabular_iid_detailed.to_csv('debug_selective_tabular_iid_detailed.csv', index=False)
            print(f"Saved selective detailed data to debug_selective_tabular_iid_detailed.csv")

if __name__ == "__main__":
    main()
