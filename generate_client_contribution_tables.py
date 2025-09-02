#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from data_reader import load_results, create_dataframe_from_results

def main():
    print("=== Generating Client Contribution Difference LaTeX Tables ===\n")
    
    # Load regular results (for "All" column - continuous attacks rounds 2-5)
    print("Loading regular results...")
    results_dir = Path("results")
    regular_data = load_results(results_dir, selective_rounds=False)
    df_regular = create_dataframe_from_results(regular_data)
    
    # Load selective results (for individual round columns R2, R3, R4, R5)
    print("Loading selective results...")
    selective_dir = Path("results_selective")
    selective_data = load_results(selective_dir, selective_rounds=True)
    df_selective = create_dataframe_from_results(selective_data)
    
    # Create output directory
    output_dir = Path("client_contribution_tables")
    output_dir.mkdir(exist_ok=True)
    print(f"Tables will be saved to: {output_dir}")
    
    # Generate tables for each scenario
    print("\n=== Generating Target Decrease Tables ===")
    generate_target_decrease_tables(df_regular, df_selective, output_dir)
    
    print("\n=== Generating Self-Promotion Tables ===")
    generate_self_promotion_tables(df_regular, df_selective, output_dir)
    
    print(f"\n=== All tables saved to {output_dir} ===")

def generate_target_decrease_tables(df_regular, df_selective, output_dir):
    """Generate tables for target decrease scenario (with_attack vs no_attack)"""
    
    # Filter for target decrease data (both datasets)
    target_data_regular = df_regular[df_regular['attack_status'].isin(['no_attack', 'with_attack'])].copy()
    target_data_selective = df_selective[df_selective['attack_status'].isin(['no_attack', 'with_attack'])].copy()
    
    # Generate table for each client (1-5)
    for client_id in range(1, 6):
        print(f"Generating table for Client {client_id} (Target Decrease)...")
        
        table_data = calculate_contribution_differences(
            target_data_regular, target_data_selective, client_id, 'with_attack', 'no_attack'
        )
        
        if table_data is not None:
            latex_table = generate_latex_table(
                table_data, client_id, "Target Decrease", 
                include_fashion=True  # Include both datasets
            )
            
            output_file = output_dir / f"client_{client_id}_target_decrease.tex"
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"  Saved: {output_file}")
        else:
            print(f"  No data available for Client {client_id}")

def generate_self_promotion_tables(df_regular, df_selective, output_dir):
    """Generate tables for self-promotion scenario (self_promotion vs no_attack)"""
    
    # Filter for self-promotion data (tabular dataset only)
    self_promo_data_regular = df_regular[
        (df_regular['attack_status'].isin(['no_attack', 'self_promotion'])) &
        (df_regular['dataset'] == 'tabular')
    ].copy()
    
    self_promo_data_selective = df_selective[
        (df_selective['attack_status'].isin(['no_attack', 'self_promotion'])) &
        (df_selective['dataset'] == 'tabular')
    ].copy()
    
    # Generate table for each client (1-5)
    for client_id in range(1, 6):
        print(f"Generating table for Client {client_id} (Self-Promotion)...")
        
        table_data = calculate_contribution_differences(
            self_promo_data_regular, self_promo_data_selective, client_id, 'self_promotion', 'no_attack'
        )
        
        if table_data is not None:
            latex_table = generate_latex_table(
                table_data, client_id, "Self-Promotion",
                include_fashion=False  # Only tabular dataset
            )
            
            output_file = output_dir / f"client_{client_id}_self_promotion.tex"
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"  Saved: {output_file}")
        else:
            print(f"  No data available for Client {client_id}")

def calculate_contribution_differences(df_regular, df_selective, client_id, attack_status, baseline_status):
    """Calculate contribution differences between attack and baseline scenarios
    
    df_regular: DataFrame from results/ (for "All" column - continuous attacks rounds 2-5)
    df_selective: DataFrame from results_selective/ (for R2, R3, R4, R5 columns - single round attacks)
    """
    
    contrib_col = f"client_{client_id}_contribution"
    avg_contrib_col = f"client_{client_id}_avg_contribution"
    
    if contrib_col not in df_regular.columns or contrib_col not in df_selective.columns:
        print(f"  Warning: {contrib_col} not found in data")
        return None
    
    results = {}
    
    # Process each combination of dataset, distribution, method
    combinations = [
        ('tabular', 'iid', 'loo'),
        ('tabular', 'iid', 'shapley'),
        ('tabular', 'dirichlet_1.0', 'loo'), 
        ('tabular', 'dirichlet_1.0', 'shapley'),
        ('fashion', 'iid', 'loo'),
        ('fashion', 'iid', 'shapley'),
        ('fashion', 'dirichlet_1.0', 'loo'),
        ('fashion', 'dirichlet_1.0', 'shapley'),
    ]
    
    for dataset, distribution, method in combinations:
        combo_results = {}
        
        # === Individual round columns (R2, R3, R4, R5) from selective data ===
        for round_num in [2, 3, 4, 5]:
            # Filter selective data for this combination and round
            baseline_selective = df_selective[
                (df_selective['dataset'] == dataset) &
                (df_selective['distribution'] == distribution) &
                (df_selective['method'] == method) &
                (df_selective['attack_status'] == baseline_status)
                # No attack_round filter for baseline - it's the same across all rounds
            ].copy()
            
            attack_selective = df_selective[
                (df_selective['dataset'] == dataset) &
                (df_selective['distribution'] == distribution) &
                (df_selective['method'] == method) &
                (df_selective['attack_status'] == attack_status) &
                (df_selective['attack_round'] == round_num)  # Only attacked in this round
            ].copy()
            
            if baseline_selective.empty or attack_selective.empty:
                continue
                
            # Calculate differences using avg_contribution at the specific attack round
            # (which represents the result immediately after the single-round attack)
            diff_values = []
            relative_diff_values = []
            
            for run_id in baseline_selective['run_id'].unique():
                # Get contribution at the specific round (not final round)
                # For R3 attack, we want contribution after round 3, not round 5
                baseline_run = baseline_selective[
                    (baseline_selective['round'] == round_num) & 
                    (baseline_selective['run_id'] == run_id)
                ][avg_contrib_col]
                
                attack_run = attack_selective[
                    (attack_selective['round'] == round_num) & 
                    (attack_selective['run_id'] == run_id)
                ][avg_contrib_col]
                
                if not baseline_run.empty and not attack_run.empty:
                    baseline_val = baseline_run.iloc[0]
                    attack_val = attack_run.iloc[0]
                    
                    abs_diff = abs(attack_val - baseline_val)
                    diff_values.append(abs_diff)
                    
                    if abs(baseline_val) > 1e-9:
                        rel_diff = (abs_diff / abs(baseline_val)) * 100
                    else:
                        rel_diff = 0.0 if abs_diff < 1e-9 else 100.0
                    relative_diff_values.append(rel_diff)
            
            if diff_values:
                combo_results[f'R{round_num}'] = {
                    'absolute_mean': np.mean(diff_values),
                    'absolute_std': np.std(diff_values),
                    'relative_mean': np.mean(relative_diff_values),
                    'relative_std': np.std(relative_diff_values)
                }
        
        # === "All" column from regular data (continuous attacks rounds 2-5) ===
        baseline_regular = df_regular[
            (df_regular['dataset'] == dataset) &
            (df_regular['distribution'] == distribution) &
            (df_regular['method'] == method) &
            (df_regular['attack_status'] == baseline_status)
        ].copy()
        
        attack_regular = df_regular[
            (df_regular['dataset'] == dataset) &
            (df_regular['distribution'] == distribution) &
            (df_regular['method'] == method) &
            (df_regular['attack_status'] == attack_status)
        ].copy()
        
        if not baseline_regular.empty and not attack_regular.empty:
            diff_values = []
            relative_diff_values = []
            
            for run_id in baseline_regular['run_id'].unique():
                # Get final round average contributions
                baseline_run = baseline_regular[
                    (baseline_regular['round'] == baseline_regular['round'].max()) & 
                    (baseline_regular['run_id'] == run_id)
                ][avg_contrib_col]
                
                attack_run = attack_regular[
                    (attack_regular['round'] == attack_regular['round'].max()) & 
                    (attack_regular['run_id'] == run_id)
                ][avg_contrib_col]
                
                if not baseline_run.empty and not attack_run.empty:
                    baseline_val = baseline_run.iloc[0]
                    attack_val = attack_run.iloc[0]
                    
                    abs_diff = abs(attack_val - baseline_val)
                    diff_values.append(abs_diff)
                    
                    if abs(baseline_val) > 1e-9:
                        rel_diff = (abs_diff / abs(baseline_val)) * 100
                    else:
                        rel_diff = 0.0 if abs_diff < 1e-9 else 100.0
                    relative_diff_values.append(rel_diff)
            
            if diff_values:
                combo_results['All'] = {
                    'absolute_mean': np.mean(diff_values),
                    'absolute_std': np.std(diff_values),
                    'relative_mean': np.mean(relative_diff_values),
                    'relative_std': np.std(relative_diff_values)
                }
        
        # Store results for this combination
        if combo_results:
            results[(dataset, distribution, method)] = combo_results
    
    return results if results else None

def generate_latex_table(table_data, client_id, scenario_name, include_fashion=True):
    """Generate LaTeX table from calculated data"""
    
    # Start building the table
    latex = "\\begin{table*}\n"
    latex += "    \\centering\n"
    
    # Remove R1 column from template (keeping R2, R3, R4, R5, All)
    latex += "    \\begin{tabular}{ccc|ccccc}\n"
    latex += "        \\multicolumn{3}{c|}{Setting} & R2 & R3 & R4 & R5 & All \\\\\n"
    latex += "        \\hline\n"
    
    # Dataset rows
    if include_fashion:
        datasets = [('tabular', 'ADULT'), ('fashion', 'CIFAR')]
        dataset_rowspan = 8
    else:
        datasets = [('tabular', 'ADULT')]
        dataset_rowspan = 4
    
    for dataset_idx, (dataset_key, dataset_name) in enumerate(datasets):
        # Distribution rows
        distributions = [('iid', 'IID'), ('dirichlet_1.0', 'non-IID')]
        
        for dist_idx, (dist_key, dist_name) in enumerate(distributions):
            if dist_idx == 0:
                # First distribution - starts with dataset label
                latex += f"        \\multirow{{{dataset_rowspan}}}{{*}}{{\\rotatebox{{90}}{{{dataset_name}}}}} & \\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{dist_name}}}}}"
            else:
                # Second distribution - starts new row with empty dataset cell
                latex += f"        & \\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{dist_name}}}}}"
            
            # Method rows
            methods = [('loo', 'LOO'), ('shapley', 'GTG')]
            
            for method_idx, (method_key, method_name) in enumerate(methods):
                if method_idx == 0:
                    # First method row - absolute values
                    latex += f" & \\multirow{{2}}{{*}}{{{method_name}}}"
                    
                    # Add data columns (R2, R3, R4, R5, All)
                    data_cols = []
                    for col in ['R2', 'R3', 'R4', 'R5', 'All']:
                        combo_key = (dataset_key, dist_key, method_key)
                        if combo_key in table_data and col in table_data[combo_key]:
                            mean_val = table_data[combo_key][col]['absolute_mean']
                            std_val = table_data[combo_key][col]['absolute_std']
                            data_cols.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
                        else:
                            data_cols.append("$-$")
                    
                    latex += " & " + " & ".join(data_cols) + " \\\\\n"
                    
                    # Second method row - relative values
                    latex += "        & & "
                    
                    data_cols = []
                    for col in ['R2', 'R3', 'R4', 'R5', 'All']:
                        combo_key = (dataset_key, dist_key, method_key)
                        if combo_key in table_data and col in table_data[combo_key]:
                            mean_val = table_data[combo_key][col]['relative_mean']
                            std_val = table_data[combo_key][col]['relative_std']
                            data_cols.append(f"${mean_val:.0f}\\% \\pm {std_val:.0f}$")
                        else:
                            data_cols.append("$-$")
                    
                    latex += " & " + " & ".join(data_cols) + " \\\\\n"
                    
                    # Add line between methods
                    latex += "        \\cline{3-8}\n"
                    
                else:
                    # Second method (GTG/Shapley) - starts new row with empty cells
                    latex += f"        & & \\multirow{{2}}{{*}}{{{method_name}}}"
                    
                    data_cols = []
                    for col in ['R2', 'R3', 'R4', 'R5', 'All']:
                        combo_key = (dataset_key, dist_key, method_key)
                        if combo_key in table_data and col in table_data[combo_key]:
                            mean_val = table_data[combo_key][col]['absolute_mean']
                            std_val = table_data[combo_key][col]['absolute_std']
                            data_cols.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
                        else:
                            data_cols.append("$-$")
                    
                    latex += " & " + " & ".join(data_cols) + " \\\\\n"
                    
                    # Second GTG row - relative values
                    latex += "        & & "
                    
                    data_cols = []
                    for col in ['R2', 'R3', 'R4', 'R5', 'All']:
                        combo_key = (dataset_key, dist_key, method_key)
                        if combo_key in table_data and col in table_data[combo_key]:
                            mean_val = table_data[combo_key][col]['relative_mean']
                            std_val = table_data[combo_key][col]['relative_std']
                            data_cols.append(f"${mean_val:.0f}\\% \\pm {std_val:.0f}$")
                        else:
                            data_cols.append("$-$")
                    
                    latex += " & " + " & ".join(data_cols) + " \\\\\n"
            
            # Add line between distributions (except for last distribution)
            if dist_idx == 0:
                latex += "        \\cline{2-8}\n"
        
        # Add line between datasets (except for last dataset)
        if include_fashion and dataset_idx == 0:
            latex += "        \\hline\n"
    
    # Close table
    latex += "    \\end{tabular}\n"
    latex += f"    \\label{{tab:client_{client_id}_{scenario_name.lower().replace('-', '_')}_diff}}\n"
    latex += f"    \\caption{{Client {client_id}'s absolute ($|f(a)-f(1)|$) and relative ($\\frac{{|f(a)-f(1)|}}{{f(1)}}$) contribution score difference between the {scenario_name.lower()} scenario and baseline for rounds 2-5.}}\n"
    latex += "\\end{table*}\n"
    
    return latex

if __name__ == "__main__":
    main()
