import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import modules from refactored files
from data_reader import load_results, create_dataframe_from_results
from visualization_utils import setup_plot_style, get_plot_subtitle, get_max_clients
from plotting import (
    plot_global_metric,
    plot_client_contribution,
    plot_final_contribution_boxplot,
    plot_contribution_comparison,
    plot_attack_params_heatmap,
    plot_client_contribution_percentage_comparison # Add the new function
)
from table_generator import (
    generate_global_metric_table,
    generate_client_contribution_table,
    generate_final_contribution_table,
    generate_contribution_comparison_table
)

# Constants
from visualization_utils import ATTACKER_CLIENT_ID, TARGET_CLIENT_ID, POSTER_FONT_SIZE

# Define root directories
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
TABLES_DIR = Path("tables")  # Directory for tables
PLOTS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def process_and_visualize_results(df, plots_dir, tables_dir, distributions=None):
    """Generates all plots and tables (CSV and image) for the given data.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        plots_dir (Path): Directory where to save plots
        tables_dir (Path): Directory where to save tables
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
    """
    print(f"Generating outputs in {plots_dir} and {tables_dir}...")
    if distributions:
        print(f"Filtering for distributions: {', '.join(distributions)}")
        
    setup_plot_style(POSTER_FONT_SIZE)
    subtitle = get_plot_subtitle(df)

    # Determine max clients for this group
    max_clients = get_max_clients(df)
    
    # Check if we have self-promotion data in this dataset
    has_self_promotion = 'is_self_promotion' in df.columns and df['is_self_promotion'].any()

    # --- Global Accuracy ---
    plot_global_metric(
        df, "global_accuracy", "Global Accuracy", 
        "Global Model Accuracy over Rounds", subtitle, 
        plots_dir / "global_accuracy.png", 
        distributions=distributions
    )
    generate_global_metric_table(
        df, "global_accuracy", 
        tables_dir / "global_accuracy.csv", 
        plots_dir / "global_accuracy_table.png", 
        subtitle,
        distributions=distributions
    )

    # --- Global Loss ---
    plot_global_metric(
        df, "global_loss", "Global Loss", 
        "Global Model Loss over Rounds", subtitle, 
        plots_dir / "global_loss.png",
        distributions=distributions
    )
    generate_global_metric_table(
        df, "global_loss", 
        tables_dir / "global_loss.csv", 
        plots_dir / "global_loss_table.png", 
        subtitle,
        distributions=distributions
    )

    # --- Target Client Contribution ---
    # Only generate target client plots for regular attacks (not for self-promotion)
    if not has_self_promotion or not df[df['is_self_promotion']].equals(df):
        # If mixed or only regular attacks
        target_col = f"client_{TARGET_CLIENT_ID}_contribution"
        plot_client_contribution(
            df[~df['is_self_promotion']] if has_self_promotion else df, 
            TARGET_CLIENT_ID, target_col, subtitle, 
            plots_dir / f"target_client_{TARGET_CLIENT_ID}_contribution.png",
            distributions=distributions
        )
        generate_client_contribution_table(
            df[~df['is_self_promotion']] if has_self_promotion else df, 
            TARGET_CLIENT_ID, target_col, 
            tables_dir / f"target_client_{TARGET_CLIENT_ID}_contribution.csv", 
            plots_dir / f"target_client_{TARGET_CLIENT_ID}_contribution_table.png", 
            subtitle,
            distributions=distributions
        )

    # --- Attacker Client Contribution ---
    # Always generate attacker client plots (both for regular and self-promotion attacks)
    attacker_col = f"client_{ATTACKER_CLIENT_ID}_contribution"
    plot_client_contribution(
        df, ATTACKER_CLIENT_ID, attacker_col, subtitle, 
        plots_dir / f"attacker_client_{ATTACKER_CLIENT_ID}_contribution.png",
        distributions=distributions
    )
    generate_client_contribution_table(
        df, ATTACKER_CLIENT_ID, attacker_col, 
        tables_dir / f"attacker_client_{ATTACKER_CLIENT_ID}_contribution.csv", 
        plots_dir / f"attacker_client_{ATTACKER_CLIENT_ID}_contribution_table.png", 
        subtitle,
        distributions=distributions
    )

    # --- Final Round Boxplot / Table ---
    plot_final_contribution_boxplot(
        df, max_clients, subtitle, 
        plots_dir / "final_round_contribution_boxplot.png",
        distributions=distributions
    )
    generate_final_contribution_table(
        df, max_clients, 
        tables_dir / "final_round_contribution.csv", 
        plots_dir / "final_round_contribution_table.png", 
        subtitle,
        distributions=distributions
    )

    # --- Contribution Comparison ---
    # The 'df' here is already filtered by distributions if distributions_to_include was specified upstream.

    df_without_attack = df[df['attack_status'] == 'without_attack'].copy()

    # Ensure 'is_self_promotion' column exists. It should be added by data_reader.py.
    if 'is_self_promotion' not in df.columns:
        print("Warning: 'is_self_promotion' column not found in the DataFrame. Contribution comparison might be incomplete.")
        # Fallback: assume no self-promotion if column is missing.
        df_with_regular_attack = df[df['attack_status'] == 'with_attack'].copy()
        df_with_self_promotion = pd.DataFrame() # Empty DataFrame
    else:
        df_with_regular_attack = df[(df['attack_status'] == 'with_attack') & (df['is_self_promotion'] == False)].copy()
        df_with_self_promotion = df[df['attack_status'] == 'self_promotion'].copy() # is_self_promotion is True for these

    # Regular attack comparison (target-based vs. no attack)
    if not df_with_regular_attack.empty and not df_without_attack.empty:
        print("Generating regular attack contribution comparison outputs...")
        plot_contribution_comparison(
            df_attack=df_with_regular_attack,
            df_no_attack=df_without_attack,
            subtitle=subtitle,
            output_path=plots_dir / "client_contributions_comparison_regular.png",
            is_self_promotion=False
        )
        generate_contribution_comparison_table( # Assuming analogous signature changes
            df_attack=df_with_regular_attack,
            df_no_attack=df_without_attack,
            subtitle=subtitle,
            csv_output_path=tables_dir / "client_contributions_comparison_regular.csv",
            img_output_path=plots_dir / "client_contributions_comparison_regular_table.png",
            is_self_promotion=False
        )
    else:
        print("Skipping regular contribution comparison: data missing for 'with_attack' (non-self-promotion) or 'without_attack'.")

    # Self-promotion attack comparison (self-promoter vs. no attack)
    if not df_with_self_promotion.empty and not df_without_attack.empty:
        print("Generating self-promotion attack contribution comparison outputs...")
        plot_contribution_comparison(
            df_attack=df_with_self_promotion,
            df_no_attack=df_without_attack,
            subtitle=subtitle,
            output_path=plots_dir / "client_contributions_comparison_self_promotion.png",
            is_self_promotion=True
        )
        generate_contribution_comparison_table( # Assuming analogous signature changes
            df_attack=df_with_self_promotion,
            df_no_attack=df_without_attack,
            subtitle=subtitle,
            csv_output_path=tables_dir / "client_contributions_comparison_self_promotion.csv",
            img_output_path=plots_dir / "client_contributions_comparison_self_promotion_table.png",
            is_self_promotion=True
        )
    else:
        print("Skipping self-promotion contribution comparison: data missing for 'self_promotion' or 'without_attack'.")

    # Generate attack parameter heatmaps
    # Target-based attacks
    if 'with_attack' in df['attack_status'].unique():
        plot_attack_params_heatmap(
            df, subtitle,
            plots_dir / "attack_params_heatmap",
            attack_type='with_attack',
            distributions=distributions
        )
    
    # Self-promotion attacks
    if 'self_promotion' in df['attack_status'].unique():
        plot_attack_params_heatmap(
            df, subtitle,
            plots_dir / "attack_params_heatmap", # Note: filename will include attack_type
            attack_type='self_promotion',
            distributions=distributions
        )

    # --- Client Contribution Percentage Comparison Plots ---
    # This function handles its own looping through distributions and attack types internally
    # It also creates distribution-specific subdirectories within the plots_dir
    # The plots_dir passed here should be the method-specific directory (e.g., plots/tabular/loo)
    plot_client_contribution_percentage_comparison(
        df_dataset_method=df, # Pass the full df for this dataset/method
        max_clients=max_clients,
        subtitle_base=subtitle, # Base subtitle, function will add more specifics
        output_dir_base=plots_dir, # Base directory for plots (e.g., plots/tabular/loo)
        specific_distributions=distributions # Pass the distribution filter if any
    )

    print(f"Finished processing for {subtitle}")

    print("Output generation complete for this group.")
    plt.rcdefaults()  # Reset styles


def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Aggregate and visualize federated learning results.")
    parser.add_argument('--distributions', nargs='+', default=None,
                        help='Space-separated list of distributions to include (e.g., "iid dirichlet_0.1"). '
                             'If not specified, all distributions will be included.')
    parser.add_argument('--exclude-distributions', nargs='+', default=None,
                        help='Space-separated list of distributions to exclude (e.g., "dirichlet_0.8"). '
                             'This option takes precedence over --distributions.')
    parser.add_argument('--attack-types', nargs='+', default=None,
                        choices=['with_attack', 'without_attack', 'self_promotion'],
                        help='Space-separated list of attack types to include. '
                             'If not specified, all attack types will be included.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Process distribution filtering options
    distributions_to_include = args.distributions
    distributions_to_exclude = args.exclude_distributions
    attack_types_to_include = args.attack_types
    
    print("Starting results aggregation...")
    raw_data = load_results(RESULTS_DIR)

    if not raw_data:
        print("No data loaded. Exiting.")
    else:
        df = create_dataframe_from_results(raw_data)
        print(f"Loaded {len(df)} records.")
        
        # Get all unique distributions in the data
        all_distributions = df['distribution'].unique().tolist() if 'distribution' in df.columns else []
        
        # Apply exclusion filter if specified (takes precedence)
        if distributions_to_exclude:
            print(f"Excluding distributions: {', '.join(distributions_to_exclude)}")
            distributions_to_include = [d for d in all_distributions if d not in distributions_to_exclude]
        
        if distributions_to_include:
            print(f"Including distributions: {', '.join(distributions_to_include)}")
        else:
            print("Including all distributions")
            
        # Apply attack type filter if specified
        if attack_types_to_include:
            print(f"Including attack types: {', '.join(attack_types_to_include)}")
            df = df[df['attack_status'].isin(attack_types_to_include)]
            if df.empty:
                print("No data left after filtering by attack types. Exiting.")
                exit(1)

        # Process each dataset and method combination
        for (dataset, method), group_df in df.groupby(['dataset', 'method']):
            print(f"\n--- Processing Dataset: {dataset}, Method: {method} ---")
            method_plots_dir = PLOTS_DIR / dataset / method
            method_tables_dir = TABLES_DIR / dataset / method  # Create corresponding tables dir
            method_plots_dir.mkdir(parents=True, exist_ok=True)
            method_tables_dir.mkdir(parents=True, exist_ok=True)

            if not group_df.empty:
                process_and_visualize_results(
                    group_df.copy(), 
                    method_plots_dir, 
                    method_tables_dir,
                    distributions=distributions_to_include
                )
            else:
                print("No data for this group.")

        print("\nAggregation and visualization finished.")
