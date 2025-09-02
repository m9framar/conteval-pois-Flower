import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import shared utilities
from visualization_utils import save_df_as_image, TABLE_FONT_SIZE, ATTACKER_CLIENT_ID, TARGET_CLIENT_ID


def generate_global_metric_table(df, metric_col, csv_path, image_path, subtitle, distributions=None):
    """Generate a table for a global metric (accuracy or loss) over rounds.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        metric_col (str): Name of the column to use for the table
        csv_path (Path): Path where to save the CSV file
        image_path (Path): Path where to save the table image
        subtitle (str): Subtitle for the table image
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
    """
    print(f"Generating table for {metric_col} to {csv_path} and {image_path}...")
    if metric_col not in df.columns:
        print(f"Skipping table - Metric column '{metric_col}' not found.")
        return
        
    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping table - No data for specified distributions: {distributions}")
            return
            
    try:
        agg_df = df.groupby(['distribution', 'attack_status', 'round']).agg(
            mean=(metric_col, 'mean'),
            std=(metric_col, 'std')
        ).unstack(level='round') # Pivot: rounds become columns

        # Flatten multi-index columns with simpler names
        agg_df.columns = [f'{"m" if col[0] == "mean" else "s"}{col[1]}' for col in agg_df.columns]

        # Save CSV (pivoted format)
        agg_df.to_csv(csv_path, float_format='%.4f')
        print(f"Table CSV saved to {csv_path}")

        # Update subtitle to include filtered distributions if applicable
        if distributions is not None:
            subtitle = f"{subtitle}\nDistributions: {', '.join(distributions)}"
            
        # Save Image with improved parameters
        table_title = f"{metric_col.replace('_', ' ').title()} Over Rounds\n{subtitle}"
        save_df_as_image(agg_df, image_path, title=table_title, 
                        col_width=1.0, row_height=0.3, font_size=8, max_header_len=8)

    except Exception as e:
        print(f"Error generating table for {metric_col}: {e}")


def generate_client_contribution_table(df, client_id, col_name, csv_path, image_path, subtitle, distributions=None):
    """Generate a table for a specific client's contribution over rounds.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        client_id (int): ID of the client
        col_name (str): Name of the column containing contribution data
        csv_path (Path): Path where to save the CSV file
        image_path (Path): Path where to save the table image
        subtitle (str): Subtitle for the table image
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
    """
    print(f"Generating table for Client {client_id} ({col_name}) to {csv_path} and {image_path}...")
    if col_name not in df.columns:
        print(f"Skipping table - Column '{col_name}' not found for client {client_id}.")
        return
        
    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping table - No data for specified distributions: {distributions}")
            return
        
    # Limit the rounds to display if there are too many
    all_rounds = sorted(df['round'].unique())
    if len(all_rounds) > 10:
        # Only keep first round, last round, and some intermediate rounds
        selected_rounds = [all_rounds[0]]  # First round
        
        # Add some intermediate rounds
        step = max(1, len(all_rounds) // 4)  # Show about 4 intermediate rounds
        for i in range(step, len(all_rounds) - 1, step):
            selected_rounds.append(all_rounds[i])
            
        selected_rounds.append(all_rounds[-1])  # Last round
        print(f"Limiting client contribution table to {len(selected_rounds)} selected rounds: {selected_rounds}")
        df = df[df['round'].isin(selected_rounds)]
            
    try:
        agg_df = df.groupby(['distribution', 'attack_status', 'round']).agg(
            mean=(col_name, 'mean'),
            std=(col_name, 'std')
        ).unstack(level='round') # Pivot: rounds become columns

        # Flatten multi-index columns with simpler names
        agg_df.columns = [f'{"m" if col[0] == "mean" else "s"}{col[1]}' for col in agg_df.columns]

        # Save CSV
        agg_df.to_csv(csv_path, float_format='%.4f')
        print(f"Table CSV saved to {csv_path}")

        # Update subtitle to include filtered distributions if applicable
        if distributions is not None:
            subtitle = f"{subtitle}\nDistributions: {', '.join(distributions)}"
            
        # Save Image with improved parameters
        table_title = f"Client {client_id} Contribution\n{subtitle}"
        save_df_as_image(agg_df, image_path, title=table_title, 
                        col_width=1.0, row_height=0.3, font_size=8, max_header_len=8)

    except Exception as e:
        print(f"Error generating table for Client {client_id} ({col_name}): {e}")


def generate_final_contribution_table(df, max_clients, csv_path, image_path, subtitle, distributions=None):
    """Generate a table for client contributions in the final round.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        max_clients (int): Maximum number of clients
        csv_path (Path): Path where to save the CSV file
        image_path (Path): Path where to save the table image
        subtitle (str): Subtitle for the table image
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
    """
    print(f"Generating table for final contributions to {csv_path} and {image_path}...")
    if max_clients <= 0 or 'round' not in df.columns or df['round'].empty:
        print("Skipping final round table - insufficient data.")
        return

    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping table - No data for specified distributions: {distributions}")
            return

    try:
        final_round = df['round'].max()
        df_final = df[df['round'] == final_round].copy()
        
        # Handle large client counts more efficiently
        if max_clients > 10:
            # Just include attacker, target, and a few other representative clients
            selected_clients = [ATTACKER_CLIENT_ID, TARGET_CLIENT_ID]
            # Add a few more clients (e.g., 3, 5, 8) if they exist
            for extra_client in [3, 5, 8]:
                if extra_client <= max_clients and extra_client not in selected_clients:
                    selected_clients.append(extra_client)
            
            value_vars_final = [f"client_{i}_contribution" for i in selected_clients 
                                if f"client_{i}_contribution" in df_final.columns]
            print(f"Limiting final contribution table to {len(selected_clients)} selected clients: {selected_clients}")
        else:
            value_vars_final = [f"client_{i}_contribution" for i in range(1, max_clients + 1) 
                                if f"client_{i}_contribution" in df_final.columns]

        if not value_vars_final:
            print("Skipping final round table - no client contribution columns found.")
            return

        df_melted_final = df_final.melt(id_vars=["distribution", "attack_status"],
                                  value_vars=value_vars_final,
                                  var_name="client_id_str",
                                  value_name="contribution")
        df_melted_final['client_id'] = df_melted_final['client_id_str'].str.extract('(\\d+)').astype(int)
        df_melted_final.dropna(subset=['contribution'], inplace=True)

        if df_melted_final.empty:
            print("Skipping final round table - no data after melting/filtering.")
            return

        agg_df = df_melted_final.groupby(['distribution', 'attack_status', 'client_id']).agg(
            mean_contribution=('contribution', 'mean'),
            std_contribution=('contribution', 'std')
        )
        
        # Simplify column names
        agg_df.columns = ['mean', 'std']

        # Save CSV
        agg_df.to_csv(csv_path, float_format='%.4f')
        print(f"Table CSV saved to {csv_path}")

        # Update subtitle to include filtered distributions if applicable
        if distributions is not None:
            subtitle = f"{subtitle}\nDistributions: {', '.join(distributions)}"
            
        # Save Image - Use compact format for better visualization
        table_title = f"Final Round ({final_round}) Contributions\n{subtitle}"
        
        # Always pivot for better layout - this makes a more compact table
        try:
            # Create a clean pivot table with distribution and attack_status as rows, client_id as columns
            agg_df_pivot = pd.pivot_table(
                df_melted_final,
                values='contribution',
                index=['distribution', 'attack_status'],
                columns=['client_id'],
                aggfunc='mean'
            )
            
            # Add a round(4) to simplify the numeric values
            agg_df_pivot = agg_df_pivot.round(4)
            
            # Rename columns to add 'C' prefix for clarity
            agg_df_pivot.columns = [f'C{col}' for col in agg_df_pivot.columns]
            
            save_df_as_image(agg_df_pivot, image_path, title=table_title, 
                            col_width=1.0, row_height=0.3, font_size=8)
        except Exception as pivot_err:
            print(f"Could not create pivoted final contribution table: {pivot_err}")
            # Fallback to standard table but with smaller settings
            save_df_as_image(agg_df, image_path, title=table_title, 
                            col_width=1.0, row_height=0.3, font_size=8)

    except Exception as e:
        print(f"Error generating final contribution table: {e}")


def generate_contribution_comparison_table(df_attack, df_no_attack, subtitle, csv_path, image_path, is_self_promotion=False, distributions=None):
    """Generate a comparison table for client contributions under attack vs. no attack.
    
    Args:
        df_attack (pd.DataFrame): DataFrame with attack results.
        df_no_attack (pd.DataFrame): DataFrame with no-attack results.
        subtitle (str): Subtitle for the table image.
        csv_path (Path): Path where to save the CSV file.
        image_path (Path): Path where to save the table image.
        is_self_promotion (bool): Whether the attack scenario is self-promotion.
        distributions (list, optional): List of distributions to include.
    """
    print(f"Generating table for contribution comparison to {csv_path} and {image_path}...")

    processed_dfs = []

    for df_scenario, scenario_label in [(df_attack, "With Attack"), (df_no_attack, "Without Attack")]:
        if df_scenario is None or df_scenario.empty:
            print(f"Skipping {scenario_label} part - DataFrame is empty or None.")
            continue

        current_df = df_scenario.copy()

        if distributions is not None:
            current_df = current_df[current_df['distribution'].isin(distributions)]
            if current_df.empty:
                print(f"Skipping {scenario_label} part - No data for specified distributions: {distributions}")
                continue
        
        # Dynamically find client contribution columns
        client_cols = sorted([col for col in current_df.columns if col.startswith("client_") and col.endswith("_contribution")])
        if not client_cols:
            print(f"Skipping {scenario_label} part - no client contribution columns found.")
            continue

        df_melted = current_df.melt(
            id_vars=["round", "distribution", "attack_status", "run_id"], # attack_status helps differentiate if needed later
            value_vars=client_cols,
            var_name="client_id_str",
            value_name="contribution"
        )
        df_melted.dropna(subset=['contribution'], inplace=True)
        if df_melted.empty:
            print(f"Skipping {scenario_label} part - no data after melting/filtering.")
            continue
            
        df_melted['client_id'] = df_melted['client_id_str'].str.extract('(\\d+)').astype(int)

        def categorize_client(cid):
            if is_self_promotion:
                if cid == ATTACKER_CLIENT_ID:
                    return f"Self-Promoter (C{ATTACKER_CLIENT_ID})"
                else:
                    return f"Other (C{cid})" if scenario_label == "With Attack" else f"Regular (C{cid})"
            else: # Target-based attack
                if cid == ATTACKER_CLIENT_ID:
                    return f"Attacker (C{ATTACKER_CLIENT_ID})"
                elif cid == TARGET_CLIENT_ID:
                    return f"Target (C{TARGET_CLIENT_ID})"
                else:
                    return f"Other (C{cid})" if scenario_label == "With Attack" else f"Regular (C{cid})"
        
        df_melted['client_category'] = df_melted['client_id'].apply(categorize_client)
        
        # Group "Other" and "Regular" clients together for averaging if there are many
        # For simplicity in the table, we might average all non-special clients.
        # Let's refine categorization for averaging:
        special_clients = []
        if is_self_promotion:
            special_clients.append(ATTACKER_CLIENT_ID)
        else:
            special_clients.append(ATTACKER_CLIENT_ID)
            special_clients.append(TARGET_CLIENT_ID)

        def get_display_category(row):
            cid = row['client_id']
            if is_self_promotion:
                if cid == ATTACKER_CLIENT_ID:
                    return f"Self-Promoter (C{ATTACKER_CLIENT_ID})"
                return "Avg. Others" # Group others for averaging
            else: # Target-based
                if cid == ATTACKER_CLIENT_ID:
                    return f"Attacker (C{ATTACKER_CLIENT_ID})"
                elif cid == TARGET_CLIENT_ID:
                    return f"Target (C{TARGET_CLIENT_ID})"
                return "Avg. Others" # Group others for averaging
        
        # Create a temporary column for grouping before applying get_display_category
        # This ensures that 'Avg. Others' is calculated correctly across actual other clients
        df_melted['temp_group_category'] = df_melted['client_id'].apply(
            lambda cid: 'special' if cid in special_clients else 'other_group'
        )

        # Aggregate contributions
        # For special clients, take their direct contribution. For 'other_group', average them.
        agg_list = []
        for group_keys, group_df in df_melted.groupby(['distribution', 'round', 'temp_group_category']):
            dist, rd, cat_group = group_keys
            if cat_group == 'special':
                # Process each special client individually
                for cid, special_client_df in group_df.groupby('client_id'):
                    client_display_cat = get_display_category(special_client_df.iloc[0]) # Get the proper label
                    mean_contrib = special_client_df['contribution'].mean()
                    agg_list.append({
                        'distribution': dist,
                        'round': rd,
                        'client_category': client_display_cat,
                        'contrib': mean_contrib,
                        'scenario': scenario_label
                    })
            else: # 'other_group'
                client_display_cat = "Avg. Others"
                mean_contrib = group_df['contribution'].mean()
                agg_list.append({
                    'distribution': dist,
                    'round': rd,
                    'client_category': client_display_cat,
                    'contrib': mean_contrib,
                    'scenario': scenario_label
                })
        
        if not agg_list:
            print(f"No data to aggregate for {scenario_label}.")
            continue
            
        scenario_agg_df = pd.DataFrame(agg_list)
        processed_dfs.append(scenario_agg_df)

    if not processed_dfs:
        print("No data processed for any scenario. Skipping table generation.")
        return

    final_agg_df = pd.concat(processed_dfs)

    # Limit rounds for display
    all_rounds = sorted(final_agg_df['round'].unique())
    if len(all_rounds) > 7: # Max 7 rounds in table (e.g. First, 3 intermediate, Last)
        selected_rounds = [all_rounds[0]] 
        num_intermediate = min(3, len(all_rounds) - 2)
        if num_intermediate > 0:
            step = (len(all_rounds) - 2) // (num_intermediate +1) if (len(all_rounds) -2) > num_intermediate else 1
            for i in range(1, num_intermediate + 1):
                 idx = min(i * step, len(all_rounds)-2) # ensure index is valid
                 if all_rounds[idx] not in selected_rounds:
                    selected_rounds.append(all_rounds[idx])
        if all_rounds[-1] not in selected_rounds: # Add last if not already there
             selected_rounds.append(all_rounds[-1])
        selected_rounds = sorted(list(set(selected_rounds))) # ensure uniqueness and order
        print(f"Limiting contribution comparison table to selected rounds: {selected_rounds}")
        final_agg_df = final_agg_df[final_agg_df['round'].isin(selected_rounds)]
        if final_agg_df.empty:
            print("No data after limiting rounds. Skipping table generation.")
            return

    try:
        table_df = final_agg_df.pivot_table(
            index=['distribution', 'scenario', 'client_category'],
            columns='round',
            values='contrib'
        )
        table_df.columns = [f'R{int(col)}' for col in table_df.columns] # Ensure round numbers are int
        table_df = table_df.sort_index() # Sort for consistent output

        table_df.to_csv(csv_path, float_format='%.4f')
        print(f"Table CSV saved to {csv_path}")

        img_subtitle = subtitle
        if distributions is not None:
            img_subtitle = f"{subtitle}\nDistributions: {', '.join(distributions)}"
        
        attack_type_str = "Self-Promotion" if is_self_promotion else "Target-Based Attack"
        table_title = f"Client Contribution Comparison ({attack_type_str})\n{img_subtitle}"
        
        save_df_as_image(table_df, image_path, title=table_title,
                        col_width=1.0, row_height=0.35, font_size=7, max_header_len=8) # Adjusted for potentially more rows

    except Exception as e:
        print(f"Error generating final pivoted contribution comparison table: {e}")