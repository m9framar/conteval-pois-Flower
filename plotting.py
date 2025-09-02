import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Import constants and shared utilities
from visualization_utils import (
    ATTACKER_CLIENT_ID, 
    TARGET_CLIENT_ID, 
    POSTER_FONT_SIZE
)


def create_split_legend(fig, ax, palette, statuses_to_compare):
    """Create split legend with labels on left and right sides outside the plot area.
    
    Args:
        fig: The matplotlib figure object
        ax: The matplotlib axes object
        palette: Color palette dictionary mapping status to color
        statuses_to_compare: List of attack statuses being compared
    """
    # Remove the default legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    # Create custom legend elements positioned outside the plot area
    if len(statuses_to_compare) >= 2:
        # Map attack status to human-readable labels
        status_labels = {
            'no_attack': 'Baseline',
            'with_attack': 'Targeted Decrease', 
            'self_promotion': 'Self Improvement'
        }
        
        # Create text elements for left and right legends
        left_status = statuses_to_compare[0]
        right_status = statuses_to_compare[1] if len(statuses_to_compare) > 1 else statuses_to_compare[0]
        
        left_label = status_labels.get(left_status, left_status)
        right_label = status_labels.get(right_status, right_status)
        
        left_color = palette.get(left_status, 'lightsteelblue')
        right_color = palette.get(right_status, 'salmon')
        
        # Get plot area boundaries for alignment
        bbox = ax.get_position()
        
        # Position legend just above the plot area (closer to the plot)
        legend_y = bbox.y1 + 0.02  # Just above the top of the plot area
        
        # Add left legend: colored rectangle followed by black text
        fig.text(bbox.x0-0.025, legend_y, '▬', transform=fig.transFigure, 
                fontsize=POSTER_FONT_SIZE+25, verticalalignment='bottom', color=left_color, weight='bold')
        fig.text(bbox.x0 + 0.040, legend_y, left_label, transform=fig.transFigure, 
                fontsize=POSTER_FONT_SIZE+10, verticalalignment='bottom', color='black', weight='bold', family='serif')
        
        # Add right legend: black text followed by colored rectangle
        fig.text(bbox.x1 +0.030 , legend_y, right_label, transform=fig.transFigure, 
                fontsize=POSTER_FONT_SIZE+10, verticalalignment='bottom', horizontalalignment='right', 
                color='black', weight='bold', family='serif')
        fig.text(bbox.x1 + 0.095, legend_y, '▬', transform=fig.transFigure, 
                fontsize=POSTER_FONT_SIZE+25, verticalalignment='bottom', horizontalalignment='right',
                color=right_color, weight='bold')


def apply_y_axis_centering(axes_list, center_y_axis=True):
    """Apply consistent y-axis centering across multiple plots for comparison.
    
    Args:
        axes_list: List of matplotlib axes objects
        center_y_axis: Whether to center around 0 and make limits consistent
    """
    if not center_y_axis or not axes_list:
        return
    
    # Find the maximum absolute value across all plots
    max_abs_overall = 0
    for ax in axes_list:
        if hasattr(ax, 'get_ylim'):
            ymin, ymax = ax.get_ylim()
            max_abs_overall = max(max_abs_overall, abs(ymin), abs(ymax))
    
    # Apply consistent symmetric limits to all plots
    for ax in axes_list:
        if hasattr(ax, 'set_ylim'):
            ax.set_ylim(-max_abs_overall, max_abs_overall)


# Global dictionary to store y-axis limits for consistent scaling across plots
_y_axis_limits_cache = {}


def plot_global_metric(df, metric_col, ylabel, title_prefix, subtitle, output_path, distributions=None):
    """Plots a global metric (e.g., accuracy, loss) over rounds.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        metric_col (str): Name of the column to plot
        ylabel (str): Label for the y-axis
        title_prefix (str): Prefix for the plot title
        subtitle (str): Subtitle for the plot
        output_path (Path): Path where to save the plot
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
    """
    print(f"Plotting {title_prefix}...")
    if metric_col not in df.columns:
        print(f"Skipping plot - Metric column '{metric_col}' not found.")
        return

    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping plot - No data for specified distributions: {distributions}")
            return

    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=df, 
        x="round", 
        y=metric_col, 
        hue="attack_status", 
        style="distribution", 
        errorbar="sd", 
        palette={"no_attack": "navy", "with_attack": "crimson", "self_promotion": "darkorange"}
    )
    
    # Update subtitle to include filtered distributions if applicable
    if distributions is not None:
        subtitle = f"{subtitle}\nDistributions: {', '.join(distributions)}"
        
    plt.title(f"{title_prefix}\n{subtitle}")
    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Attack Status / Distribution", title_fontsize=POSTER_FONT_SIZE - 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_client_contribution(df, client_id, col_name, subtitle, output_path, distributions=None):
    """Plots a specific client's contribution over rounds.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        client_id (int): ID of the client to plot
        col_name (str): Name of the column containing contribution data
        subtitle (str): Subtitle for the plot
        output_path (Path): Path where to save the plot
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
    """
    print(f"Plotting Client {client_id} Contribution...")
    if col_name not in df.columns:
        print(f"Skipping plot - Column '{col_name}' not found for client {client_id}.")
        return

    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping plot - No data for specified distributions: {distributions}")
            return

    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=df, 
        x="round", 
        y=col_name, 
        hue="attack_status", 
        style="distribution", 
        errorbar="sd", 
        palette={"no_attack": "navy", "with_attack": "crimson", "self_promotion": "darkorange"}
    )
    
    # Update subtitle to include filtered distributions if applicable
    if distributions is not None:
        subtitle = f"{subtitle}\nDistributions: {', '.join(distributions)}"
        
    plt.title(f"Client {client_id} Contribution over Rounds\n{subtitle}")
    plt.xlabel("Federated Round")
    plt.ylabel("Contribution Score")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Attack Status / Distribution", title_fontsize=POSTER_FONT_SIZE - 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_final_contribution_boxplot(df, max_clients, subtitle, output_path, distributions=None, 
                                   specific_round=None, use_attack_round_filtering=False, use_suptitle=False, center_y_axis=True, apply_y_limits=True):
    """Plots a boxplot of client contributions in the final round, with separate figures for each distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        max_clients (int): Maximum number of clients
        subtitle (str): Subtitle for the plot
        output_path (Path): Path where to save the plot
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
        specific_round (int, optional): If provided, use this round instead of the final round
        use_attack_round_filtering (bool): If True, filter data by attack_round for selective round attacks
        use_suptitle (bool): If True, add suptitle, axis labels and legend to plots
        center_y_axis (bool): If True, center y-axis around 0 and make symmetric limits for comparison
        apply_y_limits (bool): If True, apply cached y-axis limits for consistent scaling across plots
    """
    print("Plotting Final Round Contribution Distribution...")
    if max_clients <= 0 or 'round' not in df.columns or df['round'].empty:
        print("Skipping final round boxplot - insufficient data.")
        return

    # Handle selective round attack filtering
    if use_attack_round_filtering:
        if 'attack_round' not in df.columns:
            print("Skipping plot - 'attack_round' column not found for selective round filtering.")
            return
        
        # Get unique attack rounds (excluding None for no_attack)
        attack_rounds = sorted([r for r in df['attack_round'].unique() if r is not None])
        if not attack_rounds:
            print("Skipping plot - No attack rounds found for selective round filtering.")
            return
        
        # Get no_attack baseline data
        df_no_attack = df[df['attack_status'] == 'no_attack'].copy()
        if df_no_attack.empty:
            print("Skipping plot - No 'no_attack' baseline data found for selective round filtering.")
            return
        
        print(f"Processing selective round attacks for rounds: {attack_rounds}")
        
        # Process each attack round separately
        for attack_round in attack_rounds:
            # Filter data for this specific attack round
            df_attack_round = df[(df['attack_round'] == attack_round)].copy()
            if df_attack_round.empty:
                continue
            
            # Combine attack data with no_attack baseline for this round
            df_combined = pd.concat([df_no_attack, df_attack_round], ignore_index=True)
            
            # Recursively call this function for each attack round with specific_round set
            round_output_path = output_path.parent / "selective_rounds" / f"round_{attack_round}_{output_path.name}"
            round_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plot_final_contribution_boxplot(
                df_combined, max_clients, subtitle, round_output_path, 
                distributions, specific_round=attack_round, use_attack_round_filtering=False, 
                use_suptitle=use_suptitle, center_y_axis=center_y_axis, apply_y_limits=apply_y_limits
            )
        
        return  # Early return since we've processed all rounds recursively

    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping plot - No data for specified distributions: {distributions}")
            return

    # Determine which round to use for comparison
    if specific_round is not None:
        target_round = specific_round
        print(f"Using specific round: {target_round}")
    else:
        target_round = df['round'].max()
        print(f"Using final round: {target_round}")
    
    df_final = df[df['round'] == target_round].copy()
    
    # List distributions and their attack statuses
    for dist in df_final['distribution'].unique():
        status_in_dist = df_final[df_final['distribution'] == dist]['attack_status'].unique()
        print(f"Distribution '{dist}' has attack statuses: {status_in_dist}")
    
    # Choose which contribution columns to use based on the use case
    if specific_round is not None:
        # For specific round comparisons (including selective rounds), use direct contribution columns
        value_vars_final = [f"client_{i}_contribution" for i in range(1, max_clients + 1) 
                           if f"client_{i}_contribution" in df_final.columns]
        contrib_column_type = "contribution"
    else:
        # For final round comparisons (regular use case), use average contribution columns
        value_vars_final = [f"client_{i}_avg_contribution" for i in range(1, max_clients + 1) 
                           if f"client_{i}_avg_contribution" in df_final.columns]
        contrib_column_type = "avg_contribution"
    
    if not value_vars_final:
        print(f"Skipping final round boxplot - no {contrib_column_type} columns found.")
        return

    # DEBUG: Check which columns we're using for melting
    #print(f"Using these columns for melting: {value_vars_final}")

    df_melted_final = df_final.melt(
        id_vars=["dataset", "method", "distribution", "attack_status", "is_self_promotion", "run_id"],
        value_vars=value_vars_final, 
        var_name="client_id_str",
        value_name="contribution"
    )
    df_melted_final['client_id'] = df_melted_final['client_id_str'].str.replace(f"_{contrib_column_type}", "").str.extract('(\\d+)').astype(int)
    

    
    df_melted_final.dropna(subset=['contribution'], inplace=True)
    


    if df_melted_final.empty:
        print("Skipping final round boxplot - no data after melting/filtering.")
        return
    
    unique_distributions = df_melted_final['distribution'].unique()
    n_distributions = len(unique_distributions)
    

    if n_distributions == 0:
        print("Skipping final round boxplot - no distributions found in melted data.")
        return

    base_palette = {"no_attack": "lightsteelblue", "with_attack": "salmon", "self_promotion": "orangered"}

    # Use pre-calculated y-axis limits from global cache (optional)
    # The test script should have pre-calculated all limits before calling this function
    y_limits_for_plots = {}
    if center_y_axis and apply_y_limits:
        print("Using pre-calculated y-axis limits from global cache...")
        for dist_name in unique_distributions:
            for attack_status in df_melted_final['attack_status'].unique():
                dataset_name = df_melted_final[df_melted_final['distribution'] == dist_name]['dataset'].iloc[0] if not df_melted_final.empty else 'unknown'
                cache_key = f"{dataset_name}_{dist_name}_{attack_status}_{target_round}"
                
                if cache_key in _y_axis_limits_cache:
                    y_limits_for_plots[cache_key] = _y_axis_limits_cache[cache_key]
                    print(f"  {cache_key}: using cached limits {_y_axis_limits_cache[cache_key]}")
                else:
                    print(f"  {cache_key}: no cached limits found, using default")
                    y_limits_for_plots[cache_key] = (-0.01, 0.01)
    elif center_y_axis and not apply_y_limits:
        print("Y-axis centering enabled but cached limits disabled - using natural scaling")

    comparison_configs = [
        {
            "name": "Targeted Attack vs. No Attack",
            "statuses_to_compare": ['no_attack', 'with_attack'],
            "file_suffix": "no_vs_target_attack",
            "palette": {k: base_palette[k] for k in ['no_attack', 'with_attack'] if k in base_palette}
        },
        {
            "name": "Self-Promotion vs. No Attack",
            "statuses_to_compare": ['no_attack', 'self_promotion'],
            "file_suffix": "no_vs_self_promo",
            "palette": {k: base_palette[k] for k in ['no_attack', 'self_promotion'] if k in base_palette}
        }
    ]

    for config in comparison_configs:
        statuses_to_compare = config["statuses_to_compare"]
        file_suffix = config["file_suffix"]
        plot_title_main_segment = config["name"]
        current_palette = config["palette"]

        # --- Create separate figures for each distribution for this comparison type ---
        for dist_name in unique_distributions:
            # KEY FIX: First check if we have data for both attack statuses for this distribution
            available_statuses = df_melted_final[(df_melted_final['distribution'] == dist_name)]['attack_status'].unique()
            status_overlap = [status for status in statuses_to_compare if status in available_statuses]
            
            if len(status_overlap) < 1:
                print(f"No data for distribution '{dist_name}' for comparison '{plot_title_main_segment}'. Skipping individual plot.")
                continue
                
            # Create a filtered dataframe for this distribution and these attack statuses
            dist_df = df_melted_final[(df_melted_final['distribution'] == dist_name) & 
                                      (df_melted_final['attack_status'].isin(status_overlap))].copy()
            
            if dist_df.empty:
                print(f"No data for distribution '{dist_name}' after filtering. Skipping individual plot.")
                continue

            # Find all unique client_ids in this distribution
            unique_client_ids = dist_df['client_id'].unique()
            
            # Find which attack statuses we actually have data for
            available_statuses = dist_df['attack_status'].unique()
            
            # Check if we have at least one attack status with data
            if len(available_statuses) == 0:
                print(f"No attack statuses with data for '{dist_name}'. Skipping.")
                continue
                
            # Set categorical type for proper ordering
            dist_df['attack_status'] = pd.Categorical(dist_df['attack_status'], 
                                                     categories=statuses_to_compare, 
                                                     ordered=True)
            
            # Sort the dataframe for consistent visualization
            dist_df.sort_values(by=['client_id', 'attack_status'], inplace=True)

            # Create the plot figure and axes
            fig, ax = plt.subplots(figsize=(max(13, len(unique_client_ids) * 1.4), 8))
            
            # If we have at least one attack status, create the boxplot
            boxplot_ax = sns.boxplot(
                data=dist_df,
                x='client_id',
                y='contribution',
                hue='attack_status',
                hue_order=statuses_to_compare, # This ensures correct ordering even if we only have one status
                palette=current_palette,
                width=0.6,
                ax=ax
            )
            
            # Apply consistent y-axis limits for this specific scenario (dataset+distribution+attack_status+round)
            if center_y_axis and apply_y_limits:
                dataset_name = dist_df['dataset'].iloc[0] if 'dataset' in dist_df.columns else 'unknown'
                # Note: For individual plots, we might have multiple attack statuses, so we need to find a common limit
                # Let's use the union of all attack statuses in this plot
                all_attack_statuses_in_plot = dist_df['attack_status'].unique()
                
                # Find the maximum range across all attack statuses in this plot
                max_range = 0
                for status in all_attack_statuses_in_plot:
                    cache_key = f"{dataset_name}_{dist_name}_{status}_{target_round}"
                    if cache_key in y_limits_for_plots:
                        ymin, ymax = y_limits_for_plots[cache_key]
                        current_range = max(abs(ymin), abs(ymax))
                        max_range = max(max_range, current_range)
                
                if max_range > 0:
                    boxplot_ax.set_ylim(-max_range, max_range)
                    print(f"Applied unified y-limits for {dist_name}: [{-max_range:.6f}, {max_range:.6f}]")
            elif center_y_axis and not apply_y_limits:
                # Apply natural y-axis centering without cached limits
                apply_y_axis_centering([boxplot_ax], center_y_axis=True)
                print(f"Applied natural y-axis centering for {dist_name}")
            
            # Only show legend if we have multiple attack statuses and use_suptitle is True
            if use_suptitle:
                if len(available_statuses) > 1:
                    # Use split legend instead of default legend
                    create_split_legend(fig, ax, current_palette, statuses_to_compare)
                    round_title = f"Round {target_round}" if specific_round is not None else "Final Round"
                    contrib_title = "Client Contributions" if specific_round is not None else "Avg. Client Contributions"
                    ax.set_title(f"{round_title} {contrib_title}:\n{plot_title_main_segment}\nDistribution: {dist_name} - {subtitle}")
                else:
                    # For single attack status, add status info to title
                    status_str = available_statuses[0]
                    round_title = f"Round {target_round}" if specific_round is not None else "Final Round"
                    contrib_title = "Client Contributions" if specific_round is not None else "Avg. Client Contributions"
                    ax.set_title(f"{round_title} {contrib_title}:\n{status_str}\nDistribution: {dist_name} - {subtitle}")
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                
                ax.set_xlabel("Client ID", fontsize=POSTER_FONT_SIZE)
                ylabel = "Contribution Score" if specific_round is not None else "Average Contribution Score"
                ax.set_ylabel(ylabel, fontsize=POSTER_FONT_SIZE)
                # Increase tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE)
            else:
                # Apply split legend even when use_suptitle is False if we have multiple statuses
                if len(available_statuses) > 1:
                    create_split_legend(fig, ax, current_palette, statuses_to_compare)
                # Remove legend and axis labels if use_suptitle is False
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                ax.set_xlabel("")
                ax.set_ylabel("")
            
            # Apply tick label formatting for better readability
            ax.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE+10)
            
            ax.grid(True, linestyle='--', alpha=0.6, axis='y')

            comp_output_path = output_path.parent / f"{output_path.stem}_{dist_name}_comparative_{file_suffix}{output_path.suffix}"
            # Adjust layout to accommodate legend outside plot area
            if len(available_statuses) > 1:
                plt.tight_layout(rect=[0, 0, 1, 0.90])  # More space for external legend
            else:
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(comp_output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved comparative boxplot for '{dist_name}' ({plot_title_main_segment}) to {comp_output_path}")

        # --- Combined plot for this comparison type ---
        if n_distributions > 1:
            # Find all available attack statuses that match our comparison
            available_statuses_overall = [status for status in statuses_to_compare 
                                         if status in df_melted_final['attack_status'].unique()]
            
            if len(available_statuses_overall) < 1:
                print(f"No data for combined plot for comparison '{plot_title_main_segment}'. Skipping.")
                continue
                
            # Filter the main melted df for the current comparison statuses
            df_for_combined_plot = df_melted_final[df_melted_final['attack_status'].isin(available_statuses_overall)].copy()
            
            # Create the combined plot layout
            n_cols_combined = min(2, n_distributions)
            n_rows_combined = (n_distributions + n_cols_combined - 1) // n_cols_combined
            
            fig_combined, axes_combined_flat = plt.subplots(n_rows_combined, n_cols_combined, 
                                                           figsize=(max(18, max_clients * 1.5 * n_cols_combined), 6 * n_rows_combined), 
                                                           squeeze=False)
            axes_combined_flat = axes_combined_flat.flatten()
            
            # Collect axes for consistent y-axis centering
            valid_axes = []

            for i, dist_name_combined in enumerate(unique_distributions):
                ax_subplot = axes_combined_flat[i]
                if i >= len(axes_combined_flat): break
                
                # Get data for this distribution
                subplot_data = df_for_combined_plot[df_for_combined_plot['distribution'] == dist_name_combined].copy()
                
                if subplot_data.empty:
                    ax_subplot.set_visible(False)
                    continue

                # Find available attack statuses for this distribution
                available_statuses_subplot = [status for status in available_statuses_overall 
                                             if status in subplot_data['attack_status'].unique()]
                
                if len(available_statuses_subplot) == 0:
                    ax_subplot.set_visible(False)
                    continue
                
                # Set categorical type with proper ordering
                subplot_data['attack_status'] = pd.Categorical(subplot_data['attack_status'],
                                                             categories=statuses_to_compare,
                                                             ordered=True)
                subplot_data.sort_values(by=['client_id', 'attack_status'], inplace=True)
                
                # Create the boxplot
                boxplot_subplot_ax = sns.boxplot(
                    data=subplot_data,
                    x='client_id',
                    y='contribution',
                    hue='attack_status',
                    hue_order=statuses_to_compare,
                    palette=current_palette,
                    width=0.6,
                    ax=ax_subplot
                )
                
                # Add this subplot to valid axes list
                valid_axes.append(ax_subplot)
                
                # Apply consistent y-axis limits for this specific scenario
                if center_y_axis and apply_y_limits:
                    dataset_name = subplot_data['dataset'].iloc[0] if 'dataset' in subplot_data.columns else 'unknown'
                    # For combined plots, find the maximum range across all attack statuses in this subplot
                    all_attack_statuses_in_subplot = subplot_data['attack_status'].unique()
                    
                    max_range = 0
                    for status in all_attack_statuses_in_subplot:
                        cache_key = f"{dataset_name}_{dist_name_combined}_{status}_{target_round}"
                        if cache_key in y_limits_for_plots:
                            ymin, ymax = y_limits_for_plots[cache_key]
                            current_range = max(abs(ymin), abs(ymax))
                            max_range = max(max_range, current_range)
                    
                    if max_range > 0:
                        boxplot_subplot_ax.set_ylim(-max_range, max_range)
                elif center_y_axis and not apply_y_limits:
                    # Add this subplot to valid_axes for later centering
                    # (The centering will be applied after all subplots are created)
                    pass  # valid_axes already contains this subplot
                
                if use_suptitle:
                    ax_subplot.set_title(f"Distribution: {dist_name_combined}")
                    ax_subplot.set_xlabel("Client ID", fontsize=POSTER_FONT_SIZE)
                    ylabel = "Contribution" if specific_round is not None else "Avg. Contribution"
                    ax_subplot.set_ylabel(ylabel, fontsize=POSTER_FONT_SIZE)
                    # Increase tick label font sizes
                    ax_subplot.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE)
                else:
                    # Remove axis labels if use_suptitle is False
                    ax_subplot.set_xlabel("")
                    ax_subplot.set_ylabel("")
                    # Still apply tick label formatting for readability
                    ax_subplot.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE)
                
                ax_subplot.grid(True, linestyle='--', alpha=0.6, axis='y')
                
                # Remove individual legends from subplots (will add split legend later)
                if ax_subplot.get_legend() is not None:
                    ax_subplot.get_legend().remove()

            for j in range(i + 1, len(axes_combined_flat)):
                axes_combined_flat[j].set_visible(False)
            
            # Add split legend to the entire figure if we have multiple statuses
            if len(available_statuses_overall) > 1:
                # Map attack status to human-readable labels
                status_labels = {
                    'no_attack': 'Baseline',
                    'with_attack': 'With Target Attack', 
                    'self_promotion': 'With Self Attack'
                }
                
                left_status = statuses_to_compare[0]
                right_status = statuses_to_compare[1] if len(statuses_to_compare) > 1 else statuses_to_compare[0]
                
                left_label = status_labels.get(left_status, left_status)
                right_label = status_labels.get(right_status, right_status)
                
                left_color = current_palette.get(left_status, 'lightsteelblue')
                right_color = current_palette.get(right_status, 'salmon')
                
                # Get the position of the first subplot for alignment
                first_ax = valid_axes[0] if valid_axes else axes_combined_flat[0]
                bbox = first_ax.get_position()
                last_ax = valid_axes[-1] if valid_axes else axes_combined_flat[0]
                bbox_last = last_ax.get_position()
                
                # Position legend just above the plot area (closer to the plot)
                legend_y = max(bbox.y1, bbox_last.y1) + 0.02  # Just above the top of the plot area
                
                # Add left legend: colored square followed by black text
                fig_combined.text(bbox.x0, legend_y, '■', transform=fig_combined.transFigure, 
                                fontsize=POSTER_FONT_SIZE, verticalalignment='bottom', color=left_color, weight='bold')
                fig_combined.text(bbox.x0 + 0.025, legend_y, left_label, transform=fig_combined.transFigure, 
                                fontsize=POSTER_FONT_SIZE, verticalalignment='bottom', color='black', weight='bold')
                
                # Add right legend: black text followed by colored square
                fig_combined.text(bbox_last.x1 - 0.025, legend_y, right_label, transform=fig_combined.transFigure, 
                                fontsize=POSTER_FONT_SIZE, verticalalignment='bottom', horizontalalignment='right', 
                                color='black', weight='bold')
                fig_combined.text(bbox_last.x1, legend_y, '■', transform=fig_combined.transFigure, 
                                fontsize=POSTER_FONT_SIZE, verticalalignment='bottom', horizontalalignment='right',
                                color=right_color, weight='bold')

            # Apply y-axis centering for combined plots if requested and not using cached limits
            if center_y_axis and not apply_y_limits:
                print("DEBUG: Applying y-axis centering to combined plots (no cached limits)")
                apply_y_axis_centering(valid_axes, center_y_axis=True)

            round_title = f"Round {target_round}" if specific_round is not None else "Final Round"
            contrib_title = "Client Contributions" if specific_round is not None else "Avg. Client Contributions"
            
            if use_suptitle:
                fig_combined.suptitle(f"{round_title} {contrib_title}: {plot_title_main_segment} (All Distributions)\n{subtitle}", 
                                     fontsize=POSTER_FONT_SIZE + 2)
                
                # Adjust layout to accommodate split legend
                plt.tight_layout(rect=[0, 0.02, 1, 0.92])
            else:
                # Adjust layout to accommodate split legend even without suptitle
                plt.tight_layout(rect=[0, 0.02, 1, 0.95]) 
            combined_comp_output_path = output_path.parent / f"{output_path.stem}_combined_comparative_{file_suffix}{output_path.suffix}"
            plt.savefig(combined_comp_output_path)
            plt.close(fig_combined)
            print(f"Saved combined comparative boxplot ({plot_title_main_segment}) to {combined_comp_output_path}")


def plot_contribution_comparison(df_attack, df_no_attack, subtitle, output_path, is_self_promotion=False, center_y_axis=True):
    """Plot contribution comparison between attack and no-attack scenarios.
    
    For regular attacks, compares target client vs others.
    For self-promotion attacks, compares attacker vs others.
    
    Args:
        df_attack (pd.DataFrame): DataFrame with attack results
        df_no_attack (pd.DataFrame): DataFrame with no-attack results
        subtitle (str): Subtitle for the plot
        output_path (Path): Path to save the plot
        is_self_promotion (bool): Whether this is a self-promotion attack
        center_y_axis (bool): If True, center y-axis around 0 and make symmetric limits for comparison
    """
    print("Plotting Contribution Comparison...")
    
    # Determine the number of clients from the column names
    client_cols_attack = [col for col in df_attack.columns if col.startswith("client_") and col.endswith("_contribution")]
    client_cols_no_attack = [col for col in df_no_attack.columns if col.startswith("client_") and col.endswith("_contribution")]
    
    if not client_cols_attack or not client_cols_no_attack:
        print(f"Skipping contribution comparison - No client contribution columns found.")
        return
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # For the attack scenario
    if is_self_promotion:
        # For self-promotion: compare attacker vs. average of others
        attacker_col = f"client_{ATTACKER_CLIENT_ID}_contribution"
        other_client_cols = [col for col in client_cols_attack if col != attacker_col]
        
        if not other_client_cols:
            print(f"Skipping contribution comparison for self-promotion - need at least 2 clients.")
            return
        
        # Calculate average of other clients
        df_attack['avg_other_clients_contribution'] = df_attack[other_client_cols].mean(axis=1)
        
        # Plot attacker vs. average of others
        df_attack.plot(x='round', y=[attacker_col, 'avg_other_clients_contribution'], 
                      ax=axes[0], style=['-o', '--x'], 
                      color=['orangered', 'darkgray'])
        
        axes[0].set_title("With Self-Promotion Attack")
        axes[0].set_ylabel("Contribution Score")
        axes[0].set_xlabel("Round")
        axes[0].legend(["Self-Promoter (Attacker)", "Avg. Other Clients"])
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Calculate and plot individual client contributions
        colors = plt.cm.tab10(np.linspace(0, 1, len(client_cols_attack)))
        for i, col in enumerate(client_cols_attack):
            client_id = int(col.split('_')[1])
            if client_id == ATTACKER_CLIENT_ID:
                # Use a distinct color and line style for the attacker
                axes[0].plot(df_attack['round'], df_attack[col], 
                           linestyle='-', linewidth=2, marker='o', markersize=7,
                           color='orangered', alpha=0.8, label=f"Client {client_id} (Self-Promoter)")
            else:
                # Plot regular clients with thinner lines and different markers
                axes[0].plot(df_attack['round'], df_attack[col], 
                           linestyle='--', linewidth=1, marker='.', markersize=5,
                           color=colors[i % len(colors)], alpha=0.5, label=f"Client {client_id}")
        
    else:
        # For regular attacks: compare target vs. attacker vs. others
        target_col = f"client_{TARGET_CLIENT_ID}_contribution"
        attacker_col = f"client_{ATTACKER_CLIENT_ID}_contribution"
        other_client_cols = [col for col in client_cols_attack 
                           if col != target_col and col != attacker_col]
        
        # Calculate average of other (non-target, non-attacker) clients
        if other_client_cols:
            df_attack['avg_other_clients_contribution'] = df_attack[other_client_cols].mean(axis=1)
            
            # Plot target, attacker, and average of others
            df_attack.plot(x='round', y=[target_col, attacker_col, 'avg_other_clients_contribution'], 
                         ax=axes[0], style=['-o', '--x', '-.+'], 
                         color=['royalblue', 'salmon', 'darkgray'])
            
            axes[0].set_title("With Target-Based Attack")
            axes[0].set_ylabel("Contribution Score")
            axes[0].set_xlabel("Round")
            axes[0].legend(["Target Client", "Attacker Client", "Avg. Other Clients"])
        else:
            # Just plot target and attacker if there are no other clients
            df_attack.plot(x='round', y=[target_col, attacker_col], 
                         ax=axes[0], style=['-o', '--x'], 
                         color=['royalblue', 'salmon'])
            
            axes[0].set_title("With Target-Based Attack")
            axes[0].set_ylabel("Contribution Score")
            axes[0].set_xlabel("Round")
            axes[0].legend(["Target Client", "Attacker Client"])
        
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Calculate and plot individual client contributions
        colors = plt.cm.tab10(np.linspace(0, 1, len(client_cols_attack)))
        for i, col in enumerate(client_cols_attack):
            client_id = int(col.split('_')[1])
            if client_id == TARGET_CLIENT_ID:
                # Use a distinct color and line style for the target
                axes[0].plot(df_attack['round'], df_attack[col], 
                           linestyle='-', linewidth=2, marker='o', markersize=7,
                           color='royalblue', alpha=0.8, label=f"Client {client_id} (Target)")
            elif client_id == ATTACKER_CLIENT_ID:
                # Use a distinct color and line style for the attacker
                axes[0].plot(df_attack['round'], df_attack[col], 
                           linestyle='--', linewidth=2, marker='x', markersize=7,
                           color='salmon', alpha=0.8, label=f"Client {client_id} (Attacker)")
            else:
                # Plot regular clients with thinner lines and different markers
                axes[0].plot(df_attack['round'], df_attack[col], 
                           linestyle=':', linewidth=1, marker='.', markersize=5,
                           color=colors[i % len(colors)], alpha=0.5, label=f"Client {client_id}")
    
    # For the no-attack scenario
    if is_self_promotion:
        # For self-promotion comparison: highlight the would-be self-promoter in no-attack scenario
        would_be_promoter_col = f"client_{ATTACKER_CLIENT_ID}_contribution"
        other_client_cols = [col for col in client_cols_no_attack if col != would_be_promoter_col]
        
        if other_client_cols:
            df_no_attack['avg_other_clients_contribution'] = df_no_attack[other_client_cols].mean(axis=1)
            
            # Plot would-be self-promoter vs. average of others
            df_no_attack.plot(x='round', y=[would_be_promoter_col, 'avg_other_clients_contribution'], 
                           ax=axes[1], style=['-o', '--x'], 
                           color=['darkgray', 'darkgray'])
            
            axes[1].set_title("Without Attack")
            axes[1].set_ylabel("Contribution Score")
            axes[1].set_xlabel("Round")
            axes[1].legend([f"Client {ATTACKER_CLIENT_ID} (No Attack)", "Avg. Other Clients"])
        else:
            # Just plot would-be self-promoter if there are no other clients
            df_no_attack.plot(x='round', y=[would_be_promoter_col], 
                           ax=axes[1], style=['-o'], 
                           color=['darkgray'])
            
            axes[1].set_title("Without Attack")
            axes[1].set_ylabel("Contribution Score")
            axes[1].set_xlabel("Round")
            axes[1].legend([f"Client {ATTACKER_CLIENT_ID} (No Attack)"])
        
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # Calculate and plot individual client contributions for no-attack
        colors = plt.cm.tab10(np.linspace(0, 1, len(client_cols_no_attack)))
        for i, col in enumerate(client_cols_no_attack):
            client_id = int(col.split('_')[1])
            if client_id == ATTACKER_CLIENT_ID:
                # Use a distinct style for the would-be self-promoter
                axes[1].plot(df_no_attack['round'], df_no_attack[col], 
                          linestyle='-', linewidth=2, marker='o', markersize=7,
                          color='darkgray', alpha=0.8, label=f"Client {client_id}")
            else:
                # Plot regular clients with thinner lines
                axes[1].plot(df_no_attack['round'], df_no_attack[col], 
                          linestyle='--', linewidth=1, marker='.', markersize=5,
                          color=colors[i % len(colors)], alpha=0.5, label=f"Client {client_id}")
    else:
        # For regular attack comparison: highlight the would-be target in no-attack scenario
        would_be_target_col = f"client_{TARGET_CLIENT_ID}_contribution"
        would_be_attacker_col = f"client_{ATTACKER_CLIENT_ID}_contribution"
        other_client_cols = [col for col in client_cols_no_attack 
                           if col != would_be_target_col and col != would_be_attacker_col]
        
        if other_client_cols:
            df_no_attack['avg_other_clients_contribution'] = df_no_attack[other_client_cols].mean(axis=1)
            
            # Plot would-be target, would-be attacker, and average of others
            df_no_attack.plot(x='round', y=[would_be_target_col, would_be_attacker_col, 'avg_other_clients_contribution'], 
                           ax=axes[1], style=['-o', '--x', '-.+'], 
                           color=['darkgray', 'darkgray', 'darkgray'])
            
            axes[1].set_title("Without Attack")
            axes[1].set_ylabel("Contribution Score")
            axes[1].set_xlabel("Round")
            axes[1].legend([f"Client {TARGET_CLIENT_ID} (No Attack)", 
                          f"Client {ATTACKER_CLIENT_ID} (No Attack)", 
                          "Avg. Other Clients"])
        else:
            # Just plot would-be target and attacker if there are no other clients
            df_no_attack.plot(x='round', y=[would_be_target_col, would_be_attacker_col], 
                           ax=axes[1], style=['-o', '--x'], 
                           color=['darkgray', 'darkgray'])
            
            axes[1].set_title("Without Attack")
            axes[1].set_ylabel("Contribution Score")
            axes[1].set_xlabel("Round")
            axes[1].legend([f"Client {TARGET_CLIENT_ID} (No Attack)", 
                          f"Client {ATTACKER_CLIENT_ID} (No Attack)"])
        
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # Calculate and plot individual client contributions for no-attack
        colors = plt.cm.tab10(np.linspace(0, 1, len(client_cols_no_attack)))
        for i, col in enumerate(client_cols_no_attack):
            client_id = int(col.split('_')[1])
            # Use a gray palette for all clients in no-attack scenario
            axes[1].plot(df_no_attack['round'], df_no_attack[col], 
                       linestyle='--', linewidth=1, marker='.', markersize=5,
                       color=colors[i % len(colors)], alpha=0.5, label=f"Client {client_id}")
    
    # Add an overall title
    if is_self_promotion:
        fig.suptitle(f"Self-Promotion Attack: Contribution Comparison\n{subtitle}", fontsize=POSTER_FONT_SIZE)
    else:
        fig.suptitle(f"Target-Based Attack: Contribution Comparison\n{subtitle}", fontsize=POSTER_FONT_SIZE)
    
    # Apply y-axis centering if requested
    if center_y_axis:
        apply_y_axis_centering(axes, center_y_axis)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()
    print(f"Saved contribution comparison to {output_path}")


def plot_attack_params_heatmap(df, subtitle, output_path, attack_type='with_attack', distributions=None):
    """Creates heatmap visualizations of best attack parameters (init_method vs attack_scale)
    for a specific attack type and distributions.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data
        subtitle (str): Subtitle for the plot
        output_path (Path): Base path where to save the plots (actual filenames will include attack type and distribution)
        attack_type (str): Type of attack to analyze ('with_attack' or 'self_promotion')
        distributions (list, optional): List of distributions to include. If None, all distributions are processed.
    """
    if 'best_attack_scale' not in df.columns or 'best_init_method' not in df.columns:
        print(f"Skipping attack params heatmap - required columns 'best_attack_scale' or 'best_init_method' not found.")
        return
        
    print(f"Plotting Attack Parameters Heatmap for {attack_type}...")
    
    # Filter by attack type
    df = df[df['attack_status'] == attack_type].copy()
    if df.empty:
        print(f"Skipping attack params heatmap - no data for attack type '{attack_type}'")
        return
    
    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping attack params heatmap - no data for specified distributions: {distributions}")
            return
    
    # Get the first round where attacks start (to handle both current and future scenarios)
    # In current implementation, attacks start from round 2
    min_round_with_attacks = df.dropna(subset=['best_attack_scale', 'best_init_method']).get('round', pd.Series()).min()
    if pd.isna(min_round_with_attacks):
        print(f"Skipping attack params heatmap - could not determine first attack round.")
        return
        
    print(f"First round with attack parameters: {min_round_with_attacks}")
    
    # Filter out rows without attack parameters (typically round 1 in current implementation)
    df = df.dropna(subset=['best_attack_scale', 'best_init_method']).copy()
    
    # Get unique distributions in the filtered data
    unique_distributions = df['distribution'].unique()
    
    # Process each distribution separately
    for dist_name in unique_distributions:
        dist_df = df[df['distribution'] == dist_name].copy()
        if dist_df.empty:
            continue
            
        # Get unique values for attack scales and init methods for this distribution
        attack_scales = sorted(dist_df['best_attack_scale'].unique())
        init_methods = sorted(dist_df['best_init_method'].unique())
        
        if not attack_scales or not init_methods:
            print(f"Skipping heatmap for {dist_name} - missing attack parameters")
            continue
            
        # Create a crosstab to count frequency of each parameter combination
        param_counts = pd.crosstab(
            dist_df['best_attack_scale'], 
            dist_df['best_init_method'],
            margins=False
        )
        
        # Ensure all combinations are represented (fill missing with 0)
        for scale in attack_scales:
            if scale not in param_counts.index:
                param_counts.loc[scale] = 0
                
        for method in init_methods:
            if method not in param_counts.columns:
                param_counts[method] = 0
        
        # Sort by index (attack scale) and columns (init methods)
        param_counts = param_counts.sort_index()
        param_counts = param_counts[sorted(param_counts.columns)]
        
        # Create the heatmap - improved version with better formatting
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap
        sns.heatmap(
            param_counts,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={'label': 'Frequency', 'shrink': 0.8}
        )
        
        # Rotate x-axis labels by 45 degrees for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Rotate y-axis labels by 30 degrees for better readability
        plt.yticks(rotation=30, va='center')
        
        # Update subtitle to include distribution
        dist_subtitle = f"{subtitle}\nDistribution: {dist_name}"
        
        # Set title based on attack type
        attack_type_title = "Target-Based Attack" if attack_type == "with_attack" else "Self-Promotion Attack"
        plt.title(f"Attack Parameter Combinations: {attack_type_title}\n{dist_subtitle}")
        
        plt.xlabel("Initialization Method")
        plt.ylabel("Attack Scale")
        
        # Adjust layout to ensure everything fits
        plt.tight_layout(pad=1.2)
        
        # Create output path for this distribution with .png extension
        dist_output_path = output_path.parent / f"{output_path.stem}_{attack_type}_{dist_name}.png"
        plt.savefig(dist_output_path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        print(f"Saved attack params heatmap for '{dist_name}' ({attack_type}) to {dist_output_path}")
    
    # If there are multiple distributions, create a combined view with subplots
    if len(unique_distributions) > 1:
        n_dists = len(unique_distributions)
        n_cols = min(2, n_dists)
        n_rows = (n_dists + n_cols - 1) // n_cols
        
        # Create figure with increased spacing between subplots
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(n_cols * 7, n_rows * 6), # Increased size
                                squeeze=False, 
                                # Add more horizontal and vertical spacing between subplots
                                gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
        axes = axes.flatten()
        
        for i, dist_name in enumerate(unique_distributions):
            if i >= len(axes):
                break
                
            dist_df = df[df['distribution'] == dist_name].copy()
            if dist_df.empty:
                axes[i].set_visible(False)
                continue
                
            # Create crosstab for this distribution
            param_counts = pd.crosstab(
                dist_df['best_attack_scale'], 
                dist_df['best_init_method'],
                margins=False
            )
            
            # Fill in missing combinations
            attack_scales = sorted(df['best_attack_scale'].unique())
            init_methods = sorted(df['best_init_method'].unique())
            
            for scale in attack_scales:
                if scale not in param_counts.index:
                    param_counts.loc[scale] = 0
                    
            for method in init_methods:
                if method not in param_counts.columns:
                    param_counts[method] = 0
            
            # Sort by index and columns
            param_counts = param_counts.sort_index()
            param_counts = param_counts[sorted(param_counts.columns)]
            
            # Create heatmap for this subplot with improved formatting
            im = sns.heatmap(
                param_counts,
                annot=True,
                fmt="d",
                cmap="YlOrRd",
                ax=axes[i],
                cbar_kws={'label': 'Frequency', 'shrink': 0.8}
            )
            
            # Rotate x-axis labels
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            
            # Rotate y-axis labels
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=30, va='center')
            
            axes[i].set_title(f"Distribution: {dist_name}", pad=10)
            axes[i].set_xlabel("Initialization Method", labelpad=10)
            axes[i].set_ylabel("Attack Scale", labelpad=10)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        # Set overall title
        attack_type_title = "Target-Based Attack" if attack_type == "with_attack" else "Self-Promotion Attack"
        fig.suptitle(f"Attack Parameter Combinations: {attack_type_title}\n{subtitle}", 
                    fontsize=POSTER_FONT_SIZE,
                    y=0.98)  # Move the title up a bit to avoid overlap
        
        # Adjust layout with more space for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Create output path for combined plot with .png extension
        combined_output_path = output_path.parent / f"{output_path.stem}_{attack_type}_combined.png"
        plt.savefig(combined_output_path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        print(f"Saved combined attack params heatmap ({attack_type}) to {combined_output_path}")


def plot_client_contribution_percentage_comparison(df_dataset_method, max_clients, subtitle_base, output_dir_base, specific_distributions=None):
    """
    Plots the percentage change in client contributions for attack scenarios
    compared to the 'no_attack' scenario, on a round-by-round basis.
    A single plot is generated per distribution and per attack type comparison 
    (e.g., 'with_attack' vs 'no_attack'), showing all clients.
    """
    print(f"Plotting All Clients Contribution Percentage Comparison for {subtitle_base}...")

    df_to_process = df_dataset_method.copy()
    df_to_process['round'] = df_to_process['round'].astype(int)

    unique_dist_series = df_to_process['distribution'].unique()
    if specific_distributions:
        distributions_to_plot = [dist for dist in specific_distributions if dist in unique_dist_series]
    else:
        distributions_to_plot = list(unique_dist_series)

    if not distributions_to_plot:
        print("No distributions to plot for percentage comparison based on current filters.")
        return

    # Define distinct colors and line styles for clients
    # More can be added if max_clients > 5
    client_styles = [
        {'color': 'blue', 'linestyle': '-'},
        {'color': 'green', 'linestyle': '--'},
        {'color': 'red', 'linestyle': ':'},
        {'color': 'purple', 'linestyle': '-.'},
        {'color': 'orange', 'linestyle': '-'}
    ]

    for dist_name in distributions_to_plot:
        df_dist = df_to_process[df_to_process['distribution'] == dist_name].copy()
        if df_dist.empty:
            print(f"No data for distribution {dist_name} in percentage comparison.")
            continue

        df_no_attack_dist = df_dist[df_dist['attack_status'] == 'no_attack'].copy()
        if df_no_attack_dist.empty:
            print(f"No 'no_attack' data for distribution {dist_name}. Skipping percentage plots for this distribution.")
            continue

        attack_scenarios_dist = {
            "with_attack": df_dist[df_dist['attack_status'] == 'with_attack'].copy(),
            "self_promotion": df_dist[df_dist['attack_status'] == 'self_promotion'].copy()
        }

        for attack_key, df_attack_scenario_dist in attack_scenarios_dist.items():
            if df_attack_scenario_dist.empty:
                # print(f"No '{attack_key}' data for distribution {dist_name}.")
                continue

            fig, ax = plt.subplots(figsize=(14, 8))
            legend_handles = []
            has_data_for_plot = False

            for client_id in range(1, max_clients + 1):
                actual_contrib_col = f"client_{client_id}_contribution"
                if actual_contrib_col not in df_no_attack_dist.columns or actual_contrib_col not in df_attack_scenario_dist.columns:
                    # print(f"Contribution column {actual_contrib_col} not found for client {client_id} in no_attack or {attack_key} for dist {dist_name}. Skipping client.")
                    continue

                client_no_attack_agg = df_no_attack_dist[['round', actual_contrib_col]].groupby('round', as_index=False)[actual_contrib_col].mean()
                client_no_attack_agg = client_no_attack_agg.rename(columns={actual_contrib_col: 'contrib_no_attack'})

                if client_no_attack_agg.empty:
                    # print(f"No aggregated 'no_attack' data for client {client_id}, distribution {dist_name}.")
                    continue
                
                client_attack_agg = df_attack_scenario_dist[['round', actual_contrib_col]].groupby('round', as_index=False)[actual_contrib_col].mean()
                client_attack_agg = client_attack_agg.rename(columns={actual_contrib_col: f'contrib_{attack_key}'})

                if client_attack_agg.empty:
                    # print(f"No aggregated '{attack_key}' data for client {client_id}, distribution {dist_name}.")
                    continue
                
                merged_df = pd.merge(client_no_attack_agg, client_attack_agg, on='round', how='inner')
                if merged_df.empty:
                    # print(f"No common rounds for client {client_id}, {attack_key} vs no_attack, distribution {dist_name}.")
                    continue

                baseline = merged_df['contrib_no_attack']
                attack_values = merged_df[f'contrib_{attack_key}']
                
                percentage_change = np.zeros_like(baseline, dtype=float)
                non_zero_baseline_mask = np.abs(baseline) > 1e-9

                percentage_change[non_zero_baseline_mask] = \
                    ((attack_values[non_zero_baseline_mask] - baseline[non_zero_baseline_mask]) /
                     np.abs(baseline[non_zero_baseline_mask])) * 100

                zero_baseline_mask = ~non_zero_baseline_mask
                large_percentage_magnitude = 200000 
                
                positive_change_at_zero_baseline = zero_baseline_mask & (attack_values > 1e-9)
                percentage_change[positive_change_at_zero_baseline] = large_percentage_magnitude
                
                negative_change_at_zero_baseline = zero_baseline_mask & (attack_values < -1e-9)
                percentage_change[negative_change_at_zero_baseline] = -large_percentage_magnitude
                
                merged_df['percentage_change'] = percentage_change

                # Set percentage change for round 1 to 0, as no attack is performed
                if 1 in merged_df['round'].values:
                    merged_df.loc[merged_df['round'] == 1, 'percentage_change'] = 0
                
                has_data_for_plot = True

                style = client_styles[(client_id -1) % len(client_styles)] # Cycle through styles
                line, = ax.plot(merged_df['round'], merged_df['percentage_change'], 
                                marker='o', markersize=5, 
                                linestyle=style['linestyle'], color=style['color'], 
                                label=f"Client {client_id}")
                legend_handles.append(line)
            
            if not has_data_for_plot:
                plt.close(fig) # Close the figure if no data was plotted
                # print(f"No data to plot for {attack_key} vs no_attack, distribution {dist_name}. Skipping plot.")
                continue

            ax.set_yscale('symlog', linthresh=10)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter())
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Ensure integer round numbers

            ax.set_xlabel("Round", fontsize=POSTER_FONT_SIZE - 2)
            ax.set_ylabel("Contribution Change (%) vs. No Attack", fontsize=POSTER_FONT_SIZE - 2)
            
            attack_type_str = "Targeted Attack" if attack_key == "with_attack" else "Self-Promotion"
            plot_title = (f"All Clients: Contribution % Change\n"
                          f"{attack_type_str} vs. No Attack\n"
                          f"{subtitle_base}, Distribution: {dist_name}")
            ax.set_title(plot_title, fontsize=POSTER_FONT_SIZE)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE - 4)
            
            if legend_handles:
                ax.legend(handles=legend_handles, title="Clients", fontsize=POSTER_FONT_SIZE - 4, title_fontsize=POSTER_FONT_SIZE - 3)

            dist_plot_output_dir = Path(output_dir_base) / dist_name
            dist_plot_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Changed filename to reflect all clients
            plot_filename = dist_plot_output_dir / f"all_clients_contrib_pct_change_{attack_key}_{dist_name}.png"
            
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved all clients percentage comparison plot to {plot_filename}")


def plot_method_comparison_contribution_changes(df, max_clients, subtitle, output_path, distributions=None, 
                                              specific_round=None, use_attack_round_filtering=False, use_suptitle=False, return_data=False):
    """Plots boxplots comparing contribution score changes between evaluation methods (loo vs shapley) for each client.
    
    Change is calculated as: attack_contribution - no_attack_contribution (baseline)
    This shows how different evaluation methods capture the impact of attacks on each client.
    Creates separate plots for each distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data with both loo and shapley methods
        max_clients (int): Maximum number of clients
        subtitle (str): Subtitle for the plot
        output_path (Path): Path where to save the plot
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
        specific_round (int, optional): If provided, use this round instead of the final round
        use_attack_round_filtering (bool): If True, filter data by attack_round for selective round attacks
        use_suptitle (bool): If True, add suptitle to plots
        return_data (bool): If True, return the data used for plotting
        
    Returns:
        pd.DataFrame or None: DataFrame with plotting data if return_data=True, otherwise None
    """
    print("Plotting Method Comparison of Contribution Changes...")
    
    # Handle selective round attack filtering
    if use_attack_round_filtering:
        if 'attack_round' not in df.columns:
            print("Skipping plot - 'attack_round' column not found for selective round filtering.")
            return
        
        # Get unique attack rounds (excluding None for no_attack)
        attack_rounds = sorted([r for r in df['attack_round'].unique() if r is not None])
        if not attack_rounds:
            print("Skipping plot - No attack rounds found for selective round filtering.")
            return
        
        # Get no_attack baseline data
        df_no_attack = df[df['attack_status'] == 'no_attack'].copy()
        if df_no_attack.empty:
            print("Skipping plot - No 'no_attack' baseline data found for selective round filtering.")
            return
        
        print(f"Processing selective round attacks for rounds: {attack_rounds}")
        
        # Collect all data from recursive calls if return_data is requested
        all_recursive_data = []
        
        # Process each attack round separately
        for attack_round in attack_rounds:
            # Filter data for this specific attack round
            df_attack_round = df[(df['attack_round'] == attack_round)].copy()
            if df_attack_round.empty:
                continue
            
            # Combine attack data with no_attack baseline for this round
            df_combined = pd.concat([df_no_attack, df_attack_round], ignore_index=True)
            
            # Recursively call this function for each attack round with specific_round set
            round_output_path = output_path.parent / "selective_rounds" / f"round_{attack_round}_method_comparison_{output_path.name}"
            round_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            round_data = plot_method_comparison_contribution_changes(
                df_combined, max_clients, subtitle, round_output_path, 
                distributions, specific_round=attack_round, use_attack_round_filtering=False, use_suptitle=use_suptitle, return_data=return_data
            )
            
            # Collect data if return_data is requested
            if return_data and round_data is not None and not round_data.empty:
                # Add round information to the data
                round_data = round_data.copy()
                round_data['attack_round'] = attack_round
                all_recursive_data.append(round_data)
        
        # Return combined data from all rounds if requested
        if return_data:
            if all_recursive_data:
                return pd.concat(all_recursive_data, ignore_index=True)
            else:
                return pd.DataFrame()
        else:
            return None
    
    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping plot - No data for specified distributions: {distributions}")
            return
    
    # Check that we have both methods
    available_methods = df['method'].unique()
    if len(available_methods) < 2:
        print(f"Skipping method comparison - need at least 2 methods, found: {available_methods}")
        return
    
    if 'loo' not in available_methods or 'shapley' not in available_methods:
        print(f"Skipping method comparison - need both 'loo' and 'shapley' methods, found: {available_methods}")
        return None if not return_data else pd.DataFrame()
    
    # Determine which round to use for comparison
    if specific_round is not None:
        target_round = specific_round
        print(f"Using specific round: {target_round}")
        contrib_column_suffix = "contribution"
    else:
        target_round = df['round'].max()
        print(f"Using final round: {target_round}")
        contrib_column_suffix = "avg_contribution"
    
    df_final = df[df['round'] == target_round].copy()
    
    if df_final.empty:
        print("Skipping method comparison - no final round data")
        return None if not return_data else pd.DataFrame()
    
    # Check for required attack statuses
    required_statuses = ['no_attack', 'with_attack', 'self_promotion']
    available_statuses = df_final['attack_status'].unique()
    
    attack_comparisons = []
    if 'no_attack' in available_statuses and 'with_attack' in available_statuses:
        attack_comparisons.append(('with_attack', 'Targeted Attack'))
    if 'no_attack' in available_statuses and 'self_promotion' in available_statuses:
        attack_comparisons.append(('self_promotion', 'Self-Promotion'))
    
    if not attack_comparisons:
        print("Skipping method comparison - need 'no_attack' and at least one attack type")
        return None if not return_data else pd.DataFrame()
    
    # Collect all data for return if requested
    all_plotting_data = []
    
    # Process each attack comparison
    for attack_status, attack_name in attack_comparisons:
        print(f"Processing {attack_name} vs No Attack comparison...")
        
        # Calculate contribution changes for each client, method, and distribution
        change_data = []
        
        for distribution in df_final['distribution'].unique():
            for method in ['loo', 'shapley']:
                # Get no_attack baseline data
                df_baseline = df_final[
                    (df_final['distribution'] == distribution) & 
                    (df_final['method'] == method) & 
                    (df_final['attack_status'] == 'no_attack')
                ].copy()
                
                # Get attack data
                df_attack = df_final[
                    (df_final['distribution'] == distribution) & 
                    (df_final['method'] == method) & 
                    (df_final['attack_status'] == attack_status)
                ].copy()
                
                if df_baseline.empty or df_attack.empty:
                    continue
                
                # Calculate changes for each client, pairing runs properly (not Cartesian product)
                for client_id in range(1, max_clients + 1):
                    contrib_col = f"client_{client_id}_{contrib_column_suffix}"
                    
                    if contrib_col not in df_baseline.columns or contrib_col not in df_attack.columns:
                        continue
                    
                    # Get baseline and attack contributions, ensuring they're aligned by run
                    baseline_data = df_baseline[['run_id', contrib_col]].dropna()
                    attack_data = df_attack[['run_id', contrib_col]].dropna()
                    
                    # Merge on run_id to pair corresponding runs
                    merged_data = pd.merge(baseline_data, attack_data, on='run_id', suffixes=('_baseline', '_attack'))
                    
                    # Calculate changes for each paired run
                    for _, row in merged_data.iterrows():
                        contribution_change = row[f'{contrib_col}_attack'] - row[f'{contrib_col}_baseline']
                        
                        change_data.append({
                            'client_id': client_id,
                            'method': method,
                            'distribution': distribution,
                            'contribution_change': contribution_change,
                            'attack_type': attack_name,
                            'run_id': row['run_id']
                        })
        
        if not change_data:
            print(f"No data for {attack_name} comparison")
            continue
        
        # Convert to DataFrame
        changes_df = pd.DataFrame(change_data)
        
        # Add to all_plotting_data if return_data is requested
        if return_data:
            all_plotting_data.append(changes_df)
        
        # Create separate plots for each distribution
        unique_distributions = changes_df['distribution'].unique()
        
        for dist_name in unique_distributions:
            dist_data = changes_df[changes_df['distribution'] == dist_name]
            
            if dist_data.empty:
                continue
            
            # Create the boxplot for this distribution
            fig, ax = plt.subplots(figsize=(max(12, max_clients * 1.5), 8))
            
            # Use same color palette as other plots  
            method_palette = {'loo': 'lightsteelblue', 'shapley': 'salmon'}
            
            boxplot_method_ax = sns.boxplot(
                data=dist_data,
                x='client_id',
                y='contribution_change',
                hue='method',
                palette=method_palette,
                ax=ax,
                showfliers=False,  # Hide outliers for cleaner view
                whis=0
            )
            
            # Apply y-axis centering for consistent comparison
            apply_y_axis_centering([ax], center_y_axis=True)
            
            # Use split legend design like other boxplot functions
            methods_in_data = dist_data['method'].unique()
            if len(methods_in_data) > 1:
                # Remove the default legend
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                
                # Create custom legend for methods positioned outside the plot area
                left_color = method_palette.get('loo', 'lightsteelblue')
                right_color = method_palette.get('shapley', 'salmon')
                
                # Get plot area boundaries for alignment
                bbox = ax.get_position()
                
                # Position legend just above the plot area (closer to the plot)
                legend_y = bbox.y1 + 0.02  # Just above the top of the plot area
                
                # Add left legend: colored square followed by black text
                fig.text(bbox.x0-0.085, legend_y, '■', transform=fig.transFigure, 
                        fontsize=POSTER_FONT_SIZE+10, verticalalignment='bottom', color=left_color, weight='bold')
                fig.text(bbox.x0 - 0.040, legend_y, 'LOO', transform=fig.transFigure, 
                        fontsize=POSTER_FONT_SIZE+10, verticalalignment='bottom', color='black', weight='bold', family='serif')
                
                # Add right legend: black text followed by colored square
                fig.text(bbox.x1 +0.040, legend_y, 'GTG', transform=fig.transFigure, 
                        fontsize=POSTER_FONT_SIZE+10, verticalalignment='bottom', horizontalalignment='right',
                        color='black', weight='bold', family='serif')
                fig.text(bbox.x1 + 0.085, legend_y, '■', transform=fig.transFigure, 
                        fontsize=POSTER_FONT_SIZE+10, verticalalignment='bottom', horizontalalignment='right',
                        color=right_color, weight='bold')
            
            # Set title based on use_suptitle parameter
            if use_suptitle:
                fig.suptitle(f"Method Comparison: Contribution Changes\n{attack_name} vs No Attack\n{subtitle}", 
                            fontsize=POSTER_FONT_SIZE)
                ax.set_title(f"Distribution: {dist_name}")
                ax.set_xlabel("Client ID", fontsize=POSTER_FONT_SIZE)
                # Set ylabel based on round type
                if specific_round is not None:
                    ylabel = f"Contribution Change (Round {specific_round})"
                else:
                    ylabel = "Avg. Contribution Change (Attack - Baseline)"
                ax.set_ylabel(ylabel, fontsize=POSTER_FONT_SIZE)
                # Apply tick label formatting
                ax.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE+10)
            else:
                # Remove titles and labels but keep the split legend
                ax.set_xlabel("")
                ax.set_ylabel("")
                # The split legend was already applied above, regardless of use_suptitle
            
            ax.grid(True, linestyle='--', alpha=0.6, axis='y')
            
            # Add horizontal line at y=0 for reference
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
            
            # Adjust layout to accommodate legend outside plot area
            if len(methods_in_data) > 1:
                plt.tight_layout(rect=[0, 0.02, 1, 0.92])
            else:
                plt.tight_layout()
            
            # Apply tick label formatting after layout adjustment to ensure it's not overridden
            ax.tick_params(axis='both', which='major', labelsize=POSTER_FONT_SIZE+10)
            
            # Save plot for this distribution
            attack_suffix = attack_status.replace('_', '')  # 'withattack' or 'selfpromotion'
            if specific_round is not None:
                plot_filename = output_path.parent / f"method_comparison_contribution_changes_{attack_suffix}_{dist_name}_round_{specific_round}.png"
            else:
                plot_filename = output_path.parent / f"method_comparison_contribution_changes_{attack_suffix}_{dist_name}.png"
            
            plt.savefig(plot_filename, bbox_inches='tight', dpi=120)
            plt.close()
            print(f"Saved method comparison plot for {dist_name} to {plot_filename}")
    
    # Return data if requested
    if return_data:
        if all_plotting_data:
            return pd.concat(all_plotting_data, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        return None


def plot_method_comparison_global_loss_changes(df, subtitle, output_path, distributions=None, 
                                               use_attack_round_filtering=False, use_suptitle=False, 
                                               show_variance=True, return_data=False):
    """Plots line plots showing global loss over rounds comparing no_attack vs loo attack vs shapley attack.
    
    Creates separate plots for each dataset/distribution and attack type (self-promotion vs target-decrease).
    Shows global loss evolution over 5 rounds with optional variance bands.
    
    Args:
        df (pd.DataFrame): DataFrame containing results data with both loo and shapley methods
        subtitle (str): Subtitle for the plot
        output_path (Path): Path where to save the plot
        distributions (list, optional): List of distributions to include. If None, all distributions are plotted.
        use_attack_round_filtering (bool): If True, handle selective round attack data
        use_suptitle (bool): If True, add suptitle and labels to plots
        show_variance (bool): If True, show shaded variance bands around lines
        return_data (bool): If True, return the data used for plotting
        
    Returns:
        pd.DataFrame or None: DataFrame with plotting data if return_data=True, otherwise None
    """
    print("Plotting Method Comparison: Global Loss Over Rounds...")
    
    # Check for required columns
    loss_columns = [col for col in df.columns if 'loss' in col.lower() and ('global' in col.lower() or 'test' in col.lower() or 'val' in col.lower())]
    if not loss_columns:
        # Try common loss column names
        potential_loss_cols = ['loss', 'test_loss', 'val_loss', 'global_loss', 'accuracy']
        loss_columns = [col for col in potential_loss_cols if col in df.columns]
    
    if not loss_columns:
        print("Skipping global loss method comparison - no loss columns found")
        return None if not return_data else pd.DataFrame()
    
    # Use the first available loss column
    loss_column = loss_columns[0]
    print(f"Using loss column: {loss_column}")
    
    # Handle selective round attack filtering - process each attack round separately
    if use_attack_round_filtering:
        if 'attack_round' not in df.columns:
            print("Skipping plot - 'attack_round' column not found for selective round filtering.")
            return None if not return_data else pd.DataFrame()
        
        # Get unique attack rounds (excluding None/nan for no_attack and full attacks)
        attack_rounds = sorted([r for r in df['attack_round'].unique() if r is not None and pd.notna(r)])
        if not attack_rounds:
            print("Skipping plot - No attack rounds found for selective round filtering.")
            return None if not return_data else pd.DataFrame()
        
        # Get no_attack baseline data
        df_no_attack = df[df['attack_status'] == 'no_attack'].copy()
        if df_no_attack.empty:
            print("Skipping plot - No 'no_attack' baseline data found for selective round filtering.")
            return None if not return_data else pd.DataFrame()
        
        print(f"Processing selective round attacks for rounds: {attack_rounds}")
        
        # Collect all data from recursive calls if return_data is requested
        all_recursive_data = []
        
        # Process each attack round separately
        for attack_round in attack_rounds:
            # Filter data for this specific attack round
            df_attack_round = df[(df['attack_round'] == attack_round)].copy()
            if df_attack_round.empty:
                continue
            
            # Combine attack data with no_attack baseline for this round
            df_combined = pd.concat([df_no_attack, df_attack_round], ignore_index=True)
            
            # Recursively call this function for each attack round 
            round_output_path = output_path.parent / "selective_rounds" / f"round_{int(attack_round)}_global_loss_{output_path.name}"
            round_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            round_data = plot_method_comparison_global_loss_changes(
                df_combined, subtitle, round_output_path, 
                distributions, use_attack_round_filtering=False, use_suptitle=use_suptitle, 
                show_variance=show_variance, return_data=return_data
            )
            
            # Collect data if return_data is requested
            if return_data and round_data is not None and not round_data.empty:
                # Add round information to the data
                round_data = round_data.copy()
                round_data['attack_round'] = attack_round
                all_recursive_data.append(round_data)
        
        # Return combined data from all rounds if requested
        if return_data:
            if all_recursive_data:
                return pd.concat(all_recursive_data, ignore_index=True)
            else:
                return pd.DataFrame()
        else:
            return None
    
    # Filter by distributions if specified
    if distributions is not None:
        df = df[df['distribution'].isin(distributions)].copy()
        if df.empty:
            print(f"Skipping plot - No data for specified distributions: {distributions}")
            return None if not return_data else pd.DataFrame()
    
    # Check that we have both methods
    available_methods = df['method'].unique()
    if len(available_methods) < 2:
        print(f"Skipping global loss method comparison - need at least 2 methods, found: {available_methods}")
        return None if not return_data else pd.DataFrame()
    
    if 'loo' not in available_methods or 'shapley' not in available_methods:
        print(f"Skipping global loss method comparison - need both 'loo' and 'shapley' methods, found: {available_methods}")
        return None if not return_data else pd.DataFrame()
    
    # Check for required attack statuses
    available_statuses = df['attack_status'].unique()
    
    attack_comparisons = []
    if 'no_attack' in available_statuses and 'with_attack' in available_statuses:
        attack_comparisons.append(('with_attack', 'Target-Decrease Attack'))
    if 'no_attack' in available_statuses and 'self_promotion' in available_statuses:
        attack_comparisons.append(('self_promotion', 'Self-Promotion Attack'))
    
    if not attack_comparisons:
        print("Skipping global loss method comparison - need 'no_attack' and at least one attack type")
        return None if not return_data else pd.DataFrame()
    
    # Collect all data for return if requested
    all_plotting_data = []
    
    # Process each attack comparison
    for attack_status, attack_name in attack_comparisons:
        print(f"Processing {attack_name} global loss over rounds...")
        
        # Create separate plots for each distribution
        for distribution in df['distribution'].unique():
            dist_data = df[df['distribution'] == distribution].copy()
            if dist_data.empty:
                continue
            
            # Prepare data for line plotting
            plot_data = []
            
            # Process each method and attack status combination
            for method in ['loo', 'shapley']:
                # No attack baseline data
                no_attack_data = dist_data[
                    (dist_data['method'] == method) & 
                    (dist_data['attack_status'] == 'no_attack')
                ].copy()
                
                # Attack data
                attack_data = dist_data[
                    (dist_data['method'] == method) & 
                    (dist_data['attack_status'] == attack_status)
                ].copy()
                
                # Aggregate by round for both no_attack and attack scenarios
                for scenario, scenario_data in [('no_attack', no_attack_data), (f'{method}_attack', attack_data)]:
                    if scenario_data.empty:
                        continue
                    
                    round_stats = scenario_data.groupby('round')[loss_column].agg(['mean', 'std', 'count']).reset_index()
                    
                    for _, row in round_stats.iterrows():
                        plot_data.append({
                            'round': row['round'],
                            'loss_mean': row['mean'],
                            'loss_std': row['std'] if pd.notna(row['std']) else 0,
                            'count': row['count'],
                            'scenario': scenario,
                            'method': method if scenario != 'no_attack' else 'no_attack',
                            'distribution': distribution,
                            'attack_type': attack_name
                        })
            
            if not plot_data:
                print(f"No data for {attack_name} in distribution {distribution}")
                continue
            
            plot_df = pd.DataFrame(plot_data)
            
            # FIX: Aggregate duplicate no_attack entries (from LOO and Shapley processing)
            # The no_attack baseline should be the same regardless of method
            if not plot_df.empty and 'no_attack' in plot_df['scenario'].values:
                no_attack_data = plot_df[plot_df['scenario'] == 'no_attack'].copy()
                other_data = plot_df[plot_df['scenario'] != 'no_attack'].copy()
                
                # Aggregate no_attack data by round (take mean of LOO and Shapley baselines)
                no_attack_aggregated = no_attack_data.groupby('round').agg({
                    'loss_mean': 'mean',  # Average of LOO and Shapley baselines
                    'loss_std': 'mean',   # Average of standard deviations
                    'count': 'sum',       # Total count
                    'scenario': 'first',  # Keep 'no_attack'
                    'method': 'first',    # Keep 'no_attack'
                    'distribution': 'first',
                    'attack_type': 'first'
                }).reset_index()
                
                # Reconstruct the DataFrame with aggregated no_attack data
                plot_df = pd.concat([no_attack_aggregated, other_data], ignore_index=True)
            
            # Add to all_plotting_data if return_data is requested
            if return_data:
                all_plotting_data.append(plot_df)
            
            # Create the line plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define colors, line styles, and markers
            colors = {
                'no_attack': 'gray',
                'loo': 'lightblue', 
                'shapley': 'lightcoral'
            }
            
            line_styles = {
                'no_attack': '-',
                'loo': '--',
                'shapley': ':'
            }
            
            markers = {
                'no_attack': 'o',
                'loo': 's',
                'shapley': '^'
            }
            
            # Plot lines for each scenario
            unique_scenarios = plot_df['scenario'].unique()
            
            for scenario in unique_scenarios:
                scenario_data = plot_df[plot_df['scenario'] == scenario].sort_values('round')
                
                if scenario_data.empty:
                    continue
                
                method_key = scenario_data['method'].iloc[0]
                color = colors.get(method_key, 'black')
                linestyle = line_styles.get(method_key, '-')
                marker = markers.get(method_key, 'o')
                
                # Plot the main line
                label = 'No Attack' if method_key == 'no_attack' else f'{method_key.upper()} Attack'
                ax.plot(scenario_data['round'], scenario_data['loss_mean'], 
                       color=color, linestyle=linestyle, linewidth=2, 
                       marker=marker, markersize=6, label=label)
                
                # Add variance bands if requested
                if show_variance and not scenario_data['loss_std'].isna().all():
                    ax.fill_between(
                        scenario_data['round'],
                        scenario_data['loss_mean'] - scenario_data['loss_std'],
                        scenario_data['loss_mean'] + scenario_data['loss_std'],
                        color=color, alpha=0.2
                    )
            
            # Set title and labels based on use_suptitle parameter
            if use_suptitle:
                fig.suptitle(f"Global Loss Over Rounds\n{attack_name}\n{subtitle}", 
                            fontsize=POSTER_FONT_SIZE)
                ax.set_title(f"Distribution: {distribution}")
                ax.set_xlabel("Round")
                ax.set_ylabel("Global Loss")
                ax.legend()
            else:
                # If use_suptitle is False, remove all titles, labels, and legend
                ax.set_xlabel("")
                ax.set_ylabel("")
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(0.5, 5.5)  # Ensure we show rounds 1-5
            ax.set_xticks(range(1, 6))  # Show ticks for rounds 1-5
            
            plt.tight_layout()
            
            # Save plot for this distribution and attack type
            attack_suffix = attack_status.replace('_', '')  # 'withattack' or 'selfpromotion'
            dataset_name = dist_data['dataset'].iloc[0] if 'dataset' in dist_data.columns else 'unknown'
            
            # Check if this is a selective round call (path contains "selective_rounds")
            if "selective_rounds" in str(output_path):
                # For selective rounds, use the specific path provided and append attack type
                plot_filename = output_path.parent / f"{output_path.stem}_{attack_suffix}_{distribution}.png"
            else:
                # For regular calls, construct the filename as before
                plot_filename = output_path.parent / f"global_loss_over_rounds_{dataset_name}_{attack_suffix}_{distribution}.png"
            
            plt.savefig(plot_filename, bbox_inches='tight', dpi=120)
            plt.close()
            print(f"Saved global loss over rounds plot for {distribution} ({attack_name}) to {plot_filename}")
    
    # Return data if requested
    if return_data:
        if all_plotting_data:
            return pd.concat(all_plotting_data, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        return None