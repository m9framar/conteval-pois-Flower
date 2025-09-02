# Visualization and Data Handling Guide

This guide provides information on how experimental results are stored, loaded, and visualized in this project.

## Results Data Structure

- **Root Directory**: All raw results are stored under the `results/` directory.
- **Hierarchy**: Results are organized hierarchically:
  ```
  results/
  ├── <dataset_type>/         # e.g., "tabular", "fashion"
  │   └── <contrib_method>/     # e.g., "loo", "shapley"
  │       └── <distribution>/     # e.g., "iid", "dirichlet_0.8"
  │           └── <attack_status>/  # "with_attack", "without_attack", or "self_promotion"
  │               └── <run_id>/       # e.g., "run_1", "run_2", ...
  │                   └── results.json  # Raw metrics for the run
  ```
- **`results.json` Format**: Each `results.json` file contains a list under the `"fit"` key. Each element in the list represents a federated round and includes:
    - `round`: The round number.
    - `global_loss`: Global model loss for the round.
    - `global_accuracy`: Global model accuracy for the round.
    - `client_<id>_contribution` or `client_<id>_<method>`: Contribution score for client `<id>` using the configured method (`loo` or `shapley`).
    - `best_attack_scale` (if attacked): The attack scale parameter used in that round.
    - `best_init_method` (if attacked): The attack initialization method used.
    - (Shapley specific): `shapley_params`, `client_<id>_shapley_S` (best subset Shapley), `attack_shapley_params`.

  *Example Snippet (`results.json`):*
  ```json
  {
      "fit": [
          {
              "round": 1,
              "global_loss": 0.3809,
              "global_accuracy": 0.8157,
              "client_1_contribution": -0.0024,
              "client_2_contribution": 0.0129,
              // ... other clients and metrics ...
          },
          {
              "round": 2,
              "global_loss": 0.5621,
              "global_accuracy": 0.8091,
              "client_1_contribution": -0.2206,
              "client_2_contribution": -0.1071,
              "best_attack_scale": 0.5,
              "best_init_method": "random_noise",
              // ... other clients and metrics ...
          }
          // ... more rounds ...
      ]
      // (Optional) "shapley_values" section for Shapley method
  }
  ```

### Important Data Processing Note

For better comparability, Shapley values are flipped in sign during data loading so that higher positive values indicate better contributions, consistent with LOO values. This standardization ensures that across both methods:
- **Higher positive values = Better contribution**
- **Lower negative values = Worse contribution**

Without this standardization, Shapley values would have the opposite interpretation (where more negative values indicate better contribution), making comparisons difficult.

## Visualization System Architecture (Refactored)

The visualization system has been refactored into a modular architecture with separate components for data loading, visualization utilities, plotting, and table generation.

### Module Overview

1. **`data_reader.py`**: Handles loading and processing result files
   - `load_results(results_dir)`: Traverses the directory structure and loads all results.json files
   - `create_dataframe_from_results(results_data)`: Converts loaded data into a pandas DataFrame with proper types

2. **`visualization_utils.py`**: Contains shared styling and utility functions
   - `setup_plot_style(font_size)`: Sets global Matplotlib styles
   - `get_plot_subtitle(df)`: Generates subtitles for plots based on dataset and method
   - `save_df_as_image(df, path, ...)`: Saves a DataFrame as a table image
   - `get_max_clients(df)`: Determines the maximum number of clients in the dataset

3. **`plotting.py`**: Contains all plotting-related functions
   - `plot_global_metric(df, metric_col, ...)`: Plots global metrics (accuracy, loss) over rounds
   - `plot_client_contribution(df, client_id, ...)`: Plots a specific client's contribution
   - `plot_final_contribution_boxplot(df, max_clients, ...)`: Shows distribution of client contributions in final round
   - `plot_contribution_comparison(df, max_clients, ...)`: Compares attacker, target, and other clients' contributions

4. **`table_generator.py`**: Handles generating tables in both CSV and image formats
   - `generate_global_metric_table(df, metric_col, ...)`: Creates tables for global metrics
   - `generate_client_contribution_table(df, client_id, ...)`: Creates tables for client contributions
   - `generate_final_contribution_table(df, max_clients, ...)`: Creates tables for final round contributions
   - `generate_contribution_comparison_table(df, max_clients, ...)`: Creates comparison tables

5. **`aggregate_results.py`**: Main script that orchestrates the visualization process
   - Uses all the above modules to load data and generate visualizations
   - Handles creating output directories and processing each dataset/method combination

### Constants and Configuration

Key constants are defined in `visualization_utils.py`:
- `POSTER_FONT_SIZE`: Base font size for plots (default: 20)
- `TABLE_FONT_SIZE`: Font size for table images (default: 20)
- `TARGET_CLIENT_ID`: ID of the target client (default: 2)
- `ATTACKER_CLIENT_ID`: ID of the attacker client (default: 1)

## Running the Visualization System

To generate all plots and tables:

```bash
python aggregate_results.py
```

This will:
1. Load all results from the `results/` directory
2. Process each dataset type (tabular, fashion) and method (loo, shapley) combination
3. Generate plots in the `plots/` directory
4. Generate tables (CSVs and images) in the `tables/` directory

### Filtering Visualizations by Distribution

To create visualizations that include only specific distributions or exclude certain distributions:

```bash
# Include only specific distributions
python aggregate_results.py --distributions iid dirichlet_0.1

# Exclude specific distributions
python aggregate_results.py --exclude-distributions dirichlet_0.8
```

These flags help when comparing results across different distribution types without visual clutter.

## Output Structure

- **Plots**: `plots/<dataset_type>/<contrib_method>/`
  - `global_accuracy.png`: Global accuracy over rounds
  - `global_loss.png`: Global loss over rounds
  - `target_client_2_contribution.png`: Target client's contribution over rounds
  - `attacker_client_1_contribution.png`: Attacker client's contribution over rounds
  - `final_round_contribution_boxplot_*.png`: Distribution of client contributions in final round (per distribution)
  - `final_round_contribution_boxplot_combined.png`: Combined view of all distributions
  - `client_contributions_comparison.png`: Comparison of client contributions
  - `attack_params_heatmap_*.png`: Heatmaps showing frequency of attack parameter combinations (initialization method vs attack scale)
  - `*_table.png`: Visual representations of tables

- **Tables**: `tables/<dataset_type>/<contrib_method>/`
  - `global_accuracy.csv`: Tabulated global accuracy data
  - `global_loss.csv`: Tabulated global loss data
  - `target_client_2_contribution.csv`: Target client's contribution data
  - `attacker_client_1_contribution.csv`: Attacker client's contribution data
  - `final_round_contribution.csv`: Final round contributions for all clients
  - `client_contributions_comparison.csv`: Comparison data for client contributions

## Key Visualization Features

- **Distribution-specific analysis**: Final round contribution boxplots are generated per distribution to avoid scale issues when comparing very different distributions (e.g., IID vs highly non-IID)
- **Adaptive table formatting**: Tables adapt their column widths based on content, simplify long headers, and handle large datasets by limiting displayed rounds
- **Comparison plots**: The contribution comparison plot specifically highlights the relationships between attacker, target, and other client contributions
- **Attack parameter heatmaps**: Heatmaps visualize the frequency of each attack parameter combination (initialization method vs attack scale) for both targeted attacks and self-promotion attacks
- **Standardized contribution scale**: Both LOO and Shapley values use the same interpretation scale where higher positive values mean better contributions
- **Improved boxplot visualizations**: Side-by-side comparison boxplots for attack scenarios with proper handling of cases where only one attack status is available
- **Enhanced plot formatting**: Better subplot spacing, rotated x-axis labels (45°), improved layout with tight_layout and higher DPI for sharper images

## Extending the Visualization System

### Adding a New Plot Type

1. Add a new plotting function in `plotting.py`
2. Update `process_and_visualize_results()` in `aggregate_results.py` to call the new function

Example:
```python
# In plotting.py
def plot_new_metric(df, output_path, subtitle):
    # Implementation...
    plt.figure(figsize=(12, 8))
    # ... plotting code ...
    plt.savefig(output_path)
    plt.close()

# In aggregate_results.py
from plotting import plot_new_metric

def process_and_visualize_results(df, plots_dir, tables_dir):
    # ... existing code ...
    
    # Add new plot
    plot_new_metric(
        df, plots_dir / "new_metric_plot.png", subtitle
    )
    
    # ... existing code ...
```

### Adding a New Table Type

1. Add a new table generation function in `table_generator.py`
2. Update `process_and_visualize_results()` in `aggregate_results.py` to call the new function

### Customizing Plot Styles

Modify the `setup_plot_style()` function in `visualization_utils.py` to adjust the global plot styling.

## Common Troubleshooting

- **Table size warnings**: If large tables produce warnings about image dimensions, adjust the `col_width`, `row_height`, and `font_size` parameters in `table_generator.py`
- **Missing data points**: Check that the expected metrics exist in the original `results.json` files
- **Distribution filtering**: Use the `--distributions` or `--exclude-distributions` flags to focus on specific data distributions

## Weights & Biases (WandB) Integration

- **Activation**: Controlled by the `--wandb=true` flag in `run_experiments.sh` or the `use_wandb=True` setting in the server configuration (`server_app.py`).
- **Logged Metrics**: During individual experiment runs, key metrics are logged in real-time to WandB:
    - Time-series plots for global loss and accuracy.
    - Time-series plots for per-client contribution scores (LOO delta or Shapley values).
    - Configuration parameters (`run_config`).
    - Potentially other custom metrics logged via `wandb.log()` within the strategies (`contribution_utils.py`).
- **Purpose**: Provides real-time monitoring and comparison of individual experiment runs. The `aggregate_results.py` script is used for *post-hoc* analysis and visualization across *all* completed runs.

## Relationship with Result Generation

The visualization system is designed to work with the results generated by the `run_experiments.sh` script. Key considerations:

- Always run `aggregate_results.py` after experiments complete to generate the full suite of visualizations
- For quick visualization during experimentation, WandB provides real-time insights
- The final contribution plots are particularly useful for assessing attack effectiveness

## Method Handling and Data Separation

### LaTeX Table Generation Issue with Multiple Methods

When generating LaTeX tables using the `generate_latex_tables.py` script, a specific issue affects the `--method all` option:

**Issue**: When using `--method all`, the script would improperly mix data from different contribution methods (LOO and Shapley) or potentially skip data from one method entirely.

**Root Cause**: The original implementation passed a DataFrame containing both methods to the `process_and_generate_latex_tables()` function, which wasn't designed to handle multiple methods simultaneously.

**Solution**: The script now:
1. Processes each method separately when `--method all` is specified
2. Generates method-specific output files (e.g., `latex_tables_loo.tex` and `latex_tables_shapley.tex`)
3. Creates a combined file (`latex_tables.tex`) with clear section headers for each method
4. Adds method names to table captions for clarity

### Method Handling in the Visualization System

The visualization system in `aggregate_results.py` already properly handles method separation:

```python
# Process each dataset and method combination separately
for (dataset, method), group_df in df.groupby(['dataset', 'method']):
    print(f"\n--- Processing Dataset: {dataset}, Method: {method} ---")
    method_plots_dir = PLOTS_DIR / dataset / method
    method_tables_dir = TABLES_DIR / dataset / method
    # ... process this method only ...
```

This design ensures that:
1. All plots and tables are generated separately for each contribution method
2. The output files are organized in method-specific directories
3. Method-specific metrics are never mixed improperly

### Best Practices for Handling Multiple Methods

When working with multiple contribution methods:

1. **Always process each method separately**: Filter DataFrame by method before processing
2. **Include method identifiers in output filenames**: Use suffixes like `_loo` or `_shapley`
3. **Add method information to plot titles and table captions**: Clearly indicate which method the data represents
4. **Store outputs in method-specific directories**: Keep outputs organized by method
5. **For combined views, use clear section headers**: When combining methods in a single document, add clear headers

Following these practices ensures that different contribution methods are properly separated, preventing confusion and data misinterpretation.
