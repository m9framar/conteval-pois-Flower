import matplotlib.pyplot as plt
import textwrap
import pandas as pd
import numpy as np

# Constants for styling
POSTER_FONT_SIZE = 24
TABLE_FONT_SIZE = 20

# Define client IDs
TARGET_CLIENT_ID = 2
ATTACKER_CLIENT_ID = 1

def setup_plot_style(font_size=POSTER_FONT_SIZE):
    """Sets global Matplotlib styles for plots.
    
    Args:
        font_size (int): Base font size for plots
    """
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size + 2)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size+2)
    plt.rc('ytick', labelsize=font_size+2)
    plt.rc('legend', fontsize=font_size - 10)
    plt.rc('figure', titlesize=font_size + 4)

def get_plot_subtitle(df):
    """Generates a subtitle string based on dataset and method.
    
    Args:
        df (pd.DataFrame): DataFrame containing results
        
    Returns:
        str: Subtitle string for plot
    """
    if df.empty:
        return "Dataset: Unknown, Method: Unknown"
    dataset = df['dataset'].iloc[0]
    method = df['method'].iloc[0]
    
    # Replace "SHAPLEY" with "GTG" in the subtitle for better understanding
    method_text = method.upper()
    if method_text == "SHAPLEY":
        method_text = "GTG"
        
    return f"Dataset: {dataset.capitalize()}, Method: {method_text}"

def save_df_as_image(df, path, title="", col_width=1.5, row_height=0.3, font_size=8, max_header_len=10):
    """Saves a DataFrame as a table image using Matplotlib, wrapping long headers.
    
    Args:
        df (pd.DataFrame): DataFrame to save as image
        path (str): Path where to save the image
        title (str): Title for the table image
        col_width (float): Width of table columns
        row_height (float): Height of table rows
        font_size (int): Font size for table text
        max_header_len (int): Maximum length of header text before wrapping
    """
    if df.empty:
        print(f"Skipping empty table image: {path}")
        return
    try:
        # Format floats
        df_display = df.copy()
        
        # Simplify column names if they are multi-index
        if isinstance(df_display.columns, pd.MultiIndex):
            # Convert multi-index to simplified strings
            new_columns = []
            for col in df_display.columns:
                # For multi-index columns like ('mean_metric', 1), simplify to 'mean_1'
                if isinstance(col, tuple) and len(col) == 2:
                    prefix = col[0].split('_')[0][:4]  # Take first 4 chars of first part
                    suffix = str(col[1])
                    new_columns.append(f"{prefix}_{suffix}")
                else:
                    # Fallback for unexpected formats
                    new_columns.append(str(col).replace('_', ''))
            df_display.columns = new_columns
        
        # Format float columns to reduce width
        float_cols = df_display.select_dtypes(include='float').columns
        for col in float_cols:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'NaN')

        # Reset index to include it as columns
        index_names = []
        if df_display.index.name is not None:
            index_names.append(df_display.index.name)
            df_display.reset_index(inplace=True)
        elif isinstance(df_display.index, pd.MultiIndex):
            index_names.extend(df_display.index.names)
            df_display.reset_index(inplace=True)
            
            # Simplify index column names
            for idx_name in index_names:
                if idx_name in df_display.columns:
                    # For distribution column, shorten values like "dirichlet_0.8" to "dir_0.8"
                    if idx_name == "distribution":
                        df_display[idx_name] = df_display[idx_name].apply(
                            lambda x: x.replace("dirichlet_", "dir_") if isinstance(x, str) else x
                        )
                    # For attack_status column, shorten values
                    elif idx_name == "attack_status":
                        df_display[idx_name] = df_display[idx_name].apply(
                            lambda x: "w_atk" if x == "with_attack" else "no_atk" if x == "without_attack" else x
                        )
                    # For client_category column, shorten values
                    elif idx_name == "client_category":
                        df_display[idx_name] = df_display[idx_name].apply(
                            lambda x: x.replace("Client", "C").replace("(Avg)", "(A)")
                            if isinstance(x, str) else x
                        )

        # Wrap column headers and limit length
        wrapped_columns = []
        for col in df_display.columns:
            # Truncate long column names
            col_str = str(col)
            if len(col_str) > max_header_len:
                col_str = col_str[:max_header_len-2] + '..'
            wrapped_columns.append(textwrap.fill(col_str, max_header_len))
        df_display.columns = wrapped_columns
        
        # Calculate cell widths based on content - makes cells only as wide as needed
        col_widths = []
        for col_idx, col_name in enumerate(df_display.columns):
            # Get max width from column name and values
            col_values = df_display.iloc[:, col_idx].astype(str)
            max_content_width = max([len(str(val)) for val in col_values] + [len(col_name)])
            # Convert to inches with some padding (0.1 inch + 0.05 per character)
            width = 0.2 + (0.05 * max_content_width)
            # Set minimum width and cap maximum width
            width = max(0.3, min(width, 2.0))
            col_widths.append(width)
            
        # Calculate figure dimensions
        num_cols = len(df_display.columns)
        fig_width = min(sum(col_widths), 25)  # Cap width at 25 inches
        
        # Add space for title - very important to avoid overlay!
        title_height = 0.5 if title else 0.0
        
        # Adjust height based on wrapped headers and rows
        max_lines = max(col.count('\n') + 1 for col in wrapped_columns) if wrapped_columns else 1
        header_height = row_height * max_lines
        fig_height = min(title_height + header_height + (len(df_display) * row_height), 20)  # Cap height at 20 inches

        # Create figure with extra space at top for title
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')  # Hide axes

        # Add title with proper spacing from top
        if title:
            title_y_pos = 1 - (title_height / (2 * fig_height))  # Position title in the title space
            plt.title(title, fontsize=font_size + 2, weight='bold', y=title_y_pos)

        # Create the table - adjust position to leave room for title
        table_position = 0.5  # Center position by default
        if title:
            # Adjust table position down slightly if there's a title
            table_position = 0.45
        
        the_table = ax.table(cellText=df_display.values,
                           colLabels=df_display.columns,
                           loc='center',
                           cellLoc='center',
                           bbox=[0.0, 0.0, 1.0, 0.9],  # Use bbox to position table lower
                           colWidths=col_widths)

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(font_size)
        
        # Use less vertical scaling for tables with many rows
        if len(df_display) > 10:
            the_table.scale(1, 0.9)  # Less vertical scaling for taller tables
        else:
            the_table.scale(1, 1.0)  # Standard vertical scaling

        # Style the table cells (including header)
        for (i, j), cell in the_table.get_celld().items():
            cell.set_edgecolor('black')  # Add cell borders
            cell.set_linewidth(0.5)
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')  # Light grey header
                cell.set_height(header_height / max_lines)  # Adjust header cell height
                
                # For very wide tables, rotate the header text
                if num_cols > 15:
                    cell.set_text_props(rotation=45, ha='right')
            else:
                cell.set_height(row_height)
                # Set cell width based on content
                # We've already calculated this in col_widths

        # Use a better padding approach that gives more space for the title
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
        
        # Save with lower DPI to reduce file size
        plt.savefig(path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        print(f"Table saved to {path}")

    except Exception as e:
        print(f"Error saving table image {path}: {e}")

def get_max_clients(df):
    """Determine the maximum number of clients in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing results
        
    Returns:
        int: Maximum number of clients found
    """
    client_cols = [col for col in df.columns if col.startswith("client_") and "_contribution" in col]
    max_clients = 0
    if client_cols:
        try:
            client_ids = [int(col.split("_")[1]) for col in client_cols]
            max_clients = max(client_ids) if client_ids else 0
        except (ValueError, IndexError):
            print("Warning: Could not reliably determine max number of clients from column names.")
    return max_clients