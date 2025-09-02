# This file provides functions to load and process results data from JSON files.
# For Shapley values, we flip the sign to make them consistent with LOO values (higher = better contribution).
import os
import json
from pathlib import Path
import pandas as pd


def load_results(results_dir, selective_rounds=False):
    """Loads all results.json files into a list of records.
    
    Args:
        results_dir (Path): Path to the results directory
        selective_rounds (bool): If True, loads selective round attack data from results_selective structure
        
    Returns:
        list: List of dictionaries containing the extracted data from results.json files
    """
    all_data = []
    
    for dataset_type in ["tabular", "fashion"]:
        dataset_path = results_dir / dataset_type
        if not dataset_path.is_dir():
            continue
        for contrib_method in ["loo", "shapley"]:
            method_path = dataset_path / contrib_method
            if not method_path.is_dir():
                continue
            for dist_dir in method_path.iterdir():
                if not dist_dir.is_dir(): # Skip files like .DS_Store
                    continue
                distribution = dist_dir.name
    for dataset_type in ["tabular", "fashion"]:
        dataset_path = results_dir / dataset_type
        if not dataset_path.is_dir():
            continue
        for contrib_method in ["loo", "shapley"]:
            method_path = dataset_path / contrib_method
            if not method_path.is_dir():
                continue
            for dist_dir in method_path.iterdir():
                if not dist_dir.is_dir(): # Skip files like .DS_Store
                    continue
                distribution = dist_dir.name
                for attack_status_dir in dist_dir.iterdir():
                    if not attack_status_dir.is_dir():
                        continue
                    attack_status_raw = attack_status_dir.name # "with_attack", "without_attack", "self_promotion", "targeted", "no_attack"
                    
                    # Normalize attack status names for consistency
                    if attack_status_raw in ["with_attack", "targeted"]:
                        attack_status = "with_attack"
                    elif attack_status_raw in ["without_attack", "no_attack"]:
                        attack_status = "no_attack"
                    elif attack_status_raw == "self_promotion" or attack_status_raw == "self-promotion":
                        attack_status = "self_promotion"
                    else:
                        attack_status = attack_status_raw
                    
                    # Set attack flags based on attack_status
                    is_attacked = attack_status == "with_attack"
                    is_self_promotion = attack_status == "self_promotion"
                    
                    # Handle selective rounds structure
                    if selective_rounds:
                        if attack_status == "no_attack":
                            # No attack data doesn't have rounds subdivision - direct run folders
                            for run_dir in attack_status_dir.iterdir():
                                if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                                    continue
                                run_id = run_dir.name
                                results_file = run_dir / "results.json"
                                
                                if results_file.exists():
                                    all_data.extend(_process_results_file(
                                        results_file, dataset_type, contrib_method, distribution,
                                        attack_status, is_attacked, is_self_promotion, run_id, None
                                    ))
                                else:
                                    print(f"Warning: Missing results.json in {run_dir}")
                        else:
                            # Attack data has rounds subdivision
                            for rounds_dir in attack_status_dir.iterdir():
                                if not rounds_dir.is_dir() or not rounds_dir.name.startswith("rounds_"):
                                    continue
                                
                                # Extract attack round number from rounds_X
                                try:
                                    attack_round = int(rounds_dir.name.split("_")[1])
                                except (IndexError, ValueError):
                                    print(f"Warning: Could not parse attack round from {rounds_dir.name}")
                                    continue
                                    
                                for run_dir in rounds_dir.iterdir():
                                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                                        continue
                                    run_id = run_dir.name
                                    results_file = run_dir / "results.json"
                                    
                                    if results_file.exists():
                                        all_data.extend(_process_results_file(
                                            results_file, dataset_type, contrib_method, distribution,
                                            attack_status, is_attacked, is_self_promotion, run_id, attack_round
                                        ))
                                    else:
                                        print(f"Warning: Missing results.json in {run_dir}")
                    else:
                        # Regular structure (no rounds_X layer)
                        for run_dir in attack_status_dir.iterdir():
                            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                                continue
                            run_id = run_dir.name
                            results_file = run_dir / "results.json"
                            
                            if results_file.exists():
                                all_data.extend(_process_results_file(
                                    results_file, dataset_type, contrib_method, distribution,
                                    attack_status, is_attacked, is_self_promotion, run_id, None
                                ))
                            else:
                                print(f"Warning: Missing results.json in {run_dir}")
                            
    return all_data


def _process_results_file(results_file, dataset_type, contrib_method, distribution, 
                         attack_status, is_attacked, is_self_promotion, run_id, attack_round):
    """Helper function to process a single results.json file.
    
    Args:
        results_file (Path): Path to the results.json file
        dataset_type (str): Dataset type (tabular, fashion)
        contrib_method (str): Contribution method (loo, shapley)
        distribution (str): Distribution type
        attack_status (str): Attack status
        is_attacked (bool): Whether this is an attack scenario
        is_self_promotion (bool): Whether this is self-promotion attack
        run_id (str): Run identifier
        attack_round (int or None): Specific round when attack occurs (for selective rounds)
        
    Returns:
        list: List of processed records from this file
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        fit_results = data.get("fit", [])
        if not fit_results: # Skip if no fit results
            return []

        num_clients = 0
        # Determine number of clients from first round data if possible
        first_round_keys = fit_results[0].keys()
        client_keys_initial = [k for k in first_round_keys if k.startswith("client_") and ("_contribution" in k or "_shapley" in k)]
        if client_keys_initial:
            client_ids_found_initial = set()
            for k in client_keys_initial:
                parts = k.split("_")
                if len(parts) > 1 and parts[1].isdigit():
                    client_ids_found_initial.add(int(parts[1]))
            if client_ids_found_initial:
                num_clients = len(client_ids_found_initial)

        # Temporary list to store records for the current run before calculating cumulative averages
        current_run_records = []
        max_client_id_overall = 0 # Keep track of max client ID for this run

        for round_data in fit_results:
            round_num = round_data.get("round")
            if round_num is None: continue

            record = {
                "dataset": dataset_type,
                "method": contrib_method,
                "distribution": distribution,
                "attack_status": attack_status,
                "is_attacked": is_attacked,
                "is_self_promotion": is_self_promotion,
                "run_id": run_id,
                "round": round_num,
                "global_loss": round_data.get("global_loss"),
                "global_accuracy": round_data.get("global_accuracy"),
                "best_attack_scale": round_data.get("best_attack_scale"),
                "best_init_method": round_data.get("best_init_method"),
                "attack_round": attack_round,  # New field for selective rounds
                # num_clients will be set more robustly later if still 0
            }

            # Extract contributions for all clients
            # Determine max client ID in this specific round_data
            current_round_client_keys = [k for k in round_data.keys() if k.startswith("client_") and ("_contribution" in k or "_shapley" in k)]
            current_round_client_ids = set()
            for k in current_round_client_keys:
                parts = k.split("_")
                if len(parts) > 1 and parts[1].isdigit():
                    current_round_client_ids.add(int(parts[1]))
                                    
            max_client_id_in_round = 0
            if current_round_client_ids:
                max_client_id_in_round = max(current_round_client_ids)
                if max_client_id_in_round > max_client_id_overall:
                    max_client_id_overall = max_client_id_in_round
            
            clients_to_extract_this_round = max_client_id_in_round

            for i in range(1, clients_to_extract_this_round + 1):
                client_key_loo = f"client_{i}_contribution"
                client_key_shapley = f"client_{i}_shapley"
                
                contribution = None
                if contrib_method == "loo":
                    contribution = round_data.get(client_key_loo)
                elif contrib_method == "shapley":
                    # Get the Shapley value, or fall back to LOO value if Shapley not available
                    contribution = round_data.get(client_key_shapley, round_data.get(client_key_loo))
                    # Flip the sign of Shapley values to align with LOO values (higher = better)
                    if contribution is not None:
                        contribution = -contribution
                
                record[f"client_{i}_contribution"] = contribution
            current_run_records.append(record)
        
        # After processing all rounds for the current run_id, convert to DataFrame to calculate cumulative averages
        if current_run_records:
            temp_run_df = pd.DataFrame(current_run_records)
            temp_run_df.sort_values(by="round", inplace=True)
            
            # Update num_clients based on the overall max client ID found in this run
            # This ensures all client_X_contribution columns are processed for averaging
            if num_clients == 0 and max_client_id_overall > 0:
                num_clients = max_client_id_overall
            elif num_clients == 0 and not temp_run_df.empty: # Fallback if still 0
                # Try to infer from column names in the temp_run_df
                contrib_cols = [col for col in temp_run_df.columns if col.startswith("client_") and col.endswith("_contribution")]
                client_ids_from_cols = set()
                for col_name in contrib_cols:
                    try:
                        client_ids_from_cols.add(int(col_name.split("_")[1]))
                    except (IndexError, ValueError):
                        continue
                if client_ids_from_cols:
                    num_clients = max(client_ids_from_cols)


            temp_run_df["num_clients"] = num_clients # Add/update num_clients column

            for i in range(1, num_clients + 1):
                contrib_col = f"client_{i}_contribution"
                avg_contrib_col = f"client_{i}_avg_contribution"
                if contrib_col in temp_run_df.columns:
                    # Ensure the column is numeric before calculating expanding mean
                    temp_run_df[contrib_col] = pd.to_numeric(temp_run_df[contrib_col], errors='coerce')
                    # Calculate expanding mean, handling potential NaNs by not dropping them before mean calculation
                    # Grouping by run_id implicitly handled as temp_run_df is for a single run
                    temp_run_df[avg_contrib_col] = temp_run_df[contrib_col].expanding().mean()
                else:
                    # If original contribution column doesn't exist, avg column also won't
                    temp_run_df[avg_contrib_col] = pd.NA 

            return temp_run_df.to_dict('records')
        
        return []
        
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON in {results_file}")
        return []
    except Exception as e:
        print(f"Warning: Error processing {results_file}: {e}")
        return []


def create_dataframe_from_results(results_data):
    """Creates a pandas DataFrame from results data and applies necessary type conversions.
    
    Args:
        results_data (list): List of result records as returned by load_results
        
    Returns:
        pandas.DataFrame: DataFrame containing all results with proper types.
                         Note: For Shapley values, signs are flipped so that higher values
                         indicate better contributions, consistent with LOO values.
    """
    if not results_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(results_data)
    
    # Basic data cleaning/type conversion
    numeric_cols = ['global_loss', 'global_accuracy', 'best_attack_scale', 'attack_round'] + \
                  [col for col in df.columns if 'contribution' in col and 'client' in col] # Includes _contribution and _avg_contribution
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df