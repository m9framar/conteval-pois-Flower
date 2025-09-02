import numpy as np
import json
import os
import wandb
import pandas as pd
from pathlib import Path
from flwr.common import logger
from logging import INFO

class ContributionTracker:
    """Utility class for tracking client contributions across rounds."""
    
    def __init__(self, use_wandb=False, run_config=None, project_name="FL-Contribution-Analysis"):
        self.client_contributions = {}  # client_id -> total contribution delta
        self.client_rounds = {}         # client_id -> number of rounds seen
        self.use_wandb = use_wandb
        self.run_config = run_config if run_config else {}
        self.results = {}
        self.project_name = project_name
        
        # New fields for tracking Shapley values
        self.client_shapley_values = {}  # client_id -> list of shapley values per round
        self.client_shapley_values_S = {}  # client_id -> list of best subset shapley values per round
        
        # Create output directory - use custom directory from config if provided
        output_dir = self.run_config.get("output-dir", "output")
        self.save_path = Path(output_dir)
        os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize wandb if needed
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize W&B project."""
        partition_type = self.run_config.get("partition-type", "iid")
        dirichlet_alpha = self.run_config.get("dirichlet-alpha", 0.5)
        name = f"contrib-analysis-{partition_type}"
        if partition_type == "dirichlet":
            name += f"-alpha{dirichlet_alpha}"
        
        wandb.init(project=self.project_name, name=name)
        # Log configuration as a config object
        wandb.config.update(self.run_config)
    
    def store_results(self, tag, results_dict):
        """Store results in dictionary, then save as JSON."""
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)
            
    def compute_client_contribution(self, client_id, loo_loss, loo_acc, global_loss, global_acc, rnd):
        """Compute and update contribution metrics for a client.
        
        NOTE: This function updates the internal state. If only calculation is needed,
        consider calculating directly. Currently used primarily for potential future
        strategies or direct calls, not in the standard LOO path via ContributionFedAvg.
        """
        # Compute contribution (positive values are better)
        loss_contribution = loo_loss - global_loss
        acc_contribution = global_acc - loo_acc
        
        # Update running totals
        self.client_contributions[client_id] = self.client_contributions.get(client_id, 0.0) + loss_contribution
        self.client_rounds[client_id] = self.client_rounds.get(client_id, 0) + 1
        avg_contrib = self.client_contributions[client_id] / self.client_rounds[client_id]
        
        logger.log(INFO, f"Round {rnd} - Client {client_id}: LOO loss={loo_loss:.4f}, contribution={loss_contribution:.4f}")
        
        return {
            "loo_loss": loo_loss,
            "loo_acc": loo_acc,
            "loss_contribution": loss_contribution,
            "acc_contribution": acc_contribution,
            "avg_contribution": avg_contrib
        }

    def update_cumulative_contribution(self, client_id, loss_contribution):
        """Update the tracker's internal state for cumulative contribution."""
        self.client_contributions[client_id] = self.client_contributions.get(client_id, 0.0) + loss_contribution
        self.client_rounds[client_id] = self.client_rounds.get(client_id, 0) + 1
    
    def track_shapley_values(self, shapley_values, shapley_values_S, rnd):
        """Track Shapley values for each client over rounds.
        
        Parameters
        ----------
        shapley_values : dict
            Dictionary mapping client_id to Shapley value for full set
        shapley_values_S : dict
            Dictionary mapping client_id to Shapley value for best subset
        rnd : int
            Current round number
        """
        for client_id, sv in shapley_values.items():
            if client_id not in self.client_shapley_values:
                self.client_shapley_values[client_id] = []
            self.client_shapley_values[client_id].append((rnd, sv))
            
            logger.log(INFO, f"Round {rnd} - Client {client_id} Shapley value: {sv:.4f}")
            
        for client_id, sv in shapley_values_S.items():
            if client_id not in self.client_shapley_values_S:
                self.client_shapley_values_S[client_id] = []
            self.client_shapley_values_S[client_id].append((rnd, sv))
            
            logger.log(INFO, f"Round {rnd} - Client {client_id} best subset Shapley value: {sv:.4f}")
        
        # Store results for later analysis
        shapley_result = {
            "round": rnd,
            "shapley_values": shapley_values,
            "shapley_values_S": shapley_values_S
        }
        self.store_results("shapley_values", shapley_result)
        
        return shapley_result
    
    def log_to_wandb(self, round_metrics, client_results):
        """Log metrics to wandb. Assumes internal state has been updated."""
        if not self.use_wandb:
            return
            
        wandb.log(round_metrics)
        
        # Create custom chart data
        client_contributions_log = []
        client_loo_losses_log = []
        
        for client_id, result in client_results.items():
            # Calculate average contribution based on *already updated* internal state
            rounds_seen = self.client_rounds.get(client_id, 1) # Avoid division by zero if state not updated (shouldn't happen)
            total_contrib = self.client_contributions.get(client_id, 0.0)
            avg_contrib = total_contrib / rounds_seen
            
            # Prepare data for logging using current round's contribution and calculated average
            client_contributions_log.append({
                "client_id": str(client_id),
                "contribution": result["loss_contribution"], # Current round's contribution
                "avg_contribution": avg_contrib # Calculated running average
            })
            
            client_loo_losses_log.append({
                "client_id": str(client_id),
                "loo_loss": result["loo_loss"]
            })
        
        # Create custom chart with all clients together
        wandb.log({
            "client_contributions": wandb.Table(
                dataframe=pd.DataFrame(client_contributions_log)
            ),
            "client_loo_losses": wandb.Table(
                dataframe=pd.DataFrame(client_loo_losses_log)
            ),
            "contribution_comparison": wandb.plot.bar(
                wandb.Table(
                    columns=["client_id", "contribution"],
                    data=[[c["client_id"], c["contribution"]] for c in client_contributions_log]
                ),
                "client_id", 
                "contribution",
                title=f"Client Contributions (Round {round_metrics['round']})"
            )
        })
    
    def log_shapley_values_to_wandb(self, round_number,round_metrics):
        """Log Shapley values to wandb."""
        if not self.use_wandb:
            return
        wandb.log(round_metrics)
        # Create data for visualization
        shapley_data = []
        shapley_data_S = []
        
        for client_id, values in self.client_shapley_values.items():
            if len(values) > 0 and values[-1][0] == round_number:
                shapley_data.append({
                    "client_id": str(client_id),
                    "shapley_value": values[-1][1]
                })
        
        for client_id, values in self.client_shapley_values_S.items():
            if len(values) > 0 and values[-1][0] == round_number:
                shapley_data_S.append({
                    "client_id": str(client_id),
                    "shapley_value_S": values[-1][1]
                })
        
        if shapley_data:
            # Log standard Shapley values
            wandb.log({

                "shapley_values_chart": wandb.plot.bar(
                    wandb.Table(
                        columns=["client_id", "shapley_value"],
                        data=[[d["client_id"], d["shapley_value"]] for d in shapley_data]
                    ),
                    "client_id", 
                    "shapley_value",
                    title=f"Client Shapley Values (Round {round_number})"
                )
            })
        
        if shapley_data_S:
            # Log best subset Shapley values
            wandb.log({

                "shapley_values_S_chart": wandb.plot.bar(
                    wandb.Table(
                        columns=["client_id", "shapley_value_S"],
                        data=[[d["client_id"], d["shapley_value_S"]] for d in shapley_data_S]
                    ),
                    "client_id", 
                    "shapley_value_S",
                    title=f"Best Subset Shapley Values (Round {round_number})"
                )
            })
        
        # Log trend of Shapley values over rounds (if we have data for at least 2 rounds)
        self._log_shapley_trends_to_wandb()
    
    def _log_shapley_trends_to_wandb(self):
        """Log trends of Shapley values over rounds to wandb."""
        if not self.use_wandb:
            return
            
        # Create data for line charts
        shapley_trends = []
        
        for client_id, values in self.client_shapley_values.items():
            for round_num, sv in values:
                shapley_trends.append({
                    "client_id": str(client_id),
                    "round": round_num,
                    "shapley_value": sv
                })
        
        if shapley_trends:
            df = pd.DataFrame(shapley_trends)
            wandb.log({
                "shapley_trends": wandb.plot.line(
                    wandb.Table(dataframe=df),
                    "round", 
                    "shapley_value",
                    title="Shapley Values Over Rounds",
                    stroke="client_id"  # Use stroke to group lines by client_id
                )
            })
    
    def log_cumulative_contributions(self):
        """Log cumulative contributions to console."""
        logger.log(INFO, "==== Cumulative client contributions ====")
        for cid in sorted(self.client_contributions.keys()):
            rounds = self.client_rounds.get(cid, 0)
            avg_contrib = self.client_contributions.get(cid, 0.0) / rounds if rounds > 0 else 0.0
            logger.log(INFO, f"Client {cid}: {avg_contrib:.4f} (over {rounds} rounds)")
    
    def log_shapley_summary(self):
        """Log summary of Shapley values to console."""
        logger.log(INFO, "==== Shapley Value Summary ====")
        for cid in sorted(self.client_shapley_values.keys()):
            values = [v for _, v in self.client_shapley_values.get(cid, [])]
            if values:
                avg_sv = sum(values) / len(values)
                last_sv = values[-1]
                logger.log(INFO, f"Client {cid}: Last={last_sv:.4f}, Avg={avg_sv:.4f} (over {len(values)} rounds)")
        
        logger.log(INFO, "==== Best Subset Shapley Value Summary ====")
        for cid in sorted(self.client_shapley_values_S.keys()):
            values = [v for _, v in self.client_shapley_values_S.get(cid, [])]
            if values and cid in self.client_shapley_values_S:
                avg_sv = sum(values) / len(values)
                last_sv = values[-1]
                logger.log(INFO, f"Client {cid}: Last={last_sv:.4f}, Avg={avg_sv:.4f} (over {len(values)} rounds)")


def weighted_average(params_list, examples_list):
    """Compute weighted average of parameters."""
    weighted_params = []
    for params in zip(*params_list):
        weighted = sum(n * np.array(p) for p, n in zip(params, examples_list))
        weighted /= sum(examples_list)
        weighted_params.append(weighted)
    return weighted_params