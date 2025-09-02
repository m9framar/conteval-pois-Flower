import numpy as np
import torch
import copy
from flwr.server.strategy.fedavg import FedAvg
from fltabular.task import IncomeClassifier, set_weights, evaluate
from flwr.common import Context, logger, parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Status
from logging import INFO, WARNING
from fltabular.contribution_utils import ContributionTracker, weighted_average
from fltabular.model_poison import generate_contribution_attack, generate_self_promotion_attack
from fltabular.contribution_strategy import LeaveOneOutStrategy, TorchGTGShapleyStrategy

PROJECT_NAME = "FL-Contribution-Analysis-Tabular"

class ContributionFedAvg(FedAvg):
    """FedAvg strategy with client contribution evaluation."""
    
    def __init__(self, holdout_test_loader, use_wandb=False, run_config=None, 
                 attacker_client_id=1, target_client_id=2, **kwargs):
        super().__init__(**kwargs)
        self.holdout_test_loader = holdout_test_loader
        self.tracker = ContributionTracker(
            use_wandb=use_wandb,
            run_config=run_config,
            project_name=PROJECT_NAME
        )
        self.attacker_client_id = attacker_client_id
        self.target_client_id = target_client_id
        self.prev_global_loss = float('inf')
        # Read attack configuration, defaulting to True if not specified
        self.enable_attacks = run_config.get("enable-attacks", True) 
        
        # Read attack rounds configuration and parse from string
        attack_rounds_str = run_config.get("attack-rounds", "2,3,4,5")
        self.attack_rounds = [int(x.strip()) for x in attack_rounds_str.split(",") if x.strip()]
        
        # Configure contribution evaluation strategy
        self.contrib_method = run_config.get("contribution-method", "loo").lower()
        
        # Configure attack type, defaulting to "targeted" if not specified
        self.attack_type = run_config.get("attack-type", "targeted").lower()
        
        # Read Shapley hyperparameters from config, providing defaults
        self.shapley_params = {
            "eps": run_config.get("shapley-eps", 0.001),
            "round_trunc_threshold": run_config.get("shapley-round-trunc-threshold", 0.001),
            "convergence_criteria": run_config.get("shapley-convergence-criteria", 0.05),
            "last_k": run_config.get("shapley-last-k", 10),
            "converge_min": run_config.get("shapley-converge-min", 30),
            "max_percentage": run_config.get("shapley-max-percentage", 0.8),
            "normalize": run_config.get("shapley-normalize", False)
        }
        
        if self.contrib_method == "shapley" or self.contrib_method == "gtg-shapley":
            logger.log(INFO, "Using GTG Shapley contribution evaluation strategy")
            # Pass Shapley hyperparameters to the strategy
            self.contrib_strategy = TorchGTGShapleyStrategy(
                IncomeClassifier, 
                evaluate, 
                shapley_params=self.shapley_params
            )
        else:
            logger.log(INFO, "Using Leave-One-Out contribution evaluation strategy")
            self.contrib_strategy = LeaveOneOutStrategy(IncomeClassifier, evaluate)
            
        if not self.enable_attacks:
            logger.log(INFO, "Attack logic is disabled via configuration")
        else:
            logger.log(INFO, f"Using attack type: {self.attack_type}")
            logger.log(INFO, f"Attack rounds configured: {self.attack_rounds}")
    
    def _apply_attack_if_needed(self, rnd, results):
        """Extract client data and apply attack if attacker is present and attacks are enabled."""
        # Initialize attack hyperparams to None
        best_attack_scale = None
        best_init_method = None
        attack_shapley_params = None # Initialize attacker Shapley params
        
        # Skip attack logic entirely if disabled in config
        if self.enable_attacks== False:
            logger.log(INFO, "Attack logic is disabled via configuration")
            return results, best_attack_scale, best_init_method, attack_shapley_params # Return None for all hyperparams
            
        # Extract ALL client parameters in one pass - no separation needed
        all_client_params = []
        all_client_examples = []
        all_client_ids = []
        attacker_idx = None  # Track attacker position
        
        for idx, (client_proxy, fit_res) in enumerate(results):
            client_id = fit_res.metrics.get("partition_id")
            all_client_ids.append(client_id)
            all_client_params.append(parameters_to_ndarrays(fit_res.parameters))
            all_client_examples.append(fit_res.num_examples)
            
            # Track attacker position
            if client_id == self.attacker_client_id:
                attacker_idx = idx
                attacker_proxy = client_proxy
                attacker_fit_res = fit_res
        
        # If attacker is present and current round is an attack round, generate malicious parameters
        if attacker_idx is not None and rnd in self.attack_rounds and hasattr(self, "current_global_params"):
            logger.log(INFO, f"Generating malicious parameters for client {self.attacker_client_id} in round {rnd}")
            
            # Generate malicious parameters using the appropriate attack function
            if self.attack_type == "self-promotion":
                # Apply a self-promotion attack (no target client needed)
                malicious_params, best_attack_scale, best_init_method, attack_shapley_params = generate_self_promotion_attack(
                    all_client_params,
                    all_client_examples,
                    all_client_ids,
                    self.current_global_params,
                    self.prev_global_loss,
                    attacker_idx,
                    self.holdout_test_loader,
                    self.contrib_method  # Pass the contribution method to the attack generator
                )
            else:
                # Default to the targeted attack
                malicious_params, best_attack_scale, best_init_method, attack_shapley_params = generate_contribution_attack(
                    all_client_params,
                    all_client_examples,
                    all_client_ids,
                    self.current_global_params,
                    self.prev_global_loss,
                    attacker_idx,
                    self.target_client_id,
                    self.holdout_test_loader,
                    self.contrib_method  # Pass the contribution method to the attack generator
                )
            
            # Create new FitRes with malicious parameters
            new_fit_res = FitRes(
                status=Status(0, "Success"),
                parameters=ndarrays_to_parameters(malicious_params),
                num_examples=attacker_fit_res.num_examples,
                metrics=attacker_fit_res.metrics,
            )
            
            # Replace attacker's result directly in the results list
            results[attacker_idx] = (attacker_proxy, new_fit_res)
            logger.log(INFO, f"Replaced attacker's parameters with malicious parameters")
        
        # Return the potentially modified results and all attack hyperparameters
        return results, best_attack_scale, best_init_method, attack_shapley_params
            
    def aggregate_fit(self, rnd, results, failures):
        # First check if we have results
        if not results:
            logger.log(WARNING, "No clients found, skipping aggregation")
            return super().aggregate_fit(rnd, results, failures)
        
        # Apply attack if conditions are met and attacks are enabled
        # Capture attacker Shapley params as well
        results, best_attack_scale, best_init_method, attack_shapley_params = self._apply_attack_if_needed(rnd, results)
        
        # Call parent's aggregate_fit with possibly modified results
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if not results:
            return aggregated_result

        # Extract all client results and persistent IDs
        client_params = []
        client_examples = []
        client_ids = []
        for cp, fit_res in results:
            client_ids.append(fit_res.metrics["partition_id"])
            params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_params.append(params_ndarrays)
            client_examples.append(fit_res.num_examples)

        # Unpack aggregated_result and evaluate global model
        params, _ = aggregated_result
        global_params = parameters_to_ndarrays(params)
        # Save global parameters for next round
        self.current_global_params = global_params
        
        global_model = IncomeClassifier()
        set_weights(global_model, global_params)
        global_loss, global_acc = evaluate(global_model, self.holdout_test_loader)
        
        # Save loss for next round's constraint
        self.prev_global_loss = global_loss
        
        logger.log(INFO, f"Round {rnd} - Global model: loss={global_loss:.4f}, accuracy={global_acc:.4f}")
        
        # Evaluate client contributions using the selected strategy
        client_results, round_metrics = self.contrib_strategy.evaluate_contribution(
            client_ids,
            client_params,
            client_examples,
            global_params,
            global_loss,
            global_acc,
            self.holdout_test_loader,
            rnd
        )

        # --- Update Tracker State (if LOO) ---
        if self.contrib_method == "loo":
            for client_id, result in client_results.items():
                self.tracker.update_cumulative_contribution(client_id, result["loss_contribution"])
        # --- End Update Tracker State ---

        # Add best attack hyperparameters to round metrics if they exist
        if best_attack_scale is not None:
            round_metrics["best_attack_scale"] = best_attack_scale
        if best_init_method is not None:
            round_metrics["best_init_method"] = best_init_method
        # Add attacker's Shapley parameters used during optimization
        if attack_shapley_params is not None:
            round_metrics["attack_shapley_params"] = attack_shapley_params

        # Store round metrics (global loss/acc, per-client contrib/loo/shapley, attack hyperparams)
        self.tracker.store_results("fit", round_metrics)

        # Log to WandB based on contribution method
        if self.contrib_method == "shapley":
            # Extract Shapley values from client_results
            shapley_values = {cid: res["shapley_value"] for cid, res in client_results.items()}
            shapley_values_S = {cid: res["shapley_value_S"] for cid, res in client_results.items()}
            
            # Track Shapley values (this also stores them internally)
            self.tracker.track_shapley_values(shapley_values, shapley_values_S, rnd)
            
            # Log Shapley-specific charts to WandB
            self.tracker.log_shapley_values_to_wandb(rnd, round_metrics)
            # Optionally log Shapley summary to console (might be better at end of run)
            self.tracker.log_shapley_summary() 
        else:  # Default to LOO logging
            # Log LOO-specific charts to WandB (using round-specific client_results)
            self.tracker.log_to_wandb(round_metrics, client_results)
            # Log cumulative LOO contributions to console (uses internal tracker state)
            self.tracker.log_cumulative_contributions()
        
        return aggregated_result

# Backward compatibility: Keep LeaveOneOutFedAvg as an alias for ContributionFedAvg
LeaveOneOutFedAvg = ContributionFedAvg