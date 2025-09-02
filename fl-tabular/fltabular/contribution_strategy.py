"""
Contribution evaluation strategies for federated learning.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from flwr.common import Parameters, ndarrays_to_parameters
import sys
import os

# Add the common directory to the path to import the shapley modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from common.torch_shapley import TorchGTGShapley

class ContributionEvaluationStrategy:
    """Base class for contribution evaluation strategies."""
    
    def __init__(self, model_cls, evaluate_fn):
        """Initialize the strategy.
        
        Args:
            model_cls: The model class to use for evaluation
            evaluate_fn: Function to evaluate a model: (model, test_loader) -> (loss, accuracy)
        """
        self.model_cls = model_cls
        self.evaluate_fn = evaluate_fn
        
    def evaluate_contribution(self, 
                             client_ids: List[int],
                             client_params: List[List[np.ndarray]],
                             client_examples: List[int],
                             global_params: List[np.ndarray],
                             global_loss: float,
                             global_acc: float,
                             test_loader: Any,
                             round_num: int) -> Tuple[Dict[str, Dict], Dict]:
        """Evaluate client contributions.
        
        Args:
            client_ids: List of client IDs
            client_params: List of client parameters (each is a list of numpy arrays)
            client_examples: List of client example counts
            global_params: Global model parameters
            global_loss: Loss of global model
            global_acc: Accuracy of global model
            test_loader: Test data loader for evaluation
            round_num: Current round number
            
        Returns:
            client_results: Dict mapping client IDs to contribution results
            round_metrics: Dict of metrics for this round
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def weighted_average(self, params_list, examples_list):
        """Compute weighted average of parameters."""
        weighted_params = []
        for params in zip(*params_list):
            weighted = sum(n * np.array(p) for p, n in zip(params, examples_list))
            weighted /= sum(examples_list)
            weighted_params.append(weighted)
        return weighted_params


class LeaveOneOutStrategy(ContributionEvaluationStrategy):
    """Leave-one-out contribution evaluation strategy."""
    
    def __init__(self, model_cls, evaluate_fn):
        # Remove tracker from init
        super().__init__(model_cls, evaluate_fn)
        
    def evaluate_contribution(self, 
                             client_ids: List[int],
                             client_params: List[List[np.ndarray]],
                             client_examples: List[int],
                             global_params: List[np.ndarray],
                             global_loss: float,
                             global_acc: float,
                             test_loader: Any,
                             round_num: int) -> Tuple[Dict[str, Dict], Dict]:
        """Evaluate client contributions using leave-one-out approach."""
        client_results = {}
        round_metrics = {
            "round": round_num,
            "global_loss": global_loss,
            "global_accuracy": global_acc,
        }
        
        num_clients = len(client_ids)
        for i in range(num_clients):
            client_id = client_ids[i]
            
            # Compute leave-one-out model parameters
            leave_examples = [n for j, n in enumerate(client_examples) if j != i]
            leave_params = [p for j, p in enumerate(client_params) if j != i]
            if sum(leave_examples) == 0:
                continue  # Skip if no examples remain
            
            # Compute LOO model
            loo_params = self.weighted_average(leave_params, leave_examples)
            model = self.model_cls()
            from fltabular.task import set_weights  # Import here to avoid circular imports
            set_weights(model, loo_params)
            loo_loss, loo_acc = self.evaluate_fn(model, test_loader)
            
            # Compute contribution as the difference between LOO and global metrics
            loss_contribution = loo_loss - global_loss
            acc_contribution = global_acc - loo_acc
            
            # Store results for this client
            client_results[client_id] = {
                "loo_loss": loo_loss,
                "loo_acc": loo_acc,
                "loss_contribution": loss_contribution,
                "acc_contribution": acc_contribution
            }
            
            # Add to round metrics
            round_metrics[f"client_{client_id}_loo_loss"] = loo_loss
            round_metrics[f"client_{client_id}_contribution"] = loss_contribution
            
        return client_results, round_metrics


class TorchGTGShapleyStrategy(ContributionEvaluationStrategy):
    """PyTorch-compatible GTG Shapley contribution evaluation strategy.
    
    This is the recommended implementation for evaluating client contributions
    using GTG Shapley values, as it's more efficient and supports PyTorch operations.
    """
    
    def __init__(self, model_cls, evaluate_fn, shapley_params=None):
        super().__init__(model_cls, evaluate_fn)
        # Initialize last_round_loss as None - we'll initialize it with the first round's global loss
        # but crucially, we'll only update it AFTER each round's Shapley calculation
        self.last_round_loss = None
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Store Shapley hyperparameters, providing defaults if none passed
        self.shapley_params = shapley_params if shapley_params else {
            "eps": 0.001,
            "round_trunc_threshold": 0.001,
            "convergence_criteria": 0.05,
            "last_k": 10,
            "converge_min": 30,
            "max_percentage": 0.8,
            "normalize": False
        }
        
    def evaluate_contribution(self, 
                             client_ids: List[int],
                             client_params: List[List[np.ndarray]],
                             client_examples: List[int],
                             global_params: List[np.ndarray],
                             global_loss: float,
                             global_acc: float,
                             test_loader: Any,
                             round_num: int) -> Tuple[Dict[str, Dict], Dict]:
        """Evaluate client contributions using PyTorch-compatible GTG Shapley value approach."""
        client_results = {}
        round_metrics = {
            "round": round_num,
            "global_loss": global_loss,
            "global_accuracy": global_acc,
        }
        
        # For the very first round only, we need to initialize last_round_loss
        # BUT NOT TO THE CURRENT ROUND'S LOSS, as that would make current_utility - last_round_loss = 0
        # and trigger between-round truncation
        if self.last_round_loss is None:
            # First round initialization: use a value that's just slightly different from the current loss
            # This ensures the between-round truncation check passes with minimal distortion
            epsilon = self.shapley_params["round_trunc_threshold"] * 10  # Arbitrary small value
            self.last_round_loss = global_loss * (1.0 + epsilon)  # Slightly higher than global_loss
        
        num_clients = len(client_ids)
        
        # Convert numpy arrays to torch tensors
        torch_params_list = []
        for params in client_params:
            torch_params = [torch.tensor(p, device=self.device) for p in params]
            torch_params_list.append(torch_params)
            
        # Initialize the PyTorch GTG Shapley calculator using stored parameters
        # We pass the previous round's loss as last_round_utility to enable between-round truncation
        shapley_calculator = TorchGTGShapley(
            num_players=num_clients,
            last_round_utility=self.last_round_loss,  # This is from the previous round
            device=self.device,
            eps=self.shapley_params["eps"],
            round_trunc_threshold=self.shapley_params["round_trunc_threshold"],
            convergence_criteria=self.shapley_params["convergence_criteria"],
            last_k=self.shapley_params["last_k"],
            converge_min=self.shapley_params["converge_min"],
            max_percentage=self.shapley_params["max_percentage"],
            normalize=self.shapley_params["normalize"],
        )
        
        # Define utility function for Shapley calculation
        def utility_fn(subset):
            # For empty subset, return the baseline utility (previous round's loss)
            if not subset:  # Empty subset
                return float(self.last_round_loss)
                
            # Select parameters and weights for the subset
            subset_params = [torch_params_list[i] for i in subset]
            subset_weights = [client_examples[i] for i in subset]
            
            if sum(subset_weights) == 0:
                return float(self.last_round_loss)
                
            # Compute weighted average
            avg_params = []
            total_weight = sum(subset_weights)
            normalized_weights = [w / total_weight for w in subset_weights]
            
            for param_idx in range(len(subset_params[0])):
                weighted_sum = sum(w * subset_params[i][param_idx] 
                                  for i, w in enumerate(normalized_weights))
                avg_params.append(weighted_sum)
            
            # Convert to numpy for evaluation
            numpy_params = [p.detach().cpu().numpy() for p in avg_params]
            
            # Evaluate model
            model = self.model_cls()
            from fltabular.task import set_weights  # Import here to avoid circular imports
            set_weights(model, numpy_params)
            loss, _ = self.evaluate_fn(model, test_loader)
            return loss
        
        # Set utility function
        shapley_calculator.set_utility_function(utility_fn)
        
        # Compute Shapley values
        shapley_values, shapley_values_S = shapley_calculator.compute(round_num)
        
        # CRITICAL: Update last_round_loss for next calculation AFTER Shapley calculation
        # This ensures the current round's Shapley values use the previous round's loss as baseline
        # and next round will correctly compare against this round's loss
        self.last_round_loss = global_loss
        
        # Add the used Shapley parameters to round metrics
        round_metrics["shapley_params"] = self.shapley_params
        
        # Process results for each client
        for i, client_id in enumerate(client_ids):
            shapley_value = shapley_values.get(i, 0.0)
            shapley_value_S = shapley_values_S.get(i, 0.0)
            
            # Store results for this client
            client_results[client_id] = {
                "shapley_value": float(shapley_value),
                "shapley_value_S": float(shapley_value_S),
                "loss_contribution": float(shapley_value),  # Use shapley value as contribution
                "acc_contribution": 0.0  # Not directly comparable to the LOO accuracy contribution
            }
            
            # Add to round metrics
            round_metrics[f"client_{client_id}_shapley"] = float(shapley_value)
            round_metrics[f"client_{client_id}_contribution"] = float(shapley_value)
            round_metrics[f"client_{client_id}_shapley_S"] = float(shapley_value_S)
            
        return client_results, round_metrics