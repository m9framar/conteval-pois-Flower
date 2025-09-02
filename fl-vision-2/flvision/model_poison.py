import torch
import numpy as np
import sys, os
from logging import INFO, WARNING
from flwr.common import logger
from flvision.task import Net, set_weights, test

# Add the common directory to the path to import the shapley module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from common.torch_shapley import TorchGTGShapley, DifferentiableTorchGTGShapley

class FunctionalCNN(torch.nn.Module):
    """Functional version of the CNN model for Fashion-MNIST."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, parameters):
        """Functional forward pass using external parameters.
        
        Parameters should match the order in the state_dict:
        - conv1.weight, conv1.bias
        - conv2.weight, conv2.bias
        - fc1.weight, fc1.bias
        - fc2.weight, fc2.bias
        """
        conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b = parameters
        
        # First conv layer
        x = torch.nn.functional.conv2d(x, conv1_w, conv1_b, stride=1, padding=0)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        
        # Second conv layer
        x = torch.nn.functional.conv2d(x, conv2_w, conv2_b, stride=1, padding=0)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, 32 * 4 * 4)
        
        # First fully connected layer
        x = torch.nn.functional.linear(x, fc1_w, fc1_b)
        x = torch.nn.functional.relu(x)
        
        # Output layer
        x = torch.nn.functional.linear(x, fc2_w, fc2_b)
        
        return x

def generate_contribution_attack(all_client_params, all_client_examples, all_client_ids,
                                global_params, prev_round_loss, attacker_idx, 
                                target_client_id, holdout_test_loader, contrib_method="loo"):
    """Generate malicious parameters to decrease target client's contribution.
    
    Parameters
    ----------
    all_client_params : list
        List of client parameters (each is a list of numpy arrays)
    all_client_examples : list
        List of client example counts
    all_client_ids : list
        List of client IDs
    global_params : list
        Global model parameters
    prev_round_loss : float
        Loss of global model from previous round
    attacker_idx : int
        Index of attacker client in the lists
    target_client_id : int
        ID of target client
    holdout_test_loader : torch.utils.data.DataLoader
        Test data loader for evaluation
    contrib_method : str, optional (default: "loo")
        Contribution evaluation method being used by the server.
        Options: "loo" (Leave-One-Out), "shapley" (Shapley values)
    """
    logger.log(INFO, f"Generating attack against target client {target_client_id} using {contrib_method} method")
    
    # Find target client's index
    try:
        target_idx = all_client_ids.index(target_client_id)
    except ValueError:
        logger.log(WARNING, f"Target client {target_client_id} not found")
        return all_client_params[attacker_idx]  # Return attacker's original params
    
    # Extract target client's data for reference
    target_params = all_client_params[target_idx]
    
    # Set device based on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log(INFO, f"Using device: {device}")
    
    # Calculate baseline honest contribution
    def evaluate_model(params):
        model = Net()
        set_weights(model, params)
        loss, _ = test(model, holdout_test_loader, device)
        return loss
    
    # FedAvg-style weighted averaging
    def compute_weighted_avg(params_list, weights):
        total_weight = sum(weights)
        avg_params = []
        for param_tensors in zip(*params_list):
            weighted_sum = sum(w * p for w, p in zip(weights, param_tensors))
            avg_params.append(weighted_sum / total_weight)
        return avg_params
    
    # Compute honest global model
    honest_weights = all_client_examples.copy()
    honest_global = compute_weighted_avg(all_client_params, honest_weights)
    honest_global_loss = evaluate_model(honest_global)
    
    # Calculate baseline honest contribution based on method
    honest_contribution = 0
    
    if contrib_method.lower() == "shapley":
        # Create players list for Shapley calculation
        players = []
        for i, client_id in enumerate(all_client_ids):
            params = None  # We don't need parameters here, just IDs and weights
            players.append((i, params, all_client_examples[i]))
        
        # Create Shapley calculator - pass number of players instead of players list
        num_players = len(players)
        shapley_calculator = TorchGTGShapley(
            num_players=num_players,
            last_round_utility=prev_round_loss,
            normalize=False,        
        )
        
        # Define evaluation function for Shapley calculation
        def shapley_eval_fn(indices):
            if not indices:  # Empty subset
                return prev_round_loss
                
            # Select parameters and weights for the indices
            subset_params = [all_client_params[i] for i in indices]
            subset_weights = [all_client_examples[i] for i in indices]
            
            # Compute weighted average and evaluate
            avg_params = compute_weighted_avg(subset_params, subset_weights)
            return evaluate_model(avg_params)
        
        # Set utility function and compute Shapley values
        shapley_calculator.set_utility_function(shapley_eval_fn)
        shapley_values, _ = shapley_calculator.compute(0)  # Round number doesn't matter here
        
        # Get target client's Shapley value
        honest_contribution = shapley_values.get(target_idx, 0)
        logger.log(INFO, f"Target client's honest Shapley value: {honest_contribution:.6f}")
    else:
        # Default to LOO method
        # Compute honest LOO model (excluding target)
        loo_params = [p for i, p in enumerate(all_client_params) if i != target_idx]
        loo_weights = [w for i, w in enumerate(honest_weights) if i != target_idx]
        
        honest_loo = compute_weighted_avg(loo_params, loo_weights)
        honest_loo_loss = evaluate_model(honest_loo)
        
        # Calculate target's honest contribution
        honest_contribution = honest_loo_loss - honest_global_loss
        logger.log(INFO, f"Target client's honest LOO contribution: {honest_contribution:.6f}")
    
    # Store best parameters and score
    best_params = None
    best_score = honest_contribution
    best_attack_scale = None
    best_init_method = None
    
    # ---- DIFFERENTIABLE ATTACK OPTIMIZATION ----
    
    # 1. Convert all client parameters to tensors ONCE
    tensor_client_params = [[torch.tensor(p, dtype=torch.float32, device=device) for p in params] 
                           for params in all_client_params]
    
    # Define differentiable weighted average function
    def differentiable_weighted_avg(params_list, weights_tensor):
        total_weight = torch.sum(weights_tensor)
        agg_params = []
        for param_idx in range(len(params_list[0])):
            params = [client_params[param_idx] for client_params in params_list]
            weighted_sum = sum(w * p for w, p in zip(weights_tensor, params))
            agg_params.append(weighted_sum / total_weight)
        return agg_params
    
    # Define differentiable evaluation function
    def evaluate_diff(model, parameters, test_loader):
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images, parameters)
            batch_loss = criterion(outputs, labels)
            total_loss += batch_loss
            batch_count += 1
            
            if batch_idx >= 5:  # Process 5 batches
                break
                
        return total_loss / batch_count
    
    # Create functional model instance
    functional_model = FunctionalCNN().to(device)
    
    # CNN models often need smaller perturbations to avoid degradation
    for attack_scale in [0.005, 0.01, 0.05, 0.1]:
        logger.log(INFO, f"Trying attack with scale={attack_scale}")
        
        # Initialization strategies for CNN models
        init_methods = [
            ("away_from_target", lambda: [
                torch.tensor(p, dtype=torch.float32, device=device).add(
                    torch.tensor(p, dtype=torch.float32, device=device).sub(torch.tensor(t, dtype=torch.float32, device=device))
                    .mul(attack_scale * 0.01)
                ).requires_grad_(True)
                for p, t in zip(global_params, target_params)
            ]),
            ("random_noise", lambda: [
                torch.tensor(p, dtype=torch.float32, device=device).add(
                    torch.normal(0, 0.005 * attack_scale, size=torch.tensor(p).shape, device=device)
                ).requires_grad_(True)
                for p in global_params
            ]),
            # More targeted perturbation for different layer types
            ("layer_aware", lambda: [
                # Apply different scales to different layer types
                torch.tensor(p, dtype=torch.float32, device=device).add(
                    torch.normal(
                        0, 
                        # Conv layers need smaller perturbations than FC layers
                        0.003 * attack_scale if ('conv' in name) else 0.01 * attack_scale, 
                        size=torch.tensor(p).shape, 
                        device=device
                    )
                ).requires_grad_(True)
                for name, p in zip(Net().state_dict().keys(), global_params)
            ]),
        ]
        
        for method_name, init_fn in init_methods:
            # Proper module structure for attack parameters
            class AttackModel(torch.nn.Module):
                def __init__(self, init_params):
                    super().__init__()
                    self.params = torch.nn.ParameterList([
                        torch.nn.Parameter(p) for p in init_params
                    ])
                
                def forward(self):
                    return list(self.params)
            
            # Initialize attack parameters
            attack_model = AttackModel(init_fn())
            optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)  # Lower learning rate for CNN
            
            # Optimization loop
            for iteration in range(200):
                optimizer.zero_grad()
                
                # Get current attack parameters
                attack_params = attack_model()
                
                # Create a federated simulation with attacker's current parameters
                fed_client_params = tensor_client_params.copy()
                fed_client_params[attacker_idx] = attack_params
                
                # Convert client weights to tensor
                fed_weights_tensor = torch.tensor([float(n) for n in all_client_examples], 
                                              dtype=torch.float32, 
                                              device=device)
                
                # Compute global aggregation (all clients)
                global_agg_params = differentiable_weighted_avg(fed_client_params, fed_weights_tensor)
                
                # Evaluate global model
                global_loss = evaluate_diff(functional_model, global_agg_params, holdout_test_loader)
                
                if contrib_method.lower() == "shapley":
                    # Use the differentiable TorchGTGShapley implementation
                    num_clients = len(all_client_ids)
                    
                    # Define hyperparameters for attack's Shapley calculation
                    # These might differ from server's evaluation for performance
                    attack_shapley_params = {
                        "eps": 0.01,  # Looser tolerance for faster attack optimization
                        "round_trunc_threshold": 0.01,
                        "convergence_criteria": 0.1, 
                        "last_k": min(5, num_clients), 
                        "converge_min": max(10, num_clients), 
                        "max_percentage": 0.5, # Evaluate fewer permutations
                        "normalize": False, # Usually False for attacks
                    }
                    
                    # Create the differentiable Shapley calculator
                    shapley_calc = DifferentiableTorchGTGShapley(
                        num_players=num_clients,
                        last_round_utility=prev_round_loss,
                        device=device,
                        **attack_shapley_params # Unpack the dictionary
                    )
                    
                    # Define utility function for Shapley calculation that supports autograd
                    def shapley_utility_fn(params):
                        # Directly evaluate the provided parameters
                        # This function now expects model parameters directly instead of indices
                        return evaluate_diff(functional_model, params, holdout_test_loader)
                    
                    # We specifically want the target client's Shapley value for the attack
                    target_contribution = shapley_calc.compute_differentiable(
                        utility_fn=shapley_utility_fn,
                        params_list=fed_client_params,
                        weights=[float(w) for w in all_client_examples],
                        target_client_id=target_idx
                    )
                else:
                    # Default to LOO method
                    # Compute LOO aggregation (all except target)
                    loo_params = [p for i, p in enumerate(fed_client_params) if i != target_idx]
                    loo_weights = torch.tensor(
                        [float(n) for i, n in enumerate(all_client_examples) if i != target_idx], 
                        dtype=torch.float32,
                        device=device
                    )
                    loo_agg_params = differentiable_weighted_avg(loo_params, loo_weights)
                    
                    # Evaluate LOO model
                    loo_loss = evaluate_diff(functional_model, loo_agg_params, holdout_test_loader)
                    
                    # Target contribution (minimize this)
                    target_contribution = loo_loss - global_loss
                    attack_shapley_params = None # No Shapley params used for LOO attack
                
                # Constraint as soft penalty
                max_allowed_degradation = 0.01  
                prev_loss_tensor = torch.tensor(prev_round_loss, dtype=torch.float32, device=device)
                constraint_penalty = torch.relu(global_loss - prev_loss_tensor * (1 + max_allowed_degradation)) * 10.0
                
                # Add L2 regularization - stronger for CNN to preserve model stability
                l2_penalty = sum(p.pow(2).sum() for p in attack_params) * 0.005
                
                # We want to MINIMIZE target_contribution for LOO (make it less positive)
                # We want to MAXIMIZE target_contribution for Shapley (make it less negative/more positive)
                # So, for Shapley, we minimize its negative.
                if contrib_method.lower() == "shapley":
                    objective = -target_contribution
                else: # LOO
                    objective = target_contribution
                    
                total_loss = objective + constraint_penalty + l2_penalty
                
                # Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                # Track results (convert tensors to CPU numpy for tracking)
                numpy_params = [p.detach().cpu().numpy() for p in attack_params]
                curr_target_contrib = target_contribution.item()
                curr_global_loss = global_loss.item()
                constraint_ok = curr_global_loss <= prev_round_loss * (1 + max_allowed_degradation)
                
                # Update best parameters if better and constraint satisfied
                # "Better" means the objective function value decreased.
                # For LOO, objective = target_contribution (want smaller positive)
                # For Shapley, objective = -target_contribution (want smaller positive, meaning target_contribution is larger/less negative)
                current_objective_value = objective.item()
                # Initialize best_objective_value if it's the first valid score
                if best_params is None and constraint_ok:
                    best_objective_value = current_objective_value
                    best_params = numpy_params
                    best_attack_scale = attack_scale
                    best_init_method = method_name
                    best_score = curr_target_contrib # Store the actual contribution score
                    logger.log(INFO, f"Initial best objective: {best_objective_value:.6f}, contrib: {best_score:.6f} with scale={attack_scale}, method={method_name}")
                elif constraint_ok and current_objective_value < best_objective_value:
                    best_objective_value = current_objective_value
                    best_params = numpy_params
                    best_attack_scale = attack_scale  # Track the best scale
                    best_init_method = method_name  # Track the best init method
                    best_score = curr_target_contrib # Store the actual contribution score
                    if iteration % 40 == 0:  # Only log at intervals
                        # Log the actual contribution score, not the objective value
                        logger.log(INFO, f"New best objective: {best_objective_value:.6f}, contrib: {best_score:.6f} (original: {honest_contribution:.6f}) with scale={attack_scale}, method={method_name}")
    
    # Return best parameters or global parameters if no improvement
    final_params = best_params if best_params is not None else global_params
    # Calculate improvement based on the actual contribution score
    improvement = 0
    if best_params is not None:
        if contrib_method.lower() == "shapley":
            # Improvement means score increased (less negative)
            improvement = best_score - honest_contribution 
        else: # LOO
            # Improvement means score decreased (less positive)
            improvement = honest_contribution - best_score
    
    if best_params is not None:
        logger.log(INFO, f"Attack complete. Best score {best_score:.6f} achieved with scale={best_attack_scale}, method={best_init_method}.")
        logger.log(INFO, f"Contribution changed: {honest_contribution:.6f} -> {best_score:.6f} (improvement metric: {improvement:.6f})")
        # Return malicious params, best attack hyperparams, and Shapley params used during attack
        return final_params, best_attack_scale, best_init_method, attack_shapley_params
    else:
        logger.log(INFO, f"Attack complete. No improvement found over honest contribution {honest_contribution:.6f}.")
        # Return global params, None for attack hyperparams, and None for Shapley params
        return final_params, None, None, None