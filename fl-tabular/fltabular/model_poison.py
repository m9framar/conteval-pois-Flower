import torch
import numpy as np
import sys, os
from logging import INFO, WARNING
from flwr.common import logger, Parameters
from fltabular.task import IncomeClassifier, set_weights, evaluate

# Add the common directory to the path to import the shapley module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from common.torch_shapley import TorchGTGShapley, DifferentiableTorchGTGShapley

class FunctionalIncomeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, parameters):
        """Functional forward pass using external parameters directly"""
        fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias = parameters
        
        # First layer
        x = torch.nn.functional.linear(x, fc1_weight, fc1_bias)
        x = torch.nn.functional.relu(x)
        
        # Second layer
        x = torch.nn.functional.linear(x, fc2_weight, fc2_bias)
        x = torch.nn.functional.relu(x)
        
        # Output layer
        x = torch.nn.functional.linear(x, fc3_weight, fc3_bias)
        x = torch.sigmoid(x)
        
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
        model = IncomeClassifier()
        set_weights(model, params)
        loss, _ = evaluate(model, holdout_test_loader)
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
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch, parameters)
            outputs = outputs.view(-1)
            y_batch = y_batch.view(-1)
            batch_loss = criterion(outputs, y_batch)
            total_loss += batch_loss
            batch_count += 1
            
            if batch_idx >= 10:  # Process 10 batches
                break
                
        return total_loss / batch_count
    
    # Create functional model instance
    functional_model = FunctionalIncomeClassifier().to(device)
    
    # Try different attack initializations
    for attack_scale in [0.01, 0.1, 0.5, 1.0]:
        logger.log(INFO, f"Trying attack with scale={attack_scale}")
        
        # Initialization strategies
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
                    torch.normal(0, 0.01 * attack_scale, size=torch.tensor(p).shape, device=device)
                ).requires_grad_(True)
                for p in global_params
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
            optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.005)
            
            # Optimization loop
            for iteration in range(400):
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
                constraint_penalty = torch.relu(global_loss - prev_loss_tensor * (1 + max_allowed_degradation)) 
                
                # Add L2 regularization to prevent extreme parameter values
                l2_penalty = sum(p.pow(2).sum() for p in attack_params) * 0.001
                
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

# Helper functions for both attacks (target and self-promotion)
def _setup_attack_environment(holdout_test_loader):
    """Set up common attack environment components.
    
    Returns
    -------
    device : torch.device
        Device to use for computations
    evaluate_model : function
        Function to evaluate model parameters on the holdout set
    compute_weighted_avg : function
        Function to compute weighted average of parameters
    """
    # Set device based on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.log(INFO, f"Using device: {device}")
    
    # Define evaluation function
    def evaluate_model(params):
        model = IncomeClassifier()
        set_weights(model, params)
        loss, _ = evaluate(model, holdout_test_loader)
        return loss
    
    # FedAvg-style weighted averaging
    def compute_weighted_avg(params_list, weights):
        total_weight = sum(weights)
        avg_params = []
        for param_tensors in zip(*params_list):
            weighted_sum = sum(w * p for w, p in zip(weights, param_tensors))
            avg_params.append(weighted_sum / total_weight)
        return avg_params
    
    return device, evaluate_model, compute_weighted_avg

def _compute_honest_contribution(client_idx, all_client_params, all_client_examples, all_client_ids, 
                               evaluate_model, compute_weighted_avg, prev_round_loss, contrib_method="loo"):
    """Compute the honest contribution of a client.
    
    Parameters
    ----------
    client_idx : int
        Index of the client whose contribution to compute
    all_client_params : list
        List of client parameters
    all_client_examples : list
        List of client example counts
    all_client_ids : list
        List of client IDs
    evaluate_model : function
        Function to evaluate model parameters
    compute_weighted_avg : function
        Function to compute weighted average of parameters
    prev_round_loss : float
        Loss of global model from previous round
    contrib_method : str, optional (default: "loo")
        Contribution evaluation method ("loo" or "shapley")
    
    Returns
    -------
    honest_contribution : float
        The honest contribution of the client
    honest_global_loss : float
        The honest global model loss
    """
    # Compute honest global model
    honest_weights = all_client_examples.copy()
    honest_global = compute_weighted_avg(all_client_params, honest_weights)
    honest_global_loss = evaluate_model(honest_global)
    
    # Calculate baseline honest contribution based on method
    if contrib_method.lower() == "shapley":
        # Create players list for Shapley calculation
        players = []
        for i, client_id in enumerate(all_client_ids):
            params = None  # We don't need parameters here, just IDs and weights
            players.append((i, params, all_client_examples[i]))
        
        # Create Shapley calculator
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
        
        # Get client's Shapley value
        honest_contribution = shapley_values.get(client_idx, 0)
    else:
        # Default to LOO method
        # Compute honest LOO model (excluding client)
        loo_params = [p for i, p in enumerate(all_client_params) if i != client_idx]
        loo_weights = [w for i, w in enumerate(honest_weights) if i != client_idx]
        
        honest_loo = compute_weighted_avg(loo_params, loo_weights)
        honest_loo_loss = evaluate_model(honest_loo)
        
        # Calculate client's honest contribution
        honest_contribution = honest_loo_loss - honest_global_loss
    
    return honest_contribution, honest_global_loss

def _setup_differentiable_components(all_client_params, device, holdout_test_loader):
    """Set up differentiable components for attacks.
    
    Parameters
    ----------
    all_client_params : list
        List of client parameters
    device : torch.device
        Device to use for computations
    holdout_test_loader : torch.utils.data.DataLoader
        Test data loader for evaluation
    
    Returns
    -------
    tensor_client_params : list
        List of client parameters as tensors
    differentiable_weighted_avg : function
        Function to compute differentiable weighted average
    evaluate_diff : function
        Function to evaluate model using differentiable components
    functional_model : torch.nn.Module
        Functional model for differentiable evaluation
    """
    # Convert all client parameters to tensors
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
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch, parameters)
            outputs = outputs.view(-1)
            y_batch = y_batch.view(-1)
            batch_loss = criterion(outputs, y_batch)
            total_loss += batch_loss
            batch_count += 1
            
            if batch_idx >= 40:  # Process 10 batches
                break
                
        return total_loss / batch_count
    
    # Create functional model instance
    functional_model = FunctionalIncomeClassifier().to(device)
    
    return tensor_client_params, differentiable_weighted_avg, evaluate_diff, functional_model

def _compute_shapley_attack_contribution(shapley_calc, functional_model, fed_client_params, 
                                       all_client_examples, client_idx, holdout_test_loader, evaluate_diff):
    """Compute the Shapley contribution for an attack.
    
    Parameters
    ----------
    shapley_calc : DifferentiableTorchGTGShapley
        Shapley calculator
    functional_model : torch.nn.Module
        Functional model for differentiable evaluation
    fed_client_params : list
        List of client parameters as tensors with attack parameters
    all_client_examples : list
        List of client example counts
    client_idx : int
        Index of the client to compute contribution for
    holdout_test_loader : torch.utils.data.DataLoader
        Test data loader for evaluation
    evaluate_diff : function
        Function to evaluate model using differentiable components
    
    Returns
    -------
    client_contribution : torch.Tensor
        The client's contribution
    attack_shapley_params : dict
        Shapley parameters used for attack
    """
    # Define hyperparameters for attack's Shapley calculation
    attack_shapley_params = {
        "eps": 0.01,  # Looser tolerance for faster attack optimization
        "round_trunc_threshold": 0.01,
        "convergence_criteria": 0.1, 
        "last_k": min(5, len(all_client_examples)), 
        "converge_min": max(10, len(all_client_examples)), 
        "max_percentage": 0.5,
        "normalize": False,
    }
    
    # Define utility function for Shapley calculation that supports autograd
    def shapley_utility_fn(params):
        # Directly evaluate the provided parameters
        return evaluate_diff(functional_model, params, holdout_test_loader)
    
    # Compute client contribution
    client_contribution = shapley_calc.compute_differentiable(
        utility_fn=shapley_utility_fn,
        params_list=fed_client_params,
        weights=[float(w) for w in all_client_examples],
        target_client_id=client_idx
    )
    
    return client_contribution, attack_shapley_params

def _compute_loo_attack_contribution(fed_client_params, all_client_examples, client_idx, 
                                   differentiable_weighted_avg, functional_model, 
                                   holdout_test_loader, device, evaluate_diff):
    """Compute the LOO contribution for an attack.
    
    Parameters
    ----------
    fed_client_params : list
        List of client parameters as tensors with attack parameters
    all_client_examples : list
        List of client example counts
    client_idx : int
        Index of the client to compute contribution for
    differentiable_weighted_avg : function
        Function to compute differentiable weighted average
    functional_model : torch.nn.Module
        Functional model for differentiable evaluation
    holdout_test_loader : torch.utils.data.DataLoader
        Test data loader for evaluation
    device : torch.device
        Device to use for computations
    evaluate_diff : function
        Function to evaluate model using differentiable components
    
    Returns
    -------
    client_contribution : torch.Tensor
        The client's LOO contribution
    global_loss : torch.Tensor
        Global loss with all clients
    """
    # Convert client weights to tensor
    fed_weights_tensor = torch.tensor([float(n) for n in all_client_examples], 
                                    dtype=torch.float32,
                                    device=device)
    
    # Compute global aggregation (all clients)
    global_agg_params = differentiable_weighted_avg(fed_client_params, fed_weights_tensor)
    
    # Evaluate global model
    global_loss = evaluate_diff(functional_model, global_agg_params, holdout_test_loader)
    
    # Compute LOO aggregation (all except client)
    loo_params = [p for i, p in enumerate(fed_client_params) if i != client_idx]
    loo_weights = torch.tensor(
        [float(n) for i, n in enumerate(all_client_examples) if i != client_idx], 
        dtype=torch.float32,
        device=device
    )
    loo_agg_params = differentiable_weighted_avg(loo_params, loo_weights)
    
    # Evaluate LOO model
    loo_loss = evaluate_diff(functional_model, loo_agg_params, holdout_test_loader)
    
    # Client contribution
    client_contribution = loo_loss - global_loss
    
    return client_contribution, global_loss

def _generate_orthogonal_component(param_tensors_list, global_param, device, attack_scale=0.1):
    """Generate parameter components orthogonal to other clients' updates.
    
    Parameters
    ----------
    param_tensors_list : list
        List of parameter tensors from other clients
    global_param : torch.Tensor
        Global model parameter
    device : torch.device
        Device to use for computations
    attack_scale : float
        Scale of attack
    
    Returns
    -------
    orthogonal_component : torch.Tensor
        The orthogonal component
    """
    # Calculate the average direction of other clients
    avg_direction = torch.zeros_like(param_tensors_list[0])
    global_tensor = torch.tensor(global_param, dtype=torch.float32, device=device)
    
    for param in param_tensors_list:
        avg_direction += param - global_tensor
        
    if torch.norm(avg_direction) > 1e-6:
        # Normalize to get the direction
        avg_direction = avg_direction / torch.norm(avg_direction)
        
        # Generate a random perturbation
        random_pert = torch.normal(0, 0.01, size=avg_direction.shape, device=device)
        
        # Make it orthogonal to the average direction
        # proj = <random_pert, avg_direction> * avg_direction
        proj = torch.sum(random_pert * avg_direction) * avg_direction
        orthogonal_component = random_pert - proj
        
        # Scale the orthogonal component
        if torch.norm(orthogonal_component) > 1e-6:
            orthogonal_component = orthogonal_component / torch.norm(orthogonal_component) * attack_scale
            
        return orthogonal_component
    else:
        # If no clear direction from other clients, return random noise
        return torch.normal(0, 0.01 * attack_scale, size=param_tensors_list[0].shape, device=device)

def generate_self_promotion_attack(all_client_params, all_client_examples, all_client_ids,
                                global_params, prev_round_loss, attacker_idx,
                                holdout_test_loader, contrib_method="loo"):
    """Generate malicious parameters to increase attacker's own contribution.
    
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
    holdout_test_loader : torch.utils.data.DataLoader
        Test data loader for evaluation
    contrib_method : str, optional (default: "loo")
        Contribution evaluation method being used by the server.
        Options: "loo" (Leave-One-Out), "shapley" (Shapley values)
    """
    logger.log(INFO, f"Generating self-promotion attack for attacker client {all_client_ids[attacker_idx]} using {contrib_method} method")
    
    # Set up attack environment
    device, evaluate_model, compute_weighted_avg = _setup_attack_environment(holdout_test_loader)
    
    # Calculate baseline honest contribution for attacker
    honest_contribution, honest_global_loss = _compute_honest_contribution(
        attacker_idx, all_client_params, all_client_examples, all_client_ids,
        evaluate_model, compute_weighted_avg, prev_round_loss, contrib_method
    )
    
    if contrib_method.lower() == "shapley":
        logger.log(INFO, f"Attacker's honest Shapley value: {honest_contribution:.6f}")
    else:
        logger.log(INFO, f"Attacker's honest LOO contribution: {honest_contribution:.6f}")
    
    # Store best parameters and score
    best_params = None
    best_score = honest_contribution  
    best_attack_scale = None
    best_init_method = None
    best_objective_value = float('inf')  # Initialize with worst possible value
    
    # Set up differentiable components
    tensor_client_params, differentiable_weighted_avg, evaluate_diff, functional_model = _setup_differentiable_components(
        all_client_params, device, holdout_test_loader
    )
    
    # Try different attack initializations with special focus on self-promotion
    attack_scales = [0.01, 0.1, 0.5, 1.0]
    
    # Define initialization methods for self-promotion
    init_methods = [
        ("orthogonal_to_others", lambda: [
            # Generate parameters orthogonal to other clients' updates
            # This makes the attacker appear to provide unique information
            torch.tensor(global_params[i], dtype=torch.float32, device=device).add(
                _generate_orthogonal_component(
                    [tensor_client_params[j][i] for j in range(len(tensor_client_params)) if j != attacker_idx],
                    global_params[i], device, attack_scale
                )
            ).requires_grad_(True)
            for i in range(len(global_params))
        ]),
        # ("complementary", lambda: [
        #     # Focus on improving areas where global model is weakest
        #     torch.tensor(p, dtype=torch.float32, device=device).add(
        #         torch.normal(0, 0.01 * attack_scale, size=torch.tensor(p).shape, device=device)
        #     ).requires_grad_(True)
        #     for p in global_params
        # ]),
        ("random_noise", lambda: [
            torch.tensor(p, dtype=torch.float32, device=device).add(
                torch.normal(0, 0.01 * attack_scale, size=torch.tensor(p).shape, device=device)
            ).requires_grad_(True)
            for p in global_params
        ]),
    ]
    
    for attack_scale in attack_scales:
        logger.log(INFO, f"Trying self-promotion attack with scale={attack_scale}")
        
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
            optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.005)
            
            # Optimization loop
            for iteration in range(400):
                optimizer.zero_grad()
                
                # Get current attack parameters
                attack_params = attack_model()
                
                # Create a federated simulation with attacker's current parameters
                fed_client_params = tensor_client_params.copy()
                fed_client_params[attacker_idx] = attack_params
                
                # Calculate attacker contribution based on method
                if contrib_method.lower() == "shapley":
                    # Use the differentiable TorchGTGShapley implementation
                    num_clients = len(all_client_ids)
                    
                    # Create the differentiable Shapley calculator
                    shapley_calc = DifferentiableTorchGTGShapley(
                        num_players=num_clients,
                        last_round_utility=prev_round_loss,
                        device=device,
                        **{
                            "eps": 0.01,
                            "round_trunc_threshold": 0.01,
                            "convergence_criteria": 0.1,
                            "last_k": min(5, num_clients),
                            "converge_min": max(10, num_clients),
                            "max_percentage": 0.5,
                            "normalize": False,
                        }
                    )
                    
                    # Calculate contribution and get Shapley parameters
                    attacker_contribution, attack_shapley_params = _compute_shapley_attack_contribution(
                        shapley_calc, functional_model, fed_client_params,
                        all_client_examples, attacker_idx, holdout_test_loader, evaluate_diff
                    )
                    
                    # For Shapley, we want to minimize the attacker's contribution 
                    # (make it more negative, which is better for Shapley)
                    objective = attacker_contribution
                    
                    # Calculate global loss for constraints
                    fed_weights_tensor = torch.tensor(
                        [float(n) for n in all_client_examples],
                        dtype=torch.float32, device=device
                    )
                    global_agg_params = differentiable_weighted_avg(fed_client_params, fed_weights_tensor)
                    global_loss = evaluate_diff(functional_model, global_agg_params, holdout_test_loader)
                    
                else:  # LOO
                    # Calculate LOO contribution
                    attacker_contribution, global_loss = _compute_loo_attack_contribution(
                        fed_client_params, all_client_examples, attacker_idx,
                        differentiable_weighted_avg, functional_model,
                        holdout_test_loader, device, evaluate_diff
                    )
                    
                    # For LOO, we want to maximize the attacker's contribution
                    # (make it more positive)
                    objective = -attacker_contribution
                    attack_shapley_params = None
                
                # Constraint as soft penalty (same as in target attack)
                max_allowed_degradation = 0.02  # Increased from 0.01 to allow more flexibility
                prev_loss_tensor = torch.tensor(prev_round_loss, dtype=torch.float32, device=device)
                constraint_penalty = torch.relu(global_loss - prev_loss_tensor * (1 + max_allowed_degradation)) * 10.0
                
                # Add baseline protection constraint - never be worse than honest contribution
                honest_contrib_tensor = torch.tensor(honest_contribution, dtype=torch.float32, device=device)
                if contrib_method.lower() == "shapley":
                    # For Shapley, worse means less negative (higher value)
                    baseline_violation = torch.relu(attacker_contribution - honest_contrib_tensor) * 5.0
                    optimization_direction=1.0
                else:  # LOO
                    # For LOO, worse means less positive (lower value)
                    baseline_violation = torch.relu(honest_contrib_tensor - attacker_contribution) * 5.0
                    optimization_direction=-1.0
                # Add L2 regularization to prevent extreme parameter values
                l2_penalty = sum(p.pow(2).sum() for p in attack_params) * 0.001
                
                # For complementary initialization, add an additional term to encourage
                # the attacker to focus on weak areas of the model
                if method_name == "complementary":
                    # TODO: Future Enhancement - Implement a meaningful specialized penalty
                    # Currently the "complementary" initialization is similar to random noise
                    # A proper implementation would analyze where the global model performs poorly
                    # and focus the attack parameters on improving those areas
                    pass
                    

                # Make constraints aware of optimization direction
                total_loss = optimization_direction * attacker_contribution + \
                            constraint_penalty + l2_penalty + baseline_violation
                # Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                # Track results (convert tensors to CPU numpy for tracking)
                numpy_params = [p.detach().cpu().numpy() for p in attack_params]
                curr_attacker_contrib = attacker_contribution.item()
                curr_global_loss = global_loss.item()
                constraint_ok = curr_global_loss <= prev_round_loss * (1 + max_allowed_degradation)
                
                # Calculate improvement based on contribution method
                if contrib_method.lower() == "shapley":
                    # For Shapley, better means more negative
                    is_improvement = curr_attacker_contrib < honest_contribution
                else:
                    # For LOO, better means more positive
                    is_improvement = curr_attacker_contrib > honest_contribution

                
                
                # Only accept solutions that satisfy constraints AND improve over honest contribution
                acceptable_solution = constraint_ok and is_improvement
                
                # Update best parameters if better and constraints satisfied
                current_objective_value = objective.item()
                
                # Initialize best_objective_value if it's the first valid score
                if best_params is None and acceptable_solution:
                    best_objective_value = current_objective_value
                    best_params = numpy_params
                    best_attack_scale = attack_scale
                    best_init_method = method_name
                    best_score = curr_attacker_contrib
                    logger.log(INFO, f"Initial best objective: {best_objective_value:.6f}, contrib: {best_score:.6f}")
                elif acceptable_solution and current_objective_value < best_objective_value:
                    best_objective_value = current_objective_value
                    best_params = numpy_params
                    best_attack_scale = attack_scale
                    best_init_method = method_name
                    best_score = curr_attacker_contrib
                    if iteration % 40 == 0:  # Log at intervals
                        logger.log(INFO, f"New best objective: {best_objective_value:.6f}, contrib: {best_score:.6f}")
    
    # Return best parameters or global parameters if no improvement
    final_params = best_params if best_params is not None else global_params
    
    # Calculate improvement based on the correct metric direction
    improvement = 0
    if best_params is not None:
        if contrib_method.lower() == "shapley":
            # For Shapley, improvement means more negative (better) values
            improvement = honest_contribution - best_score
        else:
            # For LOO, improvement means more positive values
            improvement = best_score - honest_contribution
    
    if best_params is not None:
        logger.log(INFO, f"Self-promotion attack complete. Best score {best_score:.6f} achieved with scale={best_attack_scale}, method={best_init_method}.")
        logger.log(INFO, f"Contribution changed: {honest_contribution:.6f} -> {best_score:.6f} (improvement: {improvement:.6f})")
        return final_params, best_attack_scale, best_init_method, attack_shapley_params
    else:
        logger.log(INFO, f"Self-promotion attack complete. No improvement found over honest contribution {honest_contribution:.6f}.")
        return final_params, None, None, None