"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.common import ndarrays_to_parameters
from fltabular.task import IncomeClassifier, get_weights, load_data
from fltabular.leave_one_out import LeaveOneOutFedAvg

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flwr.common import logger
from logging import INFO
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def load_holdout_data(context: Context):
    # Use the whole dataset by setting num_partitions=1
    # We ignore the train part and only return the test loader.
    partition_type = context.run_config.get("partition-type", "iid")
    logger.log(INFO, f"Using partition type: {partition_type}")
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.5))
    _, test_loader = load_data( partition_id=0, 
                                num_partitions=1,
                                partition_type=partition_type, 
                                dirichlet_alpha=dirichlet_alpha)
    

    return test_loader

def on_fit_config(server_round: int):
    """Return configuration dict for client training."""
    # Note: Flower server rounds start at 1, but we want client round 0 to be the first round
    return {
        "curr_round": server_round - 1
    }

def aggregate_fit_metrics(metrics_list):
    """Aggregate fit metrics from clients."""
    
    # First check if we have label distribution metrics (first round)
    has_distribution = any("pos_count" in metrics for _, metrics in metrics_list)
    
    if has_distribution:
        # This is the first round, so print label distribution summary
        logger.log(INFO, "=== Label Distribution Across Clients ===")
        
        # Sort by partition ID for consistent display
        sorted_metrics = sorted(metrics_list, key=lambda x: x[1].get("partition_id", 0))
        
        for _, metrics in sorted_metrics:
            if "pos_count" in metrics:
                partition_id = metrics.get("partition_id", "unknown")
                pos_count = metrics.get("pos_count", 0)
                total = metrics.get("total_samples", 0)
                pos_pct = (pos_count / total * 100) if total > 0 else 0
                neg_count = total - pos_count
                
                logger.log(INFO, f"Client {partition_id}: {pos_pct:.1f}% positive income ({pos_count}/{total} samples)")
        
        logger.log(INFO, "========================================")
    
    # Return empty dict as we're just using this for logging
    return {}

def server_fn(context: Context) -> ServerAppComponents:
    net = IncomeClassifier()
    params = ndarrays_to_parameters(get_weights(net))
    holdout_test_loader = load_holdout_data(context)
    use_wandb = context.run_config.get("use-wandb", False)
    strategy = LeaveOneOutFedAvg(
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,  
        holdout_test_loader=holdout_test_loader,
        on_fit_config_fn=on_fit_config,
        use_wandb=use_wandb,
        run_config=context.run_config,
        attacker_client_id=int(context.run_config.get("attacker-client-id", 1)),
        target_client_id=int(context.run_config.get("target-client-id", 2))
    )
    
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy)

app = ServerApp(server_fn=server_fn)
