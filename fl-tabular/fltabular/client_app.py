"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger

from fltabular.task import (
    IncomeClassifier,
    evaluate,
    get_weights,
    load_data,
    set_weights,
    train,
)
from logging import INFO

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, is_attacker=False, partition_id=None, attack_rounds=None):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.is_attacker = is_attacker
        self.partition_id = partition_id
        self.attack_rounds = attack_rounds or []

    def fit(self, parameters, config):
        # Get current round from config (starts at 1)
        current_round = int(config.get("curr_round", 0)) 
        metrics = {"partition_id": self.partition_id}
        # Log label distribution in first round only
        if current_round == 0:
            pos_count = 0
            neg_count = 0
            # Count samples in each class
            for _, y_batch in self.trainloader:
                pos_count += (y_batch > 0.5).sum().item()
                neg_count += (y_batch <= 0.5).sum().item()
            
            total = pos_count + neg_count
            pos_pct = (pos_count / total * 100) if total > 0 else 0
            # Include in metrics
            metrics.update({
                "pos_count": pos_count,
                "neg_count": neg_count,
                "total_samples": total
            })
        
        # Regular training
        # Check if attacker should skip training (if current round is an attack round)
        # Note: current_round starts from 0, but attack_rounds uses 1-indexing
        should_attack = self.is_attacker and (current_round + 1) in self.attack_rounds
        if should_attack:
            # Attacker doesn't train in attack rounds, just returns received parameters
            return parameters, len(self.trainloader), metrics
        
        # Honest clients train normally in all rounds
        # Attacker trains normally in non-attack rounds
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), metrics

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_attackers = context.run_config["num-attackers"]
    # Only mark as attacker if attacks are enabled in config
    enable_attacks = context.run_config.get("enable-attacks", True)
    is_attacker = (partition_id < num_attackers) and enable_attacks
    
    # Get attack rounds configuration and parse from string
    attack_rounds_str = context.run_config.get("attack-rounds", "2,3,4,5")
    attack_rounds = [int(x.strip()) for x in attack_rounds_str.split(",") if x.strip()]
    
    # Get partitioning parameters from config
    partition_type = context.run_config.get("partition-type", "iid")
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 0.5))
    
    train_loader, test_loader = load_data(
        partition_id=partition_id, 
        num_partitions=context.node_config["num-partitions"],
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha
    )
    net = IncomeClassifier()
    client = FlowerClient(net, train_loader, test_loader, is_attacker, partition_id+1, attack_rounds)
    return client.to_client()


app = ClientApp(client_fn=client_fn)
