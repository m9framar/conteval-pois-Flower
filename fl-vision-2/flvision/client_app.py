"""Fashion MNIST: Flower CNN Vision Example."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger

from flvision.task import (
    Net,
    train,
    test,
    get_weights,
    load_data,
    set_weights,
)
from logging import INFO
import torch
import json

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, is_attacker=False, partition_id=None, attack_rounds=None):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.is_attacker = is_attacker
        self.partition_id = partition_id
        self.attack_rounds = attack_rounds or []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        # Get current round from config (starts at 0)
        current_round = int(config.get("curr_round", 0))
        epochs = int(config.get("epochs", 1))
        learning_rate = float(config.get("lr", 0.01))
        
        metrics = {"partition_id": self.partition_id}
        
        # Log label distribution in first round only
        if current_round == 0:
            label_counts = {}
            total_samples = 0
            
            # Count samples in each class
            for batch in self.trainloader:
                labels = batch["label"]
                for label in labels:
                    label_val = int(label.item())
                    # Convert integer keys to strings for serialization
                    label_key = str(label_val)
                    label_counts[label_key] = label_counts.get(label_key, 0) + 1
                    total_samples += 1
            
            # Convert dictionary to JSON string for serialization
            metrics.update({
                "label_counts_str": json.dumps(label_counts),  # Serialize to string
                "total_samples": total_samples
            })
        
        # Regular training
        # Check if attacker should skip training (if current round is an attack round)
        # Note: current_round starts from 0, but attack_rounds uses 1-indexing
        should_attack = self.is_attacker and (current_round + 1) in self.attack_rounds
        if should_attack:
            # Attacker doesn't train in attack rounds, just returns received parameters
            return parameters, len(self.trainloader.dataset), metrics
        
        # Honest clients train normally in all rounds
        # Attacker trains normally in non-attack rounds
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, epochs, learning_rate, self.device)
        metrics["train_loss"] = train_loss
        
        return get_weights(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

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
    dirichlet_alpha = float(context.run_config.get("dirichlet-alpha", 1.0))
    
    train_loader, test_loader = load_data(
        partition_id=partition_id, 
        num_partitions=context.node_config["num-partitions"],
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha
    )
    
    # Initialize model
    net = Net()
    
    # Create client
    client = FlowerClient(
        net=net, 
        trainloader=train_loader, 
        testloader=test_loader, 
        is_attacker=is_attacker, 
        partition_id=partition_id+1,
        attack_rounds=attack_rounds
    )
    
    return client.to_client()

app = ClientApp(client_fn=client_fn)
