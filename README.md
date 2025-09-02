# Contribution-Based Poisoning Attack in Federated Learning

This project implements a targeted poisoning attack in federated learning that manipulates a specific client's contribution score, supporting both leave-one-out (LOO) and Guided Truncation Gradient Shapley (GTG-Shapley) evaluation methods. The attack works by having a malicious client submit adversarial model parameters that make the target client appear less valuable to the federation according to the chosen metric.

## Overview

In federated learning, client contribution is often measured using methods like leave-one-out (LOO) evaluation or Shapley values. LOO compares the performance of the global model with and without a specific client's update, while Shapley values provide a theoretically sound measure of contribution based on evaluating various client coalitions. This project demonstrates how a malicious client can manipulate these contribution metrics by submitting carefully crafted parameters.

### Key Features

- Differentiable optimization-based attack on client contribution scores (LOO and GTG-Shapley)
- Federation-aware simulation for accurate parameter manipulation
- Implementation of GTG-Shapley contribution evaluation
- Support for both IID and non-IID (Dirichlet) data distributions
- Integration with Weights & Biases (wandb) for experiment tracking
- Automated experiment runner for systematic evaluation

## Project Structure

```
2024-balazs-frank-marcell-poisoning-shapley/
├── fl-tabular/                     # Tabular data implementation (Adult Income dataset)
│   ├── fltabular/
│   │   ├── client_app.py           # Client implementation for tabular data
│   │   ├── contribution_strategy.py # LOO and GTG-Shapley strategies
│   │   ├── contribution_utils.py   # Utilities for tracking contributions
│   │   ├── leave_one_out.py        # Server strategy integrating contribution eval
│   │   ├── model_poison.py         # Contribution attack for tabular model
│   │   ├── server_app.py           # Server implementation
│   │   └── task.py                 # Tabular model and data loading
│   └── pyproject.toml              # Project configuration
├── fl-vision-2/                    # Vision data implementation (Fashion-MNIST)
│   ├── flvision/
│   │   ├── client_app.py           # Client implementation for vision data
│   │   ├── contribution_strategy.py # LOO and GTG-Shapley strategies (vision)
│   │   ├── contribution_utils.py   # Utilities for tracking contributions
│   │   ├── leave_one_out.py        # Server strategy integrating contribution eval (vision)
│   │   ├── model_poison.py         # Contribution attack for CNN models
│   │   ├── server_app.py           # Server implementation 
│   │   └── task.py                 # CNN model and Fashion-MNIST data loading
│   └── pyproject.toml              # Project configuration
├── common/                         # Shared utilities
│   ├── torch_shapley.py          # PyTorch-compatible GTG-Shapley implementation
│   └── shapley.py                  # Deprecated Shapley value computation utilities
├── run_experiments.sh              # Experiment automation script
└── README.md                       # This file
```

## Setup

This project has been tested with Python 3.10 and 3.12 on:
- Ubuntu 22.04
- Ubuntu 24.04

Windows compatibility has not been tested.

### Environment Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the project packages:
   ```bash
   pip install -e fl-tabular/
   pip install -e fl-vision-2/
   ```

## Usage

### Running a Single Experiment

You can configure the experiment, including the contribution evaluation method, in the `pyproject.toml` file (see Attack Configuration section) or override settings via the command line.

To run a single experiment with the tabular dataset (using settings from `pyproject.toml`):

```bash
cd fl-tabular
flwr run .
```

To override settings, for example, to use LOO and disable attacks:

```bash
cd fl-tabular
flwr run . --run-config 'contribution-method="loo" enable-attacks=False'
```

For vision dataset (Fashion-MNIST):

```bash
cd fl-vision-2
flwr run . --run-config 'contribution-method="shapley" enable-attacks=True partition-type="dirichlet" dirichlet-alpha=0.5'
```

### Automated Experiments

The `run_experiments.sh` bash script automates running multiple experiments with different configurations. It handles both datasets, supports different data distributions, and organizes results in a structured directory hierarchy. **Note:** This script may need updates to correctly handle the `contribution-method` setting if you want to systematically compare LOO and Shapley attacks.

### WandB Logging
If enabled (`--wandb=true` in `run_experiments.sh` or `use_wandb=True` in server config), Weights & Biases will track experiments. Key metrics logged include:
- Time-series plots for global loss and accuracy.
- Time-series plots for per-client contribution scores (LOO delta or Shapley values).
- Tables summarizing contribution scores over rounds.
- Configuration parameters and final results summary.

```bash
./run_experiments.sh [dataset] [num-runs] [use-wandb] [distribution-modes]
```

Where:
- `dataset`: Type of dataset to use ("tabular" or "fashion")
- `num-runs`: Number of experiment runs per configuration
- `use-wandb`: Enable Weights & Biases logging ("true" or "false")
- `distribution-modes`: Space-separated list of data distribution modes in quotes (e.g., "iid dirichlet_0.1 dirichlet_0.5")

Example:
```bash
./run_experiments.sh tabular 5 true "iid dirichlet_0.5 dirichlet_1.0"
```

## Attack Configuration

The attack and contribution method can be configured in the `pyproject.toml` file (e.g., `fl-tabular/pyproject.toml`):

```toml
[tool.flwr.app.config]
num-server-rounds = 5
num-attackers = 1
attacker-client-id = 1
target-client-id = 2
enable-attacks = true  # Set to false to disable attack logic
contribution-method = "shapley" # Options: "loo", "shapley"

# Shapley Hyperparameters (only used if contribution-method is "shapley")
shapley-eps = 0.001
shapley-round-trunc-threshold = 0.001
shapley-convergence-criteria = 0.05
shapley-last-k = 10
shapley-converge-min = 30
shapley-max-percentage = 0.8
shapley-normalize = false
#shapley-calculate-best-subset = false # Optional: Calculate Shapley values for the best performing subset (shapley_values_S) Currently not implemented!
```

The attack optimization parameters can also be adjusted in `model_poison.py`:

```python
# Try different attack initializations
for attack_scale in [0.01, 0.1, 0.5, 1.0]:
    # ...

# Constraint as soft penalty
max_allowed_degradation = 0.10
```

## How the Attack Works

1.  The attacker creates a differentiable simulation of the federation using `torch.func`.
2.  It optimizes its parameters to make the target client appear worse according to the chosen contribution metric (LOO or GTG-Shapley).
    -   For LOO (loss-based): Minimize the target's positive contribution score.
    -   For GTG-Shapley (loss-based): Maximize the target's negative contribution score (minimize `-target_contribution`).
    *   **Note on Shapley Optimization**: For performance, the GTG-Shapley attack optimizes a *differentiable Monte Carlo approximation* of the target's Shapley value, not the full GTG algorithm used by the server for evaluation.
3.  The optimization includes a penalty term to discourage violating a performance constraint (`max_allowed_degradation` relative to the previous round's global loss).
    *   **Note on Constraint Evaluation**: This is a *soft constraint*. Furthermore, during optimization, the constraint is evaluated using only the *first 10 batches* of the holdout test data for speed. The server evaluates performance on the *entire* holdout set.
4.  The attack is federation-aware, modeling how client contributions are evaluated based on aggregated parameters.

## Experimental Results

The attack effectiveness can be compared between LOO and GTG-Shapley contribution methods. It is generally most effective in non-IID settings where client contributions naturally vary more significantly. In IID settings, the effect is more subtle since all clients have similar contributions.

Typical outcomes:
- Target client's contribution score is manipulated (decreased for LOO, increased/made less negative for Shapley).
- Global model performance is unstable, but does not degrade too much due to constraints.
- The attack builds up effectiveness over multiple rounds.