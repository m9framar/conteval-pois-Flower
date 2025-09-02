"""
Strategy for GTG Shapley value approximation and Adaptive Weight calculation.
"""


from logging import WARNING, DEBUG, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.shapley import GTGShapleyValue
from flwr.common.adp_weight import adp_weight
from flwr.server import strategy
from data_reader import FederatedMetricsCollector



"""
Base class for Shapley Value and Adaptive Weight claculation.

Parameters
----------
strat : str
    String denoting the aggregation strategy to be used.
    Note: Case sensitive.
model_evaluation_fn : Callable | None (default: None)
    Function evaluating a subset of players. Input must be of the form
    tuple[list[parameters, num_examples]] where parameters is a flower
    Parameters object, num_examples is the number of datapoints of a player.
SV_eps : float (default: 0.001)
    In-round truncation happens when remaining player's total contribution
    is smaller than this value.
SV_round_trunc_threshold : float (default: 0.001)
    Between-round truncation happens when the evaluated metric didn't
    improve more than this value.
SV_convergence_criteria : float (default: 0.05)
    Error tolerance for convergence.
SV_last_k : int (default: 10)
    Number of players whose marginal contribution to check for convergence.
SV_converge_min : int (default: 30)
    Minimal number of subsets to evaluate.
    Note: Default essentially computes all subsets for 5 clients or less.
SV_max_percentage : float (default: 0.8)
    Maximal percentage of subsets to evaluate from the powerset.
SV_normalize : bool (default: True)
    A boolean signalling whether to normalize the sum of individual shapley
    values to 1 (i.e. Shapley Index).
SV_normalize : bool (default: True)
    A boolean signalling whether to normalize the sum of individual shapley
    values to 1 (i.e. Shapley Index).
alpha : float (default: 5.0)
    Parameter for the exponential of the Gompertz function.
*args, **kwargs
    Arguments passed to the aggregation strategy.
"""
class ContEval(strategy.FedAvg):
    def __init__(
        self,
        # FedAvg parameters
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        # ContEval specific parameters
        metrics_collector: FederatedMetricsCollector = None,
        model_evaluation_fn: Callable = None,
        SV_eps: float = 0.001,
        SV_round_trunc_threshold: float = 0.001,
        SV_convergence_criteria: float = 0.05,
        SV_last_k: int = 10,
        SV_converge_min: int = 30,
        SV_max_percentage: float = 0.8,
        SV_normalize: bool = True,
        alpha: float = 5.0,
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        
        self.metrics_collector = metrics_collector
        self.model_evaluation_fn = model_evaluation_fn
        self.SV_eps = SV_eps
        self.SV_round_trunc_threshold = SV_round_trunc_threshold
        self.SV_convergence_criteria = SV_convergence_criteria
        self.SV_last_k = SV_last_k
        self.SV_converge_min = SV_converge_min
        self.SV_max_percentage = SV_max_percentage
        self.SV_normalize = SV_normalize
        self.previous_metric = 0
        self.alpha = alpha
        self.smoothed_angles = {}
        self.last_round_params = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Get aggregated results from FedAvg
        agg_parameters, agg_metrics = super().aggregate_fit(server_round, results, failures)
        if agg_parameters is None:
            return None, {}

        aggregated_params = parameters_to_ndarrays(agg_parameters)

        # GTG Shapley Value calculation
        if self.model_evaluation_fn:
            sv = GTGShapleyValue(
                [(res.metrics["id"], res.parameters, res.num_examples) for _, res in results],
                self.previous_metric,
                eps=self.SV_eps,
                round_trunc_threshold=self.SV_round_trunc_threshold,
                convergence_criteria=self.SV_convergence_criteria,
                last_k=self.SV_last_k,
                converge_min=self.SV_converge_min,
                max_percentage=self.SV_max_percentage,
                normalize=self.SV_normalize
            )
            
            self.previous_metric = self.model_evaluation_fn(([agg_parameters, 1],))
            sv.set_metric_function(self.model_evaluation_fn)
            agg_metrics["Shapley"], agg_metrics["Shapley best set"] = sv.compute(server_round)
        
        # Adaptive Weighting
        if server_round == 1:
            self.last_round_params = aggregated_params

        weights_dict = adp_weight(
            self.last_round_params,
            aggregated_params,
            results,
            self.smoothed_angles,
            self.alpha
        )
        agg_metrics["Adaptive Weights"] = weights_dict
        self.last_round_params = aggregated_params

        if self.metrics_collector:
            self.metrics_collector.add_round_data(server_round, agg_metrics, phase='train')

        return agg_parameters, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if self.metrics_collector and loss_aggregated is not None:
            metrics_aggregated['loss'] = loss_aggregated
            self.metrics_collector.add_round_data(server_round, metrics_aggregated, phase='evaluate')
        
        return loss_aggregated, metrics_aggregated



# Model aggregation used to evaluate subsets of clients
# def _model_aggregation(self, results, num_malicious_clients=0, num_clients_to_keep=2):
#     if isinstance(self.strategy, (strategy.FedAvg, strategy.FedProx)):
#         param_weight = [(parameters_to_ndarrays(params), num_examples) for params,num_examples in results]
#         aggregated_ndarrays = aggregate(param_weight)
#         return aggregated_ndarrays
#     elif isinstance(self.strategy, (strategy.FedNova,)):
#         # TODO: FedNova
#         pass
#     elif isinstance(self.strategy, (strategy.Krum,)):
#         param_weight = [(parameters_to_ndarrays(params), num_examples) for params,num_examples in results]
#         aggregated_ndarrays = aggregate_krum(param_weight, num_malicious_clients, num_clients_to_keep)
#         return aggregated_ndarrays
#     # elif isinstance(self.strategy, (strategy.Zeno,)):
#         # TODO: Zeno
#         # pass
#     else:
#         if not self.smoothed_angles and len(results) == 1:
#             # Logs first round, when single clients are evaluated
#             # (I can't be asked to write it out less then NUM_CLIENTS times)
#             log(WARNING, f"Aggregation method {self.strategy} not implemented, defaulting to FedAvg")
#         param_weight = [(parameters_to_ndarrays(params), num_examples) for params,num_examples in results]
#         aggregated_ndarrays = aggregate(param_weight)

#         return aggregated_ndarrays
