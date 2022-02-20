import torch
from torch import nn

from graph_pit.loss.optimized import optimized_graph_pit_source_aggregated_sdr_loss


class GraphPITLossWrapper(nn.Module):
    r"""Wrapper for the Graph-PIT loss (https://github.com/fgnt/graph_pit).

    Args:
        assignment_solver (str) : The assignment solver to use. (default: 'optimal_dynamic_programming')
    """

    def __init__(self, assignment_solver="optimal_dynamic_programming"):
        super().__init__()
        assert assignment_solver in [
            "optimal_brute_force",
            "optimal_branch_and_bound",
            "optimal_dynamic_programming",
            "dfs",
            "greedy_cop",
        ]
        self.assignment_solver = assignment_solver

    def forward(self, est_targets, targets, return_est=False):
        """Evaluate the loss using Graph-PIT algorithm.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: ([torch.Tensor], [segment_boundaries]). Batch of tuples containing
                list of target segments and their boundaries.
            return_est: Boolean. Whether to return the estimated targets
                (To compute metrics or to save example).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
                the batch, computed using Graph-PIT algorithm. torch.Tensor(loss_value)
            - The reordered targets estimates if return_est is True.
                torch.Tensor of shape :math:`(batch, nsrc, ...)`.
        """
        B = est_targets.shape[0]
        losses = torch.zeros(B)
        for i in range(len(est_targets)):
            est_target = est_targets[i]
            sources = targets[i]["sources"]
            segment_boundaries = targets[i]["boundaries"]
            length = targets[i]["length"]

            # Evaluate the loss using Graph-PIT (we remove the padding for loss computation)
            pw_losses = optimized_graph_pit_source_aggregated_sdr_loss(
                est_target[:, :length],
                sources,
                segment_boundaries,
                self.assignment_solver,
            )

            losses[i] = pw_losses

        mean_loss = torch.mean(losses)

        if return_est:
            return mean_loss, est_targets
        else:
            return mean_loss
