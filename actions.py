from __future__ import annotations

import math
import torch

Tensor: TypeAlias = torch.Tensor


class XYSpinAction:
    """Simplest XY action, with fixed nearest-neighbour coupling."""

    def __init__(self, J: float):
        self.J = J

    def __call__(self, spins: Tensor) -> Tensor:
        """Returns the action for a batch of spins."""
        out = torch.zeros_like(spins)
        for dim in range(1, spins.dim()):
            out.add_((spins - spins.roll(1, dim)).cos())
        return out.flatten(start_dim=1).sum(dim=1).mul(self.J).neg()


_ = XYSpinAction(1)(torch.empty(10, 4).uniform_(-math.pi, math.pi))  # test 1D
_ = XYSpinAction(1)(torch.empty(10, 4, 4).uniform_(-math.pi, math.pi))  # test 2D


class XYLinkAction:
    """Simplest XY action, with fixed nearest-neighbour coupling."""

    def __init__(self, J: float):
        self.J = J

    def __call__(self, links: Tensor) -> Tensor:
        """Returns the action for a batch of links."""
        # Each lattice site has n_dim links. Sum over cosine of all of these.
        return links.cos().flatten(start_dim=1).sum(dim=1).mul(self.J).neg()


_ = XYLinkAction(1)(torch.empty(10, 1, 4).uniform_(-math.pi, math.pi))  # test 1D
_ = XYLinkAction(1)(torch.empty(10, 2, 4, 4).uniform_(-math.pi, math.pi))  # test 2D
