from __future__ import annotations

import math
import logging
from random import random
from typing import Iterable

import torch
import pytorch_lightning as pl
import tqdm

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset
Distribution: TypeAlias = torch.distributions.Distribution

PI = math.pi

logging.basicConfig(level="INFO")
log = logging.getLogger(__name__)


class Prior(torch.utils.data.IterableDataset):
    """Wraps around torch.distributions.Distribution to make it iterable."""

    def __init__(self, distribution: Distribution, sample_shape: list[int]):
        super().__init__()
        self.distribution = distribution
        self.sample_shape = sample_shape

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor]:
        sample = self.sample()
        return sample, self.log_prob(sample)

    def sample(self) -> Tensor:
        return self.distribution.sample(self.sample_shape)

    def log_prob(self, sample: Tensor) -> Tensor:
        return self.distribution.log_prob(sample).flatten(start_dim=1).sum(dim=1)


class Composition(torch.nn.Sequential):
    """Compose multiple layers."""

    def forward(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        for layer in self:
            x, log_det_jacob = layer(x, log_det_jacob)
        return x, log_det_jacob

    def inverse(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        for layer in reversed(self):
            x, log_det_jacob = layer.inverse(x, log_det_jacob)
        return x, log_det_jacob


class Flow(Composition):
    """Wraps around Composition, starting with zero log det Jacobian."""

    def forward(self, x: Tensor) -> tuple[Tensor]:
        return super().forward(x, torch.zeros(x.shape[0]).to(x.device))

    def inverse(self, x: Tensor) -> tuple[Tensor]:
        return super().inverse(x, torch.zeros(x.shape[0]).to(x.device))


def make_checkerboard(lattice_shape: list[int]) -> BoolTensor:
    """Return a boolean mask that selects 'even' lattice sites."""
    assert all(
        [n % 2 == 0 for n in lattice_shape]
    ), "each lattice dimension should be even"
    checkerboard = torch.full(lattice_shape, False)
    if len(lattice_shape) == 1:
        checkerboard[::2] = True
    elif len(lattice_shape) == 2:
        checkerboard[::2, ::2] = True
        checkerboard[1::2, 1::2] = True
    else:
        raise NotImplementedError("d > 2 currently not supported")
    return checkerboard


def prod(iterable: Iterable):
    """Return product of elements of iterable."""
    out = 1
    for el in iterable:
        out *= el
    return out


class RandomRotationLayer(torch.nn.Module):
    """Applies a random global rotation to each configuration in a sample."""

    def forward(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        shape = [x.shape[0]] + [1 for _ in range(1, x.dim())]
        shifts = torch.empty(shape).uniform_(0, 2 * PI).to(x.device)
        x.add_(PI).add_(shifts).fmod_(2 * PI).sub_(PI)
        return x, log_det_jacob

    def inverse(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        return self.forward(x, log_det_jacob)


def spins_to_links(spins: Tensor) -> Tensor:
    """Return tensor of links."""
    return torch.stack(
        [spins.roll(1, dim) - spins for dim in range(1, spins.dim())],
        dim=1,
    )


def links_to_spins(links: Tensor) -> Tensor:
    """Return tensor of spins."""
    theta = torch.empty([links.shape[0], 1]).uniform_(0, 2 * PI).to(links.device)
    if links.dim() == 2:  # 1D XY model
        # links defined as theta_{i+1} - theta_i
        return (
            torch.cat(
                (theta, links.add(PI)),
                dim=1,
            )
            .cumsum(dim=1)
            .fmod(2 * PI)
            .sub(PI)
        )
    else:
        raise NotImplementedError("Currently only d=1 supported")


def add_final_link(independent_links: Tensor) -> Tensor:
    """One dimension only!"""
    final_link = (
        independent_links.add(PI).sum(dim=1, keepdim=True).fmod(2 * PI).sub(PI).neg()
    )
    links = torch.cat((independent_links, final_link), dim=1)
    return links


def metropolis_acceptance(weights: Tensor) -> float:
    """Returns the fraction of configs that pass the Metropolis test."""
    weights = weights.tolist()
    curr_weight = weights.pop(0)
    history = []

    for prop_weight in weights:
        if random() < min(1, math.exp(curr_weight - prop_weight)):
            curr_weight = prop_weight
            history.append(1)
        else:
            history.append(0)

    return sum(history) / len(history)


def magnetisation_sq(spins: Tensor) -> Tensor:
    """Squared magnetisation for each configuration."""
    spin_vectors = torch.stack((spins.cos(), spins.sin()), dim=1)
    return (
        spin_vectors.flatten(start_dim=2)
        .mean(dim=2)  # avg over volume to measure disorder
        .pow(2)
        .sum(dim=1)  # M_x^2 + M_y^2
    )


class JlabProgBar(pl.callbacks.TQDMProgressBar):
    """Disable validation progress bar since it's broken in Jupyter Lab."""

    def init_validation_tqdm(self):
        return tqdm.tqdm(disable=True)


"""
_test_J = 0.01
_test_action = XYLinkAction(_test_J)
_test_vonmises = VonMisesPrior([10], 10000, concentration=_test_J)
_test_x, _test_log_prob = next(_test_vonmises)
_test_weights = _test_log_prob + _test_action(_test_x)
assert math.isclose(metropolis_acceptance(_test_weights), 1.0)

_new_test_weights = _test_log_prob + _test_action(add_final_link(_test_x))
print(metropolis_acceptance(_new_test_weights))
"""
