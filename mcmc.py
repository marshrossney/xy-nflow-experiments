from __future__ import annotations

import itertools
import math
from random import random, randint, uniform

import scipy.stats
import torch
import torch.nn.functional as F

import actions
import utils

Tensor: TypeAlias = torch.Tensor

PI = math.pi


class MetropolisSampler:
    def __init__(self, init_config: Tensor, coupling_strength: float):
        self.lattice_shape = list(init_config.shape)
        self.lattice_dim = init_config.dim()
        self.lattice_size = utils.prod(self.lattice_shape)
        self.coupling_strength = coupling_strength

        self._build_neighbour_lists()
        self.config = init_config.fmod(2 * PI).flatten().tolist()

        self.history = {
            "energy": [],
            "magnetisation_sq": [],
        }
        self.history["energy"].append(self.global_energy())
        self.history["magnetisation_sq"].append(self.global_magnetisation_sq())

    def _build_neighbour_lists(self):
        indices = torch.arange(self.lattice_size).view(self.lattice_shape)
        self.neighbour_list = (
            torch.stack(
                [
                    indices.roll(shift, dim)
                    for (shift, dim) in itertools.product(
                        [1, -1], range(self.lattice_dim)
                    )
                ],
                dim=0,
            )
            .flatten(start_dim=1)
            .transpose(0, 1)  # (n_spins, n_neighbours)
            .tolist()
        )
        self.positive_neighbour_list = (
            torch.stack(
                [indices.roll(-1, dim) for dim in range(self.lattice_dim)],
                dim=0,
            )
            .flatten(start_dim=1)  # (n_pos_neighbours, n_spins)
            .tolist()
        )

    def global_energy(self):
        tot = 0
        for neighbour_list in self.positive_neighbour_list:
            neighbours = [
                self.config[neighbour_idx] for neighbour_idx in neighbour_list
            ]
            tot += sum(
                [
                    math.cos(central_spin - neighbour)
                    for (central_spin, neighbour) in zip(self.config, neighbours)
                ]
            )
        return -self.coupling_strength * tot

    def global_magnetisation_sq(self):
        m1 = sum([math.cos(spin) for spin in self.config])
        m2 = sum([math.sin(spin) for spin in self.config])
        return m1 ** 2 + m2 ** 2

    def sample(self, n_sweeps: int, delta_max: float = PI) -> float:

        n_accepted = 0
        for sweep in range(n_sweeps):

            for step in range(self.lattice_size):

                spin_idx = randint(0, self.lattice_size - 1)
                central_spin = self.config[spin_idx]
                neighbours = [
                    self.config[neighbour_idx]
                    for neighbour_idx in self.neighbour_list[spin_idx]
                ]

                delta_spin = uniform(-delta_max, delta_max)
                delta_local_energy = -self.coupling_strength * (
                    sum(
                        [
                            math.cos(central_spin + delta_spin - neighbour)
                            - math.cos(central_spin - neighbour)
                            for neighbour in neighbours
                        ]
                    )
                )

                if random() < math.exp(-delta_local_energy):
                    self.config[spin_idx] = (
                        (central_spin + delta_spin + PI) % (2 * PI)
                    ) - PI
                    n_accepted += 1

            self.history["energy"].append(self.global_energy())
            self.history["magnetisation_sq"].append(self.global_magnetisation_sq())

        return n_accepted / (n_sweeps * self.lattice_size)


class HeatbathSampler(MetropolisSampler):
    def sample(self, n_sweeps: int):

        for sweep in range(n_sweeps):

            for step in range(self.lattice_size):

                spin_idx = randint(0, self.lattice_size - 1)
                neighbours = [
                    self.config[neighbour_idx]
                    for neighbour_idx in self.neighbour_list[spin_idx]
                ]

                # local field strength
                m1 = sum([math.cos(spin) for spin in neighbours])
                m2 = sum([math.sin(spin) for spin in neighbours])

                # von Mises parameters
                kappa = max(0.01, self.coupling_strength * math.sqrt(m1 ** 2 + m2 ** 2))
                theta = math.atan2(m2, m1)

                new_spin = scipy.stats.vonmises.rvs(kappa, loc=theta)
                self.config[spin_idx] = new_spin

            self.history["energy"].append(self.global_energy())
            self.history["magnetisation_sq"].append(self.global_magnetisation_sq())


class CheckerboardGibbsSampler:
    def __init__(self, init_config: Tensor, coupling_strength: float):
        self.lattice_shape = list(init_config.shape)
        self.lattice_dim = init_config.dim()
        self.lattice_size = utils.prod(self.lattice_shape)
        self.coupling_strength = coupling_strength

        self.config = init_config.unsqueeze(dim=0).unsqueeze(
            dim=0
        )  # (1, 1, *lattice_shape)
        self.checker = utils.make_checkerboard(self.lattice_shape)

        self.history = {
            "energy": [],
            "magnetisation_sq": [],
        }

        if self.lattice_dim == 1:
            self.kernel = torch.Tensor([1, 0, 1]).view(1, 1, 3)
            self.conv = F.conv1d
        elif self.lattice_dim == 2:
            self.kernel = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).view(
                1, 1, 3, 3
            )
            self.conv = F.conv2d

        padding = tuple(1 for edge in range(2 * self.lattice_dim))
        self.pad = lambda config: F.pad(config, padding, "circular")

    def global_energy(self):
        # return -self.coupling_strength * (cos_config * m1 + sin_config * m2).sum().div(2)
        return float(
            -self.coupling_strength
            * torch.stack(
                [
                    torch.cos(self.config - self.config.roll(-1, dim))
                    for dim in range(2, 2 + self.lattice_dim)
                ],
                dim=0,
            ).sum()
        )

    def sample(self, n_sweeps: int):

        m1 = self.conv(self.pad(self.config.cos()), self.kernel)
        m2 = self.conv(self.pad(self.config.sin()), self.kernel)
        
        magnetisation_sq = float(self.config.cos().sum().pow(2) + self.config.sin().sum().pow(2))
        self.history["energy"].append(self.global_energy())
        self.history["magnetisation_sq"].append(magnetisation_sq)

        for sweep in range(n_sweeps):

            for mask in (self.checker, ~self.checker):
                m1 = m1[..., mask]
                m2 = m2[..., mask]
                kappa = self.coupling_strength * (m1.pow(2) + m2.pow(2)).sqrt()
                kappa.clamp_(min=0.01)  # otherwise sampling takes AGES
                theta = torch.atan2(m2, m1)

                new_spins = torch.distributions.VonMises(
                    loc=theta, concentration=kappa
                ).sample()
                self.config.masked_scatter_(mask, new_spins)

                cos_config, sin_config = self.config.cos(), self.config.sin()
                m1 = self.conv(self.pad(cos_config), self.kernel)
                m2 = self.conv(self.pad(sin_config), self.kernel)

            magnetisation_sq = float(cos_config.sum().pow(2) + sin_config.sum().pow(2))
            self.history["energy"].append(self.global_energy())
            self.history["magnetisation_sq"].append(magnetisation_sq)
