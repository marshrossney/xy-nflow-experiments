from __future__ import annotations

import math
import logging

import torch
import torch.nn.functional as F

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset
Distribution: TypeAlias = torch.distributions.Distribution

PI = math.pi

logging.basicConfig(level="INFO")
log = logging.getLogger(__name__)


class PointwiseRationalQuadraticSplineTransform:
    """Circular spline transformation."""

    def __init__(self, n_segments: int):
        self.n_segments = n_segments
        self.params_dof = 3 * n_segments
        self.pad_derivs = lambda derivs: F.pad(derivs, (0, 1), "circular")

    def __call__(self, inputs: Tensor, params: Tensor) -> tuple[Tensor]:
        return self.forward(inputs, params)

    @property
    def identity_params(self):
        """Parameters required for spline to enact an identity transform."""
        return torch.cat(
            (
                torch.full(size=(2 * self.n_segments,), fill_value=1 / self.n_segments),
                (
                    torch.ones(self.params_dof - 2 * self.n_segments).exp() - 1
                ).log(),  # one after softmax
            ),
            dim=0,
        )

    def _build_spline(
        self, inputs: Tensor, params: Tensor, inverse: bool = False
    ) -> tuple[Tensor]:
        inputs = inputs.unsqueeze(dim=-1)

        # Split first dim into (n_segments, n_segments, n_segments +/- 1)
        heights, widths, derivs = params.tensor_split(
            (self.n_segments, 2 * self.n_segments), dim=-1
        )
        heights = F.softmax(heights, dim=-1) * 2 * PI
        widths = F.softmax(widths, dim=-1) * 2 * PI
        derivs = self.pad_derivs(F.softplus(derivs))

        knots_xcoords = (
            torch.cat(
                (
                    torch.zeros_like(inputs),
                    torch.cumsum(widths, dim=-1),
                ),
                dim=-1,
            )
            - PI
        )
        knots_ycoords = (
            torch.cat(
                (
                    torch.zeros_like(inputs),
                    torch.cumsum(heights, dim=-1),
                ),
                dim=-1,
            )
            - PI
        )

        if inverse:
            bins = knots_ycoords
        else:
            bins = knots_xcoords
        segment_idx = (
            torch.searchsorted(
                bins,
                inputs,
            )
            - 1
        ).clamp(0, self.n_segments - 1)

        width_this_segment = torch.gather(widths, -1, segment_idx).squeeze(dim=-1)
        height_this_segment = torch.gather(heights, -1, segment_idx).squeeze(dim=-1)
        slope_this_segment = height_this_segment / width_this_segment
        # derivs.mul_(slope_this_segment.unsqueeze(dim=-1))  # maybe useful for learning identity transf but not really otherwise
        deriv_at_lower_knot = torch.gather(derivs, -1, segment_idx).squeeze(dim=-1)
        deriv_at_upper_knot = torch.gather(derivs, -1, segment_idx + 1).squeeze(dim=-1)
        xcoord_at_lower_knot = torch.gather(knots_xcoords, -1, segment_idx).squeeze(
            dim=-1
        )
        ycoord_at_lower_knot = torch.gather(knots_ycoords, -1, segment_idx).squeeze(
            dim=-1
        )

        return (
            width_this_segment,
            height_this_segment,
            slope_this_segment,
            deriv_at_lower_knot,
            deriv_at_upper_knot,
            xcoord_at_lower_knot,
            ycoord_at_lower_knot,
        )

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor]:
        """Applies the 'forward' transformation."""
        (w, h, s, d0, d1, x0, y0) = self._build_spline(x, params)

        alpha = (x - x0) / w
        # NOTE: this clamping will hide bugs that result in alpha < 0 or alpha > 1 ...
        alpha.clamp_(0, 1)
        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )
        beta = (s * alpha.pow(2) + d0 * alpha * (1 - alpha)) * denominator_recip
        y = y0 + h * beta

        gradient = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )
        # assert torch.all(gradient > 0)
        log_det_jacob = gradient.log().flatten(start_dim=1).sum(dim=1)

        return y, log_det_jacob

    def inverse(self, y: Tensor, params: Tensor) -> tuple[Tensor]:
        (w, h, s, d0, d1, x0, y0) = self._build_spline(y, params, inverse=True)

        beta = (y - y0) / h
        beta.clamp_(0, 1)
        b = d0 - (d1 + d0 - 2 * s) * beta
        a = s - b
        c = -s * beta
        alpha = -2 * c * torch.reciprocal(b + (b.pow(2) - 4 * a * c).sqrt())
        x = x0 + w * alpha

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )
        gradient_fwd = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )
        log_det_jacob = gradient_fwd.log().flatten(start_dim=1).sum(dim=1).neg()

        return x, log_det_jacob


_test_spline = PointwiseRationalQuadraticSplineTransform(n_segments=8)
_test_x = torch.empty(1000, 4).uniform_(-PI, PI)
_test_params = torch.rand(1000, 4, _test_spline.params_dof) + 0.01
_test_y, _test_ldj = _test_spline.forward(_test_x, _test_params)
_test_x_rt, _test_ldj_inv = _test_spline.inverse(_test_y, _test_params)
assert torch.allclose(_test_x, _test_x_rt, atol=1e-5)
assert torch.allclose(_test_ldj, -_test_ldj_inv, atol=1e-5)

class PointwiseAdditiveTransform:
    """Learnable phase shift."""

    params_dof: int = 1
    identity_params = torch.Tensor([0.0])

    def __call__(self, inputs: Tensor, params: Tensor) -> tuple[Tensor]:
        return self.forward(inputs, params)

    def forward(self, inputs: Tensor, shift: Tensor) -> tuple[Tensor]:
        # Don't do in-place operations, just to be safe
        outputs = inputs.add(shift)
        log_det_jacob = torch.zeros(inputs.shape[0]).type_as(inputs)
        return outputs, log_det_jacob

    def inverse(self, inputs: Tensor, shift: Tensor) -> tuple[Tensor]:
        outputs = inputs.sub(shift)
        log_det_jacob = torch.zeros(inputs.shape[0]).type_as(inputs)
        return outputs, log_det_jacob

class PointwisePhaseShift:
    """Learnable phase shift."""

    params_dof: int = 1
    identity_params = torch.Tensor([0.0])

    def __call__(self, inputs: Tensor, params: Tensor) -> tuple[Tensor]:
        return self.forward(inputs, params)

    def forward(self, inputs: Tensor, shift: Tensor) -> tuple[Tensor]:
        # Don't do in-place operations, just to be safe
        # angles [0, 4pi] before fmod applied
        outputs = inputs.add(2 * PI).add(shift).fmod(2 * PI).sub(PI)
        log_det_jacob = torch.zeros(inputs.shape[0]).type_as(inputs)
        return outputs, log_det_jacob

    def inverse(self, inputs: Tensor, shift: Tensor) -> tuple[Tensor]:
        outputs = inputs.add(2 * PI).sub(shift).fmod(2 * PI).sub(PI)
        log_det_jacob = torch.zeros(inputs.shape[0]).type_as(inputs)
        return outputs, log_det_jacob


_test_shift = PointwisePhaseShift()
_test_x = torch.empty(1000, 4).uniform_(-PI, PI)
_test_params = torch.rand(1000, 4) + 0.01
_test_y, _test_ldj = _test_shift.forward(_test_x, _test_params)
_test_x_rt, _test_ldj_inv = _test_shift.inverse(_test_y, _test_params)
assert torch.allclose(_test_x, _test_x_rt, atol=1e-5)
assert torch.allclose(_test_ldj, -_test_ldj_inv, atol=1e-5)

