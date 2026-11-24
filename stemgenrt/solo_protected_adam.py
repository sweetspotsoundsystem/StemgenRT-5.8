"""Constrain the actual Adam displacement using an independent isolated-Bass example."""
import math

import torch

ARMS = ("plain", "protected")
SCHEMA = "isolated-bass-protected-adam-v1"


@torch.no_grad()
def project_displacement(parameters, before, guard_gradients, *, apply):
    """Remove only the component that raises the guard loss to first order."""
    tensors = list(zip(parameters, before, guard_gradients, strict=True))
    if not tensors:
        raise ValueError("Empty parameter collection")
    inner, grad_sq, step_sq = [], [], []
    for parameter, initial, gradient in tensors:
        if (initial.shape != parameter.shape or gradient.shape != parameter.shape or parameter.dtype != torch.float32
                or initial.dtype != parameter.dtype or gradient.dtype != parameter.dtype):
            raise ValueError("Require aligned FP32 parameters, guard gradients and initial state")
        g = gradient.detach().double()
        displacement = parameter.detach().double()-initial.double()
        inner.append((g*displacement).sum())
        grad_sq.append(g.square().sum())
        step_sq.append(displacement.square().sum())
    dot, norm, distance = (float(torch.stack(parts).sum()) for parts in (inner, grad_sq, step_sq))
    if not all(math.isfinite(x) for x in (dot, norm, distance)):
        raise ValueError("Nonfinite guard gradient or Adam update")
    conflict = dot > 0. and norm > 0.
    after = dot
    tolerance = 0.
    if apply and conflict:
        factor = dot/norm
        products, bounds = [], []
        for parameter, initial, gradient in tensors:
            g = gradient.double()
            corrected = parameter.double()-factor*g
            parameter.copy_(corrected)
            products.append((g*(parameter.double()-initial.double())).sum())
            bounds.append((g.abs()*corrected.abs()).sum())
        after = float(torch.stack(products).sum())
        tolerance = 2*torch.finfo(torch.float32).eps*float(torch.stack(bounds).sum())+1e-16
        if abs(after)>tolerance:
            raise ArithmeticError("Rounded displacement violated the guard projection")
    fraction = min(1.,dot*dot/(norm*distance)) if conflict and distance > 0 else 0.
    return {"conflict":conflict,"applied":bool(apply and conflict),"before_dot":dot,"after_dot":after,"rounding_tolerance":tolerance,
            "guard_gradient_norm":math.sqrt(norm),"adam_step_norm":math.sqrt(distance),
            "conflicting_squared_step_fraction":fraction}
