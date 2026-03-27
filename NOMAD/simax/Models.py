from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp




class Model_JAX(eqx.Module):
    """Generic dynamic model based on JAX.

    This container wraps:
    - a continuous dynamics `state_fn(x, u, d, theta) -> dx/dt`,
    - an observation function `output_fn(x, u, d, theta) -> y`,
    - and optional metadata (names/units) describing states, outputs, controls and disturbances.
    """

    theta: dict
    state_fn: callable
    output_fn: callable

    # Optional metadata:
    # - state_*      : internal states x
    # - output_*     : outputs of h(x, u, d)
    # - control_*    : control inputs
    # - disturbance_*: disturbances d
    state_names: tuple[str, ...] | None = None
    state_units: tuple[str, ...] | None = None
    output_names: tuple[str, ...] | None = None
    output_units: tuple[str, ...] | None = None
    control_names: tuple[str, ...] | None = None
    control_units: tuple[str, ...] | None = None
    disturbance_names: tuple[str, ...] | None = None
    disturbance_units: tuple[str, ...] | None = None

    def state_derivative(self, x, u, d):
        """Evaluate dx/dt for a given state and inputs."""
        return self.state_fn(jnp.asarray(x), u, d, self.theta)

    def h(self, x, u, d):
        """Compute the model output for the provided state and inputs."""
        return self.output_fn(jnp.asarray(x), u, d, self.theta)

__all__ = ["Model_JAX"]
