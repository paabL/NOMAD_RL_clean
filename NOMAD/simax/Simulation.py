from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import pickle
import numpy as np

import equinox as eqx
import jax.numpy as jnp
from jax import lax
import jax
import jax.tree_util as jtu
import jaxopt
from jax.flatten_util import ravel_pytree

from .Models import Model_JAX

import pandas as pd

if TYPE_CHECKING:  # avoid circular import at runtime
    from .Controller import Controller

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class SimulationDataset:
    """Generic dataset for simulation/identification.

    This container stores:
    - a time grid,
    - sequences of controls `u`,
    - sequences of disturbances `d` (including, if desired, measurements).

    Attributes
    ---------
    time : jnp.ndarray
        Time vector (N,) in seconds.
    u : dict[str, jnp.ndarray]
        Control inputs keyed by name (N,).
    d : dict[str, jnp.ndarray]
        Disturbances and measured signals keyed by name (N,).
    """

    time: jnp.ndarray
    u: dict[str, jnp.ndarray]
    d: dict[str, jnp.ndarray]

    @classmethod
    def from_csv(cls, path, *, control_cols, disturbance_cols):
        """Load a generic dataset from a CSV file."""
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()

        def as_array(name):
            series = frame[name].to_numpy(dtype=float)
            return jnp.asarray(series, dtype=jnp.float32)

        u = {key: as_array(key) for key in control_cols}
        d = {key: as_array(key) for key in disturbance_cols}
        return cls(time=as_array("time"), u=u, d=d)

    def take_fraction(self, gamma):
        """Select a leading fraction of the dataset."""
        frac = float(gamma)
        if not 0.0 < frac <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        total = int(self.time.shape[0])
        keep = max(1, int(total * frac))
        window = slice(0, keep)
        sub_u = {k: v[window] for k, v in self.u.items()}
        sub_d = {k: v[window] for k, v in self.d.items()}
        return SimulationDataset(time=self.time[window], u=sub_u, d=sub_d)


@eqx.filter_jit
def _run_core(model, x0_arr, t_grid, ctrl, d_payloads, base_time_grid, integrator, use_subgrid: bool, return_ctrl_state: bool):
    """Jitted core of the simulation (pure numerical code)."""
    dtype = x0_arr.dtype

    dt0 = t_grid[1] - t_grid[0]
    dt_seq = jnp.diff(t_grid, prepend=t_grid[:1] - dt0)

    ctrl_state0 = ctrl.init_state()

    n = t_grid.size
    if use_subgrid:
        base = jnp.asarray(base_time_grid, dtype=dtype)
        sub0 = t_grid[0]
        idx0 = jnp.nonzero(base == sub0, size=1, fill_value=0)[0]
        offset = jnp.asarray(idx0, dtype=jnp.int32)[0]
    else:
        offset = jnp.asarray(0, dtype=jnp.int32)

    d_keys = tuple(d_payloads.keys())
    idx_seq = jnp.arange(n, dtype=jnp.int32)
    global_idx = offset + idx_seq
    d_series = []
    for key in d_keys:
        full = jnp.asarray(d_payloads[key], dtype=dtype)
        d_series.append(jnp.take(full, global_idx, axis=0))

    scan_inputs = (dt_seq, *d_series, idx_seq)

    def scan_step(carry, data):
        state, ctrl_state = carry
        dt = data[0]
        d_vals = data[1:-1]
        idx = data[-1]
        d_payload = {k: v for k, v in zip(d_keys, d_vals)}

        u_payload, next_ctrl_state = ctrl.compute_control(
            idx=idx,
            y_measurements=state,
            disturbances=d_payload,
            ctrl_state=ctrl_state,
            dt=dt,
        )
        u_payload = {k: jnp.asarray(v, dtype=dtype) for k, v in u_payload.items()}

        def rhs(y):
            return model.state_derivative(y, u_payload, d_payload)

        nxt = integrator(rhs, state, dt)

        y_out = model.h(nxt, u_payload, d_payload)
        if isinstance(y_out, (tuple, list)):
            y_vec = jnp.stack(
                [jnp.asarray(comp, dtype=dtype) for comp in y_out]
            ).astype(dtype)
        else:
            y_arr = jnp.asarray(y_out, dtype=dtype)
            y_vec = y_arr.reshape((-1,))

        record_u = {k: jnp.asarray(v, dtype=dtype) for k, v in u_payload.items()}

        return (nxt, next_ctrl_state), (y_vec, nxt, record_u)

    (_, ctrl_state_end), (y_seq, states_seq, u_records) = lax.scan(
        scan_step, (x0_arr, ctrl_state0), scan_inputs
    )

    controls = {k: jnp.asarray(v, dtype=dtype) for k, v in u_records.items()}

    if return_ctrl_state:
        return t_grid, y_seq, states_seq, controls, ctrl_state_end
    return t_grid, y_seq, states_seq, controls


class Simulation_JAX(eqx.Module):
    """JAX simulation engine for dynamic systems."""

    time_grid: jnp.ndarray
    d: dict[str, jnp.ndarray]
    model: Model_JAX
    controller: "Controller"
    x0: jnp.ndarray
    integrator: Any | None = eqx.field(static=True, default=None)

    def __post_init__(self):
        """Configure the numerical integrator after creation."""
        self.configure_integrator()

    def configure_integrator(self):
        """Select the integration scheme from `integrator`."""
        integ = self.integrator
        if integ is None or isinstance(integ, str):
            # By default, use RK2 (Heun) as a precision/cost compromise
            name = (integ or "rk2").lower()
            table = {
                "rk4": self.rk4_step,
                "rk2": self.rk2_step,  # Heun
                "euler": self.euler_step,
            }
            object.__setattr__(self, "integrator", table.get(name, self.rk2_step))

    def copy(self, **overrides):
        """Create an immutable copy with some fields overridden."""
        new = replace(self, **overrides)
        new.configure_integrator()
        return new

    def save_simulation(self, path):
        """Save the simulation to disk (pickle)."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            pickle.dump(self, out)
        return dest

    # ---------- Interface publique JAX ----------

    def run(self, theta=None, *, x0=None, time_grid=None, controller=None, return_ctrl_state=False):
        """Run the simulation on a time grid (Python interface)."""

        # Model (frozen theta or updated)
        if theta is None:
            model = self.model
        else:
            model = eqx.tree_at(lambda m: m.theta, self.model, theta)

        # Initial state
        if x0 is None:
            if self.x0 is None:
                raise ValueError("No initial state provided (x0=None and self.x0=None).")
            x0_arr = jnp.asarray(self.x0, dtype=jnp.float32)
        else:
            x0_arr = jnp.asarray(x0, dtype=jnp.float32)

        # Time grid
        t_grid = time_grid if time_grid is not None else self.time_grid
        if t_grid.size < 2:
            raise ValueError("The time grid must contain at least two points.")

        # Whether we're using a subgrid
        use_subgrid = time_grid is not None and (t_grid.size != self.time_grid.size)

        # Controller
        ctrl = controller or self.controller

        # Call the jitted core
        return _run_core(model, x0_arr, t_grid, ctrl, self.d, self.time_grid, self.integrator, use_subgrid, return_ctrl_state)

    # ---------- NumPy version (debug / tracing) ----------

    def run_numpy(self, theta=None, *, x0=None, time_grid=None, controller=None):
        """NumPy version of `run`: Python loop + JAX integrator."""
        # Model
        if theta is None:
            model = self.model
        else:
            model = eqx.tree_at(lambda m: m.theta, self.model, theta)

        # Initial state
        if x0 is None:
            if self.x0 is None:
                raise ValueError("No initial state provided (x0=None and self.x0=None).")
            x = np.asarray(self.x0, dtype=np.float32)
        else:
            x = np.asarray(x0, dtype=np.float32)

        # Time grid
        t_grid = np.asarray(
            time_grid if time_grid is not None else self.time_grid,
            dtype=np.float32,
        )
        if t_grid.size < 2:
            raise ValueError("The time grid must contain at least two points.")
        dt0 = t_grid[1] - t_grid[0]

        ctrl = controller or self.controller
        ctrl_state = ctrl.init_state()
        ctrl_states: list[Any] = []
        mpc_logs: list[dict[str, Any]] = []
        last_forecast_idx = None

        n = t_grid.size
        if time_grid is not None and n != self.time_grid.size:
            base = np.asarray(self.time_grid)
            sub = t_grid
            idx0 = np.where(base == sub[0])[0]
            if idx0.size == 0:
                raise ValueError(
                    "time_grid does not match a subset of the simulation time grid."
                )
            offset = int(idx0[0])
        else:
            offset = 0

        d_arrays = {}
        for key, full in self.d.items():
            arr = np.asarray(full, dtype=np.float32)
            if offset + n > arr.shape[0]:
                raise ValueError("Forcings are shorter than the requested time grid.")
            d_arrays[key] = arr[offset : offset + n]

        y_list: list[np.ndarray] = []
        states = np.zeros((n, x.size), dtype=np.float32)
        u_records: dict[str, list[float]] = {}

        for k in range(n):
            dt = dt0 if k == 0 else t_grid[k] - t_grid[k - 1]

            d_payload_np = {name: d_arrays[name][k] for name in d_arrays}
            d_jax = {
                name: jnp.asarray(val, dtype=jnp.float32)
                for name, val in d_payload_np.items()
            }

            u_payload, ctrl_state = ctrl.compute_control(
                idx=k,
                y_measurements=x,
                disturbances=d_payload_np,
                ctrl_state=ctrl_state,
                dt=dt,
            )
            if isinstance(ctrl_state, dict):
                ctrl_states.append(dict(ctrl_state))
            else:
                ctrl_states.append(ctrl_state)
            u_jax = {
                name: jnp.asarray(val, dtype=jnp.float32)
                for name, val in u_payload.items()
            }

            if isinstance(ctrl_state, dict) and "latest_forecast" in ctrl_state:
                fc = ctrl_state["latest_forecast"]
                fc_idx = int(fc.get("decision_idx", k))
                if fc_idx != last_forecast_idx:
                    fc_time = np.asarray(fc.get("time", []), dtype=np.float32)
                    fc_y = np.asarray(fc.get("y", []), dtype=np.float32)
                    fc_u = (
                        np.asarray(fc.get("u", []), dtype=np.float32)
                        if "u" in fc
                        else None
                    )
                    if fc_time.size > 0 and fc_y.size > 0:
                        entry = {
                            "time": fc_time,
                            "y": fc_y,
                            "decision_idx": fc_idx,
                            "decision_time": float(t_grid[k]),
                        }
                        if fc_u is not None:
                            entry["u"] = fc_u
                        mpc_logs.append(entry)
                        last_forecast_idx = fc_idx

            y0 = jnp.asarray(x, dtype=jnp.float32)
            dt_jax = jnp.asarray(dt, dtype=jnp.float32)

            def rhs(y):
                return model.state_derivative(y, u_jax, d_jax)

            nxt = self.integrator(rhs, y0, dt_jax)
            x = np.asarray(nxt, dtype=np.float32)

            y_out = model.h(nxt, u_jax, d_jax)
            if isinstance(y_out, (tuple, list)):
                y_vec = np.stack(
                    [np.asarray(comp, dtype=np.float32) for comp in y_out]
                )
            else:
                y_arr = np.asarray(y_out, dtype=np.float32)
                y_vec = y_arr.reshape((-1,))
            y_list.append(y_vec)

            states[k] = x
            for name, val in u_payload.items():
                u_records.setdefault(name, []).append(float(val))

        y = np.vstack(y_list)
        u_record_arrays = {
            name: np.asarray(vals, dtype=np.float32)
            for name, vals in u_records.items()
        }

        return t_grid, y, states, u_record_arrays, ctrl_states, mpc_logs

    # ---------- Plot (unchanged) ----------

    def plot(
        self,
        theta=None,
        *,
        path=None,
        NP=False,
        x0=None,
        y_meas=None,
        y_meas_label="meas",
        setpoints=None,
        disturbances_meas=None,
        precomputed=None,
    ):
        """Run the simulation and plot states, outputs, disturbances and control."""
        if precomputed is not None:
            res = precomputed
        else:
            run_fn = self.run_numpy if NP else self.run
            theta_used = theta if theta is not None else self.model.theta
            res = run_fn(theta_used, x0=x0)

        # Compatibility with different possible return formats
        if len(res) == 6:
            t, y_sim, states, controls, _ctrl_states, mpc_forecasts = res
        elif len(res) == 5:
            t, y_sim, states, controls, mpc_forecasts = res
        elif len(res) == 4:
            t, y_sim, states, controls = res
            mpc_forecasts = None
        else:
            t, y_sim = res[:2]
            states = controls = None
            mpc_forecasts = None

        # Main control (if available): prefer the order of `control_names`.
        u_sim = None
        u_name = None
        u_unit = None
        if isinstance(controls, dict) and controls:
            c_names = getattr(self.model, "control_names", None)
            key = None
            if isinstance(c_names, tuple):
                for name in c_names:
                    if name in controls:
                        key = name
                        break
            if key is None:
                key = next(iter(controls))
            u_sim = controls[key]
            u_name = key
            c_units = getattr(self.model, "control_units", None)
            if (
                isinstance(c_names, tuple)
                and isinstance(c_units, tuple)
                and len(c_units) == len(c_names)
                and key in c_names
            ):
                idx = c_names.index(key)
                u_unit = c_units[idx]
            elif isinstance(c_units, tuple) and len(c_units) > 0:
                u_unit = c_units[0]

        # Output metadata (names/units) if the model provides them
        y_names = getattr(self.model, "output_names", None)
        y_units = getattr(self.model, "output_units", None)

        # States
        state_names = getattr(self.model, "state_names", None)
        state_units = getattr(self.model, "state_units", None)

        # Simulated disturbances: truncate to the simulation length
        n = int(jnp.asarray(y_sim).shape[0])
        dist_all = {}
        for key, full in self.d.items():
            arr = jnp.asarray(full, dtype=jnp.float32)
            dist_all[key] = arr[:n]
        dist_names = getattr(self.model, "disturbance_names", None)
        dist_units = getattr(self.model, "disturbance_units", None)

        Sim_and_Data.render_plot(
            time=t,
            states=states,
            state_names=state_names,
            state_units=state_units,
            disturbances=dist_all,
            disturbances_meas=disturbances_meas,
            dist_names=dist_names,
            dist_units=dist_units,
            y_sim=y_sim,
            y_names=y_names,
            y_units=y_units,
            u_sim=u_sim,
            u_name=u_name,
            u_unit=u_unit,
            setpoints=setpoints,
            path=path,
            mpc_forecasts=mpc_forecasts,
            y_meas=y_meas,
            y_meas_label=y_meas_label,
        )

    # ---------- Integrators ----------

    @staticmethod
    def rk4_step(rhs, state, dt):
        """Perform a Runge–Kutta 4 step."""
        h = jnp.maximum(dt, 0.0)
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * h * k1)
        k3 = rhs(state + 0.5 * h * k2)
        k4 = rhs(state + h * k3)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def rk2_step(rhs, state, dt):
        """Perform a Runge–Kutta 2 step (Heun's method)."""
        h = jnp.maximum(dt, 0.0)
        k1 = rhs(state)
        k2 = rhs(state + h * k1)
        return state + 0.5 * h * (k1 + k2)


    @staticmethod
    def euler_step(rhs, state, dt):
        """Perform an explicit Euler step."""
        h = jnp.maximum(dt, 0.0)
        return state + h * rhs(state)

@dataclass(frozen=True)
class Sim_and_Data:
    """Minimal coupling between a JAX simulation and a structured dataset.

    This class links:
    - a `Simulation_JAX` instance,
    - a `SimulationDataset` describing forcings,
    - measurements `y_meas` used for identification or visualization,
    - a weight vector `W` to weight outputs,
    - an optional state-initialization function from data.

    Attributes
    ---------
    simulation : Simulation_JAX
        Reference simulation (model + controller).
    dataset : SimulationDataset
        Input data (u, d) and time grid.
    y_meas : jnp.ndarray
        Stacked measurements (N, ny) aligned with outputs `h`.
    W : jnp.ndarray | None
        Per-output weights (ny,), used in the identification residual.
    initial_state_fn : Callable[[Sim_and_Data, dict[str, Any]], jnp.ndarray] | None
        Function that reconstructs an initial state x0 from data and theta.
    """

    simulation: Simulation_JAX
    dataset: SimulationDataset
    y_meas: jnp.ndarray
    W: jnp.ndarray | None = None
    initial_state_fn: Callable[["Sim_and_Data", dict[str, Any]], jnp.ndarray] | None = None

    def copy(self, **overrides):
        """Create a copy with some fields overridden.

        Parameters
        ----------
        **overrides
            Attributes to override on the current instance.

        Returns
        ------
        Sim_and_Data
            New instance with updated attributes.
        """
        return replace(self, **overrides)

    def simulation_for_dataset(self):
        """Adapt the simulation to the dataset horizon and forcings.

        Returns
        ------
        Simulation_JAX
            Copy of `simulation` with `time_grid` and `d` aligned to the dataset.
        """
        data = self.dataset
        return self.simulation.copy(time_grid=data.time, d=data.d)

    @staticmethod
    def estimate_derivative(time, values, window=5):
        """Estimate an average derivative over a short initial window.

        Parameters
        ----------
        time : jnp.ndarray
            Time grid (N,).
        values : jnp.ndarray
            Scalar series (N,).
        window : int, optional
            Window length used for the estimate.

        Returns
        ------
        jnp.ndarray
            Scalar estimate of the derivative on the initial window.
        """
        n = int(min(window, values.size))
        if n < 2:
            return jnp.asarray(0.0, dtype=jnp.float32)
        span = time[n - 1] - time[0]
        delta = values[n - 1] - values[0]
        safe_span = jnp.where(span == 0.0, 1.0, span)
        deriv = delta / safe_span
        deriv = jnp.where(span == 0.0, 0.0, deriv)
        return jnp.asarray(deriv, dtype=jnp.float32)

    def build_setpoint_profile(self, length):
        """Build a setpoint trajectory aligned with the horizon.

        Parameters
        ----------
        length : int
            Desired trajectory length.

        Returns
        ------
        jnp.ndarray | None
            Setpoint vector (length,) or None if no setpoint information is available.
        """
        controller = self.simulation.controller
        raw = getattr(controller, "SetPoints", None)
        if raw is not None:
            arr = jnp.asarray(raw, dtype=jnp.float32)
            if arr.size > 0:
                if arr.shape[0] < length:
                    pad_val = arr[-1]
                    pad = jnp.full((length - arr.shape[0],), pad_val, dtype=jnp.float32)
                    return jnp.concatenate([arr, pad])
                return arr[:length]
        fallback = getattr(controller, "TSet", None)
        if fallback is None:
            return None
        return jnp.full((length,), float(fallback), dtype=jnp.float32)

    @staticmethod
    def render_plot(
        time,
        *,
        states=None,
        state_names=None,
        state_units=None,
        disturbances=None,
        disturbances_meas=None,
        dist_names=None,
        dist_units=None,
        y_sim=None,
        y_meas=None,
        y_names=None,
        y_units=None,
        u_sim=None,
        u_meas=None,
        u_name=None,
        u_unit=None,
        setpoints=None,
        path=None,
        mpc_forecasts=None,
        y_meas_label: str = "meas",
    ):
        """Generic plot for states, outputs, disturbances and controls.

        Quantities are grouped by type (states, disturbances, outputs, controls),
        then, within each group, by physical unit (K, W, -, ...). Each
        (group, unit) is plotted on a separate subplot.

        Parameters
        ----------
        time : jnp.ndarray
            Time grid shared by all signals (N,).
        states : jnp.ndarray | None
            State trajectory (N, nx) or None.
        state_names/state_units : tuple[str, ...] | None
            State names and units.
        disturbances : dict[str, jnp.ndarray] | None
            Simulated disturbances keyed by name (N,).
        disturbances_meas : dict[str, jnp.ndarray] | None
            Measured disturbances keyed by name (N,) to compare to simulated disturbances.
        dist_names/dist_units : tuple[str, ...] | None
            Disturbance names and units.
        y_sim : jnp.ndarray | None
            Simulated outputs (N, ny).
        y_meas : jnp.ndarray | None
            Measured outputs (N, ny), aligned with y_sim.
        y_meas_label : str
            Suffix used for measurement curves (default 'meas').
        y_names/y_units : tuple[str, ...] | None
            Output names and units.
        u_sim : jnp.ndarray | None
            Optional primary simulated control trajectory (N,).
        u_meas : jnp.ndarray | None
            Optional primary measured control trajectory (N,).
        u_name/u_unit : str | None
            Name and unit of the primary control.
        setpoints : jnp.ndarray | None
            Setpoint trajectory to overlay on outputs.
        path : str | Path | None
            If provided, path to save the image; otherwise the figure is shown.
        mpc_forecasts : list[dict] | None
            MPC internal forecasts (with keys time/y/u) to overlay on the first output/control.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        t_days = jnp.asarray(time, dtype=jnp.float32) / _SECONDS_PER_DAY
        mpc_forecasts = mpc_forecasts or []
        groups: list[tuple[str, list[dict[str, object]]]] = []

        # States
        state_series: list[dict[str, object]] = []
        if states is not None:
            x_arr = jnp.asarray(states, dtype=jnp.float32)
            if x_arr.ndim == 1:
                x_arr = x_arr[:, None]
            nx = int(x_arr.shape[1])
            if state_names is None or len(state_names) != nx:
                state_names = tuple(f"x{i}" for i in range(nx))
            if state_units is None or len(state_units) != nx:
                state_units = tuple("" for _ in range(nx))
            for i in range(nx):
                state_series.append(
                    {
                        "unit": state_units[i] or "",
                        "label": state_names[i] or f"x{i}",
                        "values": x_arr[:, i],
                    }
                )
        if state_series:
            groups.append(("States", state_series))

        # Perturbations
        dist_series: list[dict[str, object]] = []
        if disturbances or disturbances_meas:
            sim_keys = set(disturbances.keys()) if disturbances else set()
            meas_keys = set(disturbances_meas.keys()) if disturbances_meas else set()
            if dist_names is not None:
                keys = [name for name in dist_names if name in sim_keys or name in meas_keys]
            else:
                keys = sorted(sim_keys | meas_keys)
            unit_map: dict[str, str] = {}
            if dist_units is not None and dist_names is not None:
                for i, name in enumerate(dist_names):
                    if i < len(dist_units):
                        unit_map[name] = dist_units[i]
            for name in keys:
                unit = unit_map.get(name, "")
                if disturbances and name in disturbances:
                    vals = jnp.asarray(disturbances[name], dtype=jnp.float32)
                    dist_series.append({"unit": unit, "label": f"{name} (sim)", "values": vals})
                if disturbances_meas and name in disturbances_meas:
                    vals_m = jnp.asarray(disturbances_meas[name], dtype=jnp.float32)
                    dist_series.append({"unit": unit, "label": f"{name} (meas)", "values": vals_m})
        if dist_series:
            groups.append(("Disturbances", dist_series))

        # Outputs
        output_series: list[dict[str, object]] = []
        primary_output_unit = None
        if y_sim is not None:
            y_arr = jnp.asarray(y_sim, dtype=jnp.float32)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, None]
            ny = int(y_arr.shape[1])
            if y_names is None or len(y_names) != ny:
                y_names = tuple(f"y{i}" for i in range(ny))
            if y_units is None or len(y_units) != ny:
                y_units = tuple("" for _ in range(ny))
            if ny > 0:
                primary_output_unit = y_units[0] or ""

            y_meas_arr = None
            y_meas_cols = 0
            if y_meas is not None:
                y_meas_arr = jnp.asarray(y_meas, dtype=jnp.float32)
                if y_meas_arr.ndim == 1:
                    y_meas_arr = y_meas_arr[:, None]
                y_meas_cols = int(y_meas_arr.shape[1])

            for i in range(ny):
                unit = y_units[i] or ""
                label_sim = y_names[i] or f"y{i}"
                output_series.append(
                    {
                        "unit": unit,
                        "label": f"{label_sim} (sim)",
                        "values": y_arr[:, i],
                    }
                )
                if (
                    y_meas_arr is not None
                    and y_meas_cols > i
                    and y_meas_arr.shape[0] >= y_arr.shape[0]
                ):
                    output_series.append(
                        {
                            "unit": unit,
                            "label": f"{label_sim} ({y_meas_label})",
                            "values": y_meas_arr[: y_arr.shape[0], i],
                        }
                    )

            # Optional setpoint: same unit as the first output
            if setpoints is not None and ny > 0:
                sp = jnp.asarray(setpoints[: y_arr.shape[0]], dtype=jnp.float32)
                unit = y_units[0] or ""
                output_series.append(
                    {
                        "unit": unit,
                        "label": "setpoint",
                        "values": sp,
                    }
                )
        if output_series:
            groups.append(("Outputs", output_series))

        # Controls
        control_series: list[dict[str, object]] = []
        main_control_unit = None
        if u_sim is not None:
            u_arr = jnp.asarray(u_sim, dtype=jnp.float32)
            unit = u_unit or ""
            label = u_name or "u"
            control_series.append({"unit": unit or "cmd", "label": f"{label} (sim)", "values": u_arr})
            main_control_unit = main_control_unit or (unit or "cmd")
        if u_meas is not None and u_name is not None:
            u_m_arr = jnp.asarray(u_meas, dtype=jnp.float32)
            unit = u_unit or ""
            label = u_name
            control_series.append({"unit": unit or "cmd", "label": f"{label} (meas)", "values": u_m_arr})
            main_control_unit = main_control_unit or (unit or "cmd")
        if control_series:
            groups.append(("Controls", control_series))

        # Put disturbances last for readability
        if groups:
            dist_pairs = []
            other_pairs = []
            for name, series in groups:
                if name == "Disturbances":
                    dist_pairs.append((name, series))
                else:
                    other_pairs.append((name, series))
            groups = other_pairs + dist_pairs

        # Total number of subplots (groups × units)
        total_rows = 0
        group_units: list[list[str]] = []
        for _, series in groups:
            units = []
            for s in series:
                u = s["unit"] or ""
                if u not in units:
                    units.append(u)
            group_units.append(units)
            total_rows += max(len(units), 1)

        if total_rows == 0:
            return  # nothing to plot

        fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=(10, 2 + 2 * total_rows))
        if total_rows == 1:
            axes = [axes]

        axis_lookup: dict[tuple[str, str], object] = {}
        row_idx = 0
        for (group_name, series), units in zip(groups, group_units):
            if not units:
                continue
            for unit in units:
                ax = axes[row_idx]
                axis_lookup[(group_name, unit)] = ax
                row_idx += 1
                # title on the first unit of the group
                if unit == units[0]:
                    ax.set_title(group_name)
                for s in series:
                    if (s["unit"] or "") != unit:
                        continue
                    vals = jnp.asarray(s["values"], dtype=jnp.float32)
                    n_vals = int(vals.shape[0])
                    if group_name == "Controls":
                        ax.step(t_days[:n_vals], vals, label=s["label"], where="post")
                    elif s["label"] == "setpoint":
                        ax.plot(t_days[:n_vals], vals, label=s["label"], linestyle="--")
                    else:
                        ax.plot(t_days[:n_vals], vals, label=s["label"])
                ylabel = f"[{unit}]" if unit else "values"
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                ax.legend()

        if mpc_forecasts:
            stride = max(1, len(mpc_forecasts) // 25)
            forecasts_iter = mpc_forecasts
            ax_out = None
            if primary_output_unit is not None:
                ax_out = axis_lookup.get(("Outputs", primary_output_unit or ""))
            if ax_out is not None:
                label_done = False
                for idx_fc, fc in enumerate(forecasts_iter):
                    if idx_fc % stride != 0:
                        continue
                    t_fc = jnp.asarray(fc.get("time", []), dtype=jnp.float32)
                    y_fc = jnp.asarray(fc.get("y", []), dtype=jnp.float32)
                    if t_fc.size == 0 or y_fc.size == 0:
                        continue
                    if y_fc.ndim == 1:
                        y_fc = y_fc[:, None]
                    if y_fc.shape[1] == 0:
                        continue
                    vals = y_fc[:, 0]
                    label = None if label_done else "MPC forecast"
                    alpha = 0.3 if label_done else 0.8
                    ax_out.plot(t_fc / _SECONDS_PER_DAY, vals, color="green", linestyle=":", linewidth=0.9, alpha=alpha, label=label)
                    label_done = True
                if label_done:
                    ax_out.legend()

            ax_ctrl = None
            if main_control_unit is not None:
                ax_ctrl = axis_lookup.get(("Controls", main_control_unit))
            if ax_ctrl is not None:
                label_done = False
                for idx_fc, fc in enumerate(forecasts_iter):
                    if idx_fc % stride != 0:
                        continue
                    t_fc = jnp.asarray(fc.get("time", []), dtype=jnp.float32)
                    u_fc = fc.get("u", None)
                    if u_fc is None:
                        continue
                    u_arr = jnp.asarray(u_fc, dtype=jnp.float32)
                    m = min(int(u_arr.shape[0]), int(t_fc.shape[0]))
                    if m == 0:
                        continue
                    label = None if label_done else "u (MPC plan)"
                    alpha = 0.3 if label_done else 0.8
                    # ZOH for the MPC plan
                    ax_ctrl.step(t_fc[:m] / _SECONDS_PER_DAY, u_arr[:m], where="post", color="green", linestyle=":", linewidth=0.9, alpha=alpha, label=label)
                    label_done = True
                if label_done:
                    ax_ctrl.legend()

        axes[-1].set_xlabel("Time (days)")

        fig.tight_layout()
        if path:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    # === Generic LM identification ===

    def build_residual_lm(self):
        """-> Callable[[dict], jnp.ndarray]. General stacked residual on y."""
        y_meas = jnp.asarray(self.y_meas, dtype=jnp.float32)
        if y_meas.ndim == 1:
            y_meas = y_meas[:, None]
        n, ny = int(y_meas.shape[0]), int(y_meas.shape[1])

        eps = jnp.asarray(1e-6, dtype=jnp.float32)
        scale = jnp.maximum(jnp.var(y_meas, axis=0), eps)
        norm = jnp.sqrt(scale * jnp.asarray(float(n), dtype=jnp.float32))

        if self.W is None:
            W_vec = jnp.ones((ny,), dtype=jnp.float32)
        else:
            W_vec = jnp.asarray(self.W, dtype=jnp.float32).reshape((ny,))

        penalty = jnp.asarray(1e3, dtype=jnp.float32)

        def residual(theta):
            sim = self.simulation_for_dataset()
            x0 = None
            if self.initial_state_fn is not None:
                x0 = self.initial_state_fn(self, theta)
            t, y_sim, *_ = sim.run(theta, x0=x0)
            y_arr = jnp.asarray(y_sim, dtype=jnp.float32)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, None]
            m = int(min(y_arr.shape[0], n))
            y_s = y_arr[:m]
            y_m = y_meas[:m]
            finite = jnp.all(jnp.isfinite(y_s))
            err = y_s - y_m
            res = jnp.sqrt(W_vec) * err / norm
            vec = res.reshape((-1,))
            penalty_vec = jnp.full_like(vec, penalty)
            return jnp.where(finite, vec, penalty_vec)

        return jax.jit(residual)

    def plot(self, theta=None, *, path=None, theta_initial=None, NP=False):
        """Plot y_sim vs y_meas, along with simulated/measured disturbances and controls."""
        sim = self.simulation_for_dataset()
        theta_used = theta if theta is not None else sim.model.theta

        x0 = None
        if self.initial_state_fn is not None:
            x0 = self.initial_state_fn(self, theta_used)

        meas_len = self.y_meas.shape[0] if self.y_meas is not None else self.dataset.time.shape[0]
        setpoints = self.build_setpoint_profile(meas_len)
        run_fn = sim.run_numpy if NP else sim.run
        res = run_fn(theta_used, x0=x0)
        if len(res) == 6:
            t, y_sim, states, controls, _ctrl_states, mpc_forecasts = res
        elif len(res) == 5:
            t, y_sim, states, controls, mpc_forecasts = res
        elif len(res) == 4:
            t, y_sim, states, controls = res
            mpc_forecasts = None
        else:
            t, y_sim = res[:2]
            states = controls = None
            mpc_forecasts = None

        # Main simulated control
        u_sim = None
        u_name = None
        u_unit = None
        if isinstance(controls, dict) and controls:
            c_names = getattr(sim.model, "control_names", None)
            key = None
            if isinstance(c_names, tuple):
                for name in c_names:
                    if name in controls:
                        key = name
                        break
            if key is None:
                key = next(iter(controls))
            u_sim = controls[key]
            u_name = key
            c_units = getattr(sim.model, "control_units", None)
            if isinstance(c_names, tuple) and isinstance(c_units, tuple) and len(c_units) == len(c_names) and key in c_names:
                idx = c_names.index(key)
                u_unit = c_units[idx]
            elif isinstance(c_units, tuple) and len(c_units) > 0:
                u_unit = c_units[0]

        # Main measured control (if available in the dataset)
        u_meas = None
        if u_name is not None and u_name in self.dataset.u:
            u_meas = jnp.asarray(self.dataset.u[u_name], dtype=jnp.float32)

        # Simulated and measured disturbances
        n = int(jnp.asarray(y_sim).shape[0])
        dist_sim = {}
        for key, full in sim.d.items():
            dist_sim[key] = jnp.asarray(full, dtype=jnp.float32)[:n]
        dist_meas = {}
        for key, full in self.dataset.d.items():
            dist_meas[key] = jnp.asarray(full, dtype=jnp.float32)[:n]

        # Metadata from the model
        y_names = getattr(sim.model, "output_names", None)
        y_units = getattr(sim.model, "output_units", None)
        state_names = getattr(sim.model, "state_names", None)
        state_units = getattr(sim.model, "state_units", None)
        dist_names = getattr(sim.model, "disturbance_names", None)
        dist_units = getattr(sim.model, "disturbance_units", None)

        Sim_and_Data.render_plot(
            time=t,
            states=states,
            state_names=state_names,
            state_units=state_units,
            disturbances=dist_sim,
            disturbances_meas=dist_meas,
            dist_names=dist_names,
            dist_units=dist_units,
            y_sim=y_sim,
            y_meas=self.y_meas,
            y_names=y_names,
            y_units=y_units,
            u_sim=u_sim,
            u_meas=u_meas,
            u_name=u_name,
            u_unit=u_unit,
            setpoints=setpoints,
            path=path,
            mpc_forecasts=mpc_forecasts,
        )


@dataclass(frozen=True)
class FitResult:
    theta: dict[str, Any]
    metrics: dict[str, Any]
    simulation: Simulation_JAX


def _build_bounds_trees(theta0, bounds):
    if bounds is None:
        return None, None

    def as_bound(theta_leaf, bound_leaf, kind):
        if not isinstance(bound_leaf, dict):
            raise ValueError("Each bounds leaf must be a dict with 'lb' and 'ub'.")
        if kind not in bound_leaf:
            raise ValueError(f"Missing '{kind}' in bounds leaf.")
        value = bound_leaf[kind]
        return jnp.asarray(value, dtype=theta_leaf.dtype)

    try:
        lower = jtu.tree_map(lambda t, b: as_bound(t, b, "lb"), theta0, bounds)
        upper = jtu.tree_map(lambda t, b: as_bound(t, b, "ub"), theta0, bounds)
    except Exception as exc:
        raise ValueError("Bounds structure must match theta structure.") from exc

    def check(lo, hi):
        if not bool(jnp.all(lo <= hi)):
            raise ValueError("Lower bound greater than upper bound for at least one parameter.")
        return None

    jtu.tree_map(check, lower, upper)
    return lower, upper


def _project_bounds(theta, lower, upper):
    if lower is None or upper is None:
        return theta
    return jtu.tree_map(lambda v, lo, hi: jnp.clip(v, lo, hi), theta, lower, upper)


def _has_box_bounds(lower, upper):
    return lower is not None and upper is not None


def _soft_encode(theta, lower, upper):
    if not _has_box_bounds(lower, upper):
        return theta

    def encode(value, lo, hi):
        span = jnp.asarray(hi - lo, dtype=value.dtype)
        zero_span = jnp.all(span == 0)
        eps = jnp.asarray(1e-6, dtype=value.dtype)
        one = jnp.asarray(1.0, dtype=value.dtype)
        denom = jnp.where(zero_span, jnp.ones_like(span), span)
        rel = (value - lo) / denom
        rel = jnp.clip(rel, eps, one - eps)
        encoded = jnp.log(rel) - jnp.log1p(-rel)
        return jnp.where(zero_span, jnp.zeros_like(encoded), encoded)

    return jtu.tree_map(encode, theta, lower, upper)


def _soft_decode(free_theta, lower, upper):
    if not _has_box_bounds(lower, upper):
        return free_theta

    def decode(value, lo, hi):
        lo_arr = jnp.asarray(lo, dtype=value.dtype)
        span_arr = jnp.asarray(hi - lo, dtype=value.dtype)
        zero_span = jnp.all(span_arr == 0)
        decoded = lo_arr + span_arr * jax.nn.sigmoid(value)
        return jnp.where(zero_span, lo_arr, decoded)

    return jtu.tree_map(decode, free_theta, lower, upper)


def _bound_contacts(theta, lower, upper):
    if lower is None or upper is None:
        return {"lower": [], "upper": []}

    def iterate(path, node, lo, hi):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from iterate(path + (key,), value, lo[key], hi[key])
        else:
            yield path, node, lo, hi

    tol = 1e-6
    lower_hits = []
    upper_hits = []
    for path, value, lo, hi in iterate((), theta, lower, upper):
        val = jnp.asarray(value)
        lo_arr = jnp.asarray(lo)
        hi_arr = jnp.asarray(hi)
        if bool(jnp.all(jnp.isclose(val, lo_arr, rtol=1e-4, atol=tol))):
            lower_hits.append(".".join(path))
        elif bool(jnp.all(jnp.isclose(val, hi_arr, rtol=1e-4, atol=tol))):
            upper_hits.append(".".join(path))
    return {"lower": lower_hits, "upper": upper_hits}


def fit_lm(
    sim_data: Sim_and_Data,
    *,
    bounds: dict[str, Any] | None = None,
    maxiter: int = 60,
    tol: float = 1e-6,
    verbose: bool = True,
) -> FitResult:
    """Fit model parameters via Levenberg–Marquardt on outputs y.

    Parameters
    ----------
    sim_data : Sim_and_Data
        Simulation/data coupling containing the reference simulation and measurements.
    bounds : dict[str, Any] | None, optional
        Bounds dictionary (lb/ub) with the same structure as `theta`.
    maxiter : int, optional
        Maximum number of LM solver iterations.
    tol : float, optional
        Convergence tolerance (residual and parameters).
    verbose : bool, optional
        If True, enable LM solver verbosity.

    Returns
    ------
    FitResult
        Immutable object containing the identified `theta`, optimization metrics,
        and a copy of the simulation with the updated model.
    """
    import math
    import time as _time

    simulation = sim_data.simulation
    theta0 = simulation.model.theta
    lower, upper = _build_bounds_trees(theta0, bounds)
    theta_start = _project_bounds(theta0, lower, upper)
    residual_theta = sim_data.build_residual_lm()

    residual_init = residual_theta(theta_start)
    jax.block_until_ready(residual_init)
    loss_init = float(jnp.sum(residual_init**2))

    soft_bounds = _has_box_bounds(lower, upper)
    t0 = _time.perf_counter()
    theta_param = _soft_encode(theta_start, lower, upper) if soft_bounds else theta_start
    theta_flat, unravel = ravel_pytree(theta_param)

    def residual_flat(flat_theta):
        tree_theta = unravel(flat_theta)
        bounded_theta = _soft_decode(tree_theta, lower, upper) if soft_bounds else tree_theta
        return residual_theta(bounded_theta)

    solver = jaxopt.LevenbergMarquardt(
        residual_fun=residual_flat,
        maxiter=int(maxiter),
        tol=float(tol),
        xtol=float(tol),
        gtol=float(tol),
        verbose=bool(verbose),
    )
    theta_flat_opt, solver_state = solver.run(theta_flat)
    theta_raw = unravel(theta_flat_opt)
    if soft_bounds:
        theta_raw = _soft_decode(theta_raw, lower, upper)
    elapsed = _time.perf_counter() - t0

    iterations = int(getattr(solver_state, "iter_num", 0))
    error_val = float(getattr(solver_state, "error", jnp.asarray(jnp.nan)))
    theta_opt = _project_bounds(theta_raw, lower, upper)
    residual_final = residual_theta(theta_opt)
    jax.block_until_ready(residual_final)
    loss_final = float(jnp.sum(residual_final**2))
    finite_loss = math.isfinite(loss_final)
    if not finite_loss:
        status = "non_finite_loss"
    elif error_val > tol:
        status = "not_converged"
    else:
        status = "ok"

    contacts = _bound_contacts(theta_opt, lower, upper)
    metrics = {
        "loss_init": loss_init,
        "loss_final": loss_final,
        "iterations": iterations,
        "opt_time": float(elapsed),
        "opt_time_per_iter": float(elapsed / max(iterations, 1)),
        "opt_error": error_val,
        "opt_status": status,
        "bound_hits_lower": contacts["lower"],
        "bound_hits_upper": contacts["upper"],
    }

    tuned_model = eqx.tree_at(lambda m: m.theta, simulation.model, theta_opt)
    updated_sim = simulation.copy(model=tuned_model)
    return FitResult(theta=theta_opt, metrics=metrics, simulation=updated_sim)


def print_report(sim_data: Sim_and_Data, fit: FitResult, header: str = "Identification"):
    """Print a text report based on the `fit_lm` metrics.

    Parameters
    ----------
    sim_data : Sim_and_Data
        Simulation/data coupling (unused here; kept for a uniform signature).
    fit : FitResult
        Identification result returned by `fit_lm`.
    header : str, optional
        Title printed at the top of the report.
    """
    metrics = fit.metrics or {}
    lines = [header + ":"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    print("\n".join(lines))
