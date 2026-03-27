"""Controller implementations for SIMAX simulations (BOPTEST-compatible)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import jax.numpy as jnp
import numpy as np
import equinox as eqx

if TYPE_CHECKING:  # only for hints; avoids circular import
    from .Simulation import Simulation_JAX


# ------------------------------------------------------------------ #
# Helper Functions (Standalone)
# ------------------------------------------------------------------ #

def _prepare_sim_window(sim, i, window_size, forecast):
    """Prepare the simulation (time_grid and disturbances) for a given window."""
    num_time = sim.time_grid.shape[0]
    end_idx = min(i + window_size, num_time)
    window_grid = sim.time_grid[i:end_idx]
    
    sim_run = sim
    if forecast is not None:
        d_forecast = forecast.get("d")
        # Fallback: look up keys from sim.d in forecast
        if d_forecast is None:
             d_forecast = {k: forecast[k] for k in sim.d if k in forecast}
        
        if d_forecast:
            new_d = {}
            # 1. Slice existing disturbances
            for k, v in sim.d.items():
                v_arr = jnp.asarray(v)
                new_d[k] = v_arr[i:end_idx] if v_arr.shape[0] >= end_idx else v_arr
            # 2. Update with forecast
            new_d.update(d_forecast)
            sim_run = sim.copy(d=new_d, time_grid=window_grid)
            
    return sim_run, window_grid, end_idx

def _get_setpoints_window(setpoints, i, window_size, forecast, target_len, fallback_val=293.15):
    """Get and adapt the setpoint window."""
    if forecast is not None and "ST_window" in forecast:
        sp_window = jnp.asarray(forecast["ST_window"], dtype=jnp.float32)
    else:
        sp_full = jnp.asarray(setpoints, dtype=jnp.float32)
        end_sp = min(i + window_size, sp_full.shape[0])
        sp_window = sp_full[i:end_sp]

    # Padding/truncating to match target_len
    m = sp_window.shape[0]
    if m == 0:
        return jnp.full((target_len,), fallback_val, dtype=jnp.float32)
    if m < target_len:
        pad = jnp.full((target_len - m,), sp_window[-1], dtype=jnp.float32)
        return jnp.concatenate([sp_window, pad])
    if m > target_len:
        return sp_window[:target_len]
    return sp_window


# ------------------------------------------------------------------ #
# Controllers
# ------------------------------------------------------------------ #

class Controller(eqx.Module, ABC):
    """Base class for controllers compatible with JAX simulations."""

    def init_state(self):
        return None

    def setpoint_value(self, idx, *, fallback):
        seq = jnp.asarray(self.SetPoints, dtype=jnp.float32)
        if seq.size == 0:
            return jnp.asarray(fallback, dtype=jnp.float32)
        last = max(int(seq.shape[0]) - 1, 0)
        safe_idx = jnp.asarray(jnp.clip(idx, 0, last), dtype=jnp.int32)
        return seq[safe_idx]

    @abstractmethod
    def compute_control(self, *, idx, y_measurements, disturbances, ctrl_state, ST=None, dt=None, forecast=None, W=None):
        raise NotImplementedError


class Controller_Constant(Controller):
    u_c: float
    activate: bool = True
    n: int = 1

    def compute_control(self, *, idx, y_measurements, disturbances, ctrl_state, ST=None, dt=None, forecast=None, W=None):
        return {"oveHeaPumY_u": jnp.asarray(self.u_c, dtype=jnp.float32)}, ctrl_state


class Controller_PID(Controller):
    TSet: float = 273.15 + 21  # Default single setpoint
    k_p: float = 1.0
    k_d: float = 0.0
    k_i: float = 0.0
    n: int = 1
    verbose: bool = True
    SetPoints: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray([], dtype=jnp.float32))

    def init_state(self, state=None):  # type: ignore[override]
        return {
            "i_err": jnp.asarray(0.0, dtype=jnp.float32),
            "e_prev": jnp.asarray(0.0, dtype=jnp.float32),
            "d_err": jnp.asarray(0.0, dtype=jnp.float32),
            "u_prev": jnp.asarray(0.0, dtype=jnp.float32),
            "delta_sat": jnp.asarray(0.0, dtype=jnp.float32),
        }

    def compute_control(self, *, idx, y_measurements, disturbances, ctrl_state, ST=None, dt=None, forecast=None, W=None):
        if ctrl_state is None:
            ctrl_state = {
                "i_err": jnp.asarray(0.0, dtype=jnp.float32),
                "e_prev": jnp.asarray(0.0, dtype=jnp.float32),
                "d_err": jnp.asarray(0.0, dtype=jnp.float32),
                "u_prev": jnp.asarray(0.0, dtype=jnp.float32),
                "delta_sat": jnp.asarray(0.0, dtype=jnp.float32),
            }
        else:
            ctrl_state = dict(ctrl_state)

        # Extract Tz from measurements (or state[0] as fallback)
        if isinstance(y_measurements, dict) and "reaTZon_y" in y_measurements:
            tz = jnp.asarray(y_measurements["reaTZon_y"], dtype=jnp.float32)
        elif hasattr(y_measurements, "__getitem__") and len(y_measurements) > 0:
             # Fallback if a list/array is passed
             tz = jnp.asarray(y_measurements[0], dtype=jnp.float32)
        else:
             tz = jnp.asarray(293.15, dtype=jnp.float32) # Default fallback

        t_set = ST if ST is not None else self.setpoint_value(idx, fallback=self.TSet)
        k_p = jnp.asarray(self.k_p, dtype=jnp.float32)
        k_d = jnp.asarray(self.k_d, dtype=jnp.float32)
        k_i = jnp.asarray(self.k_i, dtype=jnp.float32)
        dt_val = 0.0 if dt is None else dt
        dt_in = jnp.asarray(dt_val, dtype=jnp.float32)

        err = t_set - tz
        i_err = jnp.asarray(ctrl_state.get("i_err", 0.0), dtype=jnp.float32)
        e_prev = jnp.asarray(ctrl_state.get("e_prev", 0.0), dtype=jnp.float32)
        d_err = jnp.asarray(ctrl_state.get("d_err", 0.0), dtype=jnp.float32)

        safe_dt = jnp.where(dt_in == 0.0, jnp.asarray(1.0, dtype=jnp.float32), dt_in)
        i_next = jnp.clip(i_err + err * dt_in, -100.0, 100.0)
        d_next = ((err - e_prev) / safe_dt + d_err) / 2.0

        n_val = jnp.asarray(self.n, dtype=jnp.int32)
        idx_int = jnp.asarray(idx, dtype=jnp.int32)
        u_prev_state = jnp.asarray(ctrl_state.get("u_prev", 0.0), dtype=jnp.float32)
        recompute = jnp.logical_or(n_val <= 1, jnp.equal(jnp.mod(idx_int, n_val), 0))
        u_raw = jnp.where(
            recompute,
            k_p * err + k_d * d_next + k_i * i_next,
            u_prev_state,
        )
        u = jnp.clip(u_raw, 0.0, 1.0)
        delta_sat = u_raw - u

        ctrl_state["i_err"] = i_next
        ctrl_state["e_prev"] = err
        ctrl_state["d_err"] = d_next
        ctrl_state["u_prev"] = u
        ctrl_state["delta_sat"] = delta_sat

        if self.verbose:
            msg = f"Index: {idx} | Tz: {float(tz):.2f} | Tset: {float(t_set):.2f} | Err: {float(err):.2f} | u: {float(u):.2f}"
            ctrl_state["last_log"] = msg
            print(msg)

        return {
            "oveHeaPumY_u": u,
            "delta_sat": delta_sat,
        }, ctrl_state





class Controller_constSeq(Controller):
    oveHeaPumY_u: jnp.ndarray

    def compute_control(self, *, idx, y_measurements, disturbances, ctrl_state, ST=None, dt=None, forecast=None, W=None):
        seq = jnp.asarray(self.oveHeaPumY_u, dtype=jnp.float32)
        clipped = jnp.clip(idx, 0, max(seq.shape[0] - 1, 0))
        return {
            "oveHeaPumY_u": seq[clipped],
            "oveHeaPumY_activate": jnp.ones_like(seq[clipped]),
        }, ctrl_state




# ------------------------------------------------------------------ #
# MPC Core & Controller
# ------------------------------------------------------------------ #

def mpc_cost_core(u_window_bloc, x_i, i, setpoints, sim, time_grid, window_size, n, forecast=None):
    """
    Compute MPC-window costs [energy, comfort] (pure JAX function).

    From a block control vector `u_window_bloc` (one value per block), we build
    the fine control `u_window` by repeating each value `n` times and truncating
    to `horizon_len`, then simulate the system and integrate (trapezoids) the
    net power and the comfort error.

    Parameters
    ----------
    u_window_bloc : array_like
        Reduced control, repeated in blocks of size n to form `u_window`.
    x_i : array_like
        Initial state at the start of the window (index i).
    i : int
        Current index in the global time grid.
    setpoints : array_like or callable
        Temperature setpoints over the horizon.
    sim : Simulation
        Simulation object compatible with `run(...)`.
    time_grid : array_like
        Global time grid.
    window_size : int
        Prediction window size (in time steps).
    n : int
        Repetition factor for values of `u_window_bloc`.
    forecast : optional
        Optional forecast data for the window.

    Returns
    -------
    jnp.ndarray
        Array (2,): [energy_cost, comfort_cost].
    """
    i = int(i)
    window_size = int(window_size)
    n = int(n)

    # Prepare simulation
    sim_run, window_grid, end_idx = _prepare_sim_window(sim, i, window_size, forecast)

    # Reconstruct the control
    horizon_len = end_idx - i
    u_window = jnp.repeat(u_window_bloc, n)[:horizon_len]
    u_window = jnp.clip(u_window, 0.0, 1.0)

    # Simulation interne (JAX)
    x_i = jnp.asarray(x_i, dtype=jnp.float32)
    controller = Controller_constSeq(oveHeaPumY_u=u_window)
    t, y_sim, _state, _controls = sim_run.run(time_grid=window_grid, x0=x_i, controller=controller)

    # Extract results
    y_arr = jnp.asarray(y_sim, dtype=jnp.float32)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    tz_sim = y_arr[:, 0]
    qc_sim = y_arr[:, 1] if y_arr.shape[1] > 1 else jnp.zeros_like(tz_sim)
    qe_sim = y_arr[:, 2] if y_arr.shape[1] > 2 else jnp.zeros_like(tz_sim)

    # Costs
    P_heatpump = qc_sim - qe_sim
    sp_window = _get_setpoints_window(setpoints, i, window_size, forecast, tz_sim.shape[0])
    delta_T = jnp.abs(sp_window - tz_sim)

    energy_cost = jnp.trapezoid(P_heatpump, t)
    comfort_cost = jnp.trapezoid(delta_T, t)

    return jnp.array([energy_cost, comfort_cost])




class Controller_MPC(Controller):
    """
    MPC skeleton controller.
    """
    # NOTE: This controller cannot be jitted as-is
    # because it depends on the SciPy optimizer (Python side).
    # TODO: Make the controller generic for any simulation
    sim: "Simulation_JAX"
    window_size: int
    n: int = 1 # Control every n steps
    W: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray([0.2/1000.0, 10.0], dtype=jnp.float32))
    cost_core: Callable[..., Any] = eqx.field(default=mpc_cost_core, static=True, repr=False)
    verbose: bool = True

    SetPoints: jnp.ndarray = eqx.field(default_factory=lambda: jnp.asarray([],dtype=jnp.float32))
    i: int = 0
    _objective_jit: Callable[..., Any] = eqx.field(init=False, repr=False, static=True)
    _objective_jit_grad: Callable[..., Any] = eqx.field(init=False, repr=False, static=True)

    def init_state(self):
        """Initialize the MPC state."""
        return {"i": 0, "x_i": self.sim.x0, "u_prev": 0.0, "u_window": None}

    def __post_init__(self):
        objective = type(self)._objective_bound
        def objective_u_first(u_window_bloc, self_ref, x_i, i, setpoints, W, forecast):
            return objective(self_ref, u_window_bloc, x_i, i, setpoints, W, forecast)

        objective_grad = eqx.filter_grad(objective_u_first)
        object.__setattr__(self, "_objective_jit", eqx.filter_jit(objective_u_first))
        object.__setattr__(self, "_objective_jit_grad", eqx.filter_jit(objective_grad))

    def _core_func(self, u_window_bloc, x_i, i, setpoints, forecast):
        return self.cost_core(
            u_window_bloc,
            x_i,
            i,
            setpoints,
            self.sim,
            self.sim.time_grid,
            self.window_size,
            self.n,
            forecast=forecast,
        )

    def _objective_bound(self, u_window_bloc, x_i, i, setpoints, W, forecast):
        costs = self._core_func(u_window_bloc, x_i, i, setpoints, forecast)
        return jnp.dot(W, costs)

    def objective_np(self, u, ctrl_state, W, forecast=None):
        i = int(ctrl_state.get("i", self.i))
        x_i = jnp.asarray(ctrl_state.get("x_i", self.sim.x0), dtype=jnp.float32)
        setpoints = jnp.asarray(ctrl_state.get("setpoints", self.SetPoints), dtype=jnp.float32)
        u_jax = jnp.asarray(u, dtype=jnp.float32)
        return float(self._objective_jit(u_jax, self, x_i, i, setpoints, W, forecast))

    def objective_np_grad(self, u, ctrl_state, W, forecast=None):
        i = int(ctrl_state.get("i", self.i))
        x_i = jnp.asarray(ctrl_state.get("x_i", self.sim.x0), dtype=jnp.float32)
        setpoints = jnp.asarray(ctrl_state.get("setpoints", self.SetPoints), dtype=jnp.float32)
        u_jax = jnp.asarray(u, dtype=jnp.float32)
        return np.asarray(self._objective_jit_grad(u_jax, self, x_i, i, setpoints, W, forecast), dtype=np.float64)

    def get_forecast_trajectory(self, u_window, x_i, i, forecast=None):
        """
        Simulate the trajectory predicted by the MPC for a given window."""
        sim_run, window_grid, _ = _prepare_sim_window(self.sim, int(i), self.window_size, forecast)
        x_i = jnp.asarray(x_i, dtype=jnp.float32)
        u_window = jnp.clip(jnp.asarray(u_window, dtype=jnp.float32), 0.0, 1.0)
        controller = Controller_constSeq(oveHeaPumY_u=u_window)
        t, y_sim, x_sim, _controls = sim_run.run(time_grid=window_grid, x0=x_i, controller=controller)
        return t, y_sim, x_sim


    # Warmstart before each optimization, every nZOH
    def warmstart_PID(self, ctrl_state, window_grid, x_i, window_len, end_idx, forecast, *, i, setpoints):
        """
        Minimal MPC warmstart from a PID and/or the previous MPC plan.

        Two cases:
          - first call or no previous MPC plan: full PID horizon,
            then reduction into ZOH blocks;
          - otherwise: recycle the previous MPC plan, shifted by one block and
            completed by repeating the last value.

        Parameters
        ----------
        ctrl_state : dict
            Controller internal state (x_i, latest_forecast, etc.).
        window_grid : array_like
            Local time grid for the current window.
        x_i : array_like
            Current estimated state at the start of the window.
        window_len : int
            Length of the reduced decision vector (number of ZOH blocks).
        end_idx : int
            End index of the window in the global grid (for consistency).
        forecast : dict ou None
            Forecast data passed to the simulator.

        Returns
        ------
        jnp.ndarray
            1D vector of size `window_len` used as initialization.
        """
        # Optionally retrieve the previous plan
        u_window_prev = None
        if "latest_forecast" in ctrl_state:
            u_window_prev = ctrl_state["latest_forecast"].get("u_plan_window", None)

        # Case 1: full PID (first call or no previous MPC plan)
        if u_window_prev is None or int(i) == 0:
            if self.verbose:
                print("WS full PID")
            x0 = jnp.asarray(ctrl_state.get("x_i", x_i), dtype=jnp.float32)

            sp_window = _get_setpoints_window(setpoints, int(i), self.window_size, forecast, window_grid.shape[0], fallback_val=float(x0[0]))
            controller = Controller_PID(k_p=0.6, k_d=0.0, k_i=0.6 / 800.0, SetPoints=sp_window, n=self.n, verbose=False)

            sim_run, _, _ = _prepare_sim_window(self.sim, int(i), self.window_size, forecast)
            _, _, _states, control, *_ = sim_run.run_numpy(time_grid=window_grid, x0=x0, controller=controller)
            u_pid = jnp.asarray(control["oveHeaPumY_u"], dtype=jnp.float32)

            u_pid_bloc = u_pid[::self.n]
            return u_pid_bloc[:window_len]

        # Case 2: truncated (recycle the previous MPC plan)
        if self.verbose:
            print("WS truncated")
        u_prev_bloc = jnp.asarray(u_window_prev, dtype=jnp.float32)[::self.n]

        # If we have at least 2 blocks, drop the first; otherwise keep the last (single) one
        base = u_prev_bloc[1:] if u_prev_bloc.size > 1 else u_prev_bloc[-1:]

        needed = max(window_len - base.shape[0], 0)
        pad = jnp.full((needed,), base[-1], dtype=jnp.float32)
        return jnp.concatenate([base, pad])[:window_len]






    def _estimate_state(self, x_prev, y_meas):
        """
        Minimal estimation of the MPC internal state.

        - `y_meas` is a measurement, not the state: we receive it but do not
          use it for now.
        - The internal state remains purely open-loop:
            * if `x_prev` exists, keep it as-is
        """
        return jnp.asarray(x_prev, dtype=jnp.float32)


    def compute_control(self, *, idx, y_measurements, disturbances, ctrl_state, ST=None, dt=None, ST_window=None, forecast=None, W=None):
        if ctrl_state is None: ctrl_state = self.init_state()
        else: ctrl_state = dict(ctrl_state)

        if ST is not None:
            setpoints = jnp.full((len(self.sim.time_grid),), ST, dtype=jnp.float32)
        elif ST_window is not None:
            setpoints = jnp.asarray(ST_window, dtype=jnp.float32)
        else:
            setpoints = jnp.asarray(self.SetPoints, dtype=jnp.float32)
        ctrl_state["setpoints"] = setpoints

        W_arr = jnp.asarray(W if W is not None else self.W, dtype=jnp.float32)
        i = int(idx)
        ctrl_state["i"] = i
        
        # --- State Estimation ---
        x_prev = ctrl_state.get("x_i")
        x_curr = self._estimate_state(x_prev, y_measurements)
        ctrl_state["x_i"] = x_curr
        x_i = x_curr
        # --------------------------------------
        

        ctrl_state["y_meas"] = y_measurements

        end_idx = min(i + self.window_size, len(self.sim.time_grid))
        window_grid = self.sim.time_grid[i:end_idx]


        n = int(getattr(self,"n",1))
        u_prev = ctrl_state.get("u_prev", None)
        


        horizon_len = end_idx - i
        window_len = int(np.ceil(horizon_len / max(n, 1)))
        
        # Only attempt MPC if we have at least 2 grid points
        recompute = (i % n == 0) and (horizon_len >= 2)

        if recompute:
            # Warmstart
            U0 = np.asarray(
                self.warmstart_PID(
                    ctrl_state,
                    window_grid,
                    x_i,
                    window_len,
                    forecast=forecast,
                    end_idx=end_idx,
                    i=i,
                    setpoints=setpoints,
                ),
                dtype=np.float64,
            )

            from scipy.optimize import minimize
            res = minimize(
                fun=self.objective_np,
                jac=self.objective_np_grad,
                x0=U0,
                args=(ctrl_state, W_arr, forecast),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * window_len,
                options={"maxiter": 50, "ftol": 1e-6, "disp": bool(self.verbose)}
            )
            u_window_zoh = res.x
            u_window = np.repeat(u_window_zoh, self.n)[:horizon_len]

            u_mpc = float(np.clip(u_window[0], 0.0, 1.0))
            ctrl_state["last_cost"] = float(res.fun)

            # Forecast trajectory for logging/debugging
            t_pred, y_pred, x_pred = self.get_forecast_trajectory(u_window, x_i, i, forecast)
            y_np = np.asarray(y_pred)
            x_np = np.asarray(x_pred)
            u_np = np.asarray(u_window)
            ctrl_state["latest_forecast"] = {
                "time": np.asarray(t_pred),
                # Keys expected by Simulation.run_numpy (mpc_logs)
                "y": y_np,
                "u": u_np,
                "x": x_np,
                # Compat: old keys used elsewhere
                "x_plan_window": x_np,
                "y_plan_window": y_np,
                "u_plan_window": u_np,
                "decision_idx": int(i),
            }
        else:
            u_window = ctrl_state.get("u_window")
            u_mpc = float(u_prev)

        ctrl_state["u_prev"] = u_mpc
        ctrl_state["u_window"] = u_window

        # --- Advance one step with sim.run and x0/time_grid ---
        dt_val = float(dt) if dt is not None else float(self.sim.time_grid[1] - self.sim.time_grid[0])
        d_step = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in (disturbances or {}).items()}
        u_step = {
            "oveHeaPumY_u": jnp.asarray(u_mpc, dtype=jnp.float32),
            "oveHeaPumY_activate": jnp.asarray(1.0, dtype=jnp.float32),
        }

        def rhs(y):
            return self.sim.model.state_derivative(y, u_step, d_step)

        x_next = self.sim.integrator(rhs, x_i, jnp.asarray(dt_val, dtype=jnp.float32))
        ctrl_state["x_i"] = x_next

        u_arr = jnp.asarray(u_mpc, dtype=jnp.float32)
        if setpoints.size == 0:
            tset_display = jnp.asarray(float(x_i[0]), dtype=jnp.float32)
        else:
            last = max(int(setpoints.shape[0]) - 1, 0)
            safe_idx = jnp.asarray(jnp.clip(idx, 0, last), dtype=jnp.int32)
            tset_display = setpoints[safe_idx]

        cost = float(ctrl_state.get("last_cost", 0.0))
        msg = (
            f"MPC | Index: {idx} | X_MPC: {x_i[0]} | "
            f"u_mpc: {float(u_arr):.2f} | Cost: {float(cost):.2f} | "
            f"Tset: {float(tset_display):.2f} | "
            f"Y_meas: {y_measurements[0]}"
        )
        ctrl_state["last_log"] = msg
        if self.verbose:
            print(msg)

        return {"oveHeaPumY_u": u_arr, "oveHeaPumY_activate": 1.0}, ctrl_state
        # For an MPC, ctrl_state contains the MPC internal state (i, x_i, u_prev, u_window, last_log, latest_forecast)


__all__ = [
    "Controller",
    "Controller_Proportional",
    "Controller_Constant",
    "Controller_PID",
    "Controller_constSeq",
    "Controller_MPC",
    "PID_Boptest",
]
