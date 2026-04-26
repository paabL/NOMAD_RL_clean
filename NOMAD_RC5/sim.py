from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from NOMAD.simax import Controller_constSeq, Model_JAX, SimulationDataset, Simulation_JAX

RAW_DT = 30.0
DT = 120.0
FUTURE_STEPS = 24 #24H de forecast

Q_INTERNAL_CON_W = 146.0
Q_INTERNAL_RAD_W = 219.0

BASE_SETPOINT = 273.15 + 21.0
LOWER_K = 273.15 + 20.0
UPPER_K = 273.15 + 23.0
TZ_MIN_K = 273.15 + 15.0
TZ_MAX_K = 273.15 + 30.0

PAC_TAN_K = 273.15 + 7.0
PAC_TCN_K = 273.15 + 35.0

TH = dict(
    C_c=1501928.375,
    C_f=36471268.0,
    C_i=61500000.0,
    C_w=6666669.0,
    C_z=1349999.75,
    R_c=0.0006453233654610813,
    R_f=0.00013333333481568843,
    R_i=0.000607668946031481,
    R_inf=0.004688282497227192,
    R_w1=0.00899999774992466,
    R_w2=0.0010000000474974513,
    gA=34.10103225708008,
)
PAC = dict(
    a_c=6000.00048828125,
    a_e=7851.01708984375,
    b_c=-449.9999694824219,
    b_e=-305.3724060058594,
    c_c=383.71771240234375,
    c_e=430.73504638671875,
    k_c=1.574265718460083,
    k_e=1.0164787769317627,
    Tcn=PAC_TCN_K,
    Tan=PAC_TAN_K,
)
PID = dict(kp=0.6, ki=0.6 / 800.0, kd=0.0)


def _bounds_from_nominal(values, b1=0.5, b2=2.0, rel=None):
    out = {}
    for key, value in values.items():
        if rel is not None:
            lo = float(value) * (1.0 - float(rel))
            hi = float(value) * (1.0 + float(rel))
        else:
            lo = float(value) * float(b1)
            hi = float(value) * float(b2)
        out[key] = (min(lo, hi), max(lo, hi))
    return out


TH_BOUNDS = _bounds_from_nominal(TH, rel=0.2)
PAC_BOUNDS = _bounds_from_nominal({k: v for k, v in PAC.items() if k not in ("Tcn", "Tan")}, rel=0.2)
PAC_BOUNDS["Tcn"] = (PAC["Tcn"] - 5.0, PAC["Tcn"] + 5.0)
PAC_BOUNDS["Tan"] = (PAC["Tan"] - 5.0, PAC["Tan"] + 5.0)
PID_BOUNDS = dict(
    kp=(PID["kp"] * 0.8, PID["kp"] * 1.2),
    ki=(PID["ki"] * 0.8, PID["ki"] * 1.2),
    kd=(0.0, 0.1),
)

TH_KEYS = list(TH_BOUNDS.keys())
PAC_KEYS = list(PAC_BOUNDS.keys())
PID_KEYS = list(PID_BOUNDS.keys())
CTX_NAMES = TH_KEYS + PAC_KEYS + PID_KEYS

FEATURE_NAMES = (
    "ta",
    "qsol",
    "occ",
    "occ_exp",
    "price",
    "week_idx",
    "week_sin",
    "week_cos",
    "dow_sin",
    "dow_cos",
    "hour_sin",
    "hour_cos",
)

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "rc5_data.npz"


@dataclass(frozen=True)
class RC5Data:
    dt: float
    time: jnp.ndarray
    time_np: np.ndarray
    time_h: np.ndarray
    dist_matrix: np.ndarray
    dist_h: np.ndarray
    dataset: SimulationDataset
    n_hours: int


def _as_jnp_dict(values):
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in values.items()}


def nominal_theta():
    return {"th": _as_jnp_dict(TH), "pac": _as_jnp_dict(PAC)}


def nominal_pid():
    return tuple(float(PID[k]) for k in PID_KEYS)


def context_low_high(device="cpu"):
    dev = torch.device(device)
    low = [TH_BOUNDS[k][0] for k in TH_KEYS] + [PAC_BOUNDS[k][0] for k in PAC_KEYS] + [PID_BOUNDS[k][0] for k in PID_KEYS]
    high = [TH_BOUNDS[k][1] for k in TH_KEYS] + [PAC_BOUNDS[k][1] for k in PAC_KEYS] + [PID_BOUNDS[k][1] for k in PID_KEYS]
    return torch.tensor(low, dtype=torch.float32, device=dev), torch.tensor(high, dtype=torch.float32, device=dev)


def pack_context(theta, pid):
    th = theta["th"]
    pac = theta["pac"]
    pid = dict(zip(PID_KEYS, pid)) if not isinstance(pid, dict) else pid
    return np.asarray(
        [*(float(th[k]) for k in TH_KEYS), *(float(pac.get(k, PAC[k])) for k in PAC_KEYS), *(float(pid[k]) for k in PID_KEYS)],
        dtype=np.float32,
    )


def unpack_context(ctx):
    x = np.asarray(ctx, dtype=np.float32).reshape(-1)
    i = 0
    th = {k: jnp.asarray(x[i + j], dtype=jnp.float32) for j, k in enumerate(TH_KEYS)}
    i += len(TH_KEYS)
    pac = {k: jnp.asarray(x[i + j], dtype=jnp.float32) for j, k in enumerate(PAC_KEYS)}
    i += len(PAC_KEYS)
    pid = tuple(float(x[i + j]) for j in range(len(PID_KEYS)))
    return {"th": th, "pac": pac}, pid


def split_context_torch(x):
    x = x.reshape(x.shape[0], -1)
    i = 0
    th = {k: x[:, i + j] for j, k in enumerate(TH_KEYS)}
    i += len(TH_KEYS)
    pac = {k: x[:, i + j] for j, k in enumerate(PAC_KEYS)}
    i += len(PAC_KEYS)
    pid = {k: x[:, i + j] for j, k in enumerate(PID_KEYS)}
    return th, pac, pid


def sample_params_uniform(n_envs, device="cpu", th_bounds=None, pac_bounds=None, pid_bounds=None):
    th_bounds = TH_BOUNDS if th_bounds is None else th_bounds
    pac_bounds = PAC_BOUNDS if pac_bounds is None else pac_bounds
    pid_bounds = PID_BOUNDS if pid_bounds is None else pid_bounds
    dev = torch.device(device)

    def sample(bounds):
        return {
            k: torch.empty((int(n_envs),), device=dev, dtype=torch.float32).uniform_(*map(float, bounds[k]))
            for k in bounds
        }

    return sample(th_bounds), sample(pac_bounds), sample(pid_bounds)


def load_rc5_data(path: str | Path | None = None, *, dt=DT) -> RC5Data:
    path = ASSET_PATH if path is None else Path(path)
    dt = float(dt)
    with np.load(path) as data:
        time_np = np.asarray(data["time_s"], dtype=np.float32)
        dist_matrix = np.asarray(data["dist_30s"], dtype=np.float32)

    downsample = max(1, int(round(dt / RAW_DT)))
    dt = float(downsample * RAW_DT)
    usable = (dist_matrix.shape[0] // downsample) * downsample
    dist_matrix = dist_matrix[:usable].reshape(-1, downsample, dist_matrix.shape[1]).mean(axis=1).astype(np.float32)
    time_np = time_np[:usable].reshape(-1, downsample)[:, 0].astype(np.float32)

    step_n = max(1, int(round(3600.0 / dt)))
    usable = (dist_matrix.shape[0] // step_n) * step_n
    dist_matrix = dist_matrix[:usable]
    time_np = time_np[:usable]
    dist_h = dist_matrix.reshape(-1, step_n, dist_matrix.shape[1]).mean(axis=1).astype(np.float32)
    time_h = time_np.reshape(-1, step_n)[:, 0].astype(np.float32)
    occ = dist_matrix[:, 2]

    dataset = SimulationDataset(
        time=jnp.asarray(time_np, dtype=jnp.float32),
        u={},
        d={
            "weaSta_reaWeaTDryBul_y": jnp.asarray(dist_matrix[:, 0], dtype=jnp.float32),
            "weaSta_reaWeaHGloHor_y": jnp.asarray(dist_matrix[:, 1], dtype=jnp.float32),
            "InternalGainsCon[1]": jnp.asarray(occ * Q_INTERNAL_CON_W, dtype=jnp.float32),
            "InternalGainsRad[1]": jnp.asarray(occ * Q_INTERNAL_RAD_W, dtype=jnp.float32),
            "occupancy": jnp.asarray(dist_matrix[:, 2], dtype=jnp.float32),
            "occupancy_exp": jnp.asarray(dist_matrix[:, 3], dtype=jnp.float32),
            "electricity_price": jnp.asarray(dist_matrix[:, 4], dtype=jnp.float32),
            "week_idx": jnp.asarray(dist_matrix[:, 5], dtype=jnp.float32),
            "week_sin": jnp.asarray(dist_matrix[:, 6], dtype=jnp.float32),
            "week_cos": jnp.asarray(dist_matrix[:, 7], dtype=jnp.float32),
            "dow_sin": jnp.asarray(dist_matrix[:, 8], dtype=jnp.float32),
            "dow_cos": jnp.asarray(dist_matrix[:, 9], dtype=jnp.float32),
            "hour_sin": jnp.asarray(dist_matrix[:, 10], dtype=jnp.float32),
            "hour_cos": jnp.asarray(dist_matrix[:, 11], dtype=jnp.float32),
        },
    )
    return RC5Data(
        dt=dt,
        time=dataset.time,
        time_np=time_np,
        time_h=time_h,
        dist_matrix=dist_matrix,
        dist_h=dist_h,
        dataset=dataset,
        n_hours=int(dist_h.shape[0]),
    )


def qc_dot(tc, ta, u_hp, pac):
    poly = pac["a_c"] + pac["b_c"] * (tc - pac.get("Tcn", PAC_TCN_K)) + pac["c_c"] * (ta - pac.get("Tan", PAC_TAN_K))
    return pac["k_c"] * poly * u_hp


def qe_dot(tc, ta, u_hp, pac):
    poly = pac["a_e"] + pac["b_e"] * (tc - pac.get("Tcn", PAC_TCN_K)) + pac["c_e"] * (ta - pac.get("Tan", PAC_TAN_K))
    return -pac["k_e"] * poly * u_hp


def rc5_state_derivative(state, theta, ta, q_solar, q_con, q_rad, u_hp):
    th = theta["th"]
    pac = theta["pac"]
    tz, tw, ti, tf, tc = state
    q_occ = q_con + q_rad
    qc = qc_dot(tc, ta, u_hp, pac)
    d_tz = ((ta - tz) / th["R_inf"] + (tw - tz) / th["R_w2"] + (tf - tz) / th["R_f"] + (ti - tz) / th["R_i"] + th["gA"] * q_solar + q_occ) / th["C_z"]
    d_tw = ((ta - tw) / th["R_w1"] + (tz - tw) / th["R_w2"]) / th["C_w"]
    d_ti = ((tz - ti) / th["R_i"]) / th["C_i"]
    d_tf = ((tz - tf) / th["R_f"] + (tc - tf) / th["R_c"]) / th["C_f"]
    d_tc = ((tf - tc) / th["R_c"] + qc) / th["C_c"]
    return jnp.asarray([d_tz, d_tw, d_ti, d_tf, d_tc], dtype=jnp.float32)


def rc5_state_fn(x, u, d, theta):
    return rc5_state_derivative(
        jnp.asarray(x, dtype=jnp.float32),
        theta,
        jnp.asarray(d["weaSta_reaWeaTDryBul_y"], dtype=jnp.float32),
        jnp.asarray(d["weaSta_reaWeaHGloHor_y"], dtype=jnp.float32),
        jnp.asarray(d["InternalGainsCon[1]"], dtype=jnp.float32),
        jnp.asarray(d["InternalGainsRad[1]"], dtype=jnp.float32),
        jnp.asarray(u["oveHeaPumY_u"], dtype=jnp.float32),
    )


def rc5_output_fn(x, u, d, theta):
    state = jnp.asarray(x, dtype=jnp.float32)
    tz = state[0]
    qc = qc_dot(state[-1], jnp.asarray(d["weaSta_reaWeaTDryBul_y"], dtype=jnp.float32), jnp.asarray(u["oveHeaPumY_u"], dtype=jnp.float32), theta["pac"])
    qe = qe_dot(state[-1], jnp.asarray(d["weaSta_reaWeaTDryBul_y"], dtype=jnp.float32), jnp.asarray(u["oveHeaPumY_u"], dtype=jnp.float32), theta["pac"])
    return tz, qc, qe


def build_rc5_model(theta=None):
    return Model_JAX(
        theta=nominal_theta() if theta is None else theta,
        state_fn=rc5_state_fn,
        output_fn=rc5_output_fn,
        state_names=("Tz", "Tw", "Ti", "Tf", "Tc"),
        state_units=("K", "K", "K", "K", "K"),
        output_names=("Tz", "Qc_dot", "Qe_dot"),
        output_units=("K", "W", "W"),
        control_names=("oveHeaPumY_u",),
        control_units=("-",),
        disturbance_names=(
            "weaSta_reaWeaTDryBul_y",
            "weaSta_reaWeaHGloHor_y",
            "InternalGainsCon[1]",
            "InternalGainsRad[1]",
        ),
        disturbance_units=("K", "W", "W", "W"),
    )


def rc5_steady_state_sys(ta, q_solar, q_con, q_rad, qc_dot_val, theta):
    th = theta["th"]
    q_occ = q_con + q_rad
    a = jnp.array(
        [
            [-(1 / th["R_inf"] + 1 / th["R_w2"] + 1 / th["R_f"] + 1 / th["R_i"]), 1 / th["R_w2"], 1 / th["R_i"], 1 / th["R_f"], 0.0],
            [1 / th["R_w2"], -(1 / th["R_w1"] + 1 / th["R_w2"]), 0.0, 0.0, 0.0],
            [1 / th["R_i"], 0.0, -(1 / th["R_i"]), 0.0, 0.0],
            [1 / th["R_f"], 0.0, 0.0, -(1 / th["R_f"] + 1 / th["R_c"]), 1 / th["R_c"]],
            [0.0, 0.0, 0.0, 1 / th["R_c"], -(1 / th["R_c"])],
        ],
        dtype=jnp.float32,
    )
    b = jnp.array(
        [-ta / th["R_inf"] - th["gA"] * q_solar - q_occ, -ta / th["R_w1"], 0.0, 0.0, -qc_dot_val],
        dtype=jnp.float32,
    )
    x = jnp.linalg.solve(a, b)
    t_min = 273.15 + jnp.asarray([15.0, 5.0, 15.0, 15.0, 15.0], dtype=jnp.float32)
    t_max = 273.15 + jnp.asarray([30.0, 35.0, 30.0, 40.0, 50.0], dtype=jnp.float32)
    return jnp.clip(x, t_min, t_max)


def rc5_steady_state_tz_fixed(ta, q_solar, q_con, q_rad, tz_set, theta):
    th = theta["th"]
    tz = jnp.asarray(tz_set, dtype=jnp.float32)
    q_occ = q_con + q_rad
    g_z = 1 / th["R_inf"] + 1 / th["R_w2"] + 1 / th["R_f"] + 1 / th["R_i"]
    a = jnp.array(
        [
            [1 / th["R_w2"], 1 / th["R_i"], 1 / th["R_f"], 0.0, 0.0],
            [-(1 / th["R_w1"] + 1 / th["R_w2"]), 0.0, 0.0, 0.0, 0.0],
            [0.0, -(1 / th["R_i"]), 0.0, 0.0, 0.0],
            [0.0, 0.0, -(1 / th["R_f"] + 1 / th["R_c"]), 1 / th["R_c"], 0.0],
            [0.0, 0.0, 1 / th["R_c"], -(1 / th["R_c"]), 1.0],
        ],
        dtype=jnp.float32,
    )
    b = jnp.array(
        [
            (-ta / th["R_inf"] - th["gA"] * q_solar - q_occ) + g_z * tz,
            (-ta / th["R_w1"]) - (1 / th["R_w2"]) * tz,
            -(1 / th["R_i"]) * tz,
            -(1 / th["R_f"]) * tz,
            0.0,
        ],
        dtype=jnp.float32,
    )
    tw, ti, tf, tc, qc_dot_val = jnp.linalg.solve(a, b)
    t = jnp.asarray([tz, tw, ti, tf, tc], dtype=jnp.float32)
    t_min = 273.15 + jnp.asarray([15.0, 5.0, 15.0, 15.0, 15.0], dtype=jnp.float32)
    t_max = 273.15 + jnp.asarray([30.0, 35.0, 30.0, 40.0, 50.0], dtype=jnp.float32)
    return jnp.clip(t, t_min, t_max), qc_dot_val


def build_rc5_simulation(data: RC5Data | None = None, *, x0=None, controller=None, theta=None, base_setpoint=BASE_SETPOINT):
    data = load_rc5_data() if data is None else data
    theta = build_rc5_model(theta).theta if theta is not None else nominal_theta()
    row = data.dist_matrix[0]
    q_con = row[2] * Q_INTERNAL_CON_W
    q_rad = row[2] * Q_INTERNAL_RAD_W
    x0_default, _ = rc5_steady_state_tz_fixed(row[0], row[1], q_con, q_rad, base_setpoint, theta)
    controller = controller or Controller_constSeq(oveHeaPumY_u=jnp.zeros((data.time.shape[0],), dtype=jnp.float32))
    return Simulation_JAX(
        time_grid=data.time,
        d=data.dataset.d,
        model=build_rc5_model(theta),
        controller=controller,
        x0=jnp.asarray(x0 if x0 is not None else x0_default, dtype=jnp.float32),
        integrator="euler",
    )
