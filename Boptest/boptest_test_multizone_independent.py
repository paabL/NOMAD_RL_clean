from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
BOPTEST_GYM = ROOT / "external" / "project1-boptest-gym"
for path in (ROOT, BOPTEST_GYM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from boptestGymEnv import BoptestGymEnv
from examples.test_and_plot import test_agent
from NOMAD_RC5.sim import BASE_SETPOINT, TZ_MAX_K, TZ_MIN_K, nominal_pid, nominal_theta, pack_context

URL = "http://127.0.0.1:8000"
TESTCASE = "multizone_residential_hydronic"
POLICY_PATH = ROOT / "NOMAD_RC5" / "runs" / "default" / "model.zip"
VECNORM_PATH = ROOT / "NOMAD_RC5" / "runs" / "default" / "vecnormalize.pkl"
MODEL_NAME = "nomad_rc5_multizone_independent"
SCENARIO = {"electricity_price": "highly_dynamic"}
STEP_PERIOD = 3600
PREDICTIVE_PERIOD = 12 * 3600
START_TIME = 31 * 24 * 3600
WARMUP_PERIOD = 3 * 24 * 3600
EPISODE_LENGTH = 28 * 7 * 3600
RESULT_DIR = ROOT / "Boptest" / f"results_tests_{MODEL_NAME}_{SCENARIO['electricity_price']}"
DATE0 = pd.Timestamp("2023-01-01")

ZONES = [
    dict(
        label="Bth",
        temp="conHeaBth_reaTZon_y",
        power="reaHeaBth_y",
        gains="InternalGainsRad[Bth]",
        occ="Occupancy[Bth]",
        lower="LowerSetp[Bth]",
        upper="UpperSetp[Bth]",
        action="conHeaBth_oveTSetHea_u",
    ),
    dict(
        label="Liv",
        temp="conHeaLiv_reaTZon_y",
        power="reaHeaLiv_y",
        gains="InternalGainsRad[Liv]",
        occ="Occupancy[Liv]",
        lower="LowerSetp[Liv]",
        upper="UpperSetp[Liv]",
        action="conHeaLiv_oveTSetHea_u",
    ),
    dict(
        label="Ro1",
        temp="conHeaRo1_reaTZon_y",
        power="reaHeaRo1_y",
        gains="InternalGainsRad[Ro1]",
        occ="Occupancy[Ro1]",
        lower="LowerSetp[Ro1]",
        upper="UpperSetp[Ro1]",
        action="conHeaRo1_oveTSetHea_u",
    ),
    dict(
        label="Ro2",
        temp="conHeaRo2_reaTZon_y",
        power="reaHeaRo2_y",
        gains="InternalGainsRad[Ro2]",
        occ="Occupancy[Ro2]",
        lower="LowerSetp[Ro2]",
        upper="UpperSetp[Ro2]",
        action="conHeaRo2_oveTSetHea_u",
    ),
    dict(
        label="Ro3",
        temp="conHeaRo3_reaTZon_y",
        power="reaHeaRo3_y",
        gains="InternalGainsRad[Ro3]",
        occ="Occupancy[Ro3]",
        lower="LowerSetp[Ro3]",
        upper="UpperSetp[Ro3]",
        action="conHeaRo3_oveTSetHea_u",
    ),
]
ACTIONS = [zone["action"] for zone in ZONES]
RESULT_POINTS = (
    [zone["temp"] for zone in ZONES]
    + [zone["power"] for zone in ZONES]
    + ACTIONS
    + ["weatherStation_reaWeaTDryBul_y", "weatherStation_reaWeaHDirNor_y", "reaTSetHea_y", "reaTSetCoo_y"]
)


def build_observations():
    obs = [("time", (0.0, 604800.0))]
    obs += [(zone["temp"], (280.0, 310.0)) for zone in ZONES]
    obs += [(zone["power"], (0.0, 20000.0)) for zone in ZONES]
    obs += [("TDryBul", (265.0, 303.0)), ("HDirNor", (0.0, 862.0)), ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4))]
    obs += [(zone["gains"], (0.0, 500.0)) for zone in ZONES]
    obs += [(zone["occ"], (0.0, 10.0)) for zone in ZONES]
    obs += [(zone["lower"], (280.0, 310.0)) for zone in ZONES]
    obs += [(zone["upper"], (280.0, 310.0)) for zone in ZONES]
    return OrderedDict(obs)


OBSERVATIONS = build_observations()
_SHARED = None


def time_features(time_s):
    week = int(time_s // 604800)
    dow = int((time_s % 604800) // 86400)
    hour = 2.0 * np.pi * ((time_s % 86400) / 86400.0)
    week_angle = 2.0 * np.pi * (week % 52) / 52.0
    dow_angle = 2.0 * np.pi * dow / 7.0
    return np.asarray(
        [week, np.sin(week_angle), np.cos(week_angle), np.sin(dow_angle), np.cos(dow_angle), np.sin(hour), np.cos(hour)],
        dtype=np.float32,
    )


def shared_policy():
    global _SHARED
    if _SHARED is None:
        model = RecurrentPPO.load(
            POLICY_PATH,
            device="cpu",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.2,
            },
        )
        with Path(VECNORM_PATH).open("rb") as f:
            vecnorm = pickle.load(f)
        vecnorm.training = False
        _SHARED = model, vecnorm, pack_context(nominal_theta(), nominal_pid())
    return _SHARED


class NomadZoneAgent:
    def __init__(self, env, zone, action_idx):
        self.env = env
        self.zone = zone
        self.action_idx = int(action_idx)
        self.model, self.vecnorm, self.ctx_dummy = shared_policy()
        self.obs_index = {name: i for i, name in enumerate(env.observations)}
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.time = int(env.start_time)

    def reset(self, start_time=None):
        self.state = None
        self.episode_start[...] = True
        self.time = int(self.env.start_time if start_time is None else start_time)
        return self

    def _get(self, obs, name):
        return float(obs[self.obs_index[name]])

    def _pred(self, obs, name, k):
        return self._get(obs, f"{name}_pred_{k * STEP_PERIOD}")

    def observation(self, obs):
        now = np.concatenate(
            [
                np.asarray(
                    [
                        self._pred(obs, "TDryBul", 0),
                        self._pred(obs, "HDirNor", 0),
                        self._pred(obs, self.zone["occ"], 0),
                        self._pred(obs, "PriceElectricPowerHighlyDynamic", 0),
                    ],
                    dtype=np.float32,
                ),
                time_features(self.time),
                np.asarray([self._get(obs, self.zone["temp"]), self._get(obs, self.zone["power"])], dtype=np.float32),
            ]
        )
        forecast = np.asarray(
            [
                np.concatenate(
                    [
                        np.asarray(
                            [
                                self._pred(obs, "TDryBul", k),
                                self._pred(obs, "HDirNor", k),
                                self._pred(obs, self.zone["occ"], k),
                                self._pred(obs, "PriceElectricPowerHighlyDynamic", k),
                                self._pred(obs, self.zone["lower"], k),
                                self._pred(obs, self.zone["upper"], k),
                            ],
                            dtype=np.float32,
                        ),
                        time_features(self.time + k * STEP_PERIOD),
                    ]
                )
                for k in range(1, 13)
            ],
            dtype=np.float32,
        )
        return {"now": now, "forecast": forecast, "ctx": self.ctx_dummy.copy()}

    def normalize(self, obs):
        eps = float(getattr(self.vecnorm, "epsilon", 1e-8))
        clip = float(self.vecnorm.clip_obs)
        return {
            key: np.clip((value - self.vecnorm.obs_rms[key].mean) / np.sqrt(self.vecnorm.obs_rms[key].var + eps), -clip, clip).astype(np.float32)
            for key, value in obs.items()
        }

    def predict(self, obs, deterministic=True):
        action, self.state = self.model.predict(
            self.normalize(self.observation(obs)),
            state=self.state,
            episode_start=self.episode_start,
            deterministic=deterministic,
        )
        self.episode_start[...] = False
        self.time += STEP_PERIOD
        setpoint = BASE_SETPOINT + 5.0 * float(np.asarray(action).reshape(-1)[0])
        low = float(self.env.action_space.low[self.action_idx])
        high = float(self.env.action_space.high[self.action_idx])
        return float(np.clip(setpoint, max(TZ_MIN_K, low), min(TZ_MAX_K, high)))


class IndependentThermostatAgent:
    def __init__(self, env):
        self.env = env
        self.agents = [NomadZoneAgent(env, zone, i) for i, zone in enumerate(ZONES)]

    def reset(self, start_time=None):
        for agent in self.agents:
            agent.reset(start_time)
        return self

    def predict(self, obs, deterministic=True):
        action = np.asarray([agent.predict(obs, deterministic=deterministic) for agent in self.agents], dtype=self.env.action_space.dtype)
        return action, None


def make_env():
    return BoptestGymEnv(
        url=URL,
        testcase=TESTCASE,
        actions=ACTIONS,
        observations=OBSERVATIONS,
        scenario=SCENARIO,
        predictive_period=PREDICTIVE_PERIOD,
        random_start_time=False,
        start_time=START_TIME,
        max_episode_length=EPISODE_LENGTH,
        warmup_period=WARMUP_PERIOD,
        step_period=STEP_PERIOD,
        render_episodes=False,
        log_dir=str(ROOT / "Boptest"),
    )


def validate_env(env):
    missing_actions = [action for action in ACTIONS if action not in env.all_input_vars]
    missing_temps = [zone["temp"] for zone in ZONES if zone["temp"] not in env.all_measurement_vars]
    if missing_actions or missing_temps:
        raise RuntimeError(f"Missing BOPTEST signals: actions={missing_actions}, temps={missing_temps}")


def day_index(start_time):
    return int(start_time / 3600 / 24)


def save_results_csv(env):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    res = requests.put(
        f"{env.url}/results/{env.testid}",
        json={"point_names": RESULT_POINTS, "start_time": env.start_time + 1, "final_time": 3.1536e7},
        timeout=120,
    ).json()["payload"]
    path = RESULT_DIR / f"results_sim_{day_index(env.start_time)}.csv"
    pd.DataFrame(res).to_csv(path, index=False)
    return path


def rollout_frame(env, observations, actions, rewards):
    obs0 = observations[:-1]
    times = [DATE0 + pd.Timedelta(seconds=START_TIME + i * STEP_PERIOD) for i in range(len(actions))]
    idx = {name: i for i, name in enumerate(env.observations)}
    rows = {
        "time": times,
        "reward": np.asarray(rewards, dtype=np.float32),
        "TDryBul": [obs[idx["TDryBul_pred_0"]] - 273.15 for obs in obs0],
        "HDirNor": [obs[idx["HDirNor_pred_0"]] for obs in obs0],
        "price": [obs[idx["PriceElectricPowerHighlyDynamic_pred_0"]] for obs in obs0],
    }
    acts = np.asarray(actions, dtype=np.float32)
    for i, zone in enumerate(ZONES):
        rows[f"{zone['label']}_temp"] = [obs[idx[zone["temp"]]] - 273.15 for obs in obs0]
        rows[f"{zone['label']}_low"] = [obs[idx[f"{zone['lower']}_pred_0"]] - 273.15 for obs in obs0]
        rows[f"{zone['label']}_high"] = [obs[idx[f"{zone['upper']}_pred_0"]] - 273.15 for obs in obs0]
        rows[f"{zone['label']}_rl"] = acts[:, i] - 273.15
    return pd.DataFrame(rows).set_index("time")


def save_plot(frame):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"results_sim_{day_index(START_TIME)}.pdf"
    fig, axs = plt.subplots(len(ZONES) + 1, sharex=True, figsize=(11, 13))
    for ax, zone in zip(axs[:-1], ZONES):
        key = zone["label"]
        ax.plot(frame.index, frame[f"{key}_temp"], color="darkorange", linewidth=1)
        ax.plot(frame.index, frame[f"{key}_low"], color="gray", linewidth=1)
        ax.plot(frame.index, frame[f"{key}_high"], color="gray", linewidth=1)
        ax.plot(frame.index, frame[f"{key}_rl"], color="black", linewidth=1)
        ax.set_ylabel(f"{key}\nTemp (C)")
        axr = ax.twinx()
        axr.plot(frame.index, frame["price"], color="dimgray", linestyle="dotted", linewidth=1)
        axr.set_ylabel("Price")
    axs[-1].plot(frame.index, frame["reward"], color="royalblue", linewidth=1)
    axs[-1].set_ylabel("Reward")
    axt = axs[-1].twinx()
    axt.plot(frame.index, frame["TDryBul"], color="seagreen", linewidth=1)
    axt.plot(frame.index, frame["HDirNor"], color="gold", linewidth=1)
    axt.set_ylabel("Weather")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    env = make_env()
    validate_env(env)
    agent = IndependentThermostatAgent(env).reset(START_TIME)
    try:
        observations, actions, rewards, kpis = test_agent(
            env,
            agent,
            start_time=START_TIME,
            episode_length=EPISODE_LENGTH,
            warmup_period=WARMUP_PERIOD,
            plot=False,
        )
        print(kpis)
        csv_path = save_results_csv(env)
        plot_path = save_plot(rollout_frame(env, observations, actions, rewards))
        print(csv_path)
        print(plot_path)
    finally:
        env.stop()


if __name__ == "__main__":
    main()
