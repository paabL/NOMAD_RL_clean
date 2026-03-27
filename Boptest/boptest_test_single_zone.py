from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle
import sys

import numpy as np
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
BOPTEST_GYM = ROOT / "external" / "project1-boptest-gym"
for path in (ROOT, BOPTEST_GYM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from boptestGymEnv import BoptestGymEnv
from examples.test_and_plot import plot_results, test_agent
from NOMAD_RC5.sim import BASE_SETPOINT, TZ_MAX_K, TZ_MIN_K, nominal_pid, nominal_theta, pack_context

URL = "https://api.boptest.net"
URL = "http://127.0.0.1:8000"  # local testing
TESTCASE = "bestest_hydronic_heat_pump"
POLICY_PATH = ROOT / "NOMAD_RC5" / "runs" / "default" / "model.zip"
VECNORM_PATH = ROOT / "NOMAD_RC5" / "runs" / "default" / "vecnormalize.pkl"
MODEL_NAME = "nomad_rc5_single_zone"
SCENARIO = {"electricity_price": "highly_dynamic"}
STEP_PERIOD = 3600
PREDICTIVE_PERIOD = 12 * 3600
START_TIME = 31 * 24 * 3600
WARMUP_PERIOD = 3 * 24 * 3600
EPISODE_LENGTH = 28 * 24 * 3600
OBSERVATIONS = OrderedDict(
    [
        ("time", (0, 604800)),
        ("reaTZon_y", (280.0, 310.0)),
        ("reaPHeaPum_y", (0.0, 20000.0)),
        ("TDryBul", (265.0, 303.0)),
        ("HDirNor", (0.0, 862.0)),
        ("InternalGainsRad[1]", (0.0, 219.0)),
        ("PriceElectricPowerHighlyDynamic", (-0.4, 0.4)),
        ("LowerSetp[1]", (280.0, 310.0)),
        ("UpperSetp[1]", (280.0, 310.0)),
    ]
)


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


class NomadBoptestAgent:
    def __init__(self, env, policy_path=POLICY_PATH, vecnorm_path=VECNORM_PATH):
        self.env = env
        self.model = RecurrentPPO.load(
            policy_path,
            device="cpu",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.2,
            },
        )
        with Path(vecnorm_path).open("rb") as f:
            self.vecnorm = pickle.load(f)
        self.vecnorm.training = False
        self.ctx_dummy = pack_context(nominal_theta(), nominal_pid())
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
                        self._pred(obs, "InternalGainsRad[1]", 0) / 219.0,
                        self._pred(obs, "PriceElectricPowerHighlyDynamic", 0),
                    ],
                    dtype=np.float32,
                ),
                time_features(self.time),
                np.asarray([self._get(obs, "reaTZon_y"), self._get(obs, "reaPHeaPum_y")], dtype=np.float32),
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
                                self._pred(obs, "InternalGainsRad[1]", k) / 219.0,
                                self._pred(obs, "PriceElectricPowerHighlyDynamic", k),
                                self._pred(obs, "LowerSetp[1]", k),
                                self._pred(obs, "UpperSetp[1]", k),
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
        out = {}
        for key, value in obs.items():
            rms = self.vecnorm.obs_rms[key]
            out[key] = np.clip((value - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
        return out

    def predict(self, obs, deterministic=True):
        action, self.state = self.model.predict(
            self.normalize(self.observation(obs)),
            state=self.state,
            episode_start=self.episode_start,
            deterministic=deterministic,
        )
        self.episode_start[...] = False
        self.time += STEP_PERIOD
        setpoint = np.clip(BASE_SETPOINT + 5.0 * float(np.asarray(action).reshape(-1)[0]), TZ_MIN_K, TZ_MAX_K)
        return np.asarray([setpoint], dtype=self.env.action_space.dtype), self.state


def make_env():
    return BoptestGymEnv(
        url=URL,
        testcase=TESTCASE,
        actions=["oveTSet_u"],
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


def main():
    env = make_env()
    agent = NomadBoptestAgent(env).reset(START_TIME)
    try:
        _, _, rewards, kpis = test_agent(
            env,
            agent,
            start_time=START_TIME,
            episode_length=EPISODE_LENGTH,
            warmup_period=WARMUP_PERIOD,
            plot=False,
        )
        print(kpis)
        plot_path = plot_results(
            env,
            rewards,
            log_dir=str(ROOT / "Boptest"),
            model_name=MODEL_NAME,
            save_to_file=True,
            action_point="oveTSet_u",
            action_label="Zone\nsetpoint\n(K)",
        )
        print(plot_path)
    finally:
        env.stop()


if __name__ == "__main__":
    main()
