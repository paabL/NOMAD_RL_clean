from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from NOMAD_RC5.test.flow_sample_common import DEFAULT_RUN_DIR


COLORS = {"NOMAD": "black", "MPC": "royalblue"}


def load_summary(path):
    data = np.load(path, allow_pickle=True)
    method = str(data["method"].item() if data["method"].shape == () else data["method"])
    return method, {k: data[k] for k in ("reward_days", "cum_rewards", "temp_days", "tz_c")}


def plot_compare(summaries, path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), dpi=180, constrained_layout=True)
    for method, data in summaries:
        color = COLORS.get(method, None)
        days, cum = data["reward_days"], data["cum_rewards"]
        temp_days, tz = data["temp_days"], data["tz_c"]
        axs[0].plot(days, cum, color=color, alpha=0.12, linewidth=0.8)
        axs[0].plot(days, cum.mean(axis=1), color=color, linewidth=2, label=method)
        axs[1].plot(temp_days, tz, color=color, alpha=0.08, linewidth=0.6)
        axs[1].plot(temp_days, tz.mean(axis=1), color=color, linewidth=1.8, label=method)
    axs[0].set_ylabel("Cumulative reward")
    axs[1].set_ylabel("Tz (degC)")
    axs[1].set_xlabel("Days")
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path)
    plt.close(fig)


def run(args):
    run_dir = Path(args.run_dir)
    nomad = Path(args.nomad_dir) if args.nomad_dir else run_dir / "flow_sample_NOMAD"
    mpc = Path(args.mpc_dir) if args.mpc_dir else run_dir / "flow_sample_mpc"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "flow_sample_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_compare([load_summary(nomad / "summary.npz"), load_summary(mpc / "summary.npz")], out_dir / "comparison.png")
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR)
    p.add_argument("--nomad-dir")
    p.add_argument("--mpc-dir")
    p.add_argument("--out-dir")
    print(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
