from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from NOMAD_RC5.env import RC5TorchBatch
    from NOMAD_RC5.latent_vae import BoundedContextDecoder, TrajectoryEncoder, kl_standard_normal, reparameterize
    from NOMAD_RC5.sim import (
        BASE_SETPOINT,
        CTX_NAMES,
        PID_KEYS,
        TH_KEYS,
        TZ_MAX_K,
        TZ_MIN_K,
        context_low_high,
        load_rc5_data,
        split_context_torch,
    )
else:
    from .env import RC5TorchBatch
    from .latent_vae import BoundedContextDecoder, TrajectoryEncoder, kl_standard_normal, reparameterize
    from .sim import (
        BASE_SETPOINT,
        CTX_NAMES,
        PID_KEYS,
        TH_KEYS,
        TZ_MAX_K,
        TZ_MIN_K,
        context_low_high,
        load_rc5_data,
        split_context_torch,
    )


REQUIRED_KEYS = {
    "seed",
    "device",
    "data_path",
    "save_dir",
    "param_bounds",
    "n_traj",
    "batch_size",
    "horizon",
    "latent_dim",
    "hidden",
    "steps",
    "lr",
    "beta_KL",
    "lambda_ctx",
    "lambda_phys",
    "val_fraction",
    "log_every",
    "plot_count",
    "channels",
    "env",
}

REQUIRED_ENV_KEYS = {"dt", "step_period", "future_steps"}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "vae_poc.json"


def load_config(argv):
    path = Path(argv[0]) if argv else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    cfg = json.loads(path.read_text())
    missing = sorted(REQUIRED_KEYS - set(cfg))
    missing_env = sorted(REQUIRED_ENV_KEYS - set(cfg.get("env", {})))
    if missing or missing_env:
        raise ValueError(f"Missing config keys: {missing}; missing env keys: {missing_env}")
    return cfg


def device_from(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_setpoints(batch, horizon, step_period, device):
    t_h = torch.arange(horizon, device=device, dtype=torch.float32) * float(step_period) / 3600.0
    periods = torch.tensor([2.0, 6.0, 12.0], device=device)
    amp = 0.2 + 1.3 * torch.rand(batch, 3, device=device)
    phi = 2.0 * math.pi * torch.rand(batch, 3, device=device)
    waves = amp[:, :, None] * torch.sin(2.0 * math.pi * t_h[None, None, :] / periods[None, :, None] + phi[:, :, None])
    return torch.clamp(BASE_SETPOINT + waves.sum(dim=1), TZ_MIN_K, TZ_MAX_K)


def stack_tau(rollout, channels):
    return torch.stack([rollout[k] for k in channels], dim=-1)


def context_valid(ctx, low, high):
    th, pac, pid = split_context_torch(ctx)
    tc = torch.full_like(pac["Tcn"], BASE_SETPOINT)
    ta = pac["Tan"]
    qc = pac["k_c"] * (pac["a_c"] + pac["b_c"] * (tc - pac["Tcn"]) + pac["c_c"] * (ta - pac["Tan"]))
    qe = -pac["k_e"] * (pac["a_e"] + pac["b_e"] * (tc - pac["Tcn"]) + pac["c_e"] * (ta - pac["Tan"]))
    rc_pos = torch.stack([th[k] > 0 for k in TH_KEYS], dim=1).all(dim=1)
    cap_pos = torch.stack([th[k] > 0 for k in TH_KEYS if k.startswith("C_")], dim=1).all(dim=1)
    res_pos = torch.stack([th[k] > 0 for k in TH_KEYS if k.startswith("R_")], dim=1).all(dim=1)
    pid_ok = torch.stack([(pid[k] >= low[CTX_NAMES.index(k)]) & (pid[k] <= high[CTX_NAMES.index(k)]) for k in PID_KEYS], dim=1).all(dim=1)
    return (
        torch.isfinite(ctx).all(dim=1)
        & (ctx >= low).all(dim=1)
        & (ctx <= high).all(dim=1)
        & rc_pos
        & cap_pos
        & res_pos
        & pid_ok
        & torch.isfinite(qc - qe.abs())
    )


def rollout_valid(tau):
    finite = torch.isfinite(torch.stack(list(tau.values()), dim=-1)).all(dim=(1, 2))
    temp_ok = ((tau["Tz"] > TZ_MIN_K - 30.0) & (tau["Tz"] < TZ_MAX_K + 30.0)).all(dim=1)
    sat = ((tau["u_hp"] < 1e-4) | (tau["u_hp"] > 1.0 - 1e-4)).float().mean(dim=1)
    return finite & temp_ok & (sat < 0.999)


def phys_loss(tau_hat):
    tau = torch.nan_to_num(stack_tau(tau_hat, ["Tz", "u_hp", "P_hp"]), nan=1e6, posinf=1e6, neginf=-1e6)
    temp = tau[..., 0]
    u = tau[..., 1]
    temp_pen = F.relu(temp - (TZ_MAX_K + 30.0)).square().mean() + F.relu((TZ_MIN_K - 30.0) - temp).square().mean()
    sat_pen = F.relu(((u < 1e-4) | (u > 1.0 - 1e-4)).float().mean(dim=1) - 0.95).square().mean()
    finite_pen = (~torch.isfinite(stack_tau(tau_hat, ["Tz", "u_hp", "P_hp"]))).float().mean()
    return temp_pen + sat_pen + 100.0 * finite_pen


def vae_loss(encoder, decoder, env, batch, low, high, tau_mean, tau_std, channels, cfg, sample=True):
    tau_mb, ctx_mb, setpoints_mb, starts_mb = batch
    mu, logvar = encoder(tau_mb)
    ctx_hat = decoder(reparameterize(mu, logvar) if sample else mu)
    tau_hat_dict = env.probe_rollout(ctx_hat, setpoints_mb, starts_mb)
    tau_hat = torch.nan_to_num(stack_tau(tau_hat_dict, channels), nan=1e6, posinf=1e6, neginf=-1e6)
    loss_traj = F.mse_loss((tau_hat - tau_mean) / tau_std, tau_mb)
    loss_kl = kl_standard_normal(mu, logvar)
    loss_ctx = F.mse_loss((ctx_hat - low) / (high - low), ctx_mb)
    loss_phys = phys_loss(tau_hat_dict)
    loss = loss_traj + cfg["beta_KL"] * loss_kl + cfg["lambda_ctx"] * loss_ctx + cfg["lambda_phys"] * loss_phys
    return loss, loss_traj, loss_kl, loss_ctx, loss_phys


def make_probe_batch(env, low, high, cfg, device):
    b, h = int(cfg["batch_size"]), int(cfg["horizon"])
    for _ in range(100):
        ctx = low + (high - low) * torch.rand(b, low.numel(), device=device)
        setpoints = sample_setpoints(b, h, cfg["env"]["step_period"], device)
        starts = torch.randint(0, env.max_start_h + 1, (b,), device=device)
        with torch.no_grad():
            tau = env.probe_rollout(ctx, setpoints, starts)
        valid = context_valid(ctx, low, high) & rollout_valid(tau)
        if valid.any():
            return ctx[valid], setpoints[valid], starts[valid], {k: v[valid] for k, v in tau.items()}
    raise RuntimeError("No valid VAE probe rollout found. Reduce param_bounds or horizon.")


def make_dataset(env, low, high, cfg, device):
    n = int(cfg["n_traj"])
    ctxs, setpoints, starts, taus = [], [], [], {k: [] for k in cfg["channels"]}
    while sum(x.shape[0] for x in ctxs) < n:
        ctx, sp, h0, tau = make_probe_batch(env, low, high, cfg, device)
        ctxs.append(ctx.detach())
        setpoints.append(sp.detach())
        starts.append(h0.detach())
        for k in taus:
            taus[k].append(tau[k].detach())
    ctx = torch.cat(ctxs, dim=0)[:n]
    setpoints = torch.cat(setpoints, dim=0)[:n]
    starts = torch.cat(starts, dim=0)[:n]
    tau = {k: torch.cat(v, dim=0)[:n] for k, v in taus.items()}
    if not torch.isfinite(ctx).all() or not torch.isfinite(torch.stack(list(tau.values()), dim=-1)).all():
        raise RuntimeError("Generated VAE dataset contains NaN/inf.")
    return (
        ctx,
        setpoints,
        starts,
        tau,
    )


def plot_recon(out_dir, target, recon, channels, n):
    x = torch.arange(target[channels[0]].shape[1]).cpu()
    for i in range(min(int(n), target[channels[0]].shape[0])):
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for ax, key in zip(axes, ["Tz", "u_hp", "P_hp"]):
            ax.plot(x, target[key][i].detach().cpu(), label="tau")
            ax.plot(x, recon[key][i].detach().cpu(), label="tau_hat")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("control step")
        fig.tight_layout()
        fig.savefig(out_dir / f"reconstruction_{i:02d}.png", dpi=150)
        plt.close(fig)


def plot_interpolation(out_dir, env_cfg, data, decoder, mu, setpoints, start, device):
    alphas = torch.linspace(0.0, 1.0, 7, device=device)
    z = (1.0 - alphas[:, None]) * mu[0:1] + alphas[:, None] * mu[1:2]
    ctx = decoder(z)
    env = RC5TorchBatch(data=data, device=device, n_envs=len(alphas), max_episode_length=setpoints.shape[1], **env_cfg)
    sp = setpoints[0:1].expand(len(alphas), -1)
    starts = start[0:1].expand(len(alphas))
    with torch.no_grad():
        tau = env.probe_rollout(ctx, sp, starts)
    x = torch.arange(sp.shape[1]).cpu()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, key in zip(axes, ["Tz", "u_hp", "P_hp"]):
        for i, a in enumerate(alphas.detach().cpu()):
            ax.plot(x, tau[key][i].detach().cpu(), label=f"a={float(a):.2f}")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("control step")
    fig.tight_layout()
    fig.savefig(out_dir / "interpolation.png", dpi=150)
    plt.close(fig)
    return ctx, tau


def plot_losses(out_dir, history):
    steps = [r[0] for r in history]
    train = [r[1] for r in history]
    val = [r[6] for r in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, train, label="train")
    ax.plot(steps, val, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_train_val.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config(argv)
    torch.manual_seed(int(cfg["seed"]))
    device = device_from(cfg["device"])
    out = Path(cfg["save_dir"])
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg["env"]
    data = load_rc5_data(cfg["data_path"], dt=env_cfg["dt"])
    env = RC5TorchBatch(data=data, device=device, n_envs=cfg["batch_size"], max_episode_length=cfg["horizon"], **env_cfg)
    low, high = context_low_high(device=device, param_bounds=tuple(cfg["param_bounds"]))
    ctx, setpoints, starts, tau_dict = make_dataset(env, low, high, cfg, device)
    channels = list(cfg["channels"])
    tau = stack_tau(tau_dict, channels).detach()
    tau_mean = tau.mean(dim=(0, 1), keepdim=True)
    tau_std = tau.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    tau_n = (tau - tau_mean) / tau_std
    ctx_n = (ctx - low) / (high - low)
    perm = torch.randperm(ctx.shape[0], device=device)
    n_val = max(1, int(round(float(cfg["val_fraction"]) * ctx.shape[0])))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if train_idx.numel() == 0:
        raise ValueError("val_fraction leaves no training samples.")
    print(f"dataset train={train_idx.numel()} val={val_idx.numel()} minibatch={cfg['batch_size']} horizon={cfg['horizon']} device={device}")

    encoder = TrajectoryEncoder(cfg["horizon"], len(channels), cfg["latent_dim"], cfg["hidden"]).to(device)
    decoder = BoundedContextDecoder(cfg["latent_dim"], low, high, cfg["hidden"]).to(device)
    opt = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=float(cfg["lr"]))
    history = []

    for step in range(1, int(cfg["steps"]) + 1):
        idx = train_idx[torch.randint(0, train_idx.numel(), (int(cfg["batch_size"]),), device=device)]
        batch = (tau_n[idx], ctx_n[idx], setpoints[idx], starts[idx])
        loss, loss_traj, loss_kl, loss_ctx, loss_phys = vae_loss(encoder, decoder, env, batch, low, high, tau_mean, tau_std, channels, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 10.0)
        opt.step()
        with torch.no_grad():
            v_idx = val_idx[: min(int(cfg["batch_size"]), val_idx.numel())]
            v_batch = (tau_n[v_idx], ctx_n[v_idx], setpoints[v_idx], starts[v_idx])
            val_loss, val_traj, val_kl, val_ctx, val_phys = vae_loss(encoder, decoder, env, v_batch, low, high, tau_mean, tau_std, channels, cfg, sample=False)
        row = [step, *(float(x.detach().cpu()) for x in (loss, loss_traj, loss_kl, loss_ctx, loss_phys, val_loss, val_traj, val_kl, val_ctx, val_phys))]
        history.append(row)
        if step == 1 or step % int(cfg["log_every"]) == 0:
            print("step %d train %.4g val %.4g traj %.4g val_traj %.4g" % (step, row[1], row[6], row[2], row[7]))

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        eval_idx = val_idx[: min(int(cfg["batch_size"]), val_idx.numel())]
        tau_eval = tau_n[eval_idx]
        setpoints_eval = setpoints[eval_idx]
        starts_eval = starts[eval_idx]
        tau_dict_eval = {k: v[eval_idx] for k, v in tau_dict.items()}
        mu, _ = encoder(tau_eval)
        ctx_hat = decoder(mu)
        tau_hat_dict = env.probe_rollout(ctx_hat, setpoints_eval, starts_eval)
    plot_recon(out, tau_dict_eval, tau_hat_dict, channels, cfg["plot_count"])
    _, tau_alpha = plot_interpolation(out, env_cfg, data, decoder, mu, setpoints_eval, starts_eval, device)
    plot_losses(out, history)

    torch.save(encoder.state_dict(), out / "encoder.pt")
    torch.save(decoder.state_dict(), out / "decoder.pt")
    torch.save({"mean": tau_mean.cpu(), "std": tau_std.cpu(), "channels": channels}, out / "traj_norm.pt")
    torch.save({"low": low.cpu(), "high": high.cpu(), "ctx_names": CTX_NAMES, "param_bounds": cfg["param_bounds"]}, out / "ctx_bounds.pt")
    (out / "vae_config.json").write_text(json.dumps(cfg, indent=2))
    with (out / "losses.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "train_traj", "train_kl", "train_ctx", "train_phys", "val_loss", "val_traj", "val_kl", "val_ctx", "val_phys"])
        writer.writerows(history)

    bounds_ok = bool(((ctx_hat >= low) & (ctx_hat <= high)).all().item())
    finite_ok = bool(torch.isfinite(stack_tau(tau_hat_dict, channels)).all().item())
    interp_ok = bool(torch.isfinite(stack_tau(tau_alpha, ["Tz", "u_hp", "P_hp"])).all().item())
    print(f"saved {out}")
    print(f"ctx_hat_in_bounds={bounds_ok} rollout_finite={finite_ok} interpolation_finite={interp_ok}")


if __name__ == "__main__":
    main()
