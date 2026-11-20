from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trajmatch.config import BETA_END, BETA_START, DIFF_STEPS, HEAD_LEN, LAMBDA_X0, TAIL_LEN, UNET_CH, weight_paths
from trajmatch.diffusion.unet import TailUNet


class ConditionalDDPM(nn.Module):
    def __init__(self, steps: int = DIFF_STEPS, lr: float = 1e-6, ch: int = UNET_CH, device: str = "cpu"):
        super().__init__()
        self.n_steps = int(steps)
        self.device_name = device
        beta = torch.linspace(BETA_START, BETA_END, self.n_steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1.0 - beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.unet = TailUNet(ch=ch)
        self.optim = torch.optim.AdamW(self.unet.parameters(), lr=lr)

    def to(self, device):
        self.device_name = str(device)
        return super().to(device)

    def _gather(self, consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return consts.gather(-1, t).reshape(-1, 1, 1)

    def q_sample(self, x0, t):
        mean = self._gather(self.alpha_bar, t).sqrt() * x0
        var = 1.0 - self._gather(self.alpha_bar, t)
        eps = torch.randn_like(x0)
        return mean + var.sqrt() * eps, eps

    def learn(self, traj, lam: float | None = None):
        info = traj[..., :HEAD_LEN]
        x0 = traj[..., HEAD_LEN:]
        half = len(x0) // 2 + 1
        t = torch.randint(0, self.n_steps, (half,), device=x0.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[: len(x0)]
        xt, noise = self.q_sample(x0, t)
        pred = self.unet(xt.float(), info, t)
        loss_eps = F.mse_loss(noise.float(), pred)
        x0_hat = self.predict_x0(xt, t, pred)
        recon = F.mse_loss(x0_hat, x0)
        if lam is None:
            lam = LAMBDA_X0
        loss = loss_eps + float(lam) * recon
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item()), float(recon.item())

    def predict_x0(self, xt, t, pred_eps):
        sqrt_ab = self._gather(self.alpha_bar, t).sqrt()
        sqrt_one = (1.0 - self._gather(self.alpha_bar, t)).sqrt()
        return (xt - sqrt_one * pred_eps) / (sqrt_ab + 1e-8)

    @torch.no_grad()
    def ddpm_step(self, xt, info, t, extra=None):
        eps = self.unet(xt.float(), info, t, extra=extra)
        alpha_t = self._gather(self.alpha, t)
        beta_t = self._gather(self.beta, t)
        sqrt_one = (1.0 - self._gather(self.alpha_bar, t)).sqrt()
        mu = (1.0 / (alpha_t + 1e-8)).sqrt() * (xt - (beta_t / (sqrt_one + 1e-8)) * eps)
        noise = torch.randn_like(xt) * beta_t.sqrt()
        mask = (t > 0).float().reshape(-1, 1, 1)
        return mu + mask * noise

    @torch.no_grad()
    def denoise_from(self, xt, info, t, extra=None):
        cur = xt
        max_t = int(t.max().item())
        b = xt.size(0)
        for step in range(max_t, -1, -1):
            step_t = torch.full((b,), step, device=xt.device, dtype=t.dtype)
            update = (t >= step).float().reshape(-1, 1, 1)
            prev = self.ddpm_step(cur, info, step_t, extra=extra)
            cur = update * prev + (1.0 - update) * cur
        return cur

    @torch.no_grad()
    def sample(self, info, num: int | None = None):
        b, _, _ = info.shape
        device = info.device
        last = self.n_steps - 1
        if num is None:
            xt = torch.randn(b, 2, TAIL_LEN, device=device)
            tt = torch.full((b,), last, device=device, dtype=torch.long)
            return self.denoise_from(xt, info, tt)
        info_rep = info.repeat_interleave(int(num), dim=0)
        xt = torch.randn(b * int(num), 2, TAIL_LEN, device=device)
        tt = torch.full((b * int(num),), last, device=device, dtype=torch.long)
        x0 = self.denoise_from(xt, info_rep, tt)
        return x0.reshape(b, int(num), 2, TAIL_LEN)


def load_unet(path: str, device: str, steps: int = DIFF_STEPS, lr: float = 1e-6) -> ConditionalDDPM:
    model = ConditionalDDPM(steps=steps, lr=lr, device=device).to(device)
    state = torch.load(path, map_location=device)
    model.unet.load_state_dict(state, strict=True)
    return model


def load_city(city: str, device: str, steps: int = DIFF_STEPS, lr: float = 1e-6) -> tuple[ConditionalDDPM, ConditionalDDPM]:
    pre_path, post_path = weight_paths(city)
    return load_unet(str(pre_path), device, steps, lr), load_unet(str(post_path), device, steps, lr)
