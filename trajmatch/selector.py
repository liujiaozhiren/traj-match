from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trajmatch.config import HEAD_LEN, K_SELECT, SELECTOR_ALPHA, SELECTOR_DIM, SELECTOR_LAMBDA, TAIL_LEN


def _take(hydra: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expand = index.view(index.size(0), index.size(1), *([1] * (hydra.dim() - 2)))
    expand = expand.expand(-1, -1, *hydra.shape[2:])
    return torch.gather(hydra, 1, expand)


class _Encoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        info_dim = HEAD_LEN * 2 * 2
        tail_dim = TAIL_LEN * 2
        self.hidden = hidden
        self.info_enc = nn.Sequential(nn.Linear(info_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pre_proj = nn.Sequential(nn.Linear(tail_dim, hidden), nn.ReLU())
        self.post_proj = nn.Sequential(nn.Linear(tail_dim, hidden), nn.ReLU())
        self.film = nn.Linear(hidden, hidden * 4)
        self.pre_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.post_net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, info_pre, info_post, cand_pre, cand_post):
        b, n, length, _ = cand_pre.shape
        info = self.info_enc(torch.cat([info_pre.reshape(b, -1), info_post.reshape(b, -1)], dim=-1))
        pre_z = self.pre_proj(cand_pre.reshape(b, n, length * 2))
        post_z = self.post_proj(cand_post.reshape(b, n, length * 2))
        gamma_pre, gamma_post, beta_pre, beta_post = self.film(info).split(self.hidden, dim=-1)
        pre_c = gamma_pre[:, None] * pre_z + beta_pre[:, None]
        post_c = gamma_post[:, None] * post_z + beta_post[:, None]
        return self.pre_net(pre_c), self.post_net(post_c)


@dataclass
class SelectorState:
    meta: dict
    encoder: dict
    A_pre: torch.Tensor
    b_pre: torch.Tensor
    A_post: torch.Tensor
    b_post: torch.Tensor

    def save(self, path: str):
        torch.save(
            {
                "meta": self.meta,
                "encoder": self.encoder,
                "A_pre": self.A_pre,
                "b_pre": self.b_pre,
                "A_post": self.A_post,
                "b_post": self.b_post,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "SelectorState":
        data = torch.load(path, map_location="cpu")
        return SelectorState(
            meta=data.get("meta", {}),
            encoder=data["encoder"],
            A_pre=data["A_pre"],
            b_pre=data["b_pre"],
            A_post=data["A_post"],
            b_post=data["b_post"],
        )


class TailSelector:
    def __init__(
        self,
        k_select: int = K_SELECT,
        rep_dim: int = SELECTOR_DIM,
        alpha: float = SELECTOR_ALPHA,
        ridge: float = SELECTOR_LAMBDA,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.k_select = int(k_select)
        self.rep_dim = int(rep_dim)
        self.alpha = float(alpha)
        self.device = device
        self.encoder = _Encoder(rep_dim).to(device)
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        eye = torch.eye(rep_dim, device=device)
        self.A_pre = ridge * eye.clone()
        self.A_post = ridge * eye.clone()
        self.A_inv_pre = torch.inverse(self.A_pre)
        self.A_inv_post = torch.inverse(self.A_post)
        self.b_pre = torch.zeros(rep_dim, device=device)
        self.b_post = torch.zeros(rep_dim, device=device)
        self._cache = None

    @torch.no_grad()
    def choose(self, info_pre, info_post, hydra_pre, hydra_post):
        self.encoder.eval()
        info_pre = info_pre.to(self.device)
        info_post = info_post.to(self.device)
        hydra_pre = hydra_pre.to(self.device)
        hydra_post = hydra_post.to(self.device)
        z_pre, z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
        b, n, _ = z_pre.shape
        k = min(self.k_select, n)
        theta_pre = self.A_inv_pre @ self.b_pre
        theta_post = self.A_inv_post @ self.b_post
        chosen_pre, chosen_post = [], []
        for i in range(b):
            idx_pre = _topk(z_pre[i], theta_pre, self.A_inv_pre, self.alpha, k)
            idx_post = _topk(z_post[i], theta_post, self.A_inv_post, self.alpha, k)
            chosen_pre.append(idx_pre)
            chosen_post.append(idx_post)
        idx_pre = torch.stack(chosen_pre)
        idx_post = torch.stack(chosen_post)
        self._cache = (info_pre, info_post, hydra_pre, hydra_post, idx_pre, idx_post)
        return _take(hydra_pre, idx_pre), _take(hydra_post, idx_post), idx_pre, idx_post

    def update(self, reward: torch.Tensor):
        info_pre, info_post, hydra_pre, hydra_post, idx_pre, idx_post = self._cache
        self.encoder.train()
        reward = reward.to(self.device).view(-1)
        z_pre, z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
        sel_pre = torch.cat([z_pre[i, idx_pre[i]] for i in range(z_pre.size(0))], dim=0)
        sel_post = torch.cat([z_post[i, idx_post[i]] for i in range(z_post.size(0))], dim=0)
        with torch.no_grad():
            theta_pre = self.A_inv_pre @ self.b_pre
            theta_post = self.A_inv_post @ self.b_post
        loss = F.mse_loss(sel_pre @ theta_pre, reward) + F.mse_loss(sel_post @ theta_post, reward)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            z_pre, z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
            sel_pre = torch.cat([z_pre[i, idx_pre[i]] for i in range(z_pre.size(0))], dim=0)
            sel_post = torch.cat([z_post[i, idx_post[i]] for i in range(z_post.size(0))], dim=0)
            _ridge_update(self, sel_pre, sel_post, reward)

    def state(self) -> SelectorState:
        return SelectorState(
            meta={"k_select": self.k_select, "rep_dim": self.rep_dim, "alpha": self.alpha},
            encoder=self.encoder.state_dict(),
            A_pre=self.A_pre.detach().cpu(),
            b_pre=self.b_pre.detach().cpu(),
            A_post=self.A_post.detach().cpu(),
            b_post=self.b_post.detach().cpu(),
        )

    def load_state(self, state: SelectorState):
        self.encoder.load_state_dict(state.encoder, strict=True)
        self.A_pre = state.A_pre.to(self.device)
        self.b_pre = state.b_pre.to(self.device)
        self.A_post = state.A_post.to(self.device)
        self.b_post = state.b_post.to(self.device)
        self.A_inv_pre = torch.inverse(self.A_pre)
        self.A_inv_post = torch.inverse(self.A_post)


def _topk(z, theta, a_inv, alpha, k):
    mu = z @ theta
    sigma = torch.sqrt(torch.clamp((z @ a_inv * z).sum(dim=1), min=1e-8))
    return torch.topk(mu + alpha * sigma, k=k).indices


def _ridge_update(selector: TailSelector, z_pre, z_post, reward):
    for i in range(z_pre.size(0)):
        zp = z_pre[i].view(-1, 1)
        zo = z_post[i].view(-1, 1)
        selector.A_pre = selector.A_pre + zp @ zp.t()
        selector.b_pre = selector.b_pre + reward[i] * zp.squeeze(1)
        selector.A_post = selector.A_post + zo @ zo.t()
        selector.b_post = selector.b_post + reward[i] * zo.squeeze(1)
    selector.A_inv_pre = torch.inverse(selector.A_pre)
    selector.A_inv_post = torch.inverse(selector.A_post)


def normalize_reward(reward: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
    centered = reward - reward.mean()
    scale = torch.sqrt(torch.clamp((centered ** 2).mean(), min=1e-12))
    return torch.clamp(centered / scale, -clip, clip)
