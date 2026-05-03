"""Double DQN self-play training script for SuperTicTacToe.

This script is intentionally compact so it can be adapted for an assignment.
It supports:
- a replay buffer
- target network updates
- epsilon-greedy exploration
- action masking for illegal moves

Usage:
    python train_double_dqn.py
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import importlib.util
from pathlib import Path


ENV_PATH = Path(__file__).with_name("Gymnasium environment.py")
spec = importlib.util.spec_from_file_location("gymnasium_environment", ENV_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
SuperTicTacToe = module.SuperTicTacToe


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray, np.ndarray]


@dataclass
class Config:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    min_buffer_size: int = 2_000
    target_update_every: int = 500
    train_steps: int = 30_000
    max_episode_steps: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_dim: int = 256
    log_path: str = "training_curve.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(tuple(transition))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return obs.astype(np.float32).reshape(-1)


def action_to_block_row_col(action: int) -> tuple[int, int, int]:
    block = action // 16
    row = (action % 16) // 4
    col = action % 4
    return block, row, col


def shape_reward(obs: np.ndarray, action: int, reward: float, terminated: bool) -> float:
    """Small dense reward to help the agent learn faster.

    The environment reward stays authoritative. This shaping only adds a tiny bonus
    when the new move extends a short, plausible line segment that already exists.
    It avoids rewarding isolated adjacency that does not contribute to a run.
    """
    shaped = reward
    if terminated:
        return shaped

    block, row, col = action_to_block_row_col(action)
    own = obs[block]

    def run_length(dr: int, dc: int) -> int:
        length = 1
        r, c = row - dr, col - dc
        while 0 <= r < 4 and 0 <= c < 4 and own[r, c] == 1:
            length += 1
            r -= dr
            c -= dc
        r, c = row + dr, col + dc
        while 0 <= r < 4 and 0 <= c < 4 and own[r, c] == 1:
            length += 1
            r += dr
            c += dc
        return length

    bonus = 0.0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if run_length(dr, dc) >= 2:
            bonus += 0.005
        if run_length(dr, dc) >= 3:
            bonus += 0.01

    return shaped + bonus


def epsilon_by_step(step: int, cfg: Config) -> float:
    frac = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(q_net: QNetwork, obs: np.ndarray, mask: np.ndarray, epsilon: float, device: str) -> int:
    legal_actions = np.flatnonzero(mask)
    if len(legal_actions) == 0:
        return 0

    if random.random() < epsilon:
        return int(random.choice(legal_actions))

    with torch.no_grad():
        x = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(x).squeeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        q_values = q_values.masked_fill(~mask_t, -1e9)
        return int(torch.argmax(q_values).item())


def sample_batch(batch: List[Transition], device: str):
    obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch)

    obs = torch.tensor(np.stack([flatten_obs(o) for o in obs]), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_obs = torch.tensor(np.stack([flatten_obs(o) for o in next_obs]), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    masks = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)
    next_masks = torch.tensor(np.stack(next_masks), dtype=torch.bool, device=device)
    return obs, actions, rewards, next_obs, dones, masks, next_masks


def train():
    cfg = Config()
    env = SuperTicTacToe()

    log_file = open(cfg.log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step",
        "episode",
        "epsilon",
        "loss",
        "episode_reward",
        "episode_length",
        "reward",
        "shaped_reward",
        "terminated",
        "truncated",
    ])

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(cfg.device)
    target_net = QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(cfg.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size)

    obs, info = env.reset()
    mask = env.action_mask()
    episode_reward = 0.0
    episode_len = 0
    episode_idx = 1
    last_loss = float("nan")

    for step in range(1, cfg.train_steps + 1):
        epsilon = epsilon_by_step(step, cfg)
        action = select_action(q_net, obs, mask, epsilon, cfg.device)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]
        done = terminated or truncated
        shaped_reward = shape_reward(obs, action, reward, terminated)

        replay.push(obs, action, shaped_reward, next_obs, done, mask, next_mask)

        obs = next_obs
        mask = next_mask
        episode_reward += shaped_reward
        episode_len += 1

        if len(replay) < cfg.min_buffer_size:
            loss_value = float("nan")
        else:
            batch = replay.sample(cfg.batch_size)
            obs_t, actions_t, rewards_t, next_obs_t, dones_t, masks_t, next_masks_t = sample_batch(batch, cfg.device)

            q_values = q_net(obs_t).gather(1, actions_t)

            with torch.no_grad():
                next_q_online = q_net(next_obs_t)
                next_q_online = next_q_online.masked_fill(~next_masks_t, -1e9)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

                next_q_target = target_net(next_obs_t).gather(1, next_actions)
                targets = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q_target

            loss = nn.functional.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
            optimizer.step()
            loss_value = float(loss.item())
            last_loss = loss_value

        if done or episode_len >= cfg.max_episode_steps:
            episode_idx += 1
            obs, info = env.reset()
            mask = env.action_mask()
            episode_reward = 0.0
            episode_len = 0

        csv_writer.writerow([
            step,
            episode_idx,
            f"{epsilon:.6f}",
            "" if np.isnan(loss_value) else f"{loss_value:.6f}",
            f"{episode_reward:.6f}",
            episode_len,
            f"{reward:.6f}",
            f"{shaped_reward:.6f}",
            int(terminated),
            int(truncated),
        ])

        if step % cfg.target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % 1000 == 0:
            print(f"step={step:6d} epsilon={epsilon:.3f} loss={last_loss:.4f}")

    log_file.close()
    torch.save(q_net.state_dict(), "super_tictactoe_ddqn.pt")
    print("Training complete. Saved to super_tictactoe_ddqn.pt")
    print(f"Training curve data saved to {cfg.log_path}")


if __name__ == "__main__":
    train()
