"""
DRQN Agent — Deep Recurrent Q-Network with LSTM memory.

The LSTM lets the agent remember its recent trajectory history so it can
plan multi-step manoeuvres (deviate → avoid → return to nominal path).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque


# ── Network ──────────────────────────────────────────────────────────────────

class DRQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.hidden_size = hidden

        # Encode raw observation into a latent vector
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden),
            nn.ReLU(),
        )

        # LSTM maintains episodic memory across timesteps
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)

        # Dueling heads: separate value and advantage streams
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x, hidden=None):
        """
        x      : (batch, seq_len, obs_dim)  during training
                 (1,     1,       obs_dim)  during single-step inference
        hidden : (h, c) LSTM state, or None to zero-initialise
        Returns: q_values (batch, seq_len, n_actions), new hidden
        """
        B, T, _ = x.shape
        enc = self.encoder(x.reshape(B * T, -1)).reshape(B, T, -1)
        lstm_out, new_hidden = self.lstm(enc, hidden)          # (B, T, H)
        flat = lstm_out.reshape(B * T, -1)
        val  = self.value_head(flat)                           # (B*T, 1)
        adv  = self.adv_head(flat)                             # (B*T, n_actions)
        # Dueling aggregation  Q = V + (A - mean(A))
        q    = val + (adv - adv.mean(dim=1, keepdim=True))
        return q.reshape(B, T, -1), new_hidden

    def init_hidden(self, batch: int = 1, device: str = 'cpu'):
        h = torch.zeros(1, batch, self.hidden_size, device=device)
        c = torch.zeros(1, batch, self.hidden_size, device=device)
        return (h, c)


# ── Episode Replay Buffer ─────────────────────────────────────────────────────

class EpisodeReplayBuffer:
    """Stores complete episodes; samples random sub-sequences for DRQN training."""

    def __init__(self, max_episodes: int = 800, seq_len: int = 16):
        self.buffer  = deque(maxlen=max_episodes)
        self.seq_len = seq_len

    def add(self, episode: list):
        """episode: list of (obs, action, reward, next_obs, done) tuples."""
        if len(episode) >= 4:          # skip very short episodes
            self.buffer.append(episode)

    def sample(self, batch_size: int):
        eps = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        B_obs, B_act, B_rew, B_nxt, B_don = [], [], [], [], []
        for ep in eps:
            L = len(ep)
            if L >= self.seq_len:
                start = random.randint(0, L - self.seq_len)
                seq   = ep[start : start + self.seq_len]
            else:
                seq = list(ep) + [ep[-1]] * (self.seq_len - L)
            obs, act, rew, nxt, don = zip(*seq)
            B_obs.append(np.array(obs,  dtype=np.float32))
            B_act.append(np.array(act,  dtype=np.int64))
            B_rew.append(np.array(rew,  dtype=np.float32))
            B_nxt.append(np.array(nxt,  dtype=np.float32))
            B_don.append(np.array(don,  dtype=np.float32))
        return (
            np.stack(B_obs), np.stack(B_act),
            np.stack(B_rew), np.stack(B_nxt), np.stack(B_don)
        )

    def __len__(self): return len(self.buffer)


# ── DRQN Agent ────────────────────────────────────────────────────────────────

class DRQNAgent:
    def __init__(
        self,
        obs_dim:        int,
        n_actions:      int,
        device:         str   = 'cpu',
        lr:             float = 5e-4,
        gamma:          float = 0.99,
        eps_start:      float = 1.0,
        eps_end:        float = 0.05,
        eps_decay:      float = 0.993,
        batch_size:     int   = 32,
        seq_len:        int   = 16,
        target_update:  int   = 10,
    ):
        self.obs_dim       = obs_dim
        self.n_actions     = n_actions
        self.device        = torch.device(device)
        self.gamma         = gamma
        self.epsilon       = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.seq_len       = seq_len
        self.target_update = target_update

        self.q_net      = DRQNNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = DRQNNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer        = EpisodeReplayBuffer(seq_len=seq_len)
        self.episode_count = 0
        self._loss_history = []

    # ── Inference ─────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, hidden):
        """
        Epsilon-greedy action selection.
        Updates the LSTM hidden state even during random actions.
        Returns (action_idx, new_hidden).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals, new_hidden = self.q_net(obs_t, hidden)
        if random.random() < self.epsilon:
            if random.random() < 0.8:
                action = 0  # 80% chance to coast during exploration
            else:
                action = random.randint(1, self.n_actions - 1)
        else:
            action = int(q_vals.squeeze().argmax().item())
        return action, new_hidden

    def init_hidden(self):
        return self.q_net.init_hidden(1, str(self.device))

    # ── Training ──────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < max(self.batch_size // 2, 8):
            return None

        obs_b, act_b, rew_b, nxt_b, don_b = self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs_b).to(self.device)   # (B, T, obs_dim)
        act_t = torch.LongTensor(act_b).to(self.device)    # (B, T)
        rew_t = torch.FloatTensor(rew_b).to(self.device)   # (B, T)
        nxt_t = torch.FloatTensor(nxt_b).to(self.device)   # (B, T, obs_dim)
        don_t = torch.FloatTensor(don_b).to(self.device)   # (B, T)

        # Current Q-values for taken actions
        q_vals, _ = self.q_net(obs_t)                          # (B, T, n_act)
        q_taken   = q_vals.gather(2, act_t.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Double DQN target
        with torch.no_grad():
            next_q_online, _ = self.q_net(nxt_t)
            best_actions      = next_q_online.argmax(dim=2, keepdim=True)
            next_q_target, _  = self.target_net(nxt_t)
            max_next_q        = next_q_target.gather(2, best_actions).squeeze(-1)
            targets           = rew_t + self.gamma * max_next_q * (1.0 - don_t)

        loss = nn.SmoothL1Loss()(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        l = float(loss.item())
        self._loss_history.append(l)
        return l

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net':        self.q_net.state_dict(),
            'target_net':   self.target_net.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epsilon':      self.epsilon,
            'episode_count':self.episode_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon       = ckpt.get('epsilon', self.eps_end)
        self.episode_count = ckpt.get('episode_count', 0)
