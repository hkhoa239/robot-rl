"""
PPO Agent for Robust-Gymnasium FetchReach-v3 (continuous control).

Implements:
    * Gaussian policy with state-independent log-std (action-rescaled to env range)
    * Critic V(s)
    * Generalised Advantage Estimation (GAE)
    * Clipped surrogate objective with multi-epoch mini-batch updates

Designed to mirror the structure of `sac.py` so the two algorithms can be
benchmarked side-by-side from `benchmark.ipynb`.
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import random
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config

# ── Hyperparameters ──────────────────────────────────────────────────────────
ENV_NAME        = "FetchReach-v3"
SEED            = 42
TOTAL_EPISODES  = 1000
MAX_STEPS       = 50           # FetchReach default horizon
GAMMA           = 0.98
GAE_LAMBDA      = 0.95
LR              = 3e-4
CLIP_RANGE      = 0.2
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.0           # FetchReach is dense-reward; no extra exploration bonus
N_EPOCHS        = 10
MINI_BATCH      = 64
ROLLOUT_STEPS   = 2048          # collect this many env steps before each update
HIDDEN_DIM      = 256
MAX_GRAD_NORM   = 0.5
LOG_STD_INIT    = -0.5
LOG_STD_MIN     = -20
LOG_STD_MAX     = 2
SAVE_DIR        = "results/train_ppo_fetchreach"


# ── Observation helper ───────────────────────────────────────────────────────
def flatten_obs(obs) -> np.ndarray:
    """Concatenate observation, achieved_goal, desired_goal into a single vector."""
    if isinstance(obs, dict):
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"],
        ])
    return np.asarray(obs).flatten()


# ── Gaussian Actor (continuous) ──────────────────────────────────────────────
class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy with state-independent log-std.

    The mean is produced by an MLP; log_std is a learnable parameter shared
    across states. Actions are sampled from N(mean, std) and later clipped
    to the environment bounds (rescaling is handled by the agent).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        # Single log-std parameter per action dim (state-independent)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * LOG_STD_INIT
        )

    def forward(self, state: torch.Tensor):
        mean = self.net(state)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, state: torch.Tensor):
        dist = self.forward(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, dist

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ── Critic V(s) ──────────────────────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ── Rollout buffer (on-policy) ───────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.states     = []
        self.actions    = []  # stored in *normalised* [-1, 1] space
        self.rewards    = []
        self.dones      = []
        self.log_probs  = []
        self.values     = []

    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self):
        return (
            np.array(self.states,    dtype=np.float32),
            np.array(self.actions,   dtype=np.float32),
            np.array(self.rewards,   dtype=np.float32),
            np.array(self.dones,     dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values,    dtype=np.float32),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


# ── PPO Agent ────────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 device: torch.device):
        self.action_dim = action_dim
        self.device = device

        # Action rescaling: actor outputs in approx [-1, 1], we map to [low, high]
        self.action_scale = torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32, device=device)

        self.actor  = GaussianActor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR)

        self.buffer = RolloutBuffer()

    # ── Action selection ─────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Return env-scale action, plus (norm_action, log_prob, value) for the buffer."""
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                raw = self.actor.deterministic(state_t)
                log_prob = torch.zeros(1, device=self.device)
            else:
                raw, log_prob, _ = self.actor.sample(state_t)
                log_prob = log_prob.squeeze(0)
            value = self.critic(state_t)

        norm_action = raw.squeeze(0)                            # in [-1, 1]-ish
        # Clip BEFORE rescaling so log-prob still corresponds to the sampled value
        env_action = (torch.clamp(norm_action, -1.0, 1.0)
                      * self.action_scale + self.action_bias)
        return (
            env_action.cpu().numpy(),
            norm_action.cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    # ── GAE ──────────────────────────────────────────────────────────────
    def compute_gae(self, rewards, dones, values, next_value):
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1.0 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    # ── PPO update ───────────────────────────────────────────────────────
    def update(self, last_state):
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get()

        # Bootstrap value of the state where the rollout was cut off
        last_state_t = torch.tensor(last_state, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_value = float(self.critic(last_state_t).item())

        advantages, returns = self.compute_gae(rewards, dones, values, next_value)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t        = torch.tensor(states,        device=self.device)
        actions_t       = torch.tensor(actions,       device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, device=self.device)
        advantages_t    = torch.tensor(advantages,    device=self.device)
        returns_t       = torch.tensor(returns,       device=self.device)

        n = len(states)
        total_pl, total_vl, total_ent, n_updates = 0.0, 0.0, 0.0, 0

        for _ in range(N_EPOCHS):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, MINI_BATCH):
                idx = indices[start:start + MINI_BATCH]

                dist = self.actor(states_t[idx])
                new_log_probs = dist.log_prob(actions_t[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE,
                                    1.0 + CLIP_RANGE) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(states_t[idx])
                value_loss  = F.mse_loss(values_pred, returns_t[idx])

                # ── Actor update ─────────────────────────────────────────
                self.actor_optim.zero_grad()
                (policy_loss - ENTROPY_COEF * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optim.step()

                # ── Critic update ────────────────────────────────────────
                self.critic_optim.zero_grad()
                (VALUE_COEF * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optim.step()

                total_pl  += policy_loss.item()
                total_vl  += value_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        self.buffer.clear()
        return (total_pl / n_updates,
                total_vl / n_updates,
                total_ent / n_updates)


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_results(scores: list, avg_scores: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, alpha=0.3, color="seagreen", label="Episode reward")
    ax.plot(episodes, avg_scores, color="darkorange", linewidth=2,
            label="Avg reward (100 ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("PPO on FetchReach-v3")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_dir}/training_curves.png")


def record_animation(agent: PPOAgent, args, save_dir: str, num_episodes: int = 3):
    env = gym.make(ENV_NAME, render_mode="rgb_array", reward_type="dense")
    best_frames, best_reward = [], -float("inf")

    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=SEED + 10000 + ep)
        state = flatten_obs(obs_dict)
        frames, total_reward = [], 0.0

        for _ in range(MAX_STEPS):
            frames.append(env.render())
            env_action, _, _, _ = agent.select_action(state, deterministic=True)

            robust_input = {"action": env_action, "robust_type": "action",
                            "robust_config": args}
            obs_dict, reward, terminated, truncated, _ = env.step(robust_input)
            state = flatten_obs(obs_dict)
            total_reward += reward
            if terminated or truncated:
                frames.append(env.render())
                break

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
        print(f"[ANIM] Episode {ep+1}/{num_episodes} | Reward: {total_reward:.1f}")

    env.close()

    if not best_frames:
        print("[WARN] No frames captured – skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    img = ax.imshow(best_frames[0])
    ax.set_title(f"PPO FetchReach-v3 | Reward: {best_reward:.1f}", fontsize=12)

    def update(i):
        img.set_data(best_frames[i])
        return [img]

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(best_frames), interval=30, blit=True)
    gif_path = os.path.join(save_dir, "fetchreach_ppo.gif")
    ani.save(gif_path, writer="pillow", fps=30)
    plt.close(fig)
    print(f"[INFO] Animation saved to {gif_path}")


# ── Main Training Loop ───────────────────────────────────────────────────────
def train(total_episodes: int = TOTAL_EPISODES,
          max_steps: int = MAX_STEPS,
          rollout_steps: int = ROLLOUT_STEPS,
          seed: int = SEED,
          save_dir: str = SAVE_DIR,
          log_every: int = 20,
          verbose: bool = True):
    """Train PPO on FetchReach-v3.

    The agent collects ``rollout_steps`` env transitions (across however many
    episodes that takes) before each policy update.

    Returns
    -------
    agent      : trained PPOAgent
    scores     : list[float] – per-episode total reward
    avg_scores : list[float] – running mean (window 100) per episode
    args       : robust-gymnasium config (useful for record_animation)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Device: {device}")

    args = get_config().parse_args([])
    args.noise_factor = "none"
    args.noise_sigma  = 0.0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(ENV_NAME, reward_type="dense")

    obs_dict, _ = env.reset(seed=seed)
    state       = flatten_obs(obs_dict)
    state_dim   = state.shape[0]                     # 16
    action_dim  = env.action_space.shape[0]           # 4
    action_low  = env.action_space.low
    action_high = env.action_space.high

    if verbose:
        print(f"[INFO] State dim : {state_dim}")
        print(f"[INFO] Action dim: {action_dim}")
        print(f"[INFO] Action range: [{action_low[0]:.1f}, {action_high[0]:.1f}]")

    agent = PPOAgent(state_dim, action_dim, action_low, action_high, device)

    scores, avg_scores = [], []
    recent_scores: deque = deque(maxlen=100)
    last_pl, last_vl, last_ent = 0.0, 0.0, 0.0
    rollout_filled = 0

    for ep in range(1, total_episodes + 1):
        obs_dict, _ = env.reset(seed=seed + ep)
        state = flatten_obs(obs_dict)
        total_reward = 0.0

        for _ in range(max_steps):
            env_action, norm_action, log_prob, value = agent.select_action(state)

            robust_input = {
                "action": env_action,
                "robust_type": "action",
                "robust_config": args,
            }
            obs_dict, reward, terminated, truncated, _ = env.step(robust_input)
            next_state = flatten_obs(obs_dict)
            done = terminated or truncated

            agent.buffer.push(state, norm_action, reward, done, log_prob, value)
            rollout_filled += 1

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg = float(np.mean(recent_scores))
        avg_scores.append(avg)

        # Trigger PPO update once we've collected enough on-policy data
        if rollout_filled >= rollout_steps:
            last_pl, last_vl, last_ent = agent.update(state)
            rollout_filled = 0

        if verbose and ep % log_every == 0:
            print(f"Episode {ep:4d} | Reward: {total_reward:7.1f} "
                  f"| Avg(100): {avg:7.1f} | π-loss: {last_pl:+.4f} "
                  f"| V-loss: {last_vl:.4f} | H: {last_ent:+.3f}")

    # Final flush so we don't waste data collected since the last update
    if len(agent.buffer) > 0:
        agent.update(state)

    env.close()
    return agent, scores, avg_scores, args


def main():
    agent, scores, avg_scores, args = train()
    torch.save(agent.actor.state_dict(),
               os.path.join(SAVE_DIR, "ppo_actor_final.pth"))
    torch.save(agent.critic.state_dict(),
               os.path.join(SAVE_DIR, "ppo_critic_final.pth"))

    plot_results(scores, avg_scores, SAVE_DIR)
    record_animation(agent, args, SAVE_DIR)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
