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
MAX_STEPS       = 50            # FetchReach default horizon
GAMMA           = 0.98
LR_ACTOR        = 3e-4
LR_CRITIC       = 3e-4
LR_ALPHA        = 3e-4          # entropy-coefficient learning rate
BATCH_SIZE      = 256
BUFFER_SIZE     = 1_000_000
TAU             = 0.005         # soft-update rate for target critic
HIDDEN_DIM      = 256
UPDATE_EVERY    = 1             # learn every N steps
SAVE_DIR        = "results/train_sac_fetchreach"
LOG_STD_MIN     = -20
LOG_STD_MAX     = 2


# ── Observation helper ───────────────────────────────────────────────────────
def flatten_obs(obs) -> np.ndarray:
    """Concatenate observation, achieved_goal, desired_goal into a single vector.

    FetchReach-v3 returns a Dict observation:
        observation   (10,)  – gripper state
        achieved_goal  (3,)  – current end-effector position
        desired_goal   (3,)  – target position
    Flat state dim = 16.

    If the observation is already a flat array, return it as-is.
    """
    if isinstance(obs, dict):
        return np.concatenate([
            obs["observation"],
            obs["achieved_goal"],
            obs["desired_goal"],
        ])
    return np.asarray(obs).flatten()


# ── Gaussian Actor ───────────────────────────────────────────────────────────
class GaussianActor(nn.Module):
    """Squashed-Gaussian policy: outputs tanh-bounded continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        x = self.shared(state)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """Reparameterised sample with log-prob (tanh-squashing corrected)."""
        mean, log_std = self.forward(state)
        std    = log_std.exp()
        normal = Normal(mean, std)

        x_t    = normal.rsample()                       # pre-tanh value
        action = torch.tanh(x_t)

        # Compute log-prob with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean)


# ── Twin Q-Network (Critic) ─────────────────────────────────────────────────
class TwinCritic(nn.Module):
    """Two independent Q-networks for clipped double-Q."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.float32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── SAC Agent ────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 device: torch.device):
        self.action_dim = action_dim
        self.device = device

        # Action rescaling: actor outputs in [-1, 1], we map to [low, high]
        self.action_scale = torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias  = torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32, device=device)

        # Networks
        self.actor         = GaussianActor(state_dim, action_dim).to(device)
        self.critic        = TwinCritic(state_dim, action_dim).to(device)
        self.target_critic = TwinCritic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimisers
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Automatic entropy tuning (α)
        self.target_entropy = -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim    = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.buffer     = ReplayBuffer(BUFFER_SIZE)
        self.step_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ── Action selection ─────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                raw = self.actor.deterministic(state_t)
            else:
                raw, _ = self.actor.sample(state_t)
        # Rescale from [-1, 1] → env action range
        action = (raw.squeeze(0).cpu().numpy()
                  * self.action_scale.cpu().numpy()
                  + self.action_bias.cpu().numpy())
        return action

    # ── Store + learn ────────────────────────────────────────────────────
    def step(self, state, action, reward, next_state, done):
        # Store normalised action [-1, 1]
        norm_action = ((action - self.action_bias.cpu().numpy())
                       / self.action_scale.cpu().numpy())
        self.buffer.push(state, norm_action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % UPDATE_EVERY == 0 and len(self.buffer) >= BATCH_SIZE:
            self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        s  = torch.tensor(states,      device=self.device)
        a  = torch.tensor(actions,     device=self.device)
        r  = torch.tensor(rewards,     device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, device=self.device)
        d  = torch.tensor(dones,       device=self.device).unsqueeze(1)

        # ── Critic loss ──────────────────────────────────────────────────
        with torch.no_grad():
            na, nlp = self.actor.sample(ns)
            q1_tgt, q2_tgt = self.target_critic(ns, na)
            q_tgt    = torch.min(q1_tgt, q2_tgt) - self.alpha * nlp
            td_target = r + GAMMA * (1.0 - d) * q_tgt

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ── Actor loss ───────────────────────────────────────────────────
        new_a, log_pi = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_pi - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── Alpha (entropy coefficient) loss ─────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Soft-update target critic ────────────────────────────────────
        self._soft_update()

    def _soft_update(self):
        for tp, op in zip(self.target_critic.parameters(),
                          self.critic.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_results(scores: list, avg_scores: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    episodes = range(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(episodes, avg_scores, color="darkorange", linewidth=2,
            label="Avg reward (100 ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("SAC on FetchReach-v3")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_dir}/training_curves.png")


def record_animation(agent: SACAgent, args, save_dir: str, num_episodes: int = 3):
    env = gym.make(ENV_NAME, render_mode="rgb_array", reward_type="dense")
    best_frames, best_reward = [], -float("inf")

    for ep in range(num_episodes):
        obs_dict, _ = env.reset(seed=SEED + 10000 + ep)
        state = flatten_obs(obs_dict)
        frames, total_reward = [], 0.0

        for _ in range(MAX_STEPS):
            frames.append(env.render())
            action = agent.select_action(state, deterministic=True)

            robust_input = {"action": action, "robust_type": "action",
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
    ax.set_title(f"SAC FetchReach-v3 | Reward: {best_reward:.1f}", fontsize=12)

    def update(i):
        img.set_data(best_frames[i])
        return [img]

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(best_frames), interval=30, blit=True)
    gif_path = os.path.join(save_dir, "fetchreach_sac.gif")
    ani.save(gif_path, writer="pillow", fps=30)
    plt.close(fig)
    print(f"[INFO] Animation saved to {gif_path}")


# ── Main Training Loop ───────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Robust-Gymnasium config
    args = get_config().parse_args([])
    args.noise_factor = "none"          # disable all perturbations
    args.noise_sigma  = 0.0

    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Use dense reward so SAC can learn from shaped signal
    env = gym.make(ENV_NAME, reward_type="dense")

    # Probe dimensions from the Dict observation space
    obs_dict, _ = env.reset(seed=SEED)
    state       = flatten_obs(obs_dict)
    state_dim   = state.shape[0]                     # 16
    action_dim  = env.action_space.shape[0]           # 4
    action_low  = env.action_space.low
    action_high = env.action_space.high

    print(f"[INFO] State dim : {state_dim}")
    print(f"[INFO] Action dim: {action_dim}")
    print(f"[INFO] Action range: [{action_low[0]:.1f}, {action_high[0]:.1f}]")

    agent = SACAgent(state_dim, action_dim, action_low, action_high, device)

    scores, avg_scores = [], []
    recent_scores: deque = deque(maxlen=100)

    for ep in range(1, TOTAL_EPISODES + 1):
        obs_dict, _ = env.reset(seed=SEED + ep)
        state = flatten_obs(obs_dict)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)

            robust_input = {
                "action": action,
                "robust_type": "action",
                "robust_config": args,
            }
            obs_dict, reward, terminated, truncated, _ = env.step(robust_input)
            next_state = flatten_obs(obs_dict)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg = np.mean(recent_scores)
        avg_scores.append(avg)

        if ep % 20 == 0:
            print(f"Episode {ep:4d} | Reward: {total_reward:7.1f} "
                  f"| Avg(100): {avg:7.1f} | α: {agent.alpha.item():.4f}")

    # Save final model
    torch.save(agent.actor.state_dict(),
               os.path.join(SAVE_DIR, "sac_actor_final.pth"))
    torch.save(agent.critic.state_dict(),
               os.path.join(SAVE_DIR, "sac_critic_final.pth"))
    env.close()

    plot_results(scores, avg_scores, SAVE_DIR)
    record_animation(agent, args, SAVE_DIR)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
