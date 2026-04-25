# RL Experiments with Robust Gymnasium

This repository extends [Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning](https://github.com/fangevo/Robust-Gymnasium) with reinforcement learning experiments on two environments:

- **LunarLander-v3** (discrete actions) — classic control benchmark
- **FetchReach-v3** (continuous actions) — robotic arm reaching task using MuJoCo

It is designed as a practical benchmark to compare classic and modern RL methods under:
- standard training (no perturbation),
- perturbation-based robustness tests,
- delayed/sparse reward settings,
- value-estimation bias analysis.

## What is included

### Algorithms
- DQN
- Double DQN
- Dueling Double DQN (+ PER)
- PPO
- A2C (value critic and Q critic variants)
- REINFORCE
- **SAC (Soft Actor-Critic)** — continuous-action, maximum-entropy RL

### Experiment themes
- **Baseline learning** on `LunarLander-v3` and `FetchReach-v3`
- **Continuous control** with SAC on a robotic manipulation task
- **Robustness to perturbations** (state noise / reward noise)
- **Delayed reward credit assignment**
- **Q-value overestimation bias** (DQN vs Double DQN)
- **Value-Advantage decomposition analysis** (Dueling architecture)

## Project structure

```text
sac.py                          # SAC on FetchReach-v3 (continuous)

examples/
  LunarLander_DQN/
    train_dqn.py
    train_dqn_perturbation.py
    train_delayed_reward.py
    visualize_delayed_reward.py

  LunarLander_double_DQN&DDQN/
    train_double_dqn.py
    train_dueling_double_dqn.py
    compare_algorithms_perturbation.py

  LunarLander_PPO/
    train_PPO.py
    train_delayed_reward.py

  LunarLander_A2C/
    main.py
    evaluation.py
    evaluate_saved_model.py
    plot_results.py

  LunarLander_REINFORCE/
    train_REINFORCE.py
    train_REINFORCE_normalization.py
    train_delayed_reward.py

  LunarLander_QBias_Experiment1/
    experiment_1A_overestimation_bias.py
    experiment_1B_dueling_value_advantage.py

results/
  train_sac_fetchreach/
    training_curves.png           # reward over episodes
    sac_actor_final.pth           # trained actor weights
    sac_critic_final.pth          # trained critic weights
    fetchreach_sac.gif            # rollout animation
```

## Quick start

```bash
conda create -n robustgymnasium python=3.11
conda activate robustgymnasium

git clone https://github.com/fangevo/Robust-Gymnasium.git
cd Robust-Gymnasium

pip install -r requirements.txt
pip install -e .
```

## Run experiments

### SAC on FetchReach-v3 (continuous control)

```bash
python sac.py
```

This trains a Soft Actor-Critic agent on `FetchReach-v3` (dense reward) for 1000 episodes and produces:
- `results/train_sac_fetchreach/training_curves.png` — learning curve
- `results/train_sac_fetchreach/sac_actor_final.pth` — saved actor model
- `results/train_sac_fetchreach/sac_critic_final.pth` — saved critic model
- `results/train_sac_fetchreach/fetchreach_sac.gif` — animated rollout of the learned policy

### LunarLander-v3 experiments

```bash
# DQN baseline
python examples/LunarLander_DQN/train_dqn.py

# Double DQN
python "examples/LunarLander_double_DQN&DDQN/train_double_dqn.py"

# Dueling Double DQN (+ PER)
python "examples/LunarLander_double_DQN&DDQN/train_dueling_double_dqn.py"

# PPO baseline
python examples/LunarLander_PPO/train_PPO.py

# A2C (default args)
python examples/LunarLander_A2C/main.py

# Strict REINFORCE
python examples/LunarLander_REINFORCE/train_REINFORCE.py

# Perturbation comparison across DQN variants
python "examples/LunarLander_double_DQN&DDQN/compare_algorithms_perturbation.py"

# DQN delayed-reward experiments
python examples/LunarLander_DQN/train_delayed_reward.py
python examples/LunarLander_DQN/visualize_delayed_reward.py

# Q-bias experiments
python examples/LunarLander_QBias_Experiment1/experiment_1A_overestimation_bias.py
python examples/LunarLander_QBias_Experiment1/experiment_1B_dueling_value_advantage.py

# 6 Algorithm Robustness Tests (High Wind, Low Gravity)
python examples/LunarLander_RobustnessTest/robustness_test.py
python examples/LunarLander_RobustnessTest/visualize_robustness.py
```

## SAC implementation details

The SAC agent (`sac.py`) implements the full Soft Actor-Critic algorithm:

| Component | Description |
|---|---|
| **Actor** | Squashed-Gaussian policy (tanh-bounded actions) |
| **Critic** | Twin Q-networks (clipped double-Q) |
| **Entropy** | Automatic temperature tuning (learned α) |
| **Target** | Soft-updated target critic (τ = 0.005) |
| **Buffer** | Uniform replay buffer (1M capacity) |

Key hyperparameters: `γ = 0.98`, `lr = 3e-4`, `batch = 256`, `hidden = 256`.

## Notes on Robust Gymnasium interface

The scripts use Robust Gymnasium's dict-based step input:

```python
robust_input = {
    "action": action,
    "robust_type": "action",
    "robust_config": args,
}
next_state, reward, terminated, truncated, info = env.step(robust_input)
```

Set perturbations with fields in `robust_config` (for example `noise_factor`, `noise_type`, `noise_sigma`).

## Typical outputs

Depending on script, outputs include:
- training curves (`.png`),
- saved models (`.pth` / `.pt`),
- evaluation logs (`.csv` / `.json`),
- rollout visualizations (`.gif` / `.mp4`),
- experiment summaries (`.txt`).

## Citation

If this project is useful in your research, please cite:

```bibtex
@article{robustrl2024,
  title={Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning},
  author={Gu, Shangding and Shi, Laixi and Wen, Muning and Jin, Ming and Mazumdar, Eric and Chi, Yuejie and Wierman, Adam and Spanos, Costas},
  journal={ICLR},
  year={2025}
}
```
