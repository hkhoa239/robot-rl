### Head-to-head metrics

| Metric | SAC | PPO | Winner | Margin |
|---|---|---|---|---|
| Final-50 train reward | -19.19 | -23.71 | 🥇 SAC | Δ = 4.52 |
| Best avg-100 (training) | -21.13 | -23.54 | 🥇 SAC | Δ = 2.41 |
| Held-out eval mean | -14.90 ± 5.20 | -20.89 ± 4.07 | 🥇 SAC | Δ = 5.99 |
| Wall-time (s, lower is better) | 53.5 | 9.2 | 🥇 PPO | Δ = 44.3s |
| Episodes to reach -24.5 avg | 135 | 163 | 🥇 SAC | Δ = 28 ep |

### Overall verdict

**SAC wins on 4/5 metrics** vs PPO's 1. On `FetchReach-v3` (50-step horizon, dense reward) the off-policy replay buffer pays off in sample efficiency and final reward.

### Eval-reward breakdown
- SAC: mean **-14.90** ± 5.20, min `-23.31`, max `-7.31` (over 20 held-out episodes)
- PPO: mean **-20.89** ± 4.07, min `-28.91`, max `-15.42` (over 20 held-out episodes)

### Reproducibility
- Environment: `FetchReach-v3` (dense reward, no perturbation)
- Seed: `42`  |  Train episodes: `200`  |  Eval episodes: `20`
- Artifacts: `results/benchmark_sac_vs_ppo/`
