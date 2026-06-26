# PantheonRL

This repository is currently used for running and comparing five Overcooked baselines on the `simple` layout:

- `PPO`
- `DQN`
- `A2C`
- `HAPPO`
- `HAA2C`

The commands below are the direct entry points used in this project.

## Environment

Recommended local environment:

```bash
conda create -n pantheonrl39 python=3.9
conda activate pantheonrl39
pip install setuptools==65.5.0 "wheel<0.40.0"
pip install -e .
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai
```

For PantheonRL training, the main entry point is:

```bash
python trainer.py ...
```

For HARL training, the main entry point is:

```bash
cd HARL
python examples/train.py ...
```

## PantheonRL Baselines

All commands below train on:

- environment: `OvercookedMultiEnv-v0`
- layout: `simple`

### PPO

Default PantheonRL PPO self-play baseline:

```bash
python trainer.py OvercookedMultiEnv-v0 PPO PPO \
  --env-config '{"layout_name":"simple"}' \
  --seed 10 \
  --preset 1
```

### DQN

Best DQN configuration found during local search:

```bash
python trainer.py OvercookedMultiEnv-v0 DQN DQN \
  --env-config '{"layout_name":"simple"}' \
  --seed 10 \
  --ego-config '{"learning_rate":5e-5,"buffer_size":20000,"batch_size":64,"learning_starts":5000,"train_freq":1,"target_update_interval":500,"exploration_fraction":0.8,"exploration_final_eps":0.1}' \
  --alt-config '{"learning_rate":5e-5,"buffer_size":20000,"batch_size":64,"learning_starts":5000,"train_freq":1,"target_update_interval":500,"exploration_fraction":0.8,"exploration_final_eps":0.1}'
```

### A2C

Best A2C configuration found during local search:

```bash
python trainer.py OvercookedMultiEnv-v0 A2C A2C \
  --env-config '{"layout_name":"simple"}' \
  --seed 10 \
  --ego-config '{"learning_rate":4e-4,"n_steps":10,"gamma":0.99,"gae_lambda":0.98,"ent_coef":0.01,"vf_coef":0.4}' \
  --alt-config '{"learning_rate":4e-4,"n_steps":10,"gamma":0.99,"gae_lambda":0.98,"ent_coef":0.01,"vf_coef":0.4}'
```

## HARL Baselines

All HARL commands below use:

- environment: `overcooked_pantheon`
- layout: `simple`

Run them from the `HARL/` directory.

### HAPPO

Timestep-aligned comparison configuration:

```bash
cd HARL
python examples/train.py \
  --load_config /Users/bytedance/PycharmProjects/PantheonRL/HARL/tuned_configs/overcooked_pantheon/simple/happo/compare_timestep_config.json \
  --exp_name compare_happo_200k
```

### HAA2C

Timestep-aligned comparison configuration:

```bash
cd HARL
python examples/train.py \
  --load_config /Users/bytedance/PycharmProjects/PantheonRL/HARL/tuned_configs/overcooked_pantheon/simple/haa2c/compare_timestep_config.json \
  --exp_name compare_haa2c_200k
```

## Visualization

### PPO / A2C / DQN search curves

Merge multiple `hyper_search` runs for one algorithm and one seed:

```bash
conda run -n pantheonrl39 python plot_search_curves.py \
  search_runs/20260625-113906 \
  search_runs/20260625-115555 \
  --algo A2C \
  --seed 10 \
  --top-k 6 \
  --show
```

### PPO vs HAPPO vs HAA2C

Plot the three baseline curves together:

```bash
conda run -n pantheonrl39 python plot_baseline_compare.py \
  --ppo-logdir logs/OvercookedMultiEnv-v0-simple-PPOPPO-10_1 \
  --happo-source HARL/results/overcooked_pantheon/simple/happo/compare_happo_200k/seed-00010-2026-06-26-14-28-22 \
  --haa2c-source HARL/results/overcooked_pantheon/simple/haa2c/compare_haa2c_5m/seed-00010-2026-06-26-14-26-15 \
  --x-axis timesteps \
  --xlim 0 200000 \
  --show
```

If you want to inspect trained PPO/A2C/DQN behavior in the Overcooked Flask viewer:

```bash
python overcookedgym/overcooked-flask/app.py \
  --modelpath_p0 models/OvercookedMultiEnv-v0-simple-PPO-ego-10.zip \
  --modelpath_p1 models/OvercookedMultiEnv-v0-simple-PPO-alt-10.zip \
  --algo_p0 PPO \
  --algo_p1 PPO \
  --layout_name simple \
  --port 5001
```

The visualizer also supports:

- `--algo_p0 A2C`
- `--algo_p1 A2C`
- `--algo_p0 DQN`
- `--algo_p1 DQN`

## Notes

- PantheonRL and HARL do not log exactly the same metrics by default.
- HARL `progress.txt` stores evaluation rewards, while training curves are written to TensorBoard event files.
- For fair comparisons, align `num_env_steps` first, then compare curves on the same x-axis.
