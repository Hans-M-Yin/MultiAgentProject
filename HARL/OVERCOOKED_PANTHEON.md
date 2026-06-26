# Overcooked on HARL

This workspace now includes a HARL environment adapter that reuses PantheonRL's
existing Overcooked implementation without changing the original
`PantheonRL -> Overcooked` training path.

## Requirements

- Install PantheonRL's Overcooked dependency:
  `pip install -e overcookedgym/human_aware_rl/overcooked_ai`
- Install HARL's own package requirements from `HARL/README.md`

## Example

From the `HARL/` directory:

```bash
python examples/train.py --algo haa2c --env overcooked_pantheon --exp_name simple_haa2c --layout_name simple --episode_length 400
```

To stay closest to the upstream HARL repository, prefer using the stock
algorithm defaults from `harl/configs/algos_cfgs/*.yaml` instead of the custom
alignment configs we experimented with.

Recommended commands:

```bash
python examples/train.py --algo haa2c --env overcooked_pantheon --exp_name stock_haa2c --layout_name simple
python examples/train.py --algo had3qn --env overcooked_pantheon --exp_name stock_had3qn --layout_name simple
```

If you only want to align the total training steps with `hyper_search.py`
while keeping the rest of the HARL defaults, use:

```bash
python examples/train.py --load_config /Users/bytedance/PycharmProjects/PantheonRL/HARL/tuned_configs/overcooked_pantheon/simple/haa2c/stock_200k_config.json --exp_name stock_haa2c_200k
python examples/train.py --load_config /Users/bytedance/PycharmProjects/PantheonRL/HARL/tuned_configs/overcooked_pantheon/simple/had3qn/stock_200k_config.json --exp_name stock_had3qn_200k
```

These two configs keep the upstream HARL defaults and only change:

- `num_env_steps -> 200000`
- `layout_name -> simple`
- `seed -> 10`

Notes:

- `layout_name` is passed through to PantheonRL's `OvercookedMultiEnv`.
- The centralized critic input is the concatenation of both agents' local
  observations.
- Existing PantheonRL commands such as
  `python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}'`
  are unchanged.
