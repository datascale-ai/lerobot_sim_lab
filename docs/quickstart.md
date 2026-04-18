# Quickstart

1. Create and activate a Python 3.10+ environment.
2. Install the package with `pip install -e ".[dev]"`.
3. Install optional extras as needed, for example `pip install -e ".[motion-planning,lerobot]"`.
4. If you use LeRobot training/eval against patched upstream packages, install pinned deps and run `bash patches/apply_patches.sh` from the repo root, then verify:
   - Patched packages: `python -c "import gym_hil, lerobot; print('patches OK')"`
   - This repo: `python -c "import lerobot_sim_lab; lerobot_sim_lab.envs.register_envs(); print('lerobot_sim_lab OK')"`
5. Run `pytest` to verify the local environment (needs a working MuJoCo / GL setup for env tests).
6. Use package entry points such as `lerobot-sim-view` or `lerobot-sim-record`.

## Example shell scripts under `scripts/`

Each script resolves the **repository root** from its own path and `cd`s there, so defaults are relative to the clone (e.g. `outputs/…`, `data_grab_pen/`, `third_party/smolvla_base`). Override with the environment variables documented at the top of each script when your layout differs.
