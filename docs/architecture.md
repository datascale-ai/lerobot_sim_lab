# Architecture

The repository is organized around a single installable package, `lerobot_sim_lab`.

- `config`: shared constants and scenario definitions
- `envs`: MuJoCo gym environments, wrappers, and controllers
- `data`: recording and dataset visualization
- `control`: keyboard, interactive, and remote control tools
- `trajectory`: planning, playback, and analysis
- `sim`: direct simulation and scene inspection helpers
- `training`: policy training entry points
- `evaluation`: evaluation flows and sim/real comparison
- `tuning`: task-specific tuning utilities
- `utils`: path, rendering, and formatting helpers
