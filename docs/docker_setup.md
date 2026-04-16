# Docker Setup

`docker/` now contains the maintained container configuration for this project.

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t lerobot-sim-lab .
```

To use mainland China mirrors during image build:

```bash
docker build --build-arg USE_CN_MIRRORS=true -f docker/Dockerfile -t lerobot-sim-lab .
```

## Run With Compose

```bash
docker compose -f docker/docker-compose.yml up
```

The compose file mounts the whole repository to `/app` inside the container.

## Inside The Container

```bash
cd /app
pip install -e ".[dev]"
```

If you need optional robotics dependencies:

```bash
pip install -e ".[lerobot,motion-planning,remote]"
```

## Notes

- The legacy `/workspace` mount pattern is no longer used.
- Vendored `workspace/lerobot` and `workspace/gym-hil` copies were removed during the refactor.
- Prefer package entry points or `python -m lerobot_sim_lab...` commands from `/app`.
