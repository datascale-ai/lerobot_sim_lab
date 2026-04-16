# Docker

The Docker assets live in this directory.

- `Dockerfile`: main image build
- `docker-compose.yml`: compose entrypoint
- `.devcontainer/`: devcontainer configuration

The refactor targets `/app` as the project mount point instead of mounting only `workspace/`.
