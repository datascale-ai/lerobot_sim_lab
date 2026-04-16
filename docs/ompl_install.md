# OMPL Install Notes

`mplib` and related OMPL tooling are optional for this project.

## Recommended

Use a dedicated Python environment and install the motion-planning extra:

```bash
pip install -e ".[motion-planning]"
```

## If You Need Custom OMPL Builds

- Build OMPL in a separate workspace outside this repository.
- Keep the resulting Python bindings on your `PYTHONPATH` or install them into the active environment.
- Do not vendor the full OMPL source tree into this repository.

## Container Usage

Inside the project container:

```bash
cd /app
pip install -e ".[motion-planning]"
```
