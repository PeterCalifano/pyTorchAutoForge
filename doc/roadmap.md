# Development Roadmap

## Active Infrastructure Work

- GitHub Pages documentation with Sphinx and the PyData theme.
- Test markers for slow, GPU, visual, integration, and export paths.
- Coverage reporting through `pytest-cov` and existing coverage config.
- Better split between runnable examples and pytest modules.

## Near-Term Test Cleanup

- Replace external dataset tests with temp-fixture datasets where possible.
- Keep expensive external-data tests under `slow` and `integration` markers.
- Convert TCP server smoke checks to bounded ephemeral-port tests.
- Add seeded edge-case checks for augmentation geometry and dataset selection.
