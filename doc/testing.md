# Testing

This page is for contributors and maintainers. It documents PTAF test commands,
marker policy, and coverage checks; user-facing module workflows live in the
User Guide.

## Default Test Run

```bash
./run_tests.sh -- -q
```

`run_tests.sh` defaults to `autoforge`. Override when needed:

```bash
./run_tests.sh -e other_env -- -q
```

## Marker Policy

Default collection skips tests marked `slow`, `gpu`, or `visual`.

Use explicit flags when needed:

```bash
./run_tests.sh -- --run-slow -q
./run_tests.sh -- --run-gpu -q
./run_tests.sh -- --run-visual -q
```

Markers:

- `unit`: fast local unit test.
- `integration`: spans multiple subsystems or external-style behavior.
- `slow`: intentionally excluded from default runs.
- `gpu`: requires usable CUDA kernels, not just CUDA discovery.
- `visual`: plotting or visual-inspection-oriented test.
- `export`: model export or serialization path.

## Coverage

```bash
./run_tests.sh -- --cov=pyTorchAutoForge --cov-report=term-missing:skip-covered
```
