# ARGUS

**Adaptive Relevance-Guided Update Scheduling**

ARGUS is a scheduling and orchestration framework for large-scale
LLM-driven agent simulation.

ARGUS investigates whether per-agent adaptive reasoning frequency can
reduce expensive LLM invocations while maintaining comparable behavioral
fidelity.

## Project Status

Currently under development.

### Development stages

1. Baseline simulation core
2. Relevance scoring
3. Adaptive scheduling + dead-reckoning
4. Divergence monitoring
5. Dashboard + Unity integration
6. Baseline comparison and evaluation

## Development

### Requirements

- Python 3.12+
- Git

### Setup

```bash
git clone <repository-url>
cd argus

python -m venv .venv

# Activate the virtual environment and install the development dependencies:

pip install -e ".[dev]"

# Run tests:

pytest

# Run linting:

ruff check .
```

### Architecture

See docs/ for project design documentation.

### License

See LICENSE.
