# cellpose-astra

Cellpose v4.0.8 (frozen) with ASTRA-specific deterministic training modifications.

This repository provides a version-locked Cellpose backend for:

ASTRA — Automated Structural Tissue Research & Analysis

## Upstream base

- Cellpose v4.0.8
- Original repository: https://github.com/MouseLand/cellpose

## Installation (pinned release)

```bash
pip install "git+https://github.com/jdsuh28/cellpose-astra.git@v4.0.8+astra.1"
```

## Scope

- No upstream tracking
- No general-purpose feature development
- Modifications limited strictly to ASTRA training integration
- Tags are immutable
- `astra.N` increments only when training behavior changes

## License

This repository is a derivative work of Cellpose and retains the original
Howard Hughes Medical Institute (HHMI) license.

See LICENSE for full terms.
