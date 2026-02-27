# cellpose-astra

Cellpose v4.0.8 (frozen) with ASTRA-specific deterministic training
modifications.

This repository provides a version-locked Cellpose backend for:

ASTRA --- Automated Structural Tissue Research & Analysis

------------------------------------------------------------------------

## Upstream base

-   Cellpose v4.0.8\
-   Original repository: https://github.com/MouseLand/cellpose

------------------------------------------------------------------------

## Installation (deterministic environment)

### 1. Create environment

Using mamba (recommended):

``` bash
mamba create -n cellpose-astra python=3.10 -y
mamba activate cellpose-astra
```

Or using conda:

``` bash
conda create -n cellpose-astra python=3.10 -y
conda activate cellpose-astra
```

------------------------------------------------------------------------

### 2. Install pinned release

``` bash
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/jdsuh28/cellpose-astra.git@v4.0.8+astra.1"
```

------------------------------------------------------------------------

### 3. Verify installation

``` bash
cellpose --version
```

Expected output:

    cellpose version: 4.0.8+astra.1

This installs the exact deterministic backend used by ASTRA.

------------------------------------------------------------------------

## Scope

-   No upstream tracking\
-   No general-purpose feature development\
-   Modifications limited strictly to ASTRA training integration\
-   Tags are immutable\
-   `astra.N` increments only when training behavior changes

This repository exists solely to guarantee deterministic,
validation-restricted training semantics for ASTRA workflows.

------------------------------------------------------------------------

## License

This repository is a derivative work of Cellpose and retains the
original Howard Hughes Medical Institute (HHMI) license.

See LICENSE for full terms.
