# cellpose-astra

Cellpose v4.2.1.1 with an isolated ASTRA deterministic-training interface.
The upstream runtime includes `cpsam_v2`, `cpdino`, `cpdino-vitb`, and the
original `cpsam` model.

## Upstream base

- Cellpose v4.2.1.1
- https://github.com/MouseLand/cellpose

## Candidate installation

```bash
mamba create -n cellpose-astra python=3.11 -y
mamba activate cellpose-astra
python -m pip install \
  "git+https://github.com/jdsuh28/cellpose-astra.git@upgrade/cpsam-v2"
python -m pip install \
  "git+https://github.com/facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751"
```

Verify with:

```bash
cellpose --version
python -m cellpose.astra --version
```

## ASTRA boundary

Upstream inference and training modules remain upstream-owned. ASTRA adds a
separate `cellpose.astra` entry point for deterministic checkpoint naming and
an optional `--model_save_root` training destination. Boundary tests reject
ASTRA hooks in the upstream runtime files.

Stable releases use immutable tags after runtime validation.

## License

This derivative retains the original Cellpose license. See `LICENSE`.
