# pbg-compucell3d

Process-bigraph wrapper for [CompuCell3D](https://compucell3d.org/) — a
cellular Potts model (CPM / Glazier-Graner-Hogeweg) simulator for
multicellular systems.  This package wraps CC3D's Python API as a single
time-driven `Process`, exposing cell-population statistics and the 2-D
lattice state through typed bigraph ports.

## Installation

CompuCell3D requires conda (not pip).  Use [pixi](https://pixi.sh) or
micromamba to create an environment:

```bash
pixi init --channel compucell3d --channel conda-forge
pixi add cc3d python=3.10 pip graphviz
pixi run pip install process-bigraph bigraph-schema bigraph-viz pytest
pixi run pip install -e .
```

## Quick Start

```python
from process_bigraph import allocate_core
from pbg_compucell3d import CompuCell3DProcess

core = allocate_core()
core.register_process('CompuCell3DProcess', CompuCell3DProcess)

proc = CompuCell3DProcess(config={
    'dim_x': 80, 'dim_y': 80,
    'contact_t1_t1': 2.0,   # low homo-adhesion  → engulfs
    'contact_t1_t2': 11.0,  # hetero-adhesion
    'contact_t2_t2': 16.0,  # high homo-adhesion → engulfed
}, core=core)

state = proc.initial_state()
result = proc.update({}, interval=1000)  # advance 1000 MCS

print(f"Cells: {result['n_cells']}")
print(f"TypeA: {result['type_1_count']}, TypeB: {result['type_2_count']}")
```

## API Reference

### CompuCell3DProcess

| Config parameter       | Type    | Default | Description                                    |
|------------------------|---------|---------|------------------------------------------------|
| `dim_x` / `dim_y`     | integer | 100     | Lattice dimensions (pixels)                    |
| `fluctuation_amplitude`| float   | 10.0    | Boltzmann temperature                          |
| `neighbor_order`       | integer | 2       | Potts neighbor range                           |
| `mcs_per_step`         | integer | 100     | MCS per update() (used when interval < 1)      |
| `blob_radius`          | integer | 20      | Initial cell-blob radius                       |
| `cell_width`           | integer | 5       | Approximate seeded cell diameter               |
| `target_volume`        | integer | 25      | Equilibrium cell volume (pixels)               |
| `lambda_volume`        | float   | 2.0     | Volume-constraint strength                     |
| `contact_medium_t1`    | float   | 16.0    | Contact energy: Medium ↔ TypeA                 |
| `contact_medium_t2`    | float   | 16.0    | Contact energy: Medium ↔ TypeB                 |
| `contact_t1_t1`        | float   | 2.0     | Contact energy: TypeA ↔ TypeA                  |
| `contact_t1_t2`        | float   | 11.0    | Contact energy: TypeA ↔ TypeB                  |
| `contact_t2_t2`        | float   | 16.0    | Contact energy: TypeB ↔ TypeB                  |
| `chemotaxis_lambda`    | float   | 0.0     | Chemotactic strength (0 = disabled)            |
| `diffusion_constant`   | float   | 0.1     | Chemical field diffusion coefficient           |
| `decay_constant`       | float   | 0.0005  | Chemical field decay rate                      |
| `secretion_rate`       | float   | 0.1     | TypeA secretion rate                           |
| `division_volume`      | integer | 0       | Volume triggering mitosis (0 = no division)    |
| `growth_rate_per_mcs`  | float   | 0.0     | Target-volume increase per MCS                 |

### Output ports

| Port             | Type    | Description                          |
|------------------|---------|--------------------------------------|
| `cell_type_field`| list    | 2-D array of cell-type IDs (0/1/2)  |
| `n_cells`        | integer | Total live cells                     |
| `type_1_count`   | integer | TypeA cell count                     |
| `type_2_count`   | integer | TypeB cell count                     |
| `avg_volume`     | float   | Mean cell volume (pixels)            |
| `avg_surface`    | float   | Mean cell surface (pixels)           |
| `mcs`            | integer | Current Monte Carlo Step             |

### Extended snapshot

Call `proc.get_snapshot()` for richer per-cell data (id, type, volume,
surface, center-of-mass) plus the ID lattice field and optional
concentration field.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Composite                                      │
│                                                 │
│  ┌──────────────┐      ┌────────────────────┐   │
│  │ CompuCell3D   │─────→│ stores             │   │
│  │ Process       │      │  cell_type_field   │   │
│  │               │      │  n_cells           │   │
│  │ CC3DSimService│      │  avg_volume  ...   │   │
│  └──────────────┘      └────────┬───────────┘   │
│                                 │               │
│                        ┌────────▼───────────┐   │
│                        │ ram-emitter         │   │
│                        │  n_cells, avg_vol,  │   │
│                        │  time → timeseries  │   │
│                        └────────────────────┘   │
└─────────────────────────────────────────────────┘
```

The wrapper uses the **bridge pattern**: `CC3DSimService` manages the
full CPM simulation internally.  On each `update(state, interval)` call
the process advances CC3D by `interval` Monte Carlo Steps, then reads
back cell statistics via an internal data-collector steppable.

## Demo

Generate the interactive HTML report:

```bash
pixi run python demo/demo_report.py
```

This runs three configurations — cell sorting, chemotaxis, and growth —
and produces `demo/report.html` with:

- Interactive 2-D lattice viewers (play/pause, slider)
- Plotly time-series charts (cell counts, volume, surface)
- Bigraph architecture diagrams
- Collapsible PBG document trees

## Tests

```bash
pixi run python -m pytest tests/ -v
```
