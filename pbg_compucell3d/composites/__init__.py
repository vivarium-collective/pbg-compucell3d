"""CompuCell3D composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_cc3d_document(...)` builds a PBG
   state-dict programmatically for callers that want full control over
   CC3D parameters and emitter wiring. Preserved for backwards
   compatibility with existing demo/test code.

2. **Declarative `*.composite.yaml`** — sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The
   dashboard's composite explorer discovers these automatically once
   the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_compucell3d.processes import CompuCell3DProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factories (legacy / programmatic API)
# ---------------------------------------------------------------------------

def register_compucell3d(core=None):
    """Return a core with CompuCell3DProcess, the RAM emitter, and the
    cell-sorting Visualization registered."""
    if core is None:
        core = allocate_core()
    core.register_link('CompuCell3DProcess', CompuCell3DProcess)
    core.register_link('ram-emitter', RAMEmitter)
    core.register_link('RAMEmitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    try:
        from pbg_compucell3d.visualizations import CellSortingPlots
        core.register_link('CellSortingPlots', CellSortingPlots)
    except Exception:
        # pbg-superpowers may be unavailable in stripped-down envs; the
        # base composite still works without the viz step.
        pass
    return core


def make_cc3d_document(
    dim_x=100,
    dim_y=100,
    fluctuation_amplitude=10.0,
    target_volume=25,
    lambda_volume=2.0,
    contact_medium_t1=16.0,
    contact_medium_t2=16.0,
    contact_t1_t1=2.0,
    contact_t1_t2=11.0,
    contact_t2_t2=16.0,
    mcs_per_step=100,
    interval=100.0,
):
    """Create a composite document for a CompuCell3D cell-sorting simulation.

    Returns a document dict ready for use with ``Composite()``.

    Args:
        dim_x / dim_y: Lattice dimensions in pixels.
        fluctuation_amplitude: Boltzmann temperature.
        target_volume: Equilibrium cell volume.
        lambda_volume: Volume-constraint strength.
        contact_*: Contact energies between Medium, type-1, and type-2.
        mcs_per_step: Monte Carlo Steps per process update.
        interval: PBG interval (interpreted as MCS count).

    Returns:
        dict: Composite document with CC3D process, stores, and emitter.
    """
    return {
        'cc3d': {
            '_type': 'process',
            'address': 'local:CompuCell3DProcess',
            'config': {
                'dim_x': dim_x,
                'dim_y': dim_y,
                'fluctuation_amplitude': fluctuation_amplitude,
                'target_volume': target_volume,
                'lambda_volume': lambda_volume,
                'contact_medium_t1': contact_medium_t1,
                'contact_medium_t2': contact_medium_t2,
                'contact_t1_t1': contact_t1_t1,
                'contact_t1_t2': contact_t1_t2,
                'contact_t2_t2': contact_t2_t2,
                'mcs_per_step': mcs_per_step,
            },
            'interval': interval,
            'inputs': {},
            'outputs': {
                'cell_type_field': ['stores', 'cell_type_field'],
                'n_cells': ['stores', 'n_cells'],
                'type_1_count': ['stores', 'type_1_count'],
                'type_2_count': ['stores', 'type_2_count'],
                'avg_volume': ['stores', 'avg_volume'],
                'avg_surface': ['stores', 'avg_surface'],
                'mcs': ['stores', 'mcs'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'n_cells': 'integer',
                    'type_1_count': 'integer',
                    'type_2_count': 'integer',
                    'avg_volume': 'float',
                    'avg_surface': 'float',
                    'mcs': 'integer',
                    'time': 'float',
                },
            },
            'inputs': {
                'n_cells': ['stores', 'n_cells'],
                'type_1_count': ['stores', 'type_1_count'],
                'type_2_count': ['stores', 'type_2_count'],
                'avg_volume': ['stores', 'avg_volume'],
                'avg_surface': ['stores', 'avg_surface'],
                'mcs': ['stores', 'mcs'],
                'time': ['global_time'],
            },
        },
    }


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_compucell3d() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_compucell3d()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
