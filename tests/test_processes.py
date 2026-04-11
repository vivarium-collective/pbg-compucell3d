"""Unit tests for CompuCell3DProcess."""

import warnings
import pytest

warnings.filterwarnings('ignore')

from process_bigraph import allocate_core
from pbg_compucell3d.processes import CompuCell3DProcess


@pytest.fixture
def core():
    c = allocate_core()
    c.register_process('CompuCell3DProcess', CompuCell3DProcess)
    return c


def test_instantiation(core):
    proc = CompuCell3DProcess(
        config={'dim_x': 50, 'dim_y': 50, 'blob_radius': 10},
        core=core)
    assert proc.config['dim_x'] == 50
    assert proc.config['fluctuation_amplitude'] == 10.0


def test_initial_state(core):
    proc = CompuCell3DProcess(
        config={'dim_x': 50, 'dim_y': 50, 'blob_radius': 10, 'cell_width': 4},
        core=core)
    state = proc.initial_state()
    assert 'cell_type_field' in state
    assert 'n_cells' in state
    assert 'avg_volume' in state
    assert 'mcs' in state
    assert state['n_cells'] > 0
    assert state['mcs'] == 1
    assert len(state['cell_type_field']) == 50
    assert len(state['cell_type_field'][0]) == 50


def test_single_update(core):
    proc = CompuCell3DProcess(
        config={'dim_x': 50, 'dim_y': 50, 'blob_radius': 10, 'cell_width': 4},
        core=core)
    proc.initial_state()
    result = proc.update({}, interval=50)
    assert result['mcs'] == 51
    assert result['n_cells'] > 0
    assert isinstance(result['avg_volume'], float)
    assert isinstance(result['avg_surface'], float)


def test_outputs_schema(core):
    proc = CompuCell3DProcess(
        config={'dim_x': 50, 'dim_y': 50},
        core=core)
    outputs = proc.outputs()
    expected = [
        'cell_type_field', 'n_cells', 'type_1_count',
        'type_2_count', 'avg_volume', 'avg_surface', 'mcs',
    ]
    for port in expected:
        assert port in outputs, f'Missing output port: {port}'


def test_config_defaults(core):
    proc = CompuCell3DProcess(config={}, core=core)
    assert proc.config['dim_x'] == 100
    assert proc.config['dim_y'] == 100
    assert proc.config['fluctuation_amplitude'] == 10.0
    assert proc.config['target_volume'] == 25
    assert proc.config['chemotaxis_lambda'] == 0.0
    assert proc.config['division_volume'] == 0


def test_cell_sorting_config(core):
    """Differential adhesion produces two cell types."""
    proc = CompuCell3DProcess(
        config={
            'dim_x': 60, 'dim_y': 60, 'blob_radius': 12, 'cell_width': 4,
            'contact_t1_t1': 2.0, 'contact_t1_t2': 11.0, 'contact_t2_t2': 16.0,
        },
        core=core)
    state0 = proc.initial_state()
    assert state0['type_1_count'] > 0
    assert state0['type_2_count'] > 0
    assert state0['type_1_count'] + state0['type_2_count'] == state0['n_cells']


def test_chemotaxis_config(core):
    """Chemotaxis produces a concentration field."""
    proc = CompuCell3DProcess(
        config={
            'dim_x': 50, 'dim_y': 50, 'blob_radius': 10, 'cell_width': 4,
            'chemotaxis_lambda': 300.0,
            'secretion_rate': 0.1,
        },
        core=core)
    proc.initial_state()
    proc.update({}, interval=50)
    snap = proc.get_snapshot()
    assert 'conc_field' in snap
    assert len(snap['conc_field']) == 50


def test_get_snapshot(core):
    proc = CompuCell3DProcess(
        config={'dim_x': 50, 'dim_y': 50, 'blob_radius': 10, 'cell_width': 4},
        core=core)
    proc.initial_state()
    snap = proc.get_snapshot()
    assert 'cells' in snap
    assert 'type_field' in snap
    assert 'id_field' in snap
    assert len(snap['cells']) > 0
    cell = snap['cells'][0]
    assert 'id' in cell
    assert 'type' in cell
    assert 'volume' in cell
    assert 'x_com' in cell
