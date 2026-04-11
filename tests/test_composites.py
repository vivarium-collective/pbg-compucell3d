"""Integration tests for CompuCell3D composite assembly."""

import warnings
import pytest

warnings.filterwarnings('ignore')

from process_bigraph import allocate_core, Composite, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from pbg_compucell3d.processes import CompuCell3DProcess
from pbg_compucell3d.composites import make_cc3d_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_process('CompuCell3DProcess', CompuCell3DProcess)
    c.register_process('ram-emitter', RAMEmitter)
    return c


def test_composite_short_run(core):
    doc = make_cc3d_document(
        dim_x=50, dim_y=50, interval=50.0, mcs_per_step=50)
    sim = Composite({'state': doc}, core=core)
    sim.run(100.0)
    stores = sim.state['stores']
    assert stores['n_cells'] > 0
    assert stores['mcs'] > 0
    assert stores['avg_volume'] > 0


def test_emitter_collects_timeseries(core):
    doc = make_cc3d_document(
        dim_x=50, dim_y=50, interval=50.0, mcs_per_step=50)
    sim = Composite({'state': doc}, core=core)
    sim.run(150.0)
    raw = gather_emitter_results(sim)
    emitter_data = raw[('emitter',)]
    assert len(emitter_data) >= 2
    for entry in emitter_data:
        assert 'n_cells' in entry
        assert 'avg_volume' in entry
        assert 'time' in entry
