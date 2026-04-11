"""CompuCell3D Process wrapper for process-bigraph.

Wraps the CompuCell3D cellular Potts model (CPM / Glazier-Graner-Hogeweg)
simulator as a time-driven Process using the bridge pattern.  The internal
CC3DSimService and data-collector steppable are lazily initialized on the
first update() call.
"""

import warnings
from process_bigraph import Process


class CompuCell3DProcess(Process):
    """Bridge Process wrapping CompuCell3D lattice-based cell simulation.

    Simulates multicellular dynamics on a 2-D pixel lattice using the
    Glazier-Graner-Hogeweg (GGH) / cellular Potts model.  Each update()
    call advances the simulation by ``mcs_per_step`` Monte Carlo Steps and
    returns updated cell-population statistics and the cell-type lattice.

    The process supports three physics modules that can be combined:
      - **Differential adhesion** (always on): contact energies between
        cell types drive sorting and morphogenesis.
      - **Chemotaxis** (enabled when ``chemotaxis_lambda > 0``): cells of
        type-2 migrate up a diffusible field secreted by type-1 cells.
      - **Mitosis** (enabled when ``division_volume > 0``): cells grow
        toward ``division_volume`` and divide when they reach it.

    Config:
        dim_x / dim_y: Lattice dimensions in pixels.
        fluctuation_amplitude: Boltzmann temperature (higher = more noise).
        neighbor_order: Potts neighbor range (default 2).
        mcs_per_step: Monte Carlo Steps per update() call.
        blob_radius: Radius of the initial cell blob.
        cell_width: Approximate diameter of seeded cells.
        target_volume: Equilibrium cell volume (pixels).
        lambda_volume: Strength of the volume constraint.
        contact_*: Contact energies between Medium, type-1, and type-2.
        chemotaxis_lambda: Chemotactic strength (0 = disabled).
        diffusion_constant: Diffusion coefficient for the chemical field.
        decay_constant: Decay rate of the chemical field.
        secretion_rate: Secretion rate of type-1 cells.
        division_volume: Volume at which a cell divides (0 = no mitosis).
        growth_rate_per_mcs: Volume increase per MCS during growth.
    """

    config_schema = {
        # Lattice
        'dim_x': {'_type': 'integer', '_default': 100},
        'dim_y': {'_type': 'integer', '_default': 100},
        'fluctuation_amplitude': {'_type': 'float', '_default': 10.0},
        'neighbor_order': {'_type': 'integer', '_default': 2},
        'mcs_per_step': {'_type': 'integer', '_default': 100},
        # Cell initialisation
        'blob_radius': {'_type': 'integer', '_default': 20},
        'cell_width': {'_type': 'integer', '_default': 5},
        # Volume constraint
        'target_volume': {'_type': 'integer', '_default': 25},
        'lambda_volume': {'_type': 'float', '_default': 2.0},
        # Contact energies  (type-1 = first CellType, type-2 = second)
        'contact_medium_t1': {'_type': 'float', '_default': 16.0},
        'contact_medium_t2': {'_type': 'float', '_default': 16.0},
        'contact_t1_t1': {'_type': 'float', '_default': 2.0},
        'contact_t1_t2': {'_type': 'float', '_default': 11.0},
        'contact_t2_t2': {'_type': 'float', '_default': 16.0},
        # Chemotaxis  (0 → disabled)
        'chemotaxis_lambda': {'_type': 'float', '_default': 0.0},
        'diffusion_constant': {'_type': 'float', '_default': 0.1},
        'decay_constant': {'_type': 'float', '_default': 0.0005},
        'secretion_rate': {'_type': 'float', '_default': 0.1},
        # Mitosis  (0 → disabled)
        'division_volume': {'_type': 'integer', '_default': 0},
        'growth_rate_per_mcs': {'_type': 'float', '_default': 0.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._sim = None
        self._mcs = 0

    # ── PBG interface ───────────────────────────────────────────────

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'cell_type_field': 'list',
            'n_cells': 'integer',
            'type_1_count': 'integer',
            'type_2_count': 'integer',
            'avg_volume': 'float',
            'avg_surface': 'float',
            'mcs': 'integer',
        }

    def initial_state(self):
        self._build_system()
        return self._read_state()

    def update(self, state, interval):
        self._build_system()
        steps = int(interval)
        if steps < 1:
            steps = self.config['mcs_per_step']
        for _ in range(steps):
            self._sim.step()
            self._mcs += 1
        return self._read_state()

    # ── Extended data for demo / analysis ───────────────────────────

    def get_snapshot(self):
        """Return a rich snapshot dict including per-cell data.

        Call after initial_state() or update() to get detailed state
        including individual cell properties and optional concentration
        field.  Intended for demo/visualization — not part of the
        formal PBG port schema.
        """
        self._build_system()
        out = self._sim.sim_output or {}
        return out

    # ── Internal ────────────────────────────────────────────────────

    def _read_state(self):
        """Read summary statistics from the CC3D simulation."""
        out = self._sim.sim_output or {}
        cells = out.get('cells', [])
        n = len(cells)
        t1 = sum(1 for c in cells if c['type'] == 1)
        t2 = sum(1 for c in cells if c['type'] == 2)
        avg_v = sum(c['volume'] for c in cells) / n if n else 0.0
        avg_s = sum(c['surface'] for c in cells) / n if n else 0.0
        return {
            'cell_type_field': out.get('type_field', []),
            'n_cells': n,
            'type_1_count': t1,
            'type_2_count': t2,
            'avg_volume': float(avg_v),
            'avg_surface': float(avg_s),
            'mcs': self._mcs,
        }

    def _build_system(self):
        """Lazily initialise the CC3D simulation service."""
        if self._sim is not None:
            return

        warnings.filterwarnings('ignore')

        from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
        from cc3d.core.PyCoreSpecs import (
            PottsCore, CellTypePlugin, VolumePlugin, SurfacePlugin,
            ContactPlugin, BlobInitializer, ChemotaxisPlugin,
            DiffusionSolverFE,
        )
        from cc3d.core.PySteppables import SteppableBasePy, MitosisSteppableBase

        cfg = self.config
        cx, cy = cfg['dim_x'] // 2, cfg['dim_y'] // 2

        # ── Core specs ──
        potts = PottsCore(
            dim_x=cfg['dim_x'], dim_y=cfg['dim_y'], dim_z=1,
            steps=10**8,
            neighbor_order=cfg['neighbor_order'],
            fluctuation_amplitude=cfg['fluctuation_amplitude'],
        )

        cell_types = CellTypePlugin('TypeA', 'TypeB')

        volume = VolumePlugin()
        has_growth = cfg['division_volume'] > 0 and cfg['growth_rate_per_mcs'] > 0
        if not has_growth:
            # Type-level defaults (overrides per-cell attributes)
            volume.param_new('TypeA',
                             target_volume=cfg['target_volume'],
                             lambda_volume=cfg['lambda_volume'])
            volume.param_new('TypeB',
                             target_volume=cfg['target_volume'],
                             lambda_volume=cfg['lambda_volume'])

        surface = SurfacePlugin()
        surface.param_new('TypeA', target_surface=0, lambda_surface=0)
        surface.param_new('TypeB', target_surface=0, lambda_surface=0)

        contact = ContactPlugin(neighbor_order=cfg['neighbor_order'])
        contact.param_new(type_1='Medium', type_2='TypeA',
                          energy=cfg['contact_medium_t1'])
        contact.param_new(type_1='Medium', type_2='TypeB',
                          energy=cfg['contact_medium_t2'])
        contact.param_new(type_1='TypeA', type_2='TypeA',
                          energy=cfg['contact_t1_t1'])
        contact.param_new(type_1='TypeA', type_2='TypeB',
                          energy=cfg['contact_t1_t2'])
        contact.param_new(type_1='TypeB', type_2='TypeB',
                          energy=cfg['contact_t2_t2'])

        blob = BlobInitializer()
        blob.region_new(
            width=cfg['cell_width'],
            radius=cfg['blob_radius'],
            center=(cx, cy, 0),
            cell_types=('TypeA', 'TypeB'),
        )

        specs = [potts, cell_types, volume, surface, contact, blob]

        # ── Optional: chemotaxis ──
        has_chemo = cfg['chemotaxis_lambda'] > 0
        if has_chemo:
            diff_solver = DiffusionSolverFE()
            f = diff_solver.field_new('Signal')
            f.diff_data.diff_global = cfg['diffusion_constant']
            f.diff_data.decay_global = cfg['decay_constant']
            f.secretion_data_new('TypeA', cfg['secretion_rate'])

            chemo = ChemotaxisPlugin()
            cp = chemo.param_new(field_name='Signal',
                                 solver_name='DiffusionSolverFE')
            cp.params_new('TypeB', lambda_chemo=cfg['chemotaxis_lambda'])

            specs.extend([diff_solver, chemo])

        # ── Steppable: data collector + optional mitosis ──
        div_vol = cfg['division_volume']
        growth = cfg['growth_rate_per_mcs']
        chemo_flag = has_chemo

        class _Collector(SteppableBasePy):
            """Extract cell data every MCS (lightweight)."""

            def __init__(self, frequency=1):
                super().__init__(frequency=frequency)

            def step(self, mcs):
                import numpy as np
                cells = []
                for cell in self.cell_list:
                    cells.append({
                        'id': int(cell.id),
                        'type': int(cell.type),
                        'volume': int(cell.volume),
                        'surface': int(cell.surface),
                        'x_com': float(cell.xCOM),
                        'y_com': float(cell.yCOM),
                    })

                dim = self.dim
                tf = np.zeros((dim.x, dim.y), dtype=int)
                idf = np.zeros((dim.x, dim.y), dtype=int)
                for x in range(dim.x):
                    for y in range(dim.y):
                        c = self.cell_field[x, y, 0]
                        if c:
                            tf[x, y] = c.type
                            idf[x, y] = c.id

                result = {
                    'mcs': mcs,
                    'cells': cells,
                    'type_field': tf.tolist(),
                    'id_field': idf.tolist(),
                }

                # Optional concentration field
                if chemo_flag:
                    try:
                        cf = self.field.Signal
                        conc = np.zeros((dim.x, dim.y))
                        for x in range(dim.x):
                            for y in range(dim.y):
                                conc[x, y] = cf[x, y, 0]
                        result['conc_field'] = conc.tolist()
                    except Exception:
                        pass

                self.external_output = result

        init_tvol = cfg['target_volume']

        class _Grower(MitosisSteppableBase):
            """Grow cells and trigger mitosis when volume exceeds threshold."""

            def __init__(self, frequency=1):
                super().__init__(frequency=frequency)

            def start(self):
                for cell in self.cell_list:
                    cell.targetVolume = init_tvol
                    cell.lambdaVolume = cfg['lambda_volume']

            def step(self, mcs):
                cells_to_divide = []
                for cell in self.cell_list:
                    cell.targetVolume += growth
                    if div_vol > 0 and cell.volume >= div_vol:
                        cells_to_divide.append(cell)
                for cell in cells_to_divide:
                    self.divide_cell_random_orientation(cell)

            def update_attributes(self):
                self.parent_cell.targetVolume = init_tvol
                self.clone_parent_2_child()

        # ── Build and launch ──
        sim = CC3DSimService()
        sim.register_specs(specs)
        sim.register_steppable(steppable=_Collector(frequency=1))
        if div_vol > 0 and growth > 0:
            sim.register_steppable(steppable=_Grower(frequency=1))
        sim.run()
        sim.init()
        sim.start()

        # Run one MCS so that the collector fires and initial data is available
        sim.step()
        self._mcs = 1
        self._sim = sim

    def __del__(self):
        try:
            if self._sim is not None:
                self._sim.finish()
        except Exception:
            pass
