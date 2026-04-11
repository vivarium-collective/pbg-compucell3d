"""Pre-built composite document factories for CompuCell3D simulations."""


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
