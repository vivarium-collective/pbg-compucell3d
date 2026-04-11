"""Demo: CompuCell3D multi-configuration cellular Potts model report.

Runs three distinct CPM simulations (cell sorting, chemotaxis, growth &
division), generates interactive 2-D lattice viewers with HTML5 Canvas,
Plotly charts, bigraph-viz diagrams, and navigatable PBG document trees
— all in a single self-contained HTML.
"""

import json
import os
import base64
import time as _time
import tempfile
import subprocess
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from process_bigraph import allocate_core
from pbg_compucell3d.processes import CompuCell3DProcess
from pbg_compucell3d.composites import make_cc3d_document


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'sorting',
        'title': 'Differential Adhesion Cell Sorting',
        'subtitle': 'Two cell types self-organise via Steinberg sorting',
        'description': (
            'Two cell populations (TypeA and TypeB) are initialised as a '
            'mixed blob on an 80x80 pixel lattice.  Differential contact '
            'energies cause TypeA cells (low homo-adhesion, J=2) to engulf '
            'TypeB cells (higher homo-adhesion, J=16), reproducing the '
            'Steinberg differential adhesion hypothesis.  After ~2000 MCS '
            'the cells segregate into distinct clusters.'
        ),
        'config': {
            'dim_x': 80, 'dim_y': 80,
            'fluctuation_amplitude': 10.0,
            'blob_radius': 20, 'cell_width': 5,
            'target_volume': 25, 'lambda_volume': 2.0,
            'contact_medium_t1': 16.0, 'contact_medium_t2': 16.0,
            'contact_t1_t1': 2.0, 'contact_t1_t2': 11.0,
            'contact_t2_t2': 16.0,
        },
        'n_snapshots': 20,
        'total_mcs': 3000,
        'color_scheme': 'indigo',
    },
    {
        'id': 'chemotaxis',
        'title': 'Chemotactic Migration',
        'subtitle': 'Cells chase a secreted chemical signal',
        'description': (
            'TypeA cells (green) secrete a diffusible factor "Signal" '
            'that TypeB cells (orange) follow via chemotaxis '
            '(lambda_chemo = 500).  The diffusion solver runs a '
            'forward-Euler PDE on the same lattice.  Over time, TypeB '
            'cells cluster around the TypeA secretors, demonstrating '
            'gradient-driven cell migration on a 2-D CPM lattice.'
        ),
        'config': {
            'dim_x': 80, 'dim_y': 80,
            'fluctuation_amplitude': 10.0,
            'blob_radius': 20, 'cell_width': 5,
            'target_volume': 25, 'lambda_volume': 2.0,
            'contact_medium_t1': 16.0, 'contact_medium_t2': 16.0,
            'contact_t1_t1': 16.0, 'contact_t1_t2': 16.0,
            'contact_t2_t2': 16.0,
            'chemotaxis_lambda': 500.0,
            'diffusion_constant': 0.1,
            'decay_constant': 0.0005,
            'secretion_rate': 0.1,
        },
        'n_snapshots': 20,
        'total_mcs': 2000,
        'color_scheme': 'emerald',
    },
    {
        'id': 'growth',
        'title': 'Growth & Division',
        'subtitle': 'Population expansion via cell mitosis',
        'description': (
            'Cells start from a small blob and grow by incrementing their '
            'target volume each MCS (+0.1 px/MCS).  When a cell\'s actual '
            'volume exceeds 50 pixels it divides randomly, producing two '
            'daughter cells that reset to the base target volume.  The '
            'colony expands outward as the population doubles repeatedly, '
            'demonstrating tissue growth on a 150x150 lattice.'
        ),
        'config': {
            'dim_x': 150, 'dim_y': 150,
            'fluctuation_amplitude': 10.0,
            'blob_radius': 15, 'cell_width': 5,
            'target_volume': 25, 'lambda_volume': 5.0,
            'contact_medium_t1': 10.0, 'contact_medium_t2': 10.0,
            'contact_t1_t1': 5.0, 'contact_t1_t2': 8.0,
            'contact_t2_t2': 5.0,
            'division_volume': 50,
            'growth_rate_per_mcs': 0.1,
        },
        'n_snapshots': 20,
        'total_mcs': 3000,
        'color_scheme': 'rose',
    },
]


def run_simulation(cfg_entry):
    """Run a single CC3D simulation, returning snapshots and runtime."""
    core = allocate_core()
    core.register_process('CompuCell3DProcess', CompuCell3DProcess)

    t0 = _time.perf_counter()
    proc = CompuCell3DProcess(config=cfg_entry['config'], core=core)
    state0 = proc.initial_state()
    snap0 = proc.get_snapshot()

    mcs_per_snap = cfg_entry['total_mcs'] // cfg_entry['n_snapshots']
    snapshots = [_snap(0, snap0, cfg_entry['config'])]

    for i in range(cfg_entry['n_snapshots']):
        proc.update({}, interval=mcs_per_snap)
        snap = proc.get_snapshot()
        mcs = (i + 1) * mcs_per_snap
        snapshots.append(_snap(mcs, snap, cfg_entry['config']))

    runtime = _time.perf_counter() - t0
    return snapshots, runtime


def _snap(mcs, s, cfg):
    """Extract relevant fields into a snapshot dict."""
    cells = s.get('cells', [])
    n = len(cells)
    t1 = sum(1 for c in cells if c['type'] == 1)
    t2 = sum(1 for c in cells if c['type'] == 2)
    avg_v = sum(c['volume'] for c in cells) / n if n else 0.0
    avg_s = sum(c['surface'] for c in cells) / n if n else 0.0
    return {
        'mcs': mcs,
        'n_cells': n,
        'type_1_count': t1,
        'type_2_count': t2,
        'avg_volume': avg_v,
        'avg_surface': avg_s,
        'type_field': s.get('type_field', []),
        'id_field': s.get('id_field', []),
        'conc_field': s.get('conc_field', None),
    }


def generate_bigraph_image(cfg_entry):
    """Generate a colored bigraph-viz PNG for the composite document."""
    from bigraph_viz import plot_bigraph

    has_chemo = cfg_entry['config'].get('chemotaxis_lambda', 0) > 0
    outputs = {
        'cell_type_field': ['stores', 'cell_type_field'],
        'n_cells': ['stores', 'n_cells'],
        'avg_volume': ['stores', 'avg_volume'],
        'mcs': ['stores', 'mcs'],
    }

    doc = {
        'cc3d': {
            '_type': 'process',
            'address': 'local:CompuCell3DProcess',
            'outputs': outputs,
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                'n_cells': ['stores', 'n_cells'],
                'avg_volume': ['stores', 'avg_volume'],
                'time': ['global_time'],
            },
        },
    }

    node_colors = {
        ('cc3d',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg_entry):
    """Build the PBG composite document dict for display."""
    kw = {}
    for k in ('dim_x', 'dim_y', 'fluctuation_amplitude', 'target_volume',
              'lambda_volume', 'contact_medium_t1', 'contact_medium_t2',
              'contact_t1_t1', 'contact_t1_t2', 'contact_t2_t2'):
        if k in cfg_entry['config']:
            kw[k] = cfg_entry['config'][k]
    mcs_per_snap = cfg_entry['total_mcs'] // cfg_entry['n_snapshots']
    kw['interval'] = float(mcs_per_snap)
    kw['mcs_per_step'] = mcs_per_snap
    return make_cc3d_document(**kw)


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
}

# Cell colors for 2D lattice viewer: medium=bg, type1=green, type2=orange
CELL_COLORS = {
    0: [245, 248, 252],   # medium (light gray)
    1: [34, 197, 94],     # TypeA (green)
    2: [249, 115, 22],    # TypeB (orange)
}


def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        dim_x = cfg['config']['dim_x']
        dim_y = cfg['config']['dim_y']
        n_initial = snapshots[0]['n_cells']
        n_final = snapshots[-1]['n_cells']

        # Time series
        mcs_list = [s['mcs'] for s in snapshots]
        n_cells_list = [s['n_cells'] for s in snapshots]
        t1_list = [s['type_1_count'] for s in snapshots]
        t2_list = [s['type_2_count'] for s in snapshots]
        avg_vol_list = [s['avg_volume'] for s in snapshots]
        avg_surf_list = [s['avg_surface'] for s in snapshots]

        # Check for concentration field
        has_conc = snapshots[-1]['conc_field'] is not None

        # JS data — lattice snapshots
        js_snapshots = []
        for s in snapshots:
            entry = {
                'mcs': s['mcs'],
                'type_field': s['type_field'],
                'id_field': s['id_field'],
            }
            if has_conc and s['conc_field'] is not None:
                entry['conc_field'] = s['conc_field']
            js_snapshots.append(entry)

        all_js_data[sid] = {
            'snapshots': js_snapshots,
            'dim': [dim_x, dim_y],
            'has_conc': has_conc,
            'charts': {
                'mcs': mcs_list,
                'n_cells': n_cells_list,
                'type_1': t1_list,
                'type_2': t2_list,
                'avg_volume': avg_vol_list,
                'avg_surface': avg_surf_list,
            },
        }

        # Bigraph PNG
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document JSON
        pbg_doc = build_pbg_document(cfg)

        # Metrics
        cell_pct = (f'{n_final / n_initial * 100:.0f}%'
                    if n_initial > 0 else 'N/A')

        # Concentration toggle HTML
        if has_conc:
            conc_toggle = (
                "<div class='conc-toggle'><label>"
                "<input type='checkbox' id='conc-cb-" + sid + "' "
                "onchange='toggleConc(\"" + sid + "\")' checked> "
                "Show concentration</label></div>"
            )
        else:
            conc_toggle = ""

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Lattice</span><span class="metric-value">{dim_x}&times;{dim_y}</span></div>
        <div class="metric"><span class="metric-label">Initial Cells</span><span class="metric-value">{n_initial}</span></div>
        <div class="metric"><span class="metric-label">Final Cells</span><span class="metric-value">{n_final}</span><span class="metric-sub">{cell_pct} of initial</span></div>
        <div class="metric"><span class="metric-label">Avg Volume</span><span class="metric-value">{avg_vol_list[-1]:.1f}</span></div>
        <div class="metric"><span class="metric-label">Total MCS</span><span class="metric-value">{cfg['total_mcs']:,}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">2D Lattice Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="lattice-canvas"></canvas>
        <div class="viewer-info">
          <strong>{dim_x}&times;{dim_y}</strong> lattice &middot;
          <span style="color:#22c55e;">&#9632;</span> TypeA &middot;
          <span style="color:#f97316;">&#9632;</span> TypeB &middot;
          <span style="color:#e2e8f0;">&#9632;</span> Medium
        </div>
        {conc_toggle}
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>MCS</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">MCS = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Population &amp; Morphometry</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-cells-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-types-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-volume-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-surface-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    # Navigation
    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    # PBG docs for JSON viewer
    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CompuCell3D Cellular Potts Model Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.lattice-canvas {{ width:100%; aspect-ratio:1/1; max-height:600px; display:block; image-rendering:pixelated; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#1e293b; }}
.conc-toggle {{ position:absolute; top:.8rem; right:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.4rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(241,245,249,.97));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#64748b; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#334155; min-width:120px; text-align:right; }}
.play-btn {{ background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>CompuCell3D Cellular Potts Model Report</h1>
  <p>Three Glazier-Graner-Hogeweg (GGH) simulations wrapped as <strong>process-bigraph</strong>
  Processes using CompuCell3D. Each configuration demonstrates a distinct multicellular
  behaviour with interactive 2-D lattice visualization.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-compucell3d</strong> &mdash;
  CompuCell3D + process-bigraph &mdash;
  Cellular Potts Model / Glazier-Graner-Hogeweg Framework
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};
const CELL_COLORS = {{0:[245,248,252], 1:[34,197,94], 2:[249,115,22]}};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── 2D Lattice Canvas Viewers ───
const viewers = {{}};
const playStates = {{}};
const concStates = {{}};

function turboColormap(t) {{
  // t in [0,1] -> [r,g,b] in [0,255]
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.25) {{
    const s = t / 0.25;
    r = 48 + 20*s; g = 18 + 161*s; b = 252 - 48*s;
  }} else if (t < 0.5) {{
    const s = (t - 0.25) / 0.25;
    r = 68 + 28*s; g = 179 + 38*s; b = 204 - 140*s;
  }} else if (t < 0.75) {{
    const s = (t - 0.5) / 0.25;
    r = 96 + 153*s; g = 217 - 25*s; b = 64 - 38*s;
  }} else {{
    const s = (t - 0.75) / 0.25;
    r = 249; g = 192 - 140*s; b = 26 - 13*s;
  }}
  return [Math.round(r), Math.round(g), Math.round(b)];
}}

function isBoundary(idField, x, y, W, H) {{
  // A pixel is on a cell boundary if any 4-neighbour has a different cell ID
  const cid = idField[x][y];
  if (cid === 0) return false;
  if (x > 0   && idField[x-1][y] !== cid) return true;
  if (x < W-1 && idField[x+1][y] !== cid) return true;
  if (y > 0   && idField[x][y-1] !== cid) return true;
  if (y < H-1 && idField[x][y+1] !== cid) return true;
  return false;
}}

function renderLattice(sid, frameIdx) {{
  const d = DATA[sid];
  const snap = d.snapshots[frameIdx];
  const [W, H] = d.dim;
  const canvas = document.getElementById('canvas-' + sid);
  const ctx = canvas.getContext('2d');

  // Scale factor: render at higher resolution for sharp display
  const scale = Math.max(1, Math.floor(600 / Math.max(W, H)));
  const cW = W * scale;
  const cH = H * scale;
  canvas.width = cW;
  canvas.height = cH;

  const img = ctx.createImageData(cW, cH);

  const showConc = d.has_conc && concStates[sid];
  let concMax = 0;
  if (showConc && snap.conc_field) {{
    for (let x = 0; x < W; x++)
      for (let y = 0; y < H; y++)
        if (snap.conc_field[x][y] > concMax) concMax = snap.conc_field[x][y];
  }}

  // Pre-compute boundary map
  const idField = snap.id_field;
  const bnd = new Uint8Array(W * H);
  for (let x = 0; x < W; x++)
    for (let y = 0; y < H; y++)
      if (isBoundary(idField, x, y, W, H)) bnd[x * H + y] = 1;

  for (let x = 0; x < W; x++) {{
    for (let y = 0; y < H; y++) {{
      const cellType = snap.type_field[x][y];
      let r, g, b;

      // Boundary pixel → dark outline
      if (bnd[x * H + y]) {{
        r = 30; g = 30; b = 30;
      }} else if (cellType > 0) {{
        const c = CELL_COLORS[cellType] || [150,150,150];
        r = c[0]; g = c[1]; b = c[2];
      }} else if (showConc && snap.conc_field && concMax > 0) {{
        const t = snap.conc_field[x][y] / concMax;
        const tc = turboColormap(t);
        r = tc[0]; g = tc[1]; b = tc[2];
      }} else {{
        r = 245; g = 248; b = 252;
      }}

      // Fill scaled block
      for (let sx = 0; sx < scale; sx++) {{
        for (let sy = 0; sy < scale; sy++) {{
          const px = x * scale + sx;
          const py = y * scale + sy;
          const idx = (py * cW + px) * 4;
          img.data[idx]   = r;
          img.data[idx+1] = g;
          img.data[idx+2] = b;
          img.data[idx+3] = 255;
        }}
      }}
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}

function initViewer(sid) {{
  concStates[sid] = DATA[sid].has_conc;
  renderLattice(sid, 0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    renderLattice(sid, idx);
    tval.textContent = 'MCS = ' + DATA[sid].snapshots[idx].mcs;
  }});

  viewers[sid] = {{ slider, tval }};
  playStates[sid] = {{ playing: false, interval: null }};
}}

function toggleConc(sid) {{
  const cb = document.getElementById('conc-cb-' + sid);
  concStates[sid] = cb.checked;
  const idx = parseInt(viewers[sid].slider.value);
  renderLattice(sid, idx);
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const v = viewers[sid];
  const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      renderLattice(sid, idx);
      v.tval.textContent = 'MCS = ' + d.snapshots[idx].mcs;
    }}, 500);
  }} else {{
    btn.textContent = 'Play';
    clearInterval(ps.interval);
  }}
}}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:50, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'MCS', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts;

  Plotly.newPlot('chart-cells-'+sid, [{{
    x:c.mcs, y:c.n_cells, type:'scatter', mode:'lines+markers',
    line:{{ color:'#6366f1', width:2 }}, marker:{{ size:4 }},
  }}], {{...pLayout, title:{{ text:'Total Cells', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }}
  }}, pCfg);

  Plotly.newPlot('chart-types-'+sid, [
    {{ x:c.mcs, y:c.type_1, type:'scatter', mode:'lines+markers',
       line:{{ color:'#22c55e', width:1.5 }}, marker:{{ size:3 }}, name:'TypeA' }},
    {{ x:c.mcs, y:c.type_2, type:'scatter', mode:'lines+markers',
       line:{{ color:'#f97316', width:1.5 }}, marker:{{ size:3 }}, name:'TypeB' }},
  ], {{...pLayout, title:{{ text:'Cell Types', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
  }}, pCfg);

  Plotly.newPlot('chart-volume-'+sid, [{{
    x:c.mcs, y:c.avg_volume, type:'scatter', mode:'lines+markers',
    line:{{ color:'#10b981', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
  }}], {{...pLayout, title:{{ text:'Average Volume', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Pixels', font:{{ size:10 }} }} }}, showlegend:false
  }}, pCfg);

  Plotly.newPlot('chart-surface-'+sid, [{{
    x:c.mcs, y:c.avg_surface, type:'scatter', mode:'lines+markers',
    line:{{ color:'#f43f5e', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(244,63,94,0.06)',
  }}], {{...pLayout, title:{{ text:'Average Surface', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Pixels', font:{{ size:10 }} }} }}, showlegend:false
  }}, pCfg);
}});

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s')
        print(f'  {len(snapshots)} snapshots collected')
        print(f'  Final cells: {snapshots[-1]["n_cells"]}')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Open in Safari
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
