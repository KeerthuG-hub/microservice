"""
Visualizer - generates a clear, readable interactive HTML visualization.

Key clarity choices (zero dependency changes):
  • Edge labels removed from canvas  — they create clutter.  All info is in
    the hover tooltip which appears on mouse-over.
  • Confidence-based opacity  — confirmed edges are bold/opaque, uncertain
    edges are faint so they don't compete visually.
  • Node grouping by role  — business services use large circles, infra/
    external nodes use small squares in a visually distinct colour.
  • Physics tuned for spacing  — higher spring length + repulsion keeps nodes
    apart so labels don't overlap.
  • Interactive filter panel  — user can toggle visibility of uncertain/
    implicit/build edges to reduce clutter on demand without reloading.
  • Per-tier edge width  — confirmed 3 px, probable 1.5 px, uncertain 0.6 px
    dashed.  Makes the important edges immediately obvious.
"""

from pathlib import Path
from typing import Optional
from agent.models import AnalysisResult

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# ── colour palette ────────────────────────────────────────────────────────────

DEP_COLORS = {
    'endpoint':       '#4FC3F7',   # sky-blue
    'async':          '#FFB74D',   # amber
    'data':           '#EF5350',   # red
    'semantic':       '#CE93D8',   # lavender
    'build':          '#A5D6A7',   # light-green
    'deployment':     '#80DEEA',   # cyan
    'infrastructure': '#78909C',   # blue-grey
    'observability':  '#546E7A',   # dark blue-grey
    'external':       '#66BB6A',   # green
}

NODE_COLORS = {
    'go':         '#00ADD8',
    'python':     '#4B8BBE',
    'java':       '#ED8B00',
    'javascript': '#F7DF1E',
    'typescript': '#3178C6',
    'rust':       '#CE422B',
    'ruby':       '#CC342D',
    'csharp':     '#239120',
    'php':        '#787CB4',
    'unknown':    '#90A4AE',
}

# Infra/external nodes  – distinct from business services
INFRA_COLOR = '#37474F'
EXTERNAL_COLOR = '__external__'


def _confidence_to_opacity(conf: float) -> str:
    """Return a hex alpha suffix ('ff', 'aa', '55') based on confidence."""
    if conf >= 0.85:
        return 'ff'
    if conf >= 0.65:
        return 'bb'
    return '66'


def _tier(dep) -> str:
    if dep.source in ('string_cooccurrence', 'git_change_coupling'):
        return 'implicit'
    if dep.confidence >= 0.85:
        return 'confirmed'
    if dep.confidence >= 0.65:
        return 'probable'
    return 'uncertain'


class Visualizer:

    def __init__(self, result: AnalysisResult):
        self.result = result

    # ── public ───────────────────────────────────────────────────────────────

    def save_html(self, output_path: str) -> Optional[str]:
        if not HAS_PYVIS:
            print("⚠️  pyvis not installed. Skipping HTML visualization.")
            return None

        net = Network(
            height='820px',
            width='100%',
            directed=True,
            notebook=False,
            bgcolor='#0f1117',
            font_color='white',
        )

        # ── physics: spread nodes out, reduce overlap ──────────────────────
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -120,
              "centralGravity": 0.005,
              "springLength": 220,
              "springConstant": 0.08,
              "damping": 0.5,
              "avoidOverlap": 1.0
            },
            "stabilization": {
              "enabled": true,
              "iterations": 300,
              "updateInterval": 25
            }
          },
          "edges": {
            "smooth": {"type": "dynamic"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
            "font": {"size": 0}
          },
          "nodes": {
            "font": {"size": 14, "face": "Inter, system-ui, sans-serif"}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 150,
            "navigationButtons": true,
            "keyboard": {"enabled": true}
          }
        }
        """)

        # ── connectivity → node sizing ─────────────────────────────────────
        connectivity: dict = {svc.name: 0 for svc in self.result.services}
        for dep in self.result.dependencies:
            if dep.dep_type == 'observability':
                continue
            connectivity[dep.from_service] = connectivity.get(dep.from_service, 0) + 1
            connectivity[dep.to_service]   = connectivity.get(dep.to_service,   0) + 1

        all_service_names = {svc.name for svc in self.result.services}
        infra_names       = {svc.name for svc in self.result.services if svc.is_infrastructure}

        # ── business service nodes ─────────────────────────────────────────
        for svc in self.result.services:
            if svc.is_infrastructure:
                continue   # drawn separately below
            conn  = connectivity.get(svc.name, 0)
            size  = 22 + min(conn * 4, 28)
            color = NODE_COLORS.get(svc.language, '#90A4AE')
            stats = self.result.service_stats.get(svc.name, {})
            tip   = (
                f"<b>{svc.name}</b><br>"
                f"Language: {svc.language}<br>"
                f"Outgoing: {stats.get('outgoing', 0)} | "
                f"Incoming: {stats.get('incoming', 0)}<br>"
                f"Max depth: {stats.get('max_depth', 0)} | "
                f"Transitive: {stats.get('transitive_deps', 0)}<br>"
                f"Anchor: {svc.anchor}"
            )
            net.add_node(
                svc.name,
                label=svc.name,
                color={'background': color, 'border': '#ffffff33',
                       'highlight': {'background': color, 'border': '#ffffff'}},
                size=size,
                title=tip,
                borderWidth=1.5,
                borderWidthSelected=3,
                font={'size': 13, 'color': 'white',
                      'face': 'Inter, system-ui, sans-serif'},
                shape='ellipse',
                group='service',
            )

        # ── infrastructure nodes (known to explorer) ───────────────────────
        for svc in self.result.services:
            if not svc.is_infrastructure:
                continue
            tip = (
                f"<b>{svc.name}</b> [infrastructure]<br>"
                f"Language: {svc.language}"
            )
            net.add_node(
                svc.name,
                label=svc.name,
                color={'background': '#2E3F4F', 'border': '#78909C',
                       'highlight': {'background': '#3D5166', 'border': '#90A4AE'}},
                size=14,
                title=tip,
                shape='box',
                font={'size': 11, 'color': '#90A4AE',
                      'face': 'Inter, system-ui, sans-serif'},
                borderWidth=1,
                group='infrastructure',
            )

        # ── extra nodes that appear only as dep targets ────────────────────
        seen_extra: set = set()
        for dep in self.result.dependencies:
            for node in (dep.from_service, dep.to_service):
                if node not in all_service_names and node not in seen_extra:
                    seen_extra.add(node)
                    is_ext = node == '__external__'
                    label  = '🌐 external' if is_ext else node
                    tip    = 'External user traffic' if is_ext else f'External/Infra node: {node}'
                    net.add_node(
                        node,
                        label=label,
                        color={'background': '#1B2838', 'border': '#546E7A',
                               'highlight': {'background': '#263545', 'border': '#78909C'}},
                        size=12,
                        title=tip,
                        shape='box',
                        font={'size': 11, 'color': '#78909C',
                              'face': 'Inter, system-ui, sans-serif'},
                        borderWidth=1,
                        group='external',
                    )

        # ── edges ──────────────────────────────────────────────────────────
        for dep in self.result.dependencies:
            if dep.dep_type == 'observability':
                continue

            t      = _tier(dep)
            alpha  = _confidence_to_opacity(dep.confidence)
            base   = DEP_COLORS.get(dep.dep_type, '#888888')
            color  = base + alpha

            # Width encodes confidence clearly
            if t == 'confirmed':
                width = 2.5
                dashes = False
            elif t == 'probable':
                width = 1.2
                dashes = False
            elif t == 'implicit':
                width = 0.8
                dashes = [5, 8]
            else:   # uncertain
                width = 0.7
                dashes = [3, 6]

            # Rich tooltip — no inline label on canvas
            sources_str = ', '.join(dep.sources or [dep.source or '?'])
            tip = (
                f"<b>{dep.from_service}</b> → <b>{dep.to_service}</b><br>"
                f"Type: <b>{dep.dep_type}</b> / {dep.subtype or '?'}<br>"
                f"Confidence: <b>{dep.confidence:.2f}</b> ({t})<br>"
                f"Sources: {sources_str}<br>"
                f"Evidence: {(dep.evidence or '?')[:140]}"
            )
            if dep.conditional:
                tip += "<br><span style='color:#FFB74D'>⚠ Conditional (feature flag)</span>"

            net.add_edge(
                dep.from_service,
                dep.to_service,
                color={'color': color, 'highlight': base + 'ff', 'hover': base + 'ff'},
                width=width,
                dashes=dashes,
                title=tip,
                # No label= so nothing is drawn on the canvas edge
            )

        # ── save & inject UI chrome ────────────────────────────────────────
        output_path = str(output_path)
        net.save_graph(output_path)
        self._inject_ui(output_path)
        print(f"   🌐 Graph saved: {output_path}")
        return output_path

    # ── UI injection ──────────────────────────────────────────────────────────

    def _inject_ui(self, output_path: str):
        """Inject legend, filter controls and stats bar into the saved HTML."""
        try:
            html = Path(output_path).read_text()

            legend_items = ''.join(
                f'<span class="leg-item" style="--c:{c}" data-type="{t}" '
                f'title="Click to highlight {t} edges">{t}</span>'
                for t, c in DEP_COLORS.items()
                if t != 'observability'
            )

            stats = self.result
            overlay = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
  #ctrl-panel {{
    position:fixed; top:12px; left:12px; z-index:9999;
    background:#111827cc; backdrop-filter:blur(10px);
    border:1px solid #ffffff18; border-radius:12px;
    padding:14px 16px; min-width:260px; max-width:300px;
    font-family:Inter,system-ui,sans-serif; font-size:12px; color:#e2e8f0;
    box-shadow:0 4px 24px #00000066;
  }}
  #ctrl-panel h3 {{
    margin:0 0 10px; font-size:13px; font-weight:600;
    color:#60a5fa; letter-spacing:.4px;
  }}
  .stat-row {{
    display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;
  }}
  .stat-pill {{
    flex:1; min-width:60px; background:#1e293b; border-radius:8px;
    padding:6px 8px; text-align:center;
  }}
  .stat-pill .n {{ font-size:18px; font-weight:600; color:#60a5fa; }}
  .stat-pill .l {{ font-size:9px; color:#94a3b8; margin-top:1px; }}
  .stat-pill.green .n {{ color:#4ade80; }}
  .stat-pill.amber .n {{ color:#fbbf24; }}
  .stat-pill.red   .n {{ color:#f87171; }}
  .sep {{ border:none; border-top:1px solid #ffffff18; margin:8px 0; }}
  .leg-wrap {{ display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px; }}
  .leg-item {{
    padding:3px 9px; border-radius:20px; cursor:pointer;
    background:color-mix(in srgb, var(--c) 25%, #1e293b);
    border:1px solid color-mix(in srgb, var(--c) 60%, transparent);
    color:color-mix(in srgb, var(--c) 90%, white);
    font-size:10px; user-select:none; transition:opacity .15s;
  }}
  .leg-item.muted {{ opacity:.35; }}
  .filter-row {{ display:flex; flex-direction:column; gap:5px; }}
  .filter-row label {{
    display:flex; align-items:center; gap:6px; cursor:pointer;
    font-size:11px; color:#cbd5e1;
  }}
  .filter-row input[type=checkbox] {{ accent-color:#60a5fa; }}
  #search-box {{
    width:100%; box-sizing:border-box; margin-top:8px;
    background:#1e293b; border:1px solid #334155;
    border-radius:6px; padding:5px 8px; color:#e2e8f0; font-size:11px;
  }}
  #search-box::placeholder {{ color:#64748b; }}
  .hint {{ color:#64748b; font-size:10px; margin-top:6px; }}
</style>

<div id="ctrl-panel">
  <h3>🔗 Dependency Map</h3>

  <div class="stat-row">
    <div class="stat-pill">
      <div class="n">{stats.total_services}</div>
      <div class="l">Services</div>
    </div>
    <div class="stat-pill green">
      <div class="n">{stats.confirmed_deps}</div>
      <div class="l">Confirmed</div>
    </div>
    <div class="stat-pill amber">
      <div class="n">{stats.probable_deps}</div>
      <div class="l">Probable</div>
    </div>
    <div class="stat-pill red">
      <div class="n">{stats.uncertain_deps}</div>
      <div class="l">Uncertain</div>
    </div>
  </div>

  <hr class="sep">

  <div style="font-size:10px;color:#94a3b8;margin-bottom:5px;font-weight:600;
              letter-spacing:.5px;text-transform:uppercase">Dep Types</div>
  <div class="leg-wrap">{legend_items}</div>

  <hr class="sep">

  <div style="font-size:10px;color:#94a3b8;margin-bottom:6px;font-weight:600;
              letter-spacing:.5px;text-transform:uppercase">Show / Hide</div>
  <div class="filter-row">
    <label><input type="checkbox" id="chk-confirmed" checked> Confirmed ≥ 0.85</label>
    <label><input type="checkbox" id="chk-probable"  checked> Probable 0.65–0.85</label>
    <label><input type="checkbox" id="chk-uncertain" checked> Uncertain &lt; 0.65</label>
    <label><input type="checkbox" id="chk-implicit">          Implicit (co-occurrence)</label>
    <label><input type="checkbox" id="chk-infra"    checked>  Infra / External targets</label>
  </div>

  <input id="search-box" type="text" placeholder="🔍  Highlight a service…">
  <div class="hint">Hover any edge for details • Scroll to zoom • Drag to pan</div>
</div>

<script>
(function() {{
  // Wait until vis Network is available on the page
  let attempts = 0;
  const ready = setInterval(() => {{
    if (typeof network === 'undefined' || ++attempts > 60) {{ clearInterval(ready); return; }}
    clearInterval(ready);
    init();
  }}, 200);

  function init() {{
    const allEdges = network.body.data.edges.get();
    const allNodes = network.body.data.nodes.get();

    // Tag each edge with its tier for filtering
    allEdges.forEach(e => {{
      const conf = e.title ? (e.title.match(/Confidence.*?([\d.]+)/) || [])[1] : '0';
      const c = parseFloat(conf) || 0;
      e._tier   = (e.dashes && Array.isArray(e.dashes)) ? 'implicit'
                : c >= 0.85 ? 'confirmed'
                : c >= 0.65 ? 'probable'
                : 'uncertain';
    }});

    function applyFilters() {{
      const show = {{
        confirmed: document.getElementById('chk-confirmed').checked,
        probable:  document.getElementById('chk-probable').checked,
        uncertain: document.getElementById('chk-uncertain').checked,
        implicit:  document.getElementById('chk-implicit').checked,
      }};
      const showInfra = document.getElementById('chk-infra').checked;

      const edgeUpdates = allEdges.map(e => {{
        const visible = show[e._tier] !== false;
        return {{ id: e.id, hidden: !visible }};
      }});
      network.body.data.edges.update(edgeUpdates);

      // Hide/show infra & external nodes
      const infraUpdates = allNodes
        .filter(n => n.group === 'infrastructure' || n.group === 'external')
        .map(n => ({{ id: n.id, hidden: !showInfra }}));
      network.body.data.nodes.update(infraUpdates);

      network.redraw();
    }}

    ['chk-confirmed','chk-probable','chk-uncertain','chk-implicit','chk-infra']
      .forEach(id => document.getElementById(id).addEventListener('change', applyFilters));

    // Search / highlight
    document.getElementById('search-box').addEventListener('input', function() {{
      const q = this.value.trim().toLowerCase();
      if (!q) {{
        network.body.data.nodes.update(allNodes.map(n => ({{ id: n.id, opacity: 1 }})));
        return;
      }}
      const updates = allNodes.map(n => {{
        const match = n.label && n.label.toLowerCase().includes(q);
        return {{ id: n.id, opacity: match ? 1 : 0.15 }};
      }});
      network.body.data.nodes.update(updates);
    }});

    // Legend click: filter by dep type string in tooltip
    document.querySelectorAll('.leg-item').forEach(el => {{
      el.addEventListener('click', function() {{
        this.classList.toggle('muted');
        const hidden = this.classList.contains('muted');
        const t = this.dataset.type;
        const updates = allEdges
          .filter(e => e.title && e.title.includes('Type: <b>' + t + '</b>'))
          .map(e => ({{ id: e.id, hidden }}));
        network.body.data.edges.update(updates);
        network.redraw();
      }});
    }});
  }}
}})();
</script>
"""
            html = html.replace('</body>', overlay + '\n</body>')
            Path(output_path).write_text(html)
        except Exception as exc:
            print(f"   ⚠️  Could not inject UI chrome: {exc}")
