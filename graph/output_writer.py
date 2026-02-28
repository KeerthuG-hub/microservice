"""
Output Writer - saves results as JSON, CSV, and HTML report.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional
from agent.models import AnalysisResult


class OutputWriter:

    def __init__(self, result: AnalysisResult, output_dir: str):
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_all(self) -> dict:
        paths = {}
        paths['json'] = self.save_json()
        paths['csv'] = self.save_csv()
        paths['html'] = self.save_html_report()
        return paths

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    def save_json(self) -> str:
        path = self.output_dir / f'report_{self.ts}.json'

        def dep_tier(dep):
            if dep.source in ('string_cooccurrence', 'git_change_coupling'):
                return 'implicit'
            if dep.dep_type == 'observability':
                return 'observability'
            if dep.confidence >= 0.85:
                return 'confirmed'
            if dep.confidence >= 0.65:
                return 'probable'
            return 'uncertain'

        data = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'total_services': self.result.total_services,
                'infrastructure_services': self.result.infrastructure_services,
                'total_dependencies': self.result.total_dependencies,
                'confirmed_deps': self.result.confirmed_deps,
                'probable_deps': self.result.probable_deps,
                'uncertain_deps': self.result.uncertain_deps,
                'implicit_deps': self.result.implicit_deps,
                'env_only_deps': self.result.env_only_deps,
                'learned_port_map': {str(k): v
                                      for k, v in self.result.learned_port_map.items()},
                'learned_env_patterns': list(self.result.learned_env_patterns),
            },
            'services': [
                {
                    'name': s.name,
                    'language': s.language,
                    'path': s.path,
                    'anchor': s.anchor,
                    'ports': s.ports,
                    'is_infrastructure': s.is_infrastructure,
                }
                for s in self.result.services
            ],
            'infra_nodes': [
                s.name for s in self.result.services if s.is_infrastructure
            ],
            'per_service_stats': self.result.service_stats,
            'depth_stats': self.result.depth_stats,
            'confirmed': [],
            'probable': [],
            'uncertain': [],
            'implicit': [],
            'observability': [],
        }

        for dep in sorted(self.result.dependencies,
                          key=lambda d: d.confidence, reverse=True):
            tier = dep_tier(dep)
            data[tier].append({
                'from': dep.from_service,
                'to': dep.to_service,
                'type': dep.dep_type,
                'subtype': dep.subtype,
                'confidence': round(dep.confidence, 3),
                'evidence': dep.evidence,
                'sources': dep.sources or [dep.source],
                'topic_or_table': dep.topic_or_table,
                'conditional': dep.conditional,
            })

        path.write_text(json.dumps(data, indent=2))
        print(f"   📄 JSON saved: {path}")
        return str(path)

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    def save_csv(self) -> str:
        path = self.output_dir / f'dependencies_{self.ts}.csv'
        fields = ['from_service', 'to_service', 'dep_type', 'subtype',
                  'confidence', 'tier', 'evidence', 'sources',
                  'topic_or_table', 'conditional']

        def tier(dep):
            if dep.source in ('string_cooccurrence', 'git_change_coupling'):
                return 'implicit'
            if dep.dep_type == 'observability':
                return 'observability'
            if dep.confidence >= 0.85:
                return 'confirmed'
            if dep.confidence >= 0.65:
                return 'probable'
            return 'uncertain'

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for dep in sorted(self.result.dependencies,
                               key=lambda d: d.confidence, reverse=True):
                writer.writerow({
                    'from_service': dep.from_service,
                    'to_service': dep.to_service,
                    'dep_type': dep.dep_type,
                    'subtype': dep.subtype or '',
                    'confidence': round(dep.confidence, 3),
                    'tier': tier(dep),
                    'evidence': dep.evidence or '',
                    'sources': ','.join(dep.sources or [dep.source]),
                    'topic_or_table': dep.topic_or_table or '',
                    'conditional': dep.conditional,
                })

        print(f"   📊 CSV saved: {path}")
        return str(path)

    # -------------------------------------------------------------------------
    # HTML REPORT
    # -------------------------------------------------------------------------

    def save_html_report(self) -> str:
        path = self.output_dir / f'report_{self.ts}.html'

        rows = ''
        for dep in sorted(self.result.dependencies,
                           key=lambda d: d.confidence, reverse=True):
            tier = 'implicit' if dep.source in ('string_cooccurrence', 'git_change_coupling') else (
                'observability' if dep.dep_type == 'observability' else (
                    'confirmed' if dep.confidence >= 0.85 else (
                        'probable' if dep.confidence >= 0.65 else 'uncertain'
                    )
                )
            )
            tier_color = {
                'confirmed': '#27ae60',
                'probable': '#f39c12',
                'uncertain': '#e74c3c',
                'implicit': '#9b59b6',
                'observability': '#95a5a6',
            }.get(tier, '#888')

            cond = '⚠️' if dep.conditional else ''
            env_flag = '💬 env-only' if '[env-only' in (dep.evidence or '') else ''
            rows += f"""
            <tr>
              <td>{dep.from_service}</td>
              <td>{dep.to_service}</td>
              <td>{dep.dep_type}</td>
              <td>{dep.subtype or ''}</td>
              <td><span style="color:{tier_color};font-weight:bold">{dep.confidence:.2f}</span></td>
              <td><span style="color:{tier_color}">{tier}</span></td>
              <td style="font-size:11px">{dep.evidence or ''}</td>
              <td>{dep.topic_or_table or ''}</td>
              <td>{cond}{env_flag}</td>
            </tr>"""

        svc_rows = ''
        for svc in sorted(self.result.services, key=lambda s: s.name):
            tag = '<span style="color:#e67e22;font-weight:bold">[INFRA]</span>' if svc.is_infrastructure else ''
            svc_rows += f"<tr><td>{svc.name} {tag}</td><td>{svc.language}</td><td>{svc.anchor}</td></tr>"

        # Per-service stats rows
        stat_rows = ''
        for svc_name, stats in sorted(self.result.service_stats.items()):
            infra_tag = ' <span style="color:#e67e22">[INFRA]</span>' if stats.get('is_infrastructure') else ''
            stat_rows += f"""
            <tr>
              <td>{svc_name}{infra_tag}</td>
              <td>{stats.get('outgoing', 0)}</td>
              <td>{stats.get('incoming', 0)}</td>
              <td>{stats.get('max_depth', 0)}</td>
              <td>{stats.get('transitive_deps', 0)}</td>
              <td style="font-size:10px">{', '.join(f"{k}:{v}" for k,v in stats.get('outgoing_types',{}).items())}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Microservice Dependency Report</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }}
    h1 {{ color: #3498db; }}
    h2 {{ color: #aaa; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th {{ background: #2c3e50; color: #3498db; padding: 8px; text-align: left; }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #333; font-size: 12px; }}
    tr:hover {{ background: #2c2c3e; }}
    .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
    .stat {{ background: #2c3e50; padding: 12px 20px; border-radius: 8px; min-width: 120px; }}
    .stat .num {{ font-size: 28px; font-weight: bold; color: #3498db; }}
    .stat .label {{ font-size: 11px; color: #aaa; }}
    input {{ background:#2c3e50; color:#eee; border:1px solid #444; padding:6px; border-radius:4px; width:300px; }}
    .infra {{ color: #e67e22; }}
  </style>
</head>
<body>
  <h1>🔗 Microservice Dependency Report</h1>
  <p style="color:#aaa">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

  <div class="stats">
    <div class="stat"><div class="num">{self.result.total_services}</div><div class="label">Business Services</div></div>
    <div class="stat"><div class="num" style="color:#e67e22">{self.result.infrastructure_services}</div><div class="label">Infrastructure</div></div>
    <div class="stat"><div class="num">{self.result.total_dependencies}</div><div class="label">Total Deps</div></div>
    <div class="stat"><div class="num" style="color:#27ae60">{self.result.confirmed_deps}</div><div class="label">Confirmed ≥0.85</div></div>
    <div class="stat"><div class="num" style="color:#f39c12">{self.result.probable_deps}</div><div class="label">Probable 0.65-0.85</div></div>
    <div class="stat"><div class="num" style="color:#e74c3c">{self.result.uncertain_deps}</div><div class="label">Uncertain &lt;0.65</div></div>
    <div class="stat"><div class="num" style="color:#9b59b6">{self.result.implicit_deps}</div><div class="label">Implicit</div></div>
    <div class="stat"><div class="num" style="color:#7f8c8d">{self.result.env_only_deps}</div><div class="label">Env-Only (capped)</div></div>
  </div>

  <h2>🔉 Per-Service Stats</h2>
  <table>
    <tr><th>Service</th><th>Out</th><th>In</th><th>Max Depth</th><th>Transitive</th><th>Outgoing Types</th></tr>
    {stat_rows}
  </table>

  <h2>🔧 Discovered Nodes</h2>
  <table>
    <tr><th>Name</th><th>Language</th><th>Discovery Anchor</th></tr>
    {svc_rows}
  </table>

  <h2>🔗 Dependencies</h2>
  <input type="text" id="search" placeholder="Search dependencies..." onkeyup="filterTable()">
  <table id="deptable">
    <tr>
      <th>From</th><th>To</th><th>Type</th><th>Subtype</th>
      <th>Conf</th><th>Tier</th><th>Evidence</th><th>Topic/Table</th><th>Flags</th>
    </tr>
    {rows}
  </table>

  <script>
    function filterTable() {{
      const q = document.getElementById('search').value.toLowerCase();
      const rows = document.querySelectorAll('#deptable tr:not(:first-child)');
      rows.forEach(r => {{ r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none'; }});
    }}
  </script>
</body>
</html>"""
        path.write_text(html)
        print(f"   🌐 HTML report: {path}")
        return str(path)
