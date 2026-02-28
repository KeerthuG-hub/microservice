"""
Microservice Dependency Analyzer — Enterprise Edition
"""

import streamlit as st
import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

def _write_theme():
    d = Path(__file__).parent / ".streamlit"
    d.mkdir(exist_ok=True)
    (d / "config.toml").write_text(
        '[theme]\nbase = "dark"\n'
        'primaryColor = "#60A5FA"\n'
        'backgroundColor = "#0F172A"\n'
        'secondaryBackgroundColor = "#1E293B"\n'
        'textColor = "#F1F5F9"\n'
    )
_write_theme()

PREDICTIONS_PATH  = 'data/outputs/predictions.json'
FEATURES_PATH     = 'data/features/online-boutique_features.csv'
DEPENDENCIES_PATH = 'data/outputs/online-boutique_latest.json'
GEMINI_API_KEY    = os.getenv('GEMINI_API_KEY', '')

st.set_page_config(
    page_title="Microservice Dependency Analyzer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

RISK_COLORS = {
    'HIGH':    '#F87171',
    'MEDIUM':  '#FBBF24',
    'LOW':     '#34D399',
    'MINIMAL': '#60A5FA',
}

DEP_COLORS = {
    'Confirmed': '#60A5FA',
    'Probable':  '#FBBF24',
    'Uncertain': '#94A3B8',
}

# Infrastructure nodes that should never appear in the graph
INFRA_NODES = {
    'loadgenerator', 'load-generator',
    'opentelemetrycollector', 'otelcollector', 'jaeger', 'zipkin', 'prometheus', 'grafana',
    'health', 'healthcheck', 'health-check',
    'postgres', 'postgresql', 'mysql', 'mongodb', 'redis', 'memcached',
    'kafka', 'rabbitmq', 'pubsub', 'nats',
    'gcp', 'aws', 'azure',
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', system-ui, sans-serif !important; }
code, .mono { font-family: 'DM Mono', monospace !important; }

.pg-header {
    border-bottom: 1px solid #334155;
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.pg-title { font-size: 1.4rem; font-weight: 700; color: #F1F5F9; margin: 0; }
.pg-sub   { font-size: 0.7rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.25rem; }
.pg-ts    { font-size: 0.65rem; color: #475569; font-family: 'DM Mono', monospace; }

.sec { font-size: 0.65rem; font-weight: 700; color: #64748B; text-transform: uppercase;
       letter-spacing: 0.1em; border-bottom: 1px solid #1E293B;
       padding-bottom: 0.3rem; margin-bottom: 0.75rem; display: block; }

.alert { padding: 0.875rem 1rem; border-radius: 4px; margin-bottom: 0.5rem; border-left: 3px solid; }
.a-high   { background: #1F0F0F; border-color: #F87171; }
.a-medium { background: #1F1700; border-color: #FBBF24; }
.a-ok     { background: #0A1F14; border-color: #34D399; }
.a-info   { background: #0F1629; border-color: #60A5FA; }
.a-title  { font-weight: 700; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.a-high   .a-title { color: #F87171; }
.a-medium .a-title { color: #FBBF24; }
.a-ok     .a-title { color: #34D399; }
.a-info   .a-title { color: #60A5FA; }
.a-body   { font-size: 0.875rem; color: #CBD5E1; line-height: 1.55; margin-top: 0.25rem; }
.a-rec    { font-size: 0.8rem; color: #64748B; margin-top: 0.3rem; display: block; }

.response-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.response-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #60A5FA;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.response-card-content {
    font-size: 0.9rem;
    color: #E2E8F0;
    line-height: 1.6;
}
.response-card ul {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}
.response-card li {
    margin: 0.3rem 0;
    color: #CBD5E1;
}
.response-metric {
    display: inline-block;
    background: #0F172A;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    margin: 0.2rem 0.3rem 0.2rem 0;
    font-size: 0.85rem;
}
.metric-label { color: #64748B; font-weight: 500; }
.metric-value { color: #F1F5F9; font-weight: 700; margin-left: 0.3rem; }

button {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def base_layout(**kw):
    d = dict(
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(family="DM Sans, system-ui, sans-serif", size=11, color='#CBD5E1'),
        margin=dict(l=12, r=12, t=36, b=12),
        colorway=list(RISK_COLORS.values()),
        legend=dict(bgcolor='#1E293B', bordercolor='#334155', borderwidth=1,
                    font=dict(size=10, color='#CBD5E1')),
        xaxis=dict(gridcolor='#334155', linecolor='#334155',
                   tickfont=dict(size=10, color='#64748B'),
                   title_font=dict(size=10, color='#64748B')),
        yaxis=dict(gridcolor='#334155', linecolor='#334155',
                   tickfont=dict(size=10, color='#64748B'),
                   title_font=dict(size=10, color='#64748B')),
    )
    d.update(kw)
    return d


def _is_infra_node(name: str) -> bool:
    """Return True if node is infrastructure and should be excluded from the graph."""
    n = name.lower().strip()
    if n in INFRA_NODES:
        return True
    # prefix patterns
    for prefix in ('gcp-', 'aws-', 'gke-', 'k8s-'):
        if n.startswith(prefix):
            return True
    return False


@st.cache_data
def load_predictions():
    if not Path(PREDICTIONS_PATH).exists():
        return pd.DataFrame()
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if 'incident_probability' not in df.columns and 'incident_probability_percent' in df.columns:
        df['incident_probability'] = (
            df['incident_probability_percent'].str.replace('%', '', regex=False).astype(float) / 100
        )
    return df


@st.cache_data
def load_dependencies():
    if not Path(DEPENDENCIES_PATH).exists():
        return {}
    with open(DEPENDENCIES_PATH) as f:
        return json.load(f)


@st.cache_data
def load_features():
    if not Path(FEATURES_PATH).exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_PATH)


def build_graph(dep_data, predictions_df):
    G = nx.DiGraph()

    risk_map = {} if predictions_df.empty else dict(
        zip(predictions_df['service_name'],
            predictions_df.get('risk_level', 'MINIMAL')))

    svc_names = {
        s['name'] for s in dep_data.get('services', [])
        if not s.get('is_infrastructure', False)
        and not _is_infra_node(s['name'])   # FIX: filter infra by name pattern too
    }

    NOISE_SOURCES = {'string_cooccurrence', 'git_change_coupling'}

    for dep in dep_data.get('confirmed', []) + dep_data.get('probable', []):
        src, tgt = dep.get('from', ''), dep.get('to', '')

        if not src or not tgt:
            continue
        if src.startswith('__') or tgt.startswith('__'):
            continue
        if tgt not in svc_names:
            continue

        # FIX: also skip if src is infra
        if _is_infra_node(src) or _is_infra_node(tgt):
            continue

        if dep.get('type') == 'observability':
            continue

        sources = dep.get('sources', [dep.get('source', '')])
        if isinstance(sources, str):
            sources = [sources]
        if any(s in NOISE_SOURCES for s in sources):
            continue

        if dep.get('confidence', 0) < 0.65:
            continue

        G.add_node(src, risk=risk_map.get(src, 'MINIMAL'))
        G.add_node(tgt, risk=risk_map.get(tgt, 'MINIMAL'))
        G.add_edge(
            src, tgt,
            type=dep.get('subtype', dep.get('type', 'http')),
            dep_type=dep.get('type', 'unknown'),
            confidence=dep.get('confidence', 0),
        )

    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            svc = row['service_name']
            if not _is_infra_node(svc) and svc not in G:
                G.add_node(svc, risk=row.get('risk_level', 'MINIMAL'))

    return G


def page_header(title, sub):
    st.markdown(f"""
<div class="pg-header">
  <div><div class="pg-title">{title}</div><div class="pg-sub">{sub}</div></div>
  <div class="pg-ts">Last sync · Feb 27 2026 · 09:41 UTC</div>
</div>""", unsafe_allow_html=True)


def sec(label):
    st.markdown(f'<span class="sec">{label}</span>', unsafe_allow_html=True)


def style_risk(val):
    c = RISK_COLORS.get(val, '#60A5FA')
    return f'color:{c};font-weight:700;font-size:0.8rem'


def style_prob(val):
    try:
        v = float(str(val).replace('%', ''))
        c = '#F87171' if v >= 70 else '#FBBF24' if v >= 40 else '#34D399'
        return f'color:{c};font-weight:600'
    except:
        return ''


def render_sidebar(predictions_df, dep_data, G):
    with st.sidebar:
        st.markdown("### ◈ Navigation")
        page = st.radio("", [
            "📊 Dashboard", "🔗 Dependency Graph",
            "🤖 ML Predictions", "📈 Analytics", "💬 AI Assistant"
        ], label_visibility="collapsed")

        st.divider()
        st.markdown("### System Metrics")
        if not predictions_df.empty:
            st.metric("Total Services", len(predictions_df))
            high = len(predictions_df[predictions_df['risk_level'] == 'HIGH'])
            st.metric("High Risk Services", high,
                      delta=f"-{high}" if high > 0 else None, delta_color="inverse")

        probable  = len(dep_data.get('probable', []))
        st.metric("Dependencies", G.number_of_edges())

        st.divider()
        st.markdown("### Dependency Stats")
        st.write(f"Confirmed: **{G.number_of_edges()}**")
        st.write(f"Probable: **{probable}**")
        st.write(f"Observability: **{len(dep_data.get('observability', []))}**")

        st.divider()
        st.caption("Architecture Intelligence Platform")
        return page


def render_dashboard(predictions_df, dep_data, G):
    page_header("Microservice Dependency Analyzer",
                "Architecture Intelligence Platform — Online Boutique")

    if predictions_df.empty:
        st.error("No prediction data. Run the prediction pipeline first.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    high   = len(predictions_df[predictions_df['risk_level'] == 'HIGH'])
    medium = len(predictions_df[predictions_df['risk_level'] == 'MEDIUM'])
    deps   = G.number_of_edges()
    crits  = len(list(nx.articulation_points(G.to_undirected()))
                 if G.number_of_nodes() > 1 else [])
    c1.metric("Services",        len(predictions_df))
    c2.metric("High Risk",       high,   delta=f"-{high}"   if high   > 0 else None, delta_color="inverse")
    c3.metric("Medium Risk",     medium, delta=f"-{medium}" if medium > 0 else None, delta_color="inverse")
    c4.metric("Dependencies",    deps)
    c5.metric("Critical Points", crits)

    st.divider()
    sec("Critical Alerts")
    high_rows = predictions_df[predictions_df['risk_level'] == 'HIGH']

    if not high_rows.empty:
        for _, r in high_rows.iterrows():
            affected = r.get('affected_count', r.get('estimated_affected_services', 'N/A'))
            st.markdown(f"""
<div class="alert a-high">
  <div class="a-title">Critical — {r['service_name']}</div>
  <div class="a-body">
    Probability: <strong>{r.get('incident_probability_percent','N/A')}</strong> ·
    Severity: <strong>{r.get('severity','N/A')}</strong> ·
    Affected: <strong>~{affected} services</strong> ·
    Recovery: <strong>~{r.get('estimated_recovery_minutes','N/A')} min</strong>
  </div>
  <span class="a-rec">↳ {r.get('recommendation','Immediate investigation required.')}</span>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="alert a-ok">
  <div class="a-title">All Systems Operational</div>
  <div class="a-body">No critical services detected. System health within normal parameters.</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        sec("Risk Distribution")
        rc = predictions_df['risk_level'].value_counts().reset_index()
        rc.columns = ['Risk Level', 'Count']
        fig = px.pie(rc, values='Count', names='Risk Level',
                     color='Risk Level', color_discrete_map=RISK_COLORS, hole=0.5)
        fig.update_layout(**base_layout(height=300, showlegend=True,
            legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5,
                        bgcolor='rgba(0,0,0,0)', borderwidth=0)))
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont=dict(size=11, color='#F1F5F9'),
                          marker=dict(line=dict(color='#0F172A', width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Top Risk Services")
        cols_show = [c for c in ['service_name', 'incident_probability_percent',
                                  'risk_level', 'severity']
                     if c in predictions_df.columns]
        top5 = predictions_df.nlargest(5, 'incident_probability')[cols_show]
        styled = top5.style
        if 'risk_level' in cols_show:
            styled = styled.map(style_risk, subset=['risk_level'])
        if 'incident_probability_percent' in cols_show:
            styled = styled.map(style_prob, subset=['incident_probability_percent'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    sec("Dependency Analysis")
    c1, c2, c3 = st.columns(3)

    with c1:
        conf_df = pd.DataFrame({
            'Category': ['Confirmed', 'Probable', 'Uncertain'],
            'Count': [len(dep_data.get('confirmed', [])),
                      len(dep_data.get('probable', [])),
                      len(dep_data.get('uncertain', []))]
        })
        fig = px.bar(conf_df, x='Category', y='Count',
                     color='Category', color_discrete_map=DEP_COLORS)
        fig.update_layout(**base_layout(title_text='Confidence Distribution',
                                        height=260, showlegend=False))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tc = {}
        for dep in dep_data.get('confirmed', []):
            t = dep.get('type', 'unknown')
            tc[t] = tc.get(t, 0) + 1
        if tc:
            fig = go.Figure(go.Bar(x=list(tc.keys()), y=list(tc.values()),
                                   marker_color='#60A5FA', marker_line_width=0))
            fig.update_layout(**base_layout(title_text='Dependency Types',
                                            height=260, showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

    with c3:
        sec("Network Metrics")
        if G.number_of_nodes() > 0:
            avg = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.metric("Avg Dependencies", f"{avg:.1f}")
            st.metric("Graph Density",    f"{nx.density(G):.1%}")
            st.metric("Components",       nx.number_strongly_connected_components(G))


def render_graph_page(G, predictions_df):
    page_header("Dependency Graph", "Node color = ML-predicted risk level · Arrows = call direction")

    if not G.nodes():
        st.warning("No graph data available.")
        return

    c1, c2, c3 = st.columns([2, 2, 3])
    layout = c1.selectbox("Layout", ["Spring", "Circular", "Kamada-Kawai"])
    labels = c2.checkbox("Show Labels", True)
    risks  = c3.multiselect("Filter Risk", ['HIGH', 'MEDIUM', 'LOW', 'MINIMAL'],
                             default=['HIGH', 'MEDIUM', 'LOW', 'MINIMAL'])

    G_f = G.subgraph([n for n in G.nodes()
                      if G.nodes[n].get('risk', 'MINIMAL') in risks]).copy()

    if layout == "Spring":
        pos = nx.spring_layout(G_f, seed=42, k=2.5, iterations=50)
    elif layout == "Circular":
        pos = nx.circular_layout(G_f)
    else:
        pos = nx.kamada_kawai_layout(G_f)

    risk_map = {} if predictions_df.empty else dict(
        zip(predictions_df['service_name'],
            predictions_df.get('risk_level', 'MINIMAL')))
    prob_map = {} if predictions_df.empty else dict(
        zip(predictions_df['service_name'],
            predictions_df.get('incident_probability_percent', 'N/A')))

    # FIX: use annotations for directional arrows instead of plain lines
    annotations = []
    for s, t in G_f.edges():
        if s in pos and t in pos:
            x0, y0 = pos[s]
            x1, y1 = pos[t]
            annotations.append(dict(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor='#475569',
                opacity=0.7,
            ))

    traces = []
    for risk, nodes in {
        r: [n for n in G_f.nodes() if n in pos and risk_map.get(n, 'MINIMAL') == r]
        for r in ['HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
    }.items():
        if not nodes:
            continue
        color = RISK_COLORS[risk]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode='markers+text' if labels else 'markers',
            marker=dict(size=28, color=color,
                        line=dict(width=2, color='#0F172A'), opacity=0.95),
            text=nodes if labels else [''],
            textposition='top center',
            textfont=dict(size=9, color='#F1F5F9'),
            hovertemplate=[
                f"<b>{n}</b><br>Risk: {risk}<br>Prob: {prob_map.get(n,'N/A')}<extra></extra>"
                for n in nodes
            ],
            name=risk, showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**base_layout(
        height=640, showlegend=True, hovermode='closest',
        annotations=annotations,
        title=dict(text="Service Dependency Network  ·  arrows show caller → callee direction",
                   font=dict(color='#94A3B8', size=11)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    critical = (list(nx.articulation_points(G_f.to_undirected()))
                if G_f.number_of_nodes() > 1 else [])
    avg = (sum(dict(G_f.degree()).values()) / G_f.number_of_nodes()
           if G_f.number_of_nodes() > 0 else 0)
    c1.metric("Services",            G_f.number_of_nodes())
    c2.metric("Dependencies",        G_f.number_of_edges())
    c3.metric("Articulation Points", len(critical))
    c4.metric("Avg Connections",     f"{avg:.1f}")

    if critical:
        st.markdown(f"""
<div class="alert a-medium">
  <div class="a-title">Articulation Points Detected</div>
  <div class="a-body">Failure of these services causes network partition:</div>
  <span class="a-rec">↳ {", ".join(critical)}</span>
</div>""", unsafe_allow_html=True)


def render_predictions_page(predictions_df):
    page_header("ML Incident Predictions",
                "Trained on Bank of Anthos & TeaStore · Validated on Online Boutique")

    if predictions_df.empty:
        st.warning("No prediction data available.")
        return

    c1, c2 = st.columns([2, 1])
    risk_filter = c1.multiselect("Filter by Risk",
                                  ['HIGH', 'MEDIUM', 'LOW', 'MINIMAL'],
                                  default=['HIGH', 'MEDIUM', 'LOW', 'MINIMAL'])
    sort_by = c2.selectbox("Sort by", [
        "Incident Probability", "Recovery Time", "Affected Services", "Service Name"
    ])

    filtered = predictions_df[predictions_df['risk_level'].isin(risk_filter)].copy()

    affected_col = 'affected_count' if 'affected_count' in filtered.columns \
                   else 'estimated_affected_services'

    sort_map = {
        "Incident Probability": ('incident_probability',        False),
        "Recovery Time":        ('estimated_recovery_minutes',  False),
        "Affected Services":    (affected_col,                  False),
        "Service Name":         ('service_name',                True),
    }
    col_s, asc = sort_map[sort_by]
    if col_s in filtered.columns:
        filtered = filtered.sort_values(col_s, ascending=asc)

    cols = ['service_name', 'incident_probability_percent', 'risk_level',
            'severity', affected_col, 'estimated_recovery_minutes', 'recommendation']
    avail = [c for c in cols if c in filtered.columns]
    styled = filtered[avail].style
    if 'risk_level' in avail:
        styled = styled.map(style_risk, subset=['risk_level'])
    if 'incident_probability_percent' in avail:
        styled = styled.map(style_prob, subset=['incident_probability_percent'])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        sec("Incident Probability Distribution")
        fig = px.histogram(filtered, x='incident_probability', nbins=15,
                           color='risk_level', color_discrete_map=RISK_COLORS)
        fig.update_layout(**base_layout(height=320))
        fig.update_traces(marker_line_width=0.5, marker_line_color='#0F172A')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sec("Recovery Time — Top 10")
        if 'estimated_recovery_minutes' in filtered.columns:
            fig = px.bar(
                filtered.nlargest(10, 'estimated_recovery_minutes'),
                x='service_name', y='estimated_recovery_minutes',
                color='risk_level', color_discrete_map=RISK_COLORS,
            )
            fig.update_layout(**base_layout(height=320, showlegend=False))
            fig.update_xaxes(tickangle=-35)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)


def render_analytics_page(predictions_df, features_df, G):
    page_header("Advanced Analytics",
                "Network centrality · structural metrics · risk correlation")

    if predictions_df.empty:
        st.warning("No data available.")
        return

    if G.number_of_nodes() > 0:
        sec("Network Centrality Metrics")
        btwn = nx.betweenness_centrality(G)
        clos = nx.closeness_centrality(G)
        pr   = nx.pagerank(G)

        rows = []
        for node in G.nodes():
            r = predictions_df[predictions_df['service_name'] == node]['risk_level'].values
            rows.append({
                'Service':     node,
                'Betweenness': round(btwn.get(node, 0), 4),
                'Closeness':   round(clos.get(node, 0), 4),
                'PageRank':    round(pr.get(node, 0),   4),
                'Risk':        r[0] if len(r) > 0 else 'MINIMAL',
            })
        df = pd.DataFrame(rows)

        c1, c2, c3 = st.columns(3)
        for col, metric, label in zip(
            [c1, c2, c3],
            ['Betweenness', 'Closeness', 'PageRank'],
            ['Top 5 · Betweenness', 'Top 5 · Closeness', 'Top 5 · PageRank'],
        ):
            with col:
                st.markdown(f"**{label}**")
                top = df.nlargest(5, metric)[['Service', metric, 'Risk']]
                st.dataframe(top.style.map(style_risk, subset=['Risk']),
                             hide_index=True, use_container_width=True)

        st.divider()
        sec("Betweenness vs PageRank")
        fig = px.scatter(df, x='Betweenness', y='PageRank',
                         color='Risk', color_discrete_map=RISK_COLORS,
                         hover_name='Service', size=[12] * len(df), size_max=12)
        fig.update_layout(**base_layout(height=380))
        fig.update_traces(marker=dict(line=dict(width=1.5, color='#0F172A')))
        st.plotly_chart(fig, use_container_width=True)


def format_ai_response(text: str) -> str:
    import re
    emoji_pattern = re.compile(
        u"[\U0001F300-\U0001F9FF]|[\U0001F600-\U0001F64F]|"
        u"[\U0001F680-\U0001F6FF]|[\U00002600-\U000027BF]|"
        u"[\U0000FE00-\U0000FE0F]|[\u2700-\u27BF]",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'^(\s*)[◆◇▸▹⮕➡►]+\s*', r'\1- ', text, flags=re.MULTILINE)

    lines = text.split('\n')
    sections = []
    current_section = {'title': None, 'content': []}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('###') or line.startswith('##'):
            if current_section['title'] or current_section['content']:
                sections.append(current_section)
            current_section = {'title': line.replace('#', '').strip(), 'content': []}
        elif line.isupper() and len(line) > 5 and ':' in line:
            if current_section['title'] or current_section['content']:
                sections.append(current_section)
            current_section = {'title': line.replace(':', '').strip(), 'content': []}
        else:
            current_section['content'].append(line)

    if current_section['title'] or current_section['content']:
        sections.append(current_section)

    if not sections:
        sections = [{'title': 'Analysis', 'content': text.split('\n')}]

    html_parts = []
    for section in sections:
        title = section['title'] or 'Response'
        formatted_content = []
        in_list = False

        for line in section['content']:
            line = line.strip()
            if not line:
                if in_list:
                    formatted_content.append('</ul>')
                    in_list = False
                continue
            if ':' in line and not line.startswith('-'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    lbl, val = parts
                    formatted_content.append(
                        f'<div class="response-metric">'
                        f'<span class="metric-label">{lbl.strip()}:</span>'
                        f'<span class="metric-value">{val.strip()}</span>'
                        f'</div>'
                    )
                    continue
            if line.startswith('-') or line.startswith('•'):
                if not in_list:
                    formatted_content.append('<ul>')
                    in_list = True
                formatted_content.append(f'<li>{line[1:].strip()}</li>')
            else:
                if in_list:
                    formatted_content.append('</ul>')
                    in_list = False
                formatted_content.append(f'<p>{line}</p>')

        if in_list:
            formatted_content.append('</ul>')

        html_parts.append(f"""
<div class="response-card">
    <div class="response-card-title">{title}</div>
    <div class="response-card-content">{''.join(formatted_content)}</div>
</div>""")

    return '\n'.join(html_parts)


def render_ai_assistant():
    page_header("AI Architecture Assistant",
                "Ask about dependencies, risk, and impact — in plain English")

    st.markdown("""
<div class="alert a-info">
  <div class="a-title">Your System Advisor</div>
  <div class="a-body">Natural language access to service dependencies, failures, and risk — no code needed.</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(format_ai_response(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your architecture..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from nlp_query_handler import NLPQueryHandler
                    response = NLPQueryHandler(
                        PREDICTIONS_PATH, FEATURES_PATH,
                        DEPENDENCIES_PATH, GEMINI_API_KEY,
                    ).query(prompt)
                except Exception as e:
                    response = f"AI Assistant unavailable: {e}"

            st.markdown(format_ai_response(response), unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.messages:
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()


def main():
    predictions_df = load_predictions()
    dep_data       = load_dependencies()
    features_df    = load_features()
    G              = build_graph(dep_data, predictions_df)
    page           = render_sidebar(predictions_df, dep_data, G)

    if   page == "📊 Dashboard":       render_dashboard(predictions_df, dep_data, G)
    elif page == "🔗 Dependency Graph": render_graph_page(G, predictions_df)
    elif page == "🤖 ML Predictions":  render_predictions_page(predictions_df)
    elif page == "📈 Analytics":       render_analytics_page(predictions_df, features_df, G)
    elif page == "💬 AI Assistant":    render_ai_assistant()


if __name__ == '__main__':
    main()