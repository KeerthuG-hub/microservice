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
import re

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
.a-title  { font-weight: 700; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.a-high   .a-title { color: #F87171; }
.a-medium .a-title { color: #FBBF24; }
.a-ok     .a-title { color: #34D399; }
.a-body   { font-size: 0.875rem; color: #CBD5E1; line-height: 1.55; margin-top: 0.25rem; }
.a-rec    { font-size: 0.8rem; color: #64748B; margin-top: 0.3rem; display: block; }

/* ── AI Response Cards ── */
.ai-response-wrap { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.25rem; }

.ai-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    position: relative;
}
.ai-card-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #60A5FA;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.ai-card-title .card-icon { font-size: 0.85rem; }
.ai-bullet-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}
.ai-bullet-list li {
    font-size: 0.85rem;
    color: #CBD5E1;
    line-height: 1.5;
    padding-left: 1.1rem;
    position: relative;
}
.ai-bullet-list li::before {
    content: '›';
    position: absolute;
    left: 0;
    color: #60A5FA;
    font-weight: 700;
}
.ai-bullet-list li strong,
.ai-bullet-list li b {
    color: #F1F5F9;
    font-weight: 600;
}
.ai-card-plain {
    font-size: 0.875rem;
    color: #CBD5E1;
    line-height: 1.6;
}

/* risk badge */
.rbadge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    vertical-align: middle;
}
.rbadge-HIGH   { background:#2D1515; color:#F87171; border:1px solid #F87171; }
.rbadge-MEDIUM { background:#2D2200; color:#FBBF24; border:1px solid #FBBF24; }
.rbadge-LOW    { background:#0D2318; color:#34D399; border:1px solid #34D399; }
.rbadge-MINIMAL{ background:#0F1F35; color:#60A5FA; border:1px solid #60A5FA; }

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


@st.cache_data
def load_predictions():
    if not Path(PREDICTIONS_PATH).exists():
        return pd.DataFrame()
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if 'incident_probability' not in df.columns and 'incident_probability_percent' in df.columns:
        df['incident_probability'] = (
            df['incident_probability_percent'].str.replace('%','',regex=False).astype(float) / 100
        )
    return df

@st.cache_data
def load_dependencies():
    if not Path(DEPENDENCIES_PATH).exists(): return {}
    with open(DEPENDENCIES_PATH) as f: return json.load(f)

@st.cache_data
def load_features():
    if not Path(FEATURES_PATH).exists(): return pd.DataFrame()
    return pd.read_csv(FEATURES_PATH)

def build_graph(dep_data, predictions_df):
    G = nx.DiGraph()
    risk_map = {} if predictions_df.empty else dict(zip(predictions_df['service_name'], predictions_df.get('risk_level', 'MINIMAL')))
    svc_names = {s['name'] for s in dep_data.get('services', []) if not s.get('is_infrastructure', False)}
    for dep in dep_data.get('confirmed', []) + dep_data.get('probable', []):
        src, tgt = dep.get('from',''), dep.get('to','')
        if not src or not tgt or src.startswith('__') or tgt.startswith('__'): continue
        if tgt not in svc_names: continue
        G.add_node(src, risk=risk_map.get(src,'MINIMAL'))
        G.add_node(tgt, risk=risk_map.get(tgt,'MINIMAL'))
        G.add_edge(src, tgt, dep_type=dep.get('type','unknown'), confidence=dep.get('confidence',0))
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            if row['service_name'] not in G:
                G.add_node(row['service_name'], risk=row.get('risk_level','MINIMAL'))
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
        v = float(str(val).replace('%',''))
        c = '#F87171' if v>=70 else '#FBBF24' if v>=40 else '#34D399'
        return f'color:{c};font-weight:600'
    except: return ''


# ─────────────────────────────────────────────
#  AI Response → Structured Cards Formatter
# ─────────────────────────────────────────────

SECTION_ICONS = {
    'impact':      ('⚡', 'Impact Analysis'),
    'risk':        ('🔴', 'Risk Assessment'),
    'affected':    ('🔗', 'Affected Services'),
    'recommend':   ('✅', 'Recommendations'),
    'recovery':    ('🕐', 'Recovery'),
    'deploy':      ('🚀', 'Deployment'),
    'critical':    ('⚠️', 'Critical Points'),
    'summary':     ('◈',  'Summary'),
    'service':     ('🧩', 'Services'),
    'depend':      ('🔗', 'Dependencies'),
    'default':     ('◈',  'Analysis'),
}

def _detect_card_type(heading: str) -> tuple:
    h = heading.lower()
    for key, (icon, label) in SECTION_ICONS.items():
        if key in h:
            return icon, label
    return SECTION_ICONS['default']

def _risk_badge(text: str) -> str:
    """Inline-replace risk level keywords with styled badges."""
    for level in ['HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
        text = text.replace(level, f'<span class="rbadge rbadge-{level}">{level}</span>')
    return text

def _bullets_html(items: list) -> str:
    lis = ''
    for item in items:
        item = item.strip().lstrip('-•*›').strip()
        if not item:
            continue
        item = _risk_badge(item)
        lis += f'<li>{item}</li>'
    return f'<ul class="ai-bullet-list">{lis}</ul>'

def _sentences_to_bullets(paragraph: str) -> list:
    """Split a paragraph into sentence-level bullets."""
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [s for s in sentences if len(s) > 10]

def format_ai_response(raw: str) -> str:
    """
    Convert raw LLM text (paragraphs or loose markdown) into
    a series of structured dark cards with bullet lists.
    """
    if not raw or not raw.strip():
        return '<div class="ai-card"><div class="ai-card-plain">No response.</div></div>'

    cards_html = '<div class="ai-response-wrap">'

    # ── Try to split on markdown headings (##, ###, bold lines, or ALL-CAPS lines) ──
    heading_pattern = re.compile(
        r'^(?:#{1,4}\s+|(?=[A-Z][A-Z\s]{3,}:?\s*$))(.+)$', re.MULTILINE
    )

    # Split by lines that look like headings
    lines = raw.split('\n')
    sections = []   # list of (heading, [content_lines])
    current_heading = None
    current_body: list = []

    for line in lines:
        stripped = line.strip()
        # Detect heading: markdown # or **bold standalone** or ALL CAPS label
        is_heading = (
            re.match(r'^#{1,4}\s+\S', stripped)
            or re.match(r'^\*\*[^*]+\*\*\s*:?\s*$', stripped)
            or re.match(r'^[A-Z][A-Z &/]{3,}:?\s*$', stripped)
        )
        if is_heading:
            if current_heading is not None or current_body:
                sections.append((current_heading, current_body))
            current_heading = re.sub(r'[#*]', '', stripped).strip().rstrip(':')
            current_body = []
        else:
            current_body.append(line)

    # Flush last section
    if current_heading is not None or current_body:
        sections.append((current_heading, current_body))

    # If no headings were found, treat whole response as one card and auto-section
    if len(sections) == 1 and sections[0][0] is None:
        body_text = '\n'.join(sections[0][1]).strip()
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', body_text) if p.strip()]

        if len(paragraphs) == 1:
            # Single paragraph → sentence bullets in one card
            bullets = _sentences_to_bullets(paragraphs[0])
            icon, label = SECTION_ICONS['summary']
            cards_html += f'''
<div class="ai-card">
  <div class="ai-card-title"><span class="card-icon">{icon}</span>{label}</div>
  {_bullets_html(bullets)}
</div>'''
        else:
            # Multiple paragraphs → one card per paragraph
            for i, para in enumerate(paragraphs):
                # Check if para starts with an implicit label
                first_line = para.split('\n')[0]
                icon, label = _detect_card_type(first_line)
                if i == 0:
                    label = 'Summary'
                    icon  = '◈'
                bullets = _sentences_to_bullets(para)
                cards_html += f'''
<div class="ai-card">
  <div class="ai-card-title"><span class="card-icon">{icon}</span>{label}</div>
  {_bullets_html(bullets)}
</div>'''
        cards_html += '</div>'
        return cards_html

    # ── Render each heading-section as a card ──
    for heading, body_lines in sections:
        body_text = '\n'.join(body_lines).strip()
        if not body_text and not heading:
            continue

        icon, label = _detect_card_type(heading or '')
        display_title = heading if heading else label

        # Collect bullet items — explicit list lines OR split sentences
        explicit_bullets = []
        prose_lines = []
        for bl in body_lines:
            bl_s = bl.strip()
            if re.match(r'^[-•*›]\s+', bl_s) or re.match(r'^\d+\.\s+', bl_s):
                explicit_bullets.append(re.sub(r'^[-•*›\d.]+\s*', '', bl_s))
            elif bl_s:
                prose_lines.append(bl_s)

        if explicit_bullets:
            # Mix: use explicit bullets, fold prose into them
            all_bullets = explicit_bullets
            if prose_lines:
                all_bullets = _sentences_to_bullets(' '.join(prose_lines)) + all_bullets
        else:
            # Pure prose → sentence-split
            all_bullets = _sentences_to_bullets(' '.join(prose_lines))

        if not all_bullets and body_text:
            all_bullets = [body_text]

        cards_html += f'''
<div class="ai-card">
  <div class="ai-card-title"><span class="card-icon">{icon}</span>{display_title}</div>
  {_bullets_html(all_bullets)}
</div>'''

    cards_html += '</div>'
    return cards_html


def render_ai_message(content: str, role: str):
    """Render a chat message. Assistant messages get card formatting."""
    with st.chat_message(role):
        if role == 'assistant':
            st.markdown(format_ai_response(content), unsafe_allow_html=True)
        else:
            st.markdown(content)


# ─────────────────────────────────────────────

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
            high = len(predictions_df[predictions_df['risk_level']=='HIGH'])
            st.metric("High Risk Services", high,
                      delta=f"-{high}" if high>0 else None, delta_color="inverse")

        confirmed = len(dep_data.get('confirmed',[]))
        probable  = len(dep_data.get('probable',[]))
        st.metric("Dependencies", confirmed + probable)

        st.divider()
        st.markdown("### Dependency Stats")
        st.write(f"Confirmed: **{confirmed}**")
        st.write(f"Probable: **{probable}**")
        st.write(f"Observability: **{len(dep_data.get('observability',[]))}**")

        st.divider()
        st.caption("Architecture Intelligence Platform")
        return page


def render_dashboard(predictions_df, dep_data, G):
    page_header("Microservice Dependency Analyzer", "Architecture Intelligence Platform — Online Boutique")

    if predictions_df.empty:
        st.error("No prediction data. Run the prediction pipeline first.")
        return

    c1,c2,c3,c4,c5 = st.columns(5)
    high   = len(predictions_df[predictions_df['risk_level']=='HIGH'])
    medium = len(predictions_df[predictions_df['risk_level']=='MEDIUM'])
    deps   = len(dep_data.get('confirmed',[])) + len(dep_data.get('probable',[]))
    crits  = len(list(nx.articulation_points(G.to_undirected())) if G.number_of_nodes()>1 else [])
    c1.metric("Services",        len(predictions_df))
    c2.metric("High Risk",       high,   delta=f"-{high}"   if high>0   else None, delta_color="inverse")
    c3.metric("Medium Risk",     medium, delta=f"-{medium}" if medium>0 else None, delta_color="inverse")
    c4.metric("Dependencies",    deps)
    c5.metric("Critical Points", crits)

    st.divider()

    sec("Critical Alerts")
    high_rows = predictions_df[predictions_df['risk_level']=='HIGH']

    if not high_rows.empty:
        for _, r in high_rows.iterrows():
            st.markdown(f"""
<div class="alert a-high">
  <div class="a-title">Critical — {r['service_name']}</div>
  <div class="a-body">
    Probability: <strong>{r.get('incident_probability_percent','N/A')}</strong> ·
    Severity: <strong>{r.get('severity','N/A')}</strong> ·
    Affected: <strong>~{r.get('estimated_affected_services','N/A')} services</strong> ·
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
        rc.columns = ['Risk Level','Count']
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
        cols_show = [c for c in ['service_name','incident_probability_percent','risk_level','severity'] if c in predictions_df.columns]
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
            'Category': ['Confirmed','Probable','Uncertain'],
            'Count': [len(dep_data.get('confirmed',[])),
                      len(dep_data.get('probable',[])),
                      len(dep_data.get('uncertain',[]))]
        })
        fig = px.bar(conf_df, x='Category', y='Count', color='Category', color_discrete_map=DEP_COLORS)
        fig.update_layout(**base_layout(title_text='Confidence Distribution', height=260, showlegend=False))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tc = {}
        for dep in dep_data.get('confirmed',[]):
            t = dep.get('type','unknown'); tc[t] = tc.get(t,0)+1
        if tc:
            fig = go.Figure(go.Bar(x=list(tc.keys()), y=list(tc.values()),
                                   marker_color='#60A5FA', marker_line_width=0))
            fig.update_layout(**base_layout(title_text='Dependency Types', height=260, showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

    with c3:
        sec("Network Metrics")
        if G.number_of_nodes() > 0:
            avg = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.metric("Avg Dependencies", f"{avg:.1f}")
            st.metric("Graph Density",    f"{nx.density(G):.1%}")
            st.metric("Components",       nx.number_strongly_connected_components(G))


def render_graph_page(G, predictions_df):
    page_header("Dependency Graph", "Node color = ML-predicted risk level")

    if not G.nodes():
        st.warning("No graph data available.")
        return

    c1,c2,c3 = st.columns([2,2,3])
    layout = c1.selectbox("Layout", ["Spring","Circular","Kamada-Kawai"])
    labels = c2.checkbox("Show Labels", True)
    risks  = c3.multiselect("Filter Risk", ['HIGH','MEDIUM','LOW','MINIMAL'],
                             default=['HIGH','MEDIUM','LOW','MINIMAL'])

    G_f = G.subgraph([n for n in G.nodes() if G.nodes[n].get('risk','MINIMAL') in risks]).copy()

    if layout=="Spring":      pos = nx.spring_layout(G_f, seed=42, k=2.5, iterations=50)
    elif layout=="Circular":  pos = nx.circular_layout(G_f)
    else:                     pos = nx.kamada_kawai_layout(G_f)

    risk_map = {} if predictions_df.empty else dict(zip(predictions_df['service_name'], predictions_df.get('risk_level','MINIMAL')))
    prob_map = {} if predictions_df.empty else dict(zip(predictions_df['service_name'], predictions_df.get('incident_probability_percent','N/A')))

    ex, ey = [], []
    for s,t in G_f.edges():
        if s in pos and t in pos:
            x0,y0=pos[s]; x1,y1=pos[t]
            ex+=[x0,x1,None]; ey+=[y0,y1,None]

    traces = [go.Scatter(x=ex,y=ey,mode='lines',
                         line=dict(width=1,color='#334155'),hoverinfo='none',showlegend=False)]

    for risk, nodes in {r:[n for n in G_f.nodes() if n in pos and risk_map.get(n,'MINIMAL')==r]
                        for r in ['HIGH','MEDIUM','LOW','MINIMAL']}.items():
        if not nodes: continue
        color = RISK_COLORS[risk]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode='markers+text' if labels else 'markers',
            marker=dict(size=24, color=color, line=dict(width=2,color='#0F172A'), opacity=0.95),
            text=nodes if labels else [''], textposition='top center',
            textfont=dict(size=9, color='#F1F5F9'),
            hovertemplate=[f"<b>{n}</b><br>Risk: {risk}<br>Prob: {prob_map.get(n,'N/A')}<extra></extra>" for n in nodes],
            name=risk, showlegend=True
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**base_layout(height=640, showlegend=True, hovermode='closest',
        title=dict(text="Service Dependency Network", font=dict(color='#F1F5F9',size=13)),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    critical = list(nx.articulation_points(G_f.to_undirected())) if G_f.number_of_nodes()>1 else []
    avg = sum(dict(G_f.degree()).values())/G_f.number_of_nodes() if G_f.number_of_nodes()>0 else 0
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
    page_header("ML Incident Predictions", "Trained on Bank of Anthos & TeaStore · Validated on Online Boutique")

    if predictions_df.empty:
        st.warning("No prediction data available.")
        return

    c1,c2 = st.columns([2,1])
    risk_filter = c1.multiselect("Filter by Risk", ['HIGH','MEDIUM','LOW','MINIMAL'],
                                  default=['HIGH','MEDIUM','LOW','MINIMAL'])
    sort_by = c2.selectbox("Sort by", ["Incident Probability","Recovery Time","Affected Services","Service Name"])

    filtered = predictions_df[predictions_df['risk_level'].isin(risk_filter)].copy()
    sort_map = {"Incident Probability":('incident_probability',False),
                "Recovery Time":       ('estimated_recovery_minutes',False),
                "Affected Services":   ('estimated_affected_services',False),
                "Service Name":        ('service_name',True)}
    col_s, asc = sort_map[sort_by]
    if col_s in filtered.columns:
        filtered = filtered.sort_values(col_s, ascending=asc)

    cols = ['service_name','incident_probability_percent','risk_level',
            'severity','estimated_affected_services','estimated_recovery_minutes','recommendation']
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
            fig = px.bar(filtered.nlargest(10,'estimated_recovery_minutes'),
                         x='service_name', y='estimated_recovery_minutes',
                         color='risk_level', color_discrete_map=RISK_COLORS)
            fig.update_layout(**base_layout(height=320, showlegend=False))
            fig.update_xaxes(tickangle=-35)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)


def render_analytics_page(predictions_df, features_df, G):
    page_header("Advanced Analytics", "Network centrality · structural metrics · risk correlation")

    if predictions_df.empty:
        st.warning("No data available."); return

    if G.number_of_nodes() > 0:
        sec("Network Centrality Metrics")
        btwn = nx.betweenness_centrality(G)
        clos = nx.closeness_centrality(G)
        pr   = nx.pagerank(G)

        rows = []
        for node in G.nodes():
            r = predictions_df[predictions_df['service_name']==node]['risk_level'].values
            rows.append({'Service':node,
                         'Betweenness':round(btwn.get(node,0),4),
                         'Closeness':  round(clos.get(node,0),4),
                         'PageRank':   round(pr.get(node,0),4),
                         'Risk':       r[0] if len(r)>0 else 'MINIMAL'})
        df = pd.DataFrame(rows)

        c1,c2,c3 = st.columns(3)
        for col, metric, label in zip([c1,c2,c3],
                ['Betweenness','Closeness','PageRank'],
                ['Top 5 · Betweenness','Top 5 · Closeness','Top 5 · PageRank']):
            with col:
                st.markdown(f"**{label}**")
                top = df.nlargest(5, metric)[['Service',metric,'Risk']]
                st.dataframe(top.style.map(style_risk, subset=['Risk']),
                             hide_index=True, use_container_width=True)

        st.divider()
        sec("Betweenness vs PageRank")
        fig = px.scatter(df, x='Betweenness', y='PageRank',
                         color='Risk', color_discrete_map=RISK_COLORS,
                         hover_name='Service', size=[12]*len(df), size_max=12)
        fig.update_layout(**base_layout(height=380))
        fig.update_traces(marker=dict(line=dict(width=1.5,color='#0F172A')))
        st.plotly_chart(fig, use_container_width=True)


def render_ai_assistant():
    page_header("AI Architecture Assistant", "Natural language interface for architecture analysis")

    examples = ["What breaks if checkoutservice fails?","Which services are most critical?",
                "Is it safe to deploy frontend?","What if I modify checkoutservice?",
                "Show all HIGH risk services","Recovery time for shippingservice?"]

    sec("Common Questions")
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i%3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.prefill = ex

    st.divider()

    if "messages" not in st.session_state: st.session_state.messages = []

    # Render existing messages
    for msg in st.session_state.messages:
        render_ai_message(msg["content"], msg["role"])

    prefill = st.session_state.pop("prefill", None)
    if prompt := st.chat_input("Ask about your architecture...") or prefill:
        render_ai_message(prompt, "user")
        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from nlp_query_handler import NLPQueryHandler
                    raw_response = NLPQueryHandler(PREDICTIONS_PATH, FEATURES_PATH,
                                              DEPENDENCIES_PATH, GEMINI_API_KEY).query(prompt)
                except Exception as e:
                    raw_response = f"AI Assistant unavailable: {e}"

            # Render as structured cards
            st.markdown(format_ai_response(raw_response), unsafe_allow_html=True)

        st.session_state.messages.append({"role":"assistant","content":raw_response})

    if st.session_state.messages:
        if st.button("Clear Conversation"):
            st.session_state.messages = []; st.rerun()


def main():
    predictions_df = load_predictions()
    dep_data       = load_dependencies()
    features_df    = load_features()
    G              = build_graph(dep_data, predictions_df)
    page           = render_sidebar(predictions_df, dep_data, G)

    if   page == "📊 Dashboard":        render_dashboard(predictions_df, dep_data, G)
    elif page == "🔗 Dependency Graph":  render_graph_page(G, predictions_df)
    elif page == "🤖 ML Predictions":   render_predictions_page(predictions_df)
    elif page == "📈 Analytics":        render_analytics_page(predictions_df, features_df, G)
    elif page == "💬 AI Assistant":     render_ai_assistant()

if __name__ == '__main__':
    main()