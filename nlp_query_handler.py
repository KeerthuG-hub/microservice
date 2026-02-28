"""
NLP Query Handler v2 — Hybrid Architecture
==========================================

PIPELINE:
  User Query
      │
      ▼
  [GEMINI] Intent Classification
      │  - Identifies intent type
      │  - Extracts service name
      │  - Returns structured JSON (tiny token cost)
      │
      ▼
  Context Selector
      │  - Pulls ONLY relevant context for that intent
      │  - e.g. blast_radius query → ACCURATE graph traversal
      │  - NOT dumping all 20 services into prompt
      │
      ▼
  [GROQ LLaMA3] Reasoning & Response
      │  - Gets lean, focused context (<500 tokens typically)
      │  - Produces natural language answer fast + cheap
      │
      ▼
  Response to User

WHY THIS WORKS:
  - Gemini is great at structured extraction (JSON intent) — short task
  - Groq LLaMA3 is fast + cheap for reasoning over focused context
  - Token reduction to Groq: ~70% vs sending full context always

GRAPH EDGE DIRECTION:
  Edges go  caller → callee  (frontend → checkoutservice means frontend CALLS checkout).
  Therefore:
    blast_radius  → predecessors(service) = direct callers that break
                  → ancestors(service)    = all upstream services affected
    dependency    → successors(service)   = services it calls
    whatif        → ancestors(service)    = who is impacted upstream
"""

import os
import json
import re
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ─────────────────────────────────────────────
# Intent Schema
# ─────────────────────────────────────────────

INTENT_TYPES = [
    "blast_radius",      # "what breaks if X fails?"
    "dependency_query",  # "what does X depend on?"
    "risk_assessment",   # "is it safe to deploy X?"
    "risk_ranking",      # "which services are riskiest?"
    "whatif_simulation", # "what if I make a large change to X?"
    "health_check",      # "what is the health of X?"
    "general",           # anything else
]


# ─────────────────────────────────────────────
# Main Handler
# ─────────────────────────────────────────────

class NLPQueryHandler:
    """
    Gemini classifies intent → Groq reasons over minimal focused context.
    Falls back gracefully if either API is unavailable.
    """

    def __init__(
        self,
        predictions_path: str = "data/outputs/predictions.json",
        features_path: str = "data/features/online-boutique_features.csv",
        dependencies_path: str = "data/outputs/online-boutique_latest.json",
        gemini_api_key: str = None,
        groq_api_key: str = None,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        self.predictions: Dict = {}
        self.features: Dict = {}
        self.graph = nx.DiGraph()
        self.context_loaded = False

        self._load_context(predictions_path, features_path, dependencies_path)

    # ─────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────

    def _load_context(self, predictions_path, features_path, dependencies_path):
        try:
            if Path(predictions_path).exists():
                with open(predictions_path) as f:
                    preds_list = json.load(f)
                self.predictions = {p["service_name"]: p for p in preds_list}
                print(f"✓ Loaded {len(self.predictions)} predictions")

            if Path(features_path).exists():
                df = pd.read_csv(features_path)
                self.features = df.set_index("service_name").to_dict("index")
                print(f"✓ Loaded {len(self.features)} service features")

            if Path(dependencies_path).exists():
                with open(dependencies_path) as f:
                    dep_data = json.load(f)

                # Data-driven filtering — uses tags already in run.py output
                svc_names = {
                    s["name"] for s in dep_data.get("services", [])
                    if not s.get("is_infrastructure", False)
                }

                NOISE_SOURCES  = {"string_cooccurrence", "git_change_coupling"}
                SKIP_DEP_TYPES = {"external", "infrastructure", "observability"}

                all_deps = (
                    dep_data.get("confirmed", [])
                    + dep_data.get("probable", [])
                )

                for dep in all_deps:
                    src = dep.get("from", "")
                    tgt = dep.get("to", "")

                    if not src or not tgt:
                        continue
                    if src.startswith("__") or tgt.startswith("__"):
                        continue
                    if dep.get("confidence", 0) < 0.65:
                        continue
                    if dep.get("dep_type", dep.get("type", "")) in SKIP_DEP_TYPES:
                        continue

                    sources = dep.get("sources", [dep.get("source", "")])
                    if isinstance(sources, str):
                        sources = [sources]
                    if any(s in NOISE_SOURCES for s in sources):
                        continue

                    # Both endpoints must be real business services
                    if src not in svc_names or tgt not in svc_names:
                        continue

                    self.graph.add_edge(
                        src, tgt,
                        dep_type=dep.get("type", "unknown"),
                        confidence=dep.get("confidence", 0),
                    )

                print(
                    f"✓ Graph: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges"
                )

            self.context_loaded = True

        except Exception as e:
            print(f"✗ Context loading error: {e}")
            import traceback; traceback.print_exc()

    # ─────────────────────────────────────────
    # STEP 1 — Gemini Intent Classification
    # ─────────────────────────────────────────

    def _classify_intent_gemini(self, question: str) -> Dict:
        """
        Tiny Gemini call: classify intent + extract service name.
        Returns structured dict. Token cost: ~200 in, ~50 out.
        """
        service_list = list(self.graph.nodes())[:25]

        prompt = f"""You are classifying a microservice query for routing.

Available services: {service_list}

Intent types:
- blast_radius      → "what breaks/fails if X goes down?"
- dependency_query  → "what does X depend on / call?"
- risk_assessment   → "is it safe to deploy X?" or "what is the risk of X?"
- risk_ranking      → "which services are riskiest / most critical?"
- whatif_simulation → "what if I make a change to X?"
- health_check      → "what is the health/status of X?"
- general           → anything else

User question: "{question}"

Respond ONLY with a JSON object, no markdown, no explanation:
{{
  "intent": "<one of the intent types above>",
  "service": "<exact service name from list, or null if none mentioned>",
  "confidence": <0.0-1.0>
}}"""

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            print(f"Intent: {result}")
            return result
        except Exception as e:
            print(f"Gemini classification failed: {e} — falling back to regex")
            return self._classify_intent_regex(question)

    def _classify_intent_regex(self, question: str) -> Dict:
        """Regex fallback when Gemini is unavailable."""
        q = question.lower()

        intent = "general"
        if any(w in q for w in ["fail", "break", "down", "crash", "impact", "affect"]):
            intent = "blast_radius"
        elif any(w in q for w in ["riskiest", "most critical", "most dangerous", "worst", "ranking"]):
            intent = "risk_ranking"
        elif any(w in q for w in ["what if", "modify", "change", "deploy", "update"]):
            intent = "whatif_simulation"
        elif any(w in q for w in ["safe", "risk of", "deploy"]):
            intent = "risk_assessment"
        elif any(w in q for w in ["depend", "call", "connect", "upstream", "downstream"]):
            intent = "dependency_query"
        elif any(w in q for w in ["health", "status", "latency", "error rate"]):
            intent = "health_check"

        service = self._extract_service_name(q)
        return {"intent": intent, "service": service, "confidence": 0.7}

    def _extract_service_name(self, question: str) -> Optional[str]:
        for svc in self.graph.nodes():
            if svc.lower() in question:
                return svc
        for svc in self.graph.nodes():
            short = svc.replace("service", "").replace("-", "").strip()
            if short and short in question:
                return svc
        return None

    # ─────────────────────────────────────────
    # STEP 2 — Minimal Context Selector
    # ─────────────────────────────────────────

    def _build_minimal_context(self, intent: str, service: Optional[str]) -> str:
        """
        Build ONLY the context needed for this specific intent.
        Token-saving: Groq gets focused facts, not everything.

        GRAPH DIRECTION:
          Edges are  caller → callee.
          blast_radius uses:
            - predecessors(service) = direct callers (immediately broken)
            - ancestors(service)    = all upstream services affected
        """
        lines = []

        high_risk = [s for s, p in self.predictions.items() if p.get("risk_level") == "HIGH"]
        lines.append(
            f"System: {self.graph.number_of_nodes()} services, "
            f"{self.graph.number_of_edges()} edges. "
            f"HIGH risk services: {high_risk or 'none'}"
        )

        if intent == "blast_radius" and service:
            if service in self.graph:
                # CORRECT BLAST RADIUS CALCULATION
                # Edges go caller → callee, so:
                # - predecessors = direct callers (immediately broken)
                # - ancestors    = all upstream services affected (includes direct callers)
                
                direct_callers = list(self.graph.predecessors(service))
                all_upstream   = list(nx.ancestors(self.graph, service))
                
                # Transitive = upstream services beyond direct callers
                transitive_upstream = [s for s in all_upstream if s not in direct_callers]
                
                # What this service calls (not affected by its failure)
                downstream_deps = list(self.graph.successors(service))
                
                pred = self.predictions.get(service, {})
                
                lines += [
                    f"\n=== SERVICE: {service} ===",
                    f"",
                    f"--- IMMEDIATE IMPACT (Direct Callers Broken) ---",
                    f"Services: {direct_callers or 'none'}",
                    f"Count: {len(direct_callers)}",
                    f"",
                    f"--- CASCADING IMPACT (Transitive Upstream Broken) ---",
                    f"Services: {transitive_upstream or 'none'}",
                    f"Count: {len(transitive_upstream)}",
                    f"Explanation: These depend on the direct callers above.",
                    f"",
                    f"--- DOWNSTREAM SERVICES (Orphaned, Stay Up) ---",
                    f"Services: {downstream_deps or 'none'}",
                    f"Count: {len(downstream_deps)}",
                    f"Status: These services remain operational but receive ZERO traffic from {service}.",
                    f"Impact: No payments, emails, shipments, etc. are processed.",
                    f"",
                    f"--- TOTAL BLAST RADIUS ---",
                    f"Broken upstream: {len(all_upstream)} services",
                    f"Orphaned downstream: {len(downstream_deps)} services",
                    f"Total disruption: {len(all_upstream) + len(downstream_deps)} services affected",
                    f"",
                    f"--- ML RISK ASSESSMENT ---",
                    f"Incident probability: {pred.get('incident_probability_percent', 'N/A')}",
                    f"Risk level: {pred.get('risk_level', 'N/A')}",
                    f"Severity: {pred.get('severity', 'N/A')}",
                    f"Est. recovery time: {pred.get('estimated_recovery_minutes', 'N/A')} minutes",
                    f"",
                    f"--- SRE RECOMMENDATION ---",
                    f"{pred.get('recommendation', 'Deploy with caution and prepare rollback plan.')}",
                ]
                
                # Critical articulation point check
                if service in nx.articulation_points(self.graph.to_undirected()):
                    lines.append(
                        f"\nCRITICAL: {service} is an ARTICULATION POINT. "
                        f"Failure PARTITIONS the graph. Implement failover BEFORE any deployment."
                    )

        elif intent == "dependency_query" and service:
            if service in self.graph:
                # What does this service call? → successors (downstream)
                # Who calls this service?      → predecessors (upstream)
                downstream = list(self.graph.successors(service))
                upstream   = list(self.graph.predecessors(service))
                
                edge_details = []
                for s, t, data in self.graph.edges(data=True):
                    if s == service or t == service:
                        edge_details.append(
                            f"  {s} → {t} [{data.get('dep_type','?')} "
                            f"conf={data.get('confidence',0):.2f}]"
                        )
                
                lines += [
                    f"\n=== SERVICE: {service} ===",
                    f"",
                    f"--- DEPENDENCIES (what it calls) ---",
                    f"Downstream services: {downstream or 'none'}",
                    f"Count: {len(downstream)}",
                    f"",
                    f"--- DEPENDENTS (who calls it) ---",
                    f"Upstream callers: {upstream or 'none'}",
                    f"Count: {len(upstream)}",
                    f"",
                    f"--- EDGE DETAILS ---",
                    "\n".join(edge_details[:15]),
                ]

        elif intent == "risk_assessment" and service:
            pred = self.predictions.get(service, {})
            feat = self.features.get(service, {})
            upstream   = list(self.graph.predecessors(service))
            downstream = list(self.graph.successors(service))
            
            affected_col = pred.get('affected_count',
                          pred.get('estimated_affected_services', 'N/A'))
            
            lines += [
                f"\n=== RISK ASSESSMENT: {service} ===",
                f"",
                f"--- ML PREDICTIONS ---",
                f"Incident probability: {pred.get('incident_probability_percent', 'N/A')}",
                f"Risk level: {pred.get('risk_level', 'N/A')}",
                f"Severity: {pred.get('severity', 'N/A')}",
                f"Estimated affected services: {affected_col}",
                f"Estimated recovery time: {pred.get('estimated_recovery_minutes', 'N/A')} min",
                f"",
                f"--- BLAST RADIUS ---",
                f"Upstream callers (break if this fails): {upstream or 'none'}",
                f"Downstream dependencies (it calls): {downstream or 'none'}",
                f"",
                f"--- OPERATIONAL METRICS ---",
                f"Error rate: {feat.get('error_rate', 'N/A')}",
                f"Latency p99: {feat.get('latency_p99', 'N/A')} ms",
                f"",
                f"--- RECOMMENDATION ---",
                f"{pred.get('recommendation', 'N/A')}",
            ]

        elif intent == "risk_ranking":
            sorted_preds = sorted(
                self.predictions.values(),
                key=lambda x: float(x.get("incident_probability", 0)),
                reverse=True,
            )[:7]
            
            lines.append("\n=== TOP 7 RISKIEST SERVICES ===")
            lines.append("")
            for i, p in enumerate(sorted_preds, 1):
                lines.append(
                    f"{i}. {p['service_name']}: {p.get('incident_probability_percent','N/A')} "
                    f"[{p.get('risk_level','?')}] severity={p.get('severity','?')}"
                )

        elif intent == "whatif_simulation" and service:
            pred = self.predictions.get(service, {})
            # Who would be impacted by a change to this service?
            # = upstream callers (ancestors)
            direct_callers = list(self.graph.predecessors(service))
            all_affected   = list(nx.ancestors(self.graph, service))
            current_prob   = float(pred.get("incident_probability", 0))
            
            lines += [
                f"\n=== WHAT-IF ANALYSIS: {service} ===",
                f"",
                f"--- CURRENT STATE ---",
                f"Current incident probability: {current_prob*100:.1f}%",
                f"Risk level: {pred.get('risk_level', 'N/A')}",
                f"",
                f"--- IMPACT ZONE (who's at risk) ---",
                f"Direct callers at risk: {direct_callers or 'none'}",
                f"All transitively affected upstream: {all_affected or 'none'}",
                f"Total services at risk: {len(all_affected)}",
                f"",
                f"--- DEPLOYMENT HISTORY ---",
                f"Deployment frequency: {pred.get('deployment_frequency', 'N/A')}",
                f"Last incident: {pred.get('last_incident_days', 'N/A')} days ago",
            ]

        elif intent == "health_check" and service:
            feat = self.features.get(service, {})
            pred = self.predictions.get(service, {})
            
            lines += [
                f"\n=== HEALTH CHECK: {service} ===",
                f"",
                f"--- OPERATIONAL METRICS ---",
                f"Health score: {feat.get('health_score', 'N/A')}",
                f"Error rate: {feat.get('error_rate', 'N/A')}",
                f"Latency p99: {feat.get('latency_p99', 'N/A')} ms",
                f"Requests/day: {feat.get('traffic_requests_per_day', 'N/A')}",
                f"Complexity score: {feat.get('complexity_score', 'N/A')}",
                f"",
                f"--- ML RISK LEVEL ---",
                f"Risk level: {pred.get('risk_level', 'N/A')}",
                f"Incident probability: {pred.get('incident_probability_percent', 'N/A')}",
            ]

        else:
            sorted_preds = sorted(
                self.predictions.values(),
                key=lambda x: float(x.get("incident_probability", 0)),
                reverse=True,
            )[:3]
            
            lines.append("\n=== TOP 3 HIGHEST RISK SERVICES ===")
            lines.append("")
            for p in sorted_preds:
                lines.append(
                    f"  {p['service_name']}: {p.get('incident_probability_percent','?')} "
                    f"[{p.get('risk_level','?')}]"
                )
            lines.append(f"\nAll services: {list(self.graph.nodes())}")

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # STEP 3 — Groq Reasoning
    # ─────────────────────────────────────────

    def _reason_with_groq(self, question: str, context: str, intent: str) -> str:
        """
        Groq LLaMA3 gets a focused, minimal context.
        Fast, cheap, accurate for reasoning tasks.
        """
        system_prompt = """You are an expert SRE analyzing a microservice production system.
You receive:
- Query intent
- Target service (if any)
- Pre-filtered relevant context

You MUST adapt your structure based on the intent.

GENERAL RULES:
- Use ### headers
- Only include sections relevant to the intent
- Always include ### ML Risk Assessment if ML data is provided
- List affected services with bullet points
- Verify math matches counts
- Use exact numbers from context (never invent)
- Keep under 350 words
- End with a direct imperative SRE recommendation

INTENT-SPECIFIC STRUCTURE:

If intent = blast_radius:
Include:
- ### Immediate Impact
- ### Cascading Impact
- ### Downstream Services (Orphaned)
- ### Total Blast Radius
- ### ML Risk Assessment
- ### Recommendation

If intent = risk_assessment:
Include:
- ### Risk Overview
- ### Dependency Exposure
- ### ML Risk Assessment
- ### Operational Signals (if provided)
- ### Recommendation

If intent = risk_ranking:
Include:
- ### Top Risk Services
- ### ML Risk Comparison
- ### Recommendation

If intent = health_check:
Include:
- ### Service Health
- ### ML Risk Assessment
- ### Recommendation

If intent = dependency_query:
Include:
- ### Upstream Callers
- ### Downstream Dependencies
- ### Risk Context (if available)
- ### Recommendation

CRITICAL:
- Do NOT include sections irrelevant to the intent
- Do NOT say “no information provided”
- Do NOT fabricate metrics"""

        user_prompt = f"""Context (pre-filtered for your query):
{context}

Question: {question}

Answer:"""

        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq error: {e}\n\nContext used:\n{context}"

    # ─────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────

    def query(self, user_question: str) -> str:
        """
        Full pipeline:
          Gemini -> intent JSON -> minimal context -> Groq -> answer
        """
        if not self.context_loaded:
            return "Context not loaded. Check data paths."

        print(f"\n{'='*60}")
        print(f"Question: {user_question}")
        print(f"{'='*60}")

        if self.gemini_api_key:
            classification = self._classify_intent_gemini(user_question)
        else:
            classification = self._classify_intent_regex(user_question)

        intent  = classification.get("intent", "general")
        service = classification.get("service")
        print(f"Intent: {intent} | Service: {service}")

        context         = self._build_minimal_context(intent, service)
        token_estimate  = len(context.split()) * 1.3
        print(f"Context tokens: ~{int(token_estimate)}")

        if self.groq_api_key:
            answer = self._reason_with_groq(user_question, context, intent)
        else:
            answer = self._format_context_as_answer(intent, service, context)

        return answer

    def _format_context_as_answer(self, intent: str, service: Optional[str], context: str) -> str:
        """Simple formatted answer when no LLM is available."""
        return (
            f"### Analysis [{intent}]{f' — {service}' if service else ''}\n\n"
            f"```\n{context}\n```\n\n"
            f"_(No LLM keys configured — showing raw context)_"
        )

    def get_token_savings_stats(self, question: str) -> Dict:
        """Diagnostic: compare token usage of hybrid vs old approach."""
        classification  = self._classify_intent_regex(question)
        intent          = classification.get("intent", "general")
        service         = classification.get("service")

        focused_context = self._build_minimal_context(intent, service)
        full_context    = self._build_full_context_old_style()

        focused_tokens  = int(len(focused_context.split()) * 1.3)
        full_tokens     = int(len(full_context.split()) * 1.3)
        savings_pct     = round((1 - focused_tokens / max(full_tokens, 1)) * 100, 1)

        return {
            "intent":                   intent,
            "service":                  service,
            "focused_context_tokens":   focused_tokens,
            "full_context_tokens":      full_tokens,
            "token_savings_percent":    savings_pct,
            "focused_context_preview":  focused_context[:300],
        }

    def _build_full_context_old_style(self) -> str:
        """Simulate the old approach: dump everything."""
        lines = [f"All services: {list(self.graph.nodes())}"]
        for svc, pred in self.predictions.items():
            lines.append(str(pred))
        for svc, feat in self.features.items():
            lines.append(str(feat))
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Built-in Test
# ─────────────────────────────────────────────

def run_test():
    print("\n" + "="*70)
    print("NLP HANDLER TEST — Accurate Blast Radius")
    print("="*70)

    handler = NLPQueryHandler.__new__(NLPQueryHandler)
    handler.gemini_api_key  = os.getenv("GEMINI_API_KEY")
    handler.groq_api_key    = os.getenv("GROQ_API_KEY")
    handler.context_loaded  = True
    handler.predictions     = {}
    handler.features        = {}

    # Graph edges: caller → callee
    handler.graph = nx.DiGraph()
    edges = [
        ("frontend",              "cartservice"),
        ("frontend",              "checkoutservice"),
        ("frontend",              "productcatalogservice"),
        ("frontend",              "currencyservice"),
        ("frontend",              "recommendationservice"),
        ("checkoutservice",       "cartservice"),
        ("checkoutservice",       "paymentservice"),
        ("checkoutservice",       "emailservice"),
        ("checkoutservice",       "shippingservice"),
        ("checkoutservice",       "productcatalogservice"),
        ("checkoutservice",       "currencyservice"),
        ("recommendationservice", "productcatalogservice"),
        ("cartservice",           "redis-cart"),
    ]
    for src, tgt in edges:
        handler.graph.add_edge(src, tgt, dep_type="grpc", confidence=0.9)

    handler.predictions = {
        "checkoutservice": {
            "service_name": "checkoutservice",
            "incident_probability": 0.72,
            "incident_probability_percent": "72%",
            "risk_level": "HIGH",
            "severity": "HIGH",
            "affected_count": 1,
            "estimated_recovery_minutes": 120,
            "recommendation": "Staged rollout required — high blast radius",
        },
    }

    handler.features = {
        "checkoutservice": {
            "error_rate":              0.023,
            "latency_p99":             850,
            "traffic_requests_per_day":125000,
            "health_score":            0.61,
            "complexity_score":        0.87,
        },
    }

    test_q = "What breaks if checkoutservice fails?"
    print(f"\nQuery: '{test_q}'")
    print("-" * 50)

    classification = handler._classify_intent_regex(test_q)
    intent  = classification["intent"]
    service = classification["service"]
    context = handler._build_minimal_context(intent, service)

    print(f"Intent: {intent} | Service: {service}")
    print(f"\nContext generated:")
    print(context)
    print("\n" + "-" * 50)
    print("\n✓ Expected blast radius:")
    print("  - Direct callers: [frontend]")
    print("  - All affected: [frontend]")
    print("  - Downstream (orphaned): [cartservice, paymentservice, emailservice, ...")
    print("\n✓ Test complete!")

    if handler.gemini_api_key and handler.groq_api_key:
        print("\n" + "="*70)
        print("Running full LLM pipeline...")
        print("="*70)
        result = handler.query(test_q)
        print(f"\nLLM Answer:\n{result}")


if __name__ == "__main__":
    run_test()