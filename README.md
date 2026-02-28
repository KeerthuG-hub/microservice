# 🔗 Universal Microservice Dependency Analyzer

Detects **all 9 dependency types** across any microservice project.
Works with any language, any domain (ticketing, healthcare, banking, food delivery...).
**Zero hardcoded assumptions** — everything learned from the project itself.

---

## 🚀 Quick Start

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add your Groq API key (free at console.groq.com)
cp .env.example .env && nano .env

# 3. Run
python run.py /path/to/your/microservice/project

# 4. No LLM needed? (offline mode)
python run.py /path/to/project --no-llm
```

Outputs: JSON report + CSV + interactive HTML dependency graph in `data/outputs/`

---

## 📋 Design Principles

| ❌ Never hardcode | ✅ Always discover |
|---|---|
| Service names (no "cart", "payment") | Names FROM K8s Service / directory names |
| Port numbers (no "8080") | Ports FROM K8s Service objects |
| Env var patterns (no "_SERVICE_URL") | Patterns learned FROM actual env vars |
| Directory structure (no "src/services/") | Layout detected FROM Dockerfile positions |

---

## 🔍 9 Dependency Types Detected

| Type | Subtypes | Sources |
|------|----------|---------|
| **endpoint** | HTTP, gRPC, GraphQL, WebSocket | K8s env, config files, source code |
| **async** | Kafka, RabbitMQ, SQS, NATS | Topic strings, broker URLs |
| **data** | shared_db, shared_table, shared_redis | Connection strings, SQL queries |
| **semantic** | shared_domain, change_coupling | String co-occurrence, git history |
| **build** | proto_import, internal_lib, base_image | go.mod, pom.xml, Dockerfile FROM |
| **deployment** | startup_order, helm_hook, db_migration | depends_on, initContainers, Jobs |
| **infrastructure** | postgres, redis, mongodb, vault... | Connection strings, port numbers |
| **observability** | prometheus, jaeger, opentelemetry | ServiceMonitor, OTEL env vars |
| **external** | stripe, sendgrid, auth0, aws... | SDK imports in build files |

---

## 📊 Confidence Tiers

| Tier | Range | Meaning |
|------|-------|---------|
| ✅ Confirmed | ≥ 0.85 | Safe to trust — use in architecture docs |
| 🟡 Probable | 0.65–0.85 | Likely real — worth a quick manual check |
| 🔴 Uncertain | < 0.65 | Verify manually before trusting |
| 👁️ Implicit | string/git | Possible hidden coupling — investigate |
| 📡 Observability | any | Monitoring only — not functional deps |

Confidence **boosts** when multiple independent sources agree (+0.05 per additional source).

---

## 🤖 LLM Usage (Minimal)

LLM is invoked **only when**:
1. A service has **zero dependencies** found by static analysis AND has source code
2. Average confidence for a service is **< 0.65**
3. User noted a **custom framework** in the 3 context questions

LLM call = per-service (not per-file), max 6 files, max 1500 chars each.
Reduces LLM cost ~90% vs full agentic approach.

**Supported backends** (set `LLM_BACKEND` env var):
- `groq` — Llama 3.1 70B, free, 14,400 req/day
- `gemini` — Gemini 1.5 Flash, free, 1M tokens/day
- `ollama` — Local, fully offline

---

## 🗂️ Project Structure

```
microservice-dependency-analyzer/
├── agent/
│   ├── models.py              ← Data structures
│   ├── context_collector.py   ← 3 questions + auto-detection
│   ├── explorer.py            ← Service discovery (zero hardcoding)
│   ├── core.py                ← Main orchestration
│   ├── merger.py              ← Dedup + confidence fusion
│   ├── semantic_analyzer.py   ← String co-occurrence + git coupling
│   ├── static_analyzer/
│   │   ├── k8s_parser.py      ← K8s manifests (all object types)
│   │   ├── code_parser.py     ← Multi-language source code
│   │   ├── build_parser.py    ← go.mod, package.json, pom.xml...
│   │   ├── proto_parser.py    ← .proto files (highest confidence)
│   │   ├── compose_parser.py  ← docker-compose
│   │   └── config_parser.py   ← application.yml, .env, appsettings
│   └── llm/
│       ├── llm_client.py      ← Groq/Gemini/Ollama wrapper
│       └── llm_analyzer.py    ← Targeted LLM invocation
├── graph/
│   ├── builder.py             ← NetworkX graph
│   ├── visualizer.py          ← Interactive HTML (PyVis)
│   └── output_writer.py       ← JSON, CSV, HTML reports
├── evaluation/
│   └── evaluator.py           ← Precision/Recall/F1 measurement
├── tests/
│   └── test_analyzer.py       ← Unit tests
├── run.py                     ← CLI entry point
└── requirements.txt
```

---

## 📏 Evaluation

```bash
# With ground truth JSON
python run.py /path/to/project \
  --ground-truth data/ground_truth/my_gt.json

# Ground truth format:
{
  "edges": [
    ["frontend", "cartservice"],
    ["checkoutservice", "paymentservice"]
  ]
}
```

---

## ⚙️ Options

```bash
python run.py PROJECT_PATH [OPTIONS]

  --no-llm              Skip LLM (offline mode, faster)
  --non-interactive     Skip 3 questions, use defaults
  --output DIR          Output directory (default: data/outputs)
  --graph FILE          HTML graph output path
  --ground-truth FILE   Evaluate against ground truth JSON
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```
