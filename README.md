
# AI Powered Microservice Dependency Analyzer

An intelligent SRE platform that automatically understands microservice architectures without manual diagrams, stale documentation, or guesswork.

This system analyzes source code and configuration to discover service dependencies, predicts architectural risk using machine learning, computes exact blast radius using graph traversal, and answers plain English architecture questions through a structured query pipeline.

---

## Overview

Modern microservice systems are complex and constantly changing. Manual dependency mapping becomes outdated quickly, and deployment risk is often estimated by intuition rather than structural evidence.

This platform solves that by combining deterministic graph analysis with probabilistic risk modeling. Graph traversal handles structural truth. Machine learning handles incident likelihood. These concerns are strictly separated at the architectural level.

---

## Core Capabilities

### Automatic Dependency Discovery

The system scans the entire project and extracts service relationships from:

* Source code across multiple languages
* Kubernetes and Docker configurations
* Build manifests
* Protocol buffer contracts

All detected signals are merged and confidence scored. Only high confidence edges are used to build the final dependency graph.

### Exact Blast Radius Computation

Blast radius is computed using deterministic graph traversal. When a service fails, all direct and transitive downstream services are identified using exact graph descendant traversal.

No machine learning is used to approximate topology.

### Risk Prediction Engine

Each service is converted into a feature vector derived from graph structure and operational signals. A Random Forest classifier predicts:

* Incident probability
* Severity score
* Recovery time estimate

Services are assigned one of four risk tiers:

* HIGH
* MEDIUM
* LOW
* MINIMAL

### Intelligent Query Assistant

Users can ask natural language questions such as:

* What breaks if payment service fails?
* Is it safe to deploy auth service?
* Which services are most critical?
* What does order service depend on?

A two stage pipeline first classifies intent, then selects only the relevant context slice before generating an SRE style response.

---

## Architecture Layers

### Layer 1 Dependency Analysis

* Multi language parser suite
* Dynamic service owner inference
* Confidence scored dependency merging
* Threshold enforcement at graph boundary

Only edges with confidence of 0.65 or higher are used for topology.

### Layer 2 Machine Learning Engine

* Graph feature extraction using NetworkX
* Centrality metrics and coupling metrics
* Synthetic training data with variance injection
* Random Forest models for classification and regression
* Stratified training and cross validation

### Layer 3 Query Engine

* Intent classification before reasoning
* Minimal context assembly per question
* Structured response formatting
* Graceful degradation if models are unavailable

### Layer 4 Visualization Dashboard

* Interactive dependency graph
* Risk tier visualization
* Centrality analytics
* Prediction tables
* AI chat interface

Infrastructure components such as databases and cloud providers are filtered before analysis so they do not distort structural reasoning.

---

## Key Design Principles

Separation of concerns
Graph traversal handles structural facts. Machine learning handles probabilistic risk. Neither approximates the other.

Intent routing before reasoning
Only relevant context is sent to the reasoning model, improving quality and reducing noise.

Variance injection for generalization
Training data is perturbed to prevent memorization of service identity and encourage learning structural patterns.

Graceful degradation
If any model fails, the system falls back to deterministic structured output instead of failing silently.

Confidence threshold enforcement
Low confidence signals never enter the graph used by downstream computation.

---

## Future Improvements

* Integration with real monitoring data such as Prometheus or Datadog
* Probabilistic cascade simulation for partial failures
* Time series based incident trend prediction
* Expanded training corpus across more real world microservice architectures

---

## Summary

This project provides an automated, architecture aware SRE intelligence layer for microservice systems. It combines deterministic graph science with supervised machine learning to deliver explainable risk analysis, precise blast radius computation, and natural language architectural insight.

It replaces manual diagrams with continuously derived structural truth and transforms topology into actionable operational intelligence.
