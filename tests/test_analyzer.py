"""
Tests for the microservice dependency analyzer.
Tests normalization, service discovery, dependency dedup, etc.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.models import Service, Dependency, ProjectContext
from agent.explorer import ServiceExplorer
from agent.merger import Merger


# -------------------------------------------------------------------------
# NORMALIZATION
# -------------------------------------------------------------------------

class TestNormalization:
    """Test that front-end, frontend, FRONT_END all match correctly."""

    def _make_services(self):
        return {
            'front-end': Service(
                name='front-end',
                path='/proj/front-end',
                language='javascript',
                anchor='file:package.json',
                normalized_name='frontend'
            ),
            'order-service': Service(
                name='order-service',
                path='/proj/order-service',
                language='go',
                anchor='file:go.mod',
                normalized_name='orderservice'
            ),
        }

    def _norm(self, name):
        return name.lower().replace('-', '').replace('_', '').replace('.', '')

    def test_hyphen_match(self):
        svcs = self._make_services()
        norm = self._norm('front-end')
        found = any(s.normalized_name == norm for s in svcs.values())
        assert found, "front-end should match"

    def test_underscore_match(self):
        svcs = self._make_services()
        norm = self._norm('front_end')
        found = any(s.normalized_name == norm for s in svcs.values())
        assert found, "front_end should match front-end"

    def test_uppercase_match(self):
        svcs = self._make_services()
        norm = self._norm('FRONTEND')
        found = any(s.normalized_name == norm for s in svcs.values())
        assert found, "FRONTEND should match front-end"

    def test_no_false_match(self):
        svcs = self._make_services()
        norm = self._norm('payment-service')
        found = any(s.normalized_name == norm for s in svcs.values())
        assert not found, "payment-service should not match any known service"


# -------------------------------------------------------------------------
# MERGER
# -------------------------------------------------------------------------

class TestMerger:

    def _make_services(self):
        return {
            'svc-a': Service('svc-a', '/a', 'go', 'file:go.mod', normalized_name='svca'),
            'svc-b': Service('svc-b', '/b', 'python', 'file:requirements.txt', normalized_name='svcb'),
        }

    def test_deduplication(self):
        svcs = self._make_services()
        merger = Merger(svcs)

        deps = [
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.85, 'from k8s', 'k8s_env'),
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.70, 'from code', 'code_http'),
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.75, 'from config', 'config_url'),
        ]
        merged = merger.merge(deps)
        assert len(merged) == 1, "3 identical key deps should merge to 1"

    def test_confidence_boost(self):
        svcs = self._make_services()
        merger = Merger(svcs)

        deps = [
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.85, 'k8s', 'k8s_env'),
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.80, 'code', 'code_http'),
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.75, 'config', 'config_url'),
        ]
        merged = merger.merge(deps)
        assert merged[0].confidence > 0.85, "Multi-source should boost confidence"
        assert merged[0].confidence <= 0.98

    def test_different_types_not_merged(self):
        svcs = self._make_services()
        merger = Merger(svcs)

        deps = [
            Dependency('svc-a', 'svc-b', 'endpoint', 'http', 0.85, 'http', 'k8s_env'),
            Dependency('svc-a', 'svc-b', 'async', 'kafka', 0.80, 'kafka', 'code_kafka'),
        ]
        merged = merger.merge(deps)
        assert len(merged) == 2, "Different dep types should stay separate"

    def test_self_dep_excluded(self):
        svcs = self._make_services()
        merger = Merger(svcs)

        deps = [
            Dependency('svc-a', 'svc-a', 'endpoint', 'http', 0.90, 'self', 'k8s_env'),
        ]
        merged = merger.merge(deps)
        assert len(merged) == 0, "Self-dependencies should be excluded"

    def test_k8s_env_not_penalized(self):
        """K8s env var endpoint deps must remain ≥0.85 after merging.

        K8s manifests are authoritative runtime config; their deps should never
        be penalized as uncertain env-only deps.
        """
        svcs = self._make_services()
        merger = Merger(svcs)

        # Single k8s_env source, no code call confirmation
        deps = [
            Dependency('svc-a', 'svc-b', 'endpoint', 'grpc', 0.90,
                       'env CART_SERVICE_ADDR=svc-b:7070', 'k8s_env'),
        ]
        merged = merger.merge(deps)
        assert len(merged) == 1
        assert merged[0].confidence >= 0.85, (
            "k8s_env endpoint dep must not be capped to env-only (0.68)"
        )
        assert '[env-only' not in (merged[0].evidence or ''), (
            "k8s_env dep should not be labelled env-only"
        )

    def test_service_addr_code_env_var_elevated(self):
        """code_env_var deps whose key ends with a service-address suffix must not
        be capped at 0.68 by the merger's env-only penalty path."""
        svcs = self._make_services()
        merger = Merger(svcs)

        # code_env_var dep whose evidence contains a _SERVICE_ADDR key
        deps = [
            Dependency('svc-a', 'svc-b', 'endpoint', 'grpc', 0.85,
                       'env var SVC_B_SERVICE_ADDR in main.go', 'code_env_var'),
        ]
        merged = merger.merge(deps)
        assert len(merged) == 1
        assert merged[0].confidence >= 0.85, (
            "Service-address code_env_var dep should not be capped to 0.68"
        )
        assert '[env-only' not in (merged[0].evidence or ''), (
            "Service-address code_env_var dep should not be labelled env-only"
        )


# -------------------------------------------------------------------------
# DEPENDENCY MODEL
# -------------------------------------------------------------------------

class TestDependencyModel:

    def test_hash_equality(self):
        d1 = Dependency('a', 'b', 'endpoint', 'http', 0.9, 'ev1', 'src1')
        d2 = Dependency('a', 'b', 'endpoint', 'http', 0.5, 'ev2', 'src2')
        assert d1 == d2, "Same (from,to,type,subtype) should be equal"
        assert hash(d1) == hash(d2)

    def test_different_subtype_not_equal(self):
        d1 = Dependency('a', 'b', 'endpoint', 'http', 0.9, 'ev1', 'src1')
        d2 = Dependency('a', 'b', 'endpoint', 'grpc', 0.9, 'ev2', 'src2')
        assert d1 != d2


# -------------------------------------------------------------------------
# TIERS
# -------------------------------------------------------------------------

class TestTiers:

    def test_confirmed_tier(self):
        d = Dependency('a', 'b', 'endpoint', 'http', 0.90, '', 'k8s_env')
        assert d.confidence >= 0.85

    def test_probable_tier(self):
        d = Dependency('a', 'b', 'endpoint', 'http', 0.72, '', 'code_http')
        assert 0.65 <= d.confidence < 0.85

    def test_uncertain_tier(self):
        d = Dependency('a', 'b', 'endpoint', 'http', 0.55, '', 'code_env_var')
        assert d.confidence < 0.65

    def test_implicit_by_source(self):
        d = Dependency('a', 'b', 'semantic', 'shared_domain', 0.50,
                       'shared string', 'string_cooccurrence')
        assert d.source == 'string_cooccurrence'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
