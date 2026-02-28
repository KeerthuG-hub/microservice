"""
Proto Parser - detects gRPC dependencies from .proto files.
Finds cross-service proto imports and service definitions.
Zero hardcoded names.
"""

from pathlib import Path
from typing import List, Dict
import re
from agent.models import Dependency, Service


class ProtoParser:

    def __init__(self, services: Dict[str, Service]):
        self.services = services
        self.deps: List[Dependency] = []
        # Map: proto file path → owning service name
        self._proto_owner: Dict[str, str] = {}

    def parse(self, project_path: str) -> List[Dependency]:
        print("\n🔌 Parsing .proto files...")
        root = Path(project_path)

        proto_files = [p for p in root.rglob('*.proto')
                       if not self._is_vendor(p)]

        # First pass: build proto → service ownership map
        for proto_file in proto_files:
            owner = self._find_owner(proto_file)
            if owner:
                self._proto_owner[str(proto_file)] = owner

        # Second pass: parse imports and service definitions
        for proto_file in proto_files:
            self._parse_proto(proto_file)

        print(f"   ✅ {len(self.deps)} deps from proto files")
        return self.deps

    def _find_owner(self, proto_file: Path) -> str:
        """Find which service owns this .proto file by walking up dirs."""
        for part in reversed(proto_file.parts):
            for svc_name, svc in self.services.items():
                if svc.normalized_name == self._norm(part):
                    return svc_name
        # Fallback: check if proto file parent is inside a service dir
        for svc_name, svc in self.services.items():
            try:
                proto_file.relative_to(svc.path)
                return svc_name
            except ValueError:
                pass
        return None

    def _parse_proto(self, proto_file: Path):
        try:
            content = proto_file.read_text(errors='ignore')
        except Exception:
            return

        from_svc = self._proto_owner.get(str(proto_file))

        # Parse cross-service import statements.
        # A proto import means the owning service USES types from another
        # service's proto definition — but it does NOT by itself prove a
        # runtime gRPC call. We emit a build/proto_import dep only.
        # The code parser will add the endpoint/grpc dep if it finds an
        # actual gRPC stub call in the source code.
        for m in re.finditer(r'import\s+"([^"]+\.proto)"', content):
            imported = m.group(1)
            imported_name = Path(imported).stem
            to_svc = self._match(imported_name)
            if to_svc and from_svc and to_svc != from_svc:
                self.deps.append(Dependency(
                    from_service=from_svc,
                    to_service=to_svc,
                    dep_type='build',
                    subtype='proto_import',
                    confidence=0.75,   # lower: import alone doesn't prove runtime call
                    evidence=f'proto import: {imported} in {proto_file.name}',
                    source='proto_parser'
                ))

        # NOTE: We intentionally do NOT create edges from 'service Foo {}' blocks.
        # A 'service' block defines what THIS service *offers*, not what it calls.
        # Creating an edge from owner → Foo reverses the dependency direction.
        # Runtime call edges come from the code parser (grpc stub detection).

    def _match(self, name: str) -> str:
        norm = self._norm(name)
        for svc_name, svc in self.services.items():
            if svc.normalized_name == norm:
                return svc_name
        return None

    def _norm(self, name: str) -> str:
        return name.lower().replace('-', '').replace('_', '').replace('.', '')

    @staticmethod
    def _is_vendor(path: Path) -> bool:
        skip = {'node_modules', 'vendor', '.git', 'generated', '__pycache__'}
        return any(p in skip for p in path.parts)
