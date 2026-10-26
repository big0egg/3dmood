"""Model input types and stateless per-object seeds; no GT or dataset access."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .common import digest, file_digest

@dataclass(frozen=True)
class ObjectInput:
    rgb: np.ndarray
    intrinsics: np.ndarray
    box_xyxy: tuple
    mask: np.ndarray | None = None
    depth: np.ndarray | None = None


def object_seed(base, frame_id, target_id):
    return int(digest([int(base), frame_id, target_id])[:8], 16)


def project_source(repo, folder):
    repo = Path(repo)
    return digest({str(p.relative_to(repo)): file_digest(p) for p in sorted((repo / folder).rglob("*.py"))})
