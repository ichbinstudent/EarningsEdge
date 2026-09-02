"""Model registry: which artifact made which decision.

Training scripts register joblib artifacts (path + content hash); scanners
log the active model's sha into their outputs, so every score/decision is
attributable to an exact model version across retraining.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from earnings_edge.db import (
    model_registry_get_active,
    model_registry_promote,
    model_registry_register,
)

logger = logging.getLogger("framework.data.model_registry")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model(name: str, path: Path,
                   trained_at: Optional[str] = None, promote: bool = False) -> int:
    """Register a model artifact (idempotent per name+sha). Returns row id."""
    path = Path(path)
    sha = sha256_of(path)
    rid = model_registry_register(
        name, str(path), sha, trained_at=trained_at or _utcnow(),
    )
    if promote:
        promote_model(name, sha)
    return rid


def promote_model(name: str, sha256: str) -> None:
    model_registry_promote(name, sha256)
    logger.info("model %s promoted: %s", name, sha256[:12])


def get_active(name: str) -> Optional[dict]:
    """Most recently promoted artifact for ``name`` (fallback: latest registered)."""
    return model_registry_get_active(name)
