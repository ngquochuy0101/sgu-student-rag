from __future__ import annotations

import os
from typing import Dict, Optional


_RUNTIME_ENV_DEFAULTS: Dict[str, str] = {
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "GRPC_VERBOSITY": "ERROR",
    "GLOG_minloglevel": "2",
}


def configure_runtime_environment(overrides: Optional[Dict[str, str]] = None) -> None:
    """Apply runtime environment defaults that keep Windows execution stable."""
    values = dict(_RUNTIME_ENV_DEFAULTS)
    if overrides:
        values.update(overrides)

    for key, value in values.items():
        # Force a deterministic runtime to avoid protobuf/tensorflow import instability.
        os.environ[key] = value
