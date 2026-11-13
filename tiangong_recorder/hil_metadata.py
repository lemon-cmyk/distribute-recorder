from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# These columns are optional. Legacy episodes retain their original schema.
HIL_FEATURES = {
    "complementary_info.policy_action": {
        "dtype": "float32", "shape": (16,), "names": None,
    },
    "complementary_info.policy_action_valid": {
        "dtype": "bool", "shape": (1,), "names": None,
    },
    "complementary_info.is_intervention": {
        "dtype": "float32", "shape": (1,), "names": None,
    },
    "complementary_info.state": {
        "dtype": "float32", "shape": (1,), "names": None,
    },
    "complementary_info.collector_policy_id": {
        "dtype": "string", "shape": (1,), "names": None,
    },
    "episode_success": {
        "dtype": "string", "shape": (1,), "names": None,
    },
}


def extract_hil_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve available per-frame annotations without inferring missing labels."""
    info = entry.get("complementary_info", {})
    if not isinstance(info, Mapping):
        raise ValueError("complementary_info must be a mapping")
    result = {}
    for key, feature in HIL_FEATURES.items():
        source = info if key.startswith("complementary_info.") else entry
        source_key = key.rsplit(".", 1)[-1]
        if source_key not in source:
            continue
        raw = np.asarray(source[source_key]).reshape(-1)
        if raw.shape != feature["shape"]:
            raise ValueError(f"{key} must have shape {feature['shape']}; got {raw.shape}")
        if feature["dtype"] == "string":
            value = raw[0]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must contain one non-empty string")
            result[key] = str(value)
            continue
        if raw.dtype.kind not in "biuf" or not np.isfinite(raw).all():
            raise ValueError(f"{key} must contain finite numeric values")
        if key.endswith(("policy_action_valid", "is_intervention")):
            if not np.isin(raw, [0, 1]).all():
                raise ValueError(f"{key} must contain only 0 or 1")
        elif key == "complementary_info.state" and not np.isin(raw, [0, 1, 2]).all():
            raise ValueError(f"{key} must contain only 0, 1 or 2")
        with np.errstate(over="ignore", invalid="ignore"):
            value = raw.astype(feature["dtype"])
        if not np.isfinite(value).all():
            raise ValueError(f"{key} is outside the supported numeric range")
        result[key] = value
    return result
