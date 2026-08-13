from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping


_ROOT_ENV = {
    "model": "SAFETY_MODEL_ROOT",
    "data": "SAFETY_DATA_ROOT",
    "output": "SAFETY_OUTPUT_ROOT",
}
_MARKERS = {
    "model": {"model", "models"},
    "data": {"data", "dataset", "datasets"},
    "output": {"output", "outputs"},
}


def _portable_suffix(value: str, *, category: str) -> Path:
    normalized = value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", ".", ".."}]
    lowered = [part.lower() for part in parts]
    marker_positions = [index for index, part in enumerate(lowered) if part in _MARKERS[category]]
    if marker_positions:
        parts = parts[marker_positions[-1] + 1 :]
    if not parts:
        raise ValueError(f"cannot derive portable {category} path from {value!r}")
    return Path(*parts)


def resolve_portable_path(
    value: str,
    base_dir: str | Path,
    *,
    category: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a path with optional platform-owned category roots.

    Root overrides are intentionally category-aware so a ModelMate job can
    reuse checked-in YAML without persisting notebook-local absolute paths.
    """

    if not value or "://" in value:
        return value
    if category not in _ROOT_ENV:
        raise ValueError(f"unsupported portable path category: {category}")
    env = os.environ if environment is None else environment
    root_value = str(env.get(_ROOT_ENV[category], "")).strip()
    path = Path(value)
    if root_value:
        root = Path(root_value).expanduser().resolve()
        try:
            resolved_existing = path.expanduser().resolve() if path.is_absolute() else None
            if resolved_existing is not None and resolved_existing.is_relative_to(root):
                return str(resolved_existing)
        except (OSError, ValueError):
            pass
        return str((root / _portable_suffix(value, category=category)).resolve())
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


def configure_visible_devices(
    environment: MutableMapping[str, str] | Mapping[str, str],
    *,
    backend: str,
    requested_devices: str,
) -> dict[str, str]:
    """Return a configured environment without overriding scheduler choices."""

    result = {str(key): str(value) for key, value in environment.items()}
    normalized = str(backend).strip().lower()
    variable = {
        "npu": "ASCEND_RT_VISIBLE_DEVICES",
        "cuda": "CUDA_VISIBLE_DEVICES",
        "ppu": "PPU_VISIBLE_DEVICES",
    }.get(normalized)
    if variable and requested_devices and not str(result.get(variable, "")).strip():
        result[variable] = str(requested_devices)
    return result
