from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from .platform import resolve_portable_path
from .schema import ExperimentCell


@dataclass(frozen=True)
class AssetRequirement:
    asset_id: str
    path: Path
    kind: str
    cell_id: str = ""
    min_free_bytes: int = 0


@dataclass(frozen=True)
class PreflightIssue:
    cell_id: str
    asset_id: str
    code: str
    category: str
    message: str
    suggestion: str


@dataclass(frozen=True)
class PreflightReport:
    status: str
    issues: tuple[PreflightIssue, ...]
    checked: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checked": list(self.checked),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _training_pair_and_teacher(cell: ExperimentCell) -> tuple[str, str]:
    axes = {**dict(cell.overrides), **dict(cell.axes)}
    pair = str(axes.get("pair", "qwen35_9b_to_08b"))
    teacher_variant = ""
    if cell.experiment_id == "P2-03":
        teacher = str(axes.get("teacher", ""))
        pair = "qwen3_8b_to_06b" if teacher == "qwen3_8b" else "qwen3_4b_to_06b"
        if teacher in {"same_size_base", "safety_tuned"}:
            teacher_variant = teacher
    elif cell.experiment_id == "P2-04":
        pair = "qwen3_8b_to_llama32_1b"
    return pair, teacher_variant


def training_model_requirements(
    cell: ExperimentCell,
    *,
    project_root: str | Path,
    environment: Mapping[str, str] | None = None,
    device: str = "npu",
) -> tuple[AssetRequirement, ...]:
    """Resolve the effective teacher/student snapshots for one training cell."""

    root = Path(project_root).resolve()
    pair, teacher_variant = _training_pair_and_teacher(cell)
    config_path = root / "configs" / f"{pair}_phase1_{device}.yaml"
    if not config_path.is_file():
        raise ValueError(f"training pair config is missing: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        teacher = dict(raw["models"]["teacher"])
        student = dict(raw["models"]["student"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"training pair config has invalid model entries: {config_path}") from exc
    if teacher_variant:
        registry_path = root / "configs" / "ablations" / "teacher_registry.yaml"
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        try:
            teacher = dict(registry["teachers"][teacher_variant])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"teacher registry lacks {teacher_variant}") from exc
        teacher_base = registry_path.parent
    else:
        teacher_base = config_path.parent
    result = []
    for asset_id, entry, base in (
        ("training_teacher_model", teacher, teacher_base),
        ("training_student_model", student, config_path.parent),
    ):
        value = str(entry.get("path", "")).strip()
        if not value:
            raise ValueError(f"{asset_id} path is missing from {config_path}")
        path = Path(
            resolve_portable_path(
                value,
                base,
                category="model",
                environment=environment,
            )
        ).resolve()
        result.append(AssetRequirement(asset_id, path, "model", cell.cell_id))
    return tuple(result)


def requirements_from_manifest(
    asset_ids: Iterable[str],
    manifest: Mapping[str, object],
    *,
    cell_id: str,
    base_dir: str | Path | None = None,
) -> tuple[list[AssetRequirement], tuple[str, ...]]:
    """Translate only explicitly declared cell requirements into checks."""

    requirements: list[AssetRequirement] = []
    missing: list[str] = []
    for raw_id in asset_ids:
        asset_id = str(raw_id)
        if asset_id not in manifest:
            missing.append(asset_id)
            continue
        raw = manifest[asset_id]
        if isinstance(raw, str):
            path_text = raw.strip()
            kind = "directory" if Path(path_text).suffix == "" else "file"
        elif isinstance(raw, Mapping):
            path_text = str(raw.get("path", "")).strip()
            kind = str(raw.get("kind", "directory")).strip().lower()
        else:
            raise ValueError(f"asset {asset_id} must declare a path string or object")
        if not path_text:
            raise ValueError(f"asset {asset_id} path must be non-empty")
        if kind not in {"file", "directory", "model"}:
            raise ValueError(f"asset {asset_id} has unsupported kind: {kind}")
        path = Path(path_text).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        requirements.append(AssetRequirement(asset_id, path.resolve(), kind, cell_id))
    return requirements, tuple(missing)


def _cell_id(asset_id: str, path: Path, explicit: str = "") -> str:
    if explicit:
        return explicit
    digest = hashlib.sha256(f"{asset_id}:{path}".encode("utf-8")).hexdigest()[:12]
    return f"preflight-{digest}"


def _issue(requirement: AssetRequirement, code: str, message: str, suggestion: str) -> PreflightIssue:
    return PreflightIssue(
        cell_id=_cell_id(requirement.asset_id, requirement.path, requirement.cell_id),
        asset_id=requirement.asset_id,
        code=code,
        category=requirement.kind,
        message=message,
        suggestion=suggestion,
    )


def run_preflight(
    requirements: Iterable[AssetRequirement],
    *,
    environment: Mapping[str, str] | None = None,
) -> PreflightReport:
    # ``environment`` is accepted so callers can test scheduler state, but is
    # deliberately never serialized: tokens and passwords must not enter logs.
    del environment
    issues: list[PreflightIssue] = []
    checked: list[str] = []
    for requirement in requirements:
        path = Path(requirement.path)
        checked.append(requirement.asset_id)
        if requirement.kind == "model":
            model_report = inspect_model_directory(path, cell_id=requirement.cell_id)
            issues.extend(model_report.issues)
        elif requirement.kind == "directory":
            if not path.is_dir():
                issues.append(_issue(requirement, "ASSET_DIRECTORY_MISSING", "required directory is missing", "prepare or mount the directory before submitting"))
        elif requirement.kind == "file":
            if not path.is_file():
                issues.append(_issue(requirement, "ASSET_FILE_MISSING", "required file is missing", "prepare or upload the file before submitting"))
        elif requirement.kind == "output":
            probe = path if path.exists() else path.parent
            if not probe.exists() or not probe.is_dir():
                issues.append(_issue(requirement, "OUTPUT_PARENT_MISSING", "output parent directory is missing", "create or mount the persistent output directory"))
            else:
                free_bytes = shutil.disk_usage(probe).free
                if free_bytes < int(requirement.min_free_bytes):
                    issues.append(_issue(requirement, "OUTPUT_DISK_INSUFFICIENT", "output volume has insufficient free space", "reduce the plan or provision more persistent storage"))
        else:
            issues.append(_issue(requirement, "ASSET_KIND_UNKNOWN", "unknown requirement kind", "use file, directory, or model"))
    return PreflightReport("READY" if not issues else "BLOCKED", tuple(issues), tuple(checked))


def inspect_model_directory(path: str | Path, *, cell_id: str = "") -> PreflightReport:
    root = Path(path)
    requirement = AssetRequirement(root.name or "model", root, "model", cell_id)
    issues: list[PreflightIssue] = []
    if not root.is_dir():
        issues.append(_issue(requirement, "MODEL_DIRECTORY_MISSING", "model directory is missing", "mount the persistent model directory"))
        return PreflightReport("BLOCKED", tuple(issues), (requirement.asset_id,))
    if not (root / "config.json").is_file():
        issues.append(_issue(requirement, "MODEL_CONFIG_MISSING", "config.json is missing", "download a complete model snapshot"))
    tokenizer_candidates = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json")
    if not any((root / name).is_file() for name in tokenizer_candidates):
        issues.append(_issue(requirement, "MODEL_TOKENIZER_MISSING", "tokenizer files are missing", "download tokenizer assets into the model directory"))
    weight_files = [
        file for pattern in ("*.safetensors", "pytorch_model*.bin", "*.pth")
        for file in root.glob(pattern) if file.is_file()
    ]
    if not weight_files:
        issues.append(_issue(requirement, "MODEL_WEIGHTS_MISSING", "model weights are missing", "download all weight shards and any index JSON"))
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = {str(name) for name in dict(payload["weight_map"]).values()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            issues.append(_issue(requirement, "MODEL_INDEX_INVALID", f"invalid weight index: {index_name}", "replace it with a complete snapshot index"))
            continue
        missing_shards = sorted(name for name in shard_names if not (root / name).is_file())
        if missing_shards:
            issues.append(_issue(requirement, "MODEL_SHARD_MISSING", f"weight index references missing shards: {missing_shards[:3]}", "download every shard listed by the index"))
    return PreflightReport("READY" if not issues else "BLOCKED", tuple(issues), (requirement.asset_id,))


def inspect_submission_package(
    root: str | Path,
    *,
    max_file_bytes: int = 512 * 1024 * 1024,
    max_total_bytes: int = 512 * 1024 * 1024,
) -> PreflightReport:
    package = Path(root)
    requirement = AssetRequirement("submission_package", package, "package")
    issues: list[PreflightIssue] = []
    total = 0
    if not package.is_dir():
        issues.append(_issue(requirement, "PACKAGE_DIRECTORY_MISSING", "submission directory is missing", "select an existing code directory"))
        return PreflightReport("BLOCKED", tuple(issues), (requirement.asset_id,))
    forbidden_roots = {"models", "model", "data", "datasets", "outputs", "output"}
    forbidden_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".arrow", ".parquet"}
    for current, directories, files in os.walk(package, followlinks=False):
        current_path = Path(current)
        for name in tuple(directories):
            child = current_path / name
            if child.is_symlink():
                issues.append(_issue(requirement, "PACKAGE_SYMLINK", f"symbolic link is forbidden: {child.relative_to(package)}", "remove links from the upload package"))
                directories.remove(name)
            elif current_path == package and name.lower() in forbidden_roots:
                issues.append(_issue(requirement, "PACKAGE_ASSET_DIRECTORY", f"asset directory is inside package: {child.relative_to(package)}", "mount models/data/outputs from persistent storage"))
        for name in files:
            file = current_path / name
            if file.is_symlink():
                issues.append(_issue(requirement, "PACKAGE_SYMLINK", f"symbolic link is forbidden: {file.relative_to(package)}", "remove links from the upload package"))
                continue
            size = file.stat().st_size
            total += size
            if size > max_file_bytes:
                issues.append(_issue(requirement, "PACKAGE_FILE_TOO_LARGE", f"file exceeds upload limit: {file.relative_to(package)}", "move the file to persistent storage"))
            if file.suffix.lower() in forbidden_suffixes:
                issues.append(_issue(requirement, "PACKAGE_ASSET_FILE", f"model/data artifact is inside package: {file.relative_to(package)}", "exclude generated and weight artifacts"))
    if total > max_total_bytes:
        issues.append(_issue(requirement, "PACKAGE_TOTAL_TOO_LARGE", "package exceeds total upload limit", "upload code only and mount persistent assets"))
    return PreflightReport("READY" if not issues else "BLOCKED", tuple(issues), (requirement.asset_id,))
