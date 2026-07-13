#!/usr/bin/env python3
"""Package formal experiment artifacts without copying checkpoints or model weights."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from formal_llm_judge_targets import (
    FormalJudgeTarget,
    iter_formal_targets,
    judge_payload_is_complete,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_NAMES = frozenset({
    "checkpoint",
    "checkpoints",
    "model",
    "models",
    "weight",
    "weights",
    "hf_cache",
})
MODEL_SUFFIXES = frozenset({
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
})
REPORT_PATTERNS = (
    "llm_judge_results_*.csv",
    "llm_judge_results_audit.json",
    "sweep_results*.csv",
    "output_inventory_for_judge.txt",
)
AUXILIARY_OUTPUT_DIRS = ("judge_backfill_logs", "run_pairs_logs")


@dataclass(frozen=True)
class PackageMember:
    source_path: Path
    archive_path: Path
    size: int


@dataclass
class PackagePlan:
    project_root: Path
    outputs_root: Path
    targets: list[FormalJudgeTarget]
    members: list[PackageMember]
    missing_run_roots: list[dict[str, object]] = field(default_factory=list)
    missing_expected_pan_results: list[dict[str, str]] = field(default_factory=list)
    missing_or_invalid_judge_results: list[dict[str, str]] = field(default_factory=list)
    missing_eval_data: list[str] = field(default_factory=list)
    eval_data_fingerprints: list[dict[str, object]] = field(default_factory=list)
    excluded_file_count: int = 0
    excluded_file_bytes: int = 0

    def manifest(self) -> dict[str, object]:
        run_roots: dict[str, list[dict[str, str]]] = {}
        for target in self.targets:
            key = _portable_path(target.run_root, self.project_root, self.outputs_root)
            run_roots.setdefault(key, []).append(target.identity())
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": self.project_root.as_posix(),
            "outputs_root": self.outputs_root.as_posix(),
            "formal_target_count": len(self.targets),
            "formal_run_roots": [
                {"path": path, "targets": targets}
                for path, targets in sorted(run_roots.items())
            ],
            "included_file_count": len(self.members),
            "included_file_bytes": sum(member.size for member in self.members),
            "included_files": [
                {"path": member.archive_path.as_posix(), "size": member.size}
                for member in self.members
            ],
            "excluded_model_file_count": self.excluded_file_count,
            "excluded_model_file_bytes": self.excluded_file_bytes,
            "missing_run_roots": self.missing_run_roots,
            "missing_expected_pan_results": self.missing_expected_pan_results,
            "missing_or_invalid_judge_results": self.missing_or_invalid_judge_results,
            "missing_eval_data": self.missing_eval_data,
            "eval_data_fingerprints": self.eval_data_fingerprints,
        }


def is_model_artifact(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & MODEL_DIR_NAMES:
        return True
    return path.suffix.lower() in MODEL_SUFFIXES


def _portable_path(path: Path, project_root: Path, outputs_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        pass
    try:
        return (Path("outputs") / path.relative_to(outputs_root)).as_posix()
    except ValueError:
        return (Path("external") / path.name).as_posix()


def _member(path: Path, project_root: Path, outputs_root: Path) -> PackageMember:
    return PackageMember(
        source_path=path,
        archive_path=Path(_portable_path(path, project_root, outputs_root)),
        size=path.stat().st_size,
    )


def _iter_tree_files(root: Path) -> Iterable[tuple[Path, bool]]:
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs = []
        for name in dirs:
            candidate = current_path / name
            if candidate.is_symlink() or name.lower() in MODEL_DIR_NAMES:
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in files:
            path = current_path / name
            if path.is_symlink():
                continue
            yield path, is_model_artifact(path.relative_to(root))


def _judge_file_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "malformed"
    return "complete" if judge_payload_is_complete(payload) else "incomplete"


def _eval_data_candidates(project_root: Path, dataset: str) -> tuple[Path, Path]:
    if dataset == "pan":
        primary = project_root / "data" / "processed" / "pan_test_set.jsonl"
    else:
        primary = project_root / "data" / "processed" / "eval" / f"{dataset}_test.jsonl"
    relative = primary.relative_to(project_root / "data" / "processed")
    fallback = project_root / "data" / "processed" / "processed" / relative
    return primary, fallback


def _fingerprint(path: Path, dataset: str) -> dict[str, object]:
    digest = hashlib.sha256()
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
    return {
        "dataset": dataset,
        "path": path.as_posix(),
        "size": path.stat().st_size,
        "line_count": line_count,
        "sha256": digest.hexdigest(),
    }


def build_package_plan(
    project_root: Path,
    outputs_root: Path,
    targets: Sequence[FormalJudgeTarget],
    *,
    include_eval_data: bool = True,
    include_auxiliary: bool = True,
) -> PackagePlan:
    project_root = project_root.resolve()
    outputs_root = outputs_root.resolve()
    targets = list(targets)
    member_by_source: dict[Path, PackageMember] = {}
    missing_pan = []
    missing_judge = []
    excluded_count = 0
    excluded_bytes = 0

    for target in targets:
        identity = target.identity()
        pan_path = target.pan_results_path.resolve()
        if not pan_path.is_file():
            missing_pan.append({
                **identity,
                "pan_results_path": _portable_path(pan_path, project_root, outputs_root),
            })
        judge_path = pan_path.with_name("judge_results.json")
        judge_status = _judge_file_status(judge_path)
        if judge_status != "complete":
            missing_judge.append({
                **identity,
                "status": judge_status,
                "judge_results_path": _portable_path(judge_path, project_root, outputs_root),
            })

    target_groups: dict[Path, list[FormalJudgeTarget]] = {}
    for target in targets:
        target_groups.setdefault(target.run_root.resolve(), []).append(target)

    missing_roots = []
    for run_root, root_targets in sorted(target_groups.items(), key=lambda item: item[0].as_posix()):
        if not run_root.is_dir():
            missing_roots.append({
                "path": _portable_path(run_root, project_root, outputs_root),
                "targets": [target.identity() for target in root_targets],
            })
            continue
        for path, excluded in _iter_tree_files(run_root):
            if excluded:
                excluded_count += 1
                excluded_bytes += path.stat().st_size
                continue
            resolved = path.resolve()
            member_by_source.setdefault(
                resolved, _member(resolved, project_root, outputs_root)
            )

    missing_eval_data = []
    fingerprints = []
    if include_eval_data:
        for dataset in sorted({target.dataset for target in targets}):
            paths = [
                candidate
                for candidate in _eval_data_candidates(project_root, dataset)
                if candidate.is_file()
            ]
            if not paths:
                missing_eval_data.append(dataset)
                continue
            for path in paths:
                resolved = path.resolve()
                member_by_source.setdefault(
                    resolved, _member(resolved, project_root, outputs_root)
                )
                fingerprint = _fingerprint(resolved, dataset)
                fingerprint["path"] = _portable_path(resolved, project_root, outputs_root)
                fingerprints.append(fingerprint)

    if include_auxiliary:
        for dirname in AUXILIARY_OUTPUT_DIRS:
            root = outputs_root / dirname
            if not root.is_dir():
                continue
            for path, excluded in _iter_tree_files(root):
                if excluded:
                    excluded_count += 1
                    excluded_bytes += path.stat().st_size
                    continue
                resolved = path.resolve()
                member_by_source.setdefault(
                    resolved, _member(resolved, project_root, outputs_root)
                )
        for pattern in REPORT_PATTERNS:
            for path in project_root.glob(pattern):
                if path.is_file() and not path.is_symlink():
                    resolved = path.resolve()
                    member_by_source.setdefault(
                        resolved, _member(resolved, project_root, outputs_root)
                    )

    members = sorted(member_by_source.values(), key=lambda member: member.archive_path.as_posix())
    archive_paths = [member.archive_path.as_posix() for member in members]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("multiple source files map to the same archive path")

    return PackagePlan(
        project_root=project_root,
        outputs_root=outputs_root,
        targets=targets,
        members=members,
        missing_run_roots=missing_roots,
        missing_expected_pan_results=missing_pan,
        missing_or_invalid_judge_results=missing_judge,
        missing_eval_data=missing_eval_data,
        eval_data_fingerprints=fingerprints,
        excluded_file_count=excluded_count,
        excluded_file_bytes=excluded_bytes,
    )


def _sidecar_path(archive_path: Path) -> Path:
    name = archive_path.name
    stem = name[:-7] if name.endswith(".tar.gz") else archive_path.stem
    return archive_path.with_name(f"{stem}_manifest.json")


def write_package(plan: PackagePlan, archive_path: Path) -> Path:
    archive_path = archive_path.resolve()
    sidecar_path = _sidecar_path(archive_path)
    if archive_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"package output already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = plan.manifest()
    manifest["archive_path"] = archive_path.as_posix()
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    temp_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    temp_sidecar = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    try:
        with tarfile.open(temp_archive, "w:gz") as handle:
            for member in plan.members:
                handle.add(
                    member.source_path,
                    arcname=member.archive_path.as_posix(),
                    recursive=False,
                )
            info = tarfile.TarInfo("llm_judge_package_manifest.json")
            info.size = len(payload)
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            handle.addfile(info, io.BytesIO(payload))
        temp_sidecar.write_bytes(payload)
        temp_archive.replace(archive_path)
        temp_sidecar.replace(sidecar_path)
    except BaseException:
        temp_archive.unlink(missing_ok=True)
        temp_sidecar.unlink(missing_ok=True)
        raise
    return sidecar_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive-name", default="")
    parser.add_argument("--no-eval-data", action="store_true")
    parser.add_argument("--no-auxiliary", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    outputs_root = (args.outputs_root or project_root / "outputs").resolve()
    output_dir = (args.output_dir or project_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = args.archive_name or f"llm_judge_artifacts_{timestamp}.tar.gz"
    if not archive_name.endswith(".tar.gz"):
        parser.error("--archive-name must end with .tar.gz")
    archive_path = output_dir / archive_name

    targets = iter_formal_targets(outputs_root)
    plan = build_package_plan(
        project_root,
        outputs_root,
        targets,
        include_eval_data=not args.no_eval_data,
        include_auxiliary=not args.no_auxiliary,
    )
    print(
        f"formal_targets={len(targets)} files={len(plan.members)} "
        f"bytes={sum(member.size for member in plan.members)} "
        f"missing_pan={len(plan.missing_expected_pan_results)} "
        f"missing_judge={len(plan.missing_or_invalid_judge_results)} "
        f"excluded_model_files={plan.excluded_file_count}"
    )
    for item in plan.missing_expected_pan_results:
        print(f"[package][WARN] missing pan_results: {item}")
    for item in plan.missing_or_invalid_judge_results:
        print(f"[package][WARN] judge result issue: {item}")

    if not args.dry_run:
        manifest_path = write_package(plan, archive_path)
        print(archive_path.as_posix())
        print(manifest_path.as_posix())

    has_missing = bool(
        plan.missing_expected_pan_results
        or plan.missing_or_invalid_judge_results
        or plan.missing_run_roots
    )
    return 2 if has_missing and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
