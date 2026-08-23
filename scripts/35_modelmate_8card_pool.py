from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.catalog import load_catalog
from src.ablations.modelmate_pool import (
    ROUND_SPECS,
    PoolLayout,
    RoundSpec,
    ShardResult,
    build_round_plan,
    build_shard_command,
    derive_pool_layout,
    run_shard_pool,
    select_round_cells,
)
from src.ablations.planner import build_catalog_plan


DEFAULT_MODEL_ROOT = Path("/opt/dpcvol/models/safetytransfer")
DEFAULT_DATA_ROOT = Path("/opt/dpcvol/datasets/safetytransfer")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "ablation-outputs" / "iclr-886760f"
DEFAULT_ASSET_MANIFEST = PROJECT_ROOT / "configs" / "ablations" / "assets.modelmate.template.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_plan(path: Path, plan) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in plan.cells:
            handle.write(
                json.dumps(
                    {
                        "axes": dict(cell.axes),
                        "cell_id": cell.cell_id,
                        "depends_on": list(cell.depends_on),
                        "experiment_id": cell.experiment_id,
                        "output_dir": cell.output_dir,
                        "overrides": dict(cell.overrides),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_persistent_root(path: Path, label: str) -> Path:
    source = Path(path).expanduser()
    resolved = source.resolve()
    normalized = resolved.as_posix().lower().rstrip("/") + "/"
    forbidden = ("/tmp/", "/cache/", "/home/work/user-job-dir/app/")
    if any(normalized.startswith(prefix) for prefix in forbidden):
        raise ValueError(f"{label} root must be persistent, not {resolved}")
    return resolved


def _parse_device_ids(raw: str, requested_devices: int) -> tuple[int, ...]:
    if requested_devices <= 0:
        raise ValueError("--devices must be positive")
    if not raw.strip():
        return tuple(range(requested_devices))
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if len(values) != requested_devices:
        raise ValueError("--device-ids count must equal --devices")
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("--device-ids must be unique non-negative integers")
    return values


def _runtime_environment(model_root: Path, data_root: Path, output_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        SAFETY_MODEL_ROOT=str(model_root),
        SAFETY_DATA_ROOT=str(data_root),
        SAFETY_OUTPUT_ROOT=str(output_root),
        HF_HOME=str(data_root / "_hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return environment


def _worker_environment(
    environment: Mapping[str, str],
    *,
    device: str,
    device_id: int,
) -> dict[str, str]:
    isolated = dict(environment)
    if device != "npu":
        return isolated
    raw_visible = str(environment.get("ASCEND_RT_VISIBLE_DEVICES", "")).strip()
    if not raw_visible:
        isolated["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
        return isolated
    visible = tuple(token.strip() for token in raw_visible.split(",") if token.strip())
    if not 0 <= device_id < len(visible):
        raise RuntimeError(
            f"worker requested logical NPU {device_id}, but the scheduler exposed "
            f"only {len(visible)} visible NPU entries"
        )
    # The existing single-device launcher intentionally uses logical npu:0.
    # Narrowing each child to one scheduler-provided token maps that npu:0 to
    # a distinct physical card without changing the parent scheduler contract.
    isolated["ASCEND_RT_VISIBLE_DEVICES"] = visible[device_id]
    return isolated


def verify_npu_devices(device_ids: Sequence[int]) -> dict[str, object]:
    import torch
    import torch_npu  # noqa: F401 - importing registers the NPU backend

    available = int(torch.npu.device_count())
    if not device_ids:
        raise RuntimeError("no NPU device IDs were requested")
    if max(device_ids) >= available:
        raise RuntimeError(
            f"requested NPU {max(device_ids)}, but torch reports only {available} devices"
        )
    probes = []
    for device_id in device_ids:
        device = f"npu:{device_id}"
        torch.npu.set_device(device)
        value = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        loss = value.square().sum()
        loss.backward()
        torch.npu.synchronize()
        observed = float(loss.detach().cpu().item())
        gradient = [float(item) for item in value.grad.detach().cpu().tolist()]
        if observed != 14.0 or gradient != [2.0, 4.0, 6.0]:
            raise RuntimeError(
                f"NPU {device_id} returned unexpected probe values: loss={observed}, grad={gradient}"
            )
        probes.append(
            {
                "device_id": int(device_id),
                "loss": observed,
                "gradient": gradient,
            }
        )
    return {
        "schema_version": 1,
        "checked_at": _utc_now(),
        "torch_device_count": available,
        "requested_device_ids": [int(value) for value in device_ids],
        "probes": probes,
    }


class SubprocessShardExecutor:
    def __init__(
        self,
        *,
        project_root: Path,
        round_root: Path,
        environment: Mapping[str, str],
        plan_path: Path,
        state_root: Path,
        asset_manifest: Path,
        layout: PoolLayout,
        python_executable: str,
        device: str,
        dry_run: bool,
    ) -> None:
        self.project_root = Path(project_root)
        self.round_root = Path(round_root)
        self.environment = dict(environment)
        self.plan_path = Path(plan_path)
        self.state_root = Path(state_root)
        self.asset_manifest = Path(asset_manifest)
        self.layout = layout
        self.python_executable = str(python_executable)
        self.device = str(device)
        self.dry_run = bool(dry_run)
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen] = {}

    def __call__(self, shard_index: int, device_id: int) -> int:
        shard_root = (
            self.round_root
            / "shards"
            / f"shard-{shard_index:05d}-of-{self.layout.shard_count:05d}"
        )
        shard_root.mkdir(parents=True, exist_ok=True)
        log_path = shard_root / "worker.log"
        command = build_shard_command(
            python_executable=self.python_executable,
            project_root=self.project_root,
            plan_path=self.plan_path,
            state_root=self.state_root,
            asset_manifest=self.asset_manifest,
            layout=self.layout,
            shard_index=shard_index,
            device=self.device,
            device_id=device_id,
            dry_run=self.dry_run,
        )
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(
                json.dumps(
                    {
                        "event": "shard_start",
                        "created_at": _utc_now(),
                        "shard_index": shard_index,
                        "device_id": device_id,
                        "command": list(command),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            process = subprocess.Popen(
                list(command),
                cwd=str(self.project_root),
                env=_worker_environment(
                    self.environment,
                    device=self.device,
                    device_id=device_id,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name != "nt",
            )
            with self._lock:
                self._processes[process.pid] = process
            try:
                return int(process.wait())
            finally:
                with self._lock:
                    self._processes.pop(process.pid, None)

    def terminate_all(self) -> None:
        with self._lock:
            processes = tuple(self._processes.values())
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                continue
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            json.dumps(
                {"event": "command", "created_at": _utc_now(), "command": list(command)},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _summary_payload(
    *,
    round_name: str,
    layout: PoolLayout,
    results: Sequence[ShardResult],
    status: str,
    plan_sha256: str,
    dry_run: bool,
) -> dict[str, object]:
    executed = {result.shard_index for result in results}
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": status,
        "round": round_name,
        "dry_run": bool(dry_run),
        "layout": asdict(layout),
        "plan_sha256": plan_sha256,
        "executed_shards": sorted(executed),
        "pending_shards": sorted(set(range(layout.shard_count)).difference(executed)),
        "failed_shards": [
            result.shard_index
            for result in results
            if result.returncode != 0 or result.error
        ],
        "results": [asdict(result) for result in results],
    }


def _require_completed_prerequisites(
    output_root: Path,
    spec: RoundSpec,
    catalog,
    complete_plan,
) -> None:
    for prerequisite in spec.prerequisites:
        prerequisite_spec = ROUND_SPECS[prerequisite]
        expected_cell_ids = {
            cell.cell_id
            for cell in select_round_cells(catalog, complete_plan, prerequisite_spec)
        }
        summary_path = Path(output_root) / "jobs" / prerequisite / "pool-summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"prerequisite {prerequisite} is missing: {summary_path}; "
                "run the required round first"
            )
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"prerequisite {prerequisite} summary is unreadable: {exc}"
            ) from exc
        if payload.get("status") != "READY" or payload.get("dry_run") is not False:
            raise RuntimeError(
                f"prerequisite {prerequisite} must be a real READY run; "
                f"found status={payload.get('status')!r}, dry_run={payload.get('dry_run')!r}"
            )
        failed = payload.get("failed_shards", [])
        pending = payload.get("pending_shards", [])
        if failed or pending:
            raise RuntimeError(
                f"prerequisite {prerequisite} has unfinished shards: "
                f"failed={failed!r}, pending={pending!r}"
            )
        if payload.get("expected_cells") != prerequisite_spec.expected_cells:
            raise RuntimeError(
                f"prerequisite {prerequisite} expected_cells is "
                f"{payload.get('expected_cells')!r}, expected "
                f"{prerequisite_spec.expected_cells}"
            )
        status_path = summary_path.with_name("status.json")
        if not status_path.is_file():
            raise RuntimeError(
                f"prerequisite {prerequisite} status is missing: {status_path}"
            )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"prerequisite {prerequisite} status is unreadable: {exc}"
            ) from exc
        rows = status.get("cells") if isinstance(status, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(
                f"prerequisite {prerequisite} status lacks a cells list"
            )
        cell_ids = {
            str(row.get("cell_id", ""))
            for row in rows
            if isinstance(row, dict) and row.get("cell_id")
        }
        if (
            len(rows) != prerequisite_spec.expected_cells
            or cell_ids != expected_cell_ids
            or any(
                not isinstance(row, dict) or row.get("state") != "COMPLETED"
                for row in rows
            )
        ):
            raise RuntimeError(
                f"prerequisite {prerequisite} status must contain exactly "
                f"{prerequisite_spec.expected_cells} unique COMPLETED cells"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one allowlisted ICLR ablation round as single-NPU workers on an 8-card node."
    )
    parser.add_argument("--round", choices=tuple(ROUND_SPECS), default="p0-smoke")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--asset-manifest", default=str(DEFAULT_ASSET_MANIFEST))
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--device-ids", default="")
    parser.add_argument("--logical-shards", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "npu", "cpu"], default="auto")
    parser.add_argument("--launch-stagger-seconds", type=float, default=15.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-device-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.logical_shards <= 0:
        raise ValueError("--logical-shards must be positive")
    if args.launch_stagger_seconds < 0:
        raise ValueError("--launch-stagger-seconds cannot be negative")
    if args.skip_device_check and not args.dry_run:
        raise ValueError("--skip-device-check is allowed only with --dry-run")

    spec = ROUND_SPECS[args.round]
    device = spec.default_device if args.device == "auto" else args.device
    if device == "cpu" and spec.default_device == "npu":
        raise ValueError(f"round {spec.name} requires NPU execution")

    model_root = _validate_persistent_root(Path(args.model_root), "model")
    data_root = _validate_persistent_root(Path(args.data_root), "data")
    output_root = _validate_persistent_root(Path(args.output_root), "output")
    asset_manifest = Path(args.asset_manifest).expanduser().resolve()
    if not asset_manifest.is_file():
        raise FileNotFoundError(f"asset manifest is missing: {asset_manifest}")

    requested_device_ids = _parse_device_ids(args.device_ids, args.devices)
    catalog = load_catalog(PROJECT_ROOT / "configs" / "ablations" / "catalog.yaml")
    complete = build_catalog_plan(
        catalog,
        output_root=output_root / "cell-outputs",
        scope="all",
    )
    plan = build_round_plan(catalog, complete, spec)
    layout = derive_pool_layout(
        cell_count=len(plan.cells),
        requested_shards=args.logical_shards,
        requested_devices=args.devices,
    )
    device_ids = requested_device_ids[: layout.device_count]

    round_root = output_root / "jobs" / spec.name
    state_root = output_root / "jobs" / spec.state_group / (
        "dry-run-state" if args.dry_run else "run-state"
    )
    plan_path = round_root / "plan.jsonl"
    plan_sha256 = _write_plan(plan_path, plan)
    environment = _runtime_environment(model_root, data_root, output_root)

    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "round": spec.name,
        "state_group": spec.state_group,
        "device": device,
        "device_ids": list(device_ids),
        "layout": asdict(layout),
        "model_root": str(model_root),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "asset_manifest": str(asset_manifest),
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "dry_run": bool(args.dry_run),
    }
    _atomic_json(round_root / "job-metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True), flush=True)

    preflight_command = (
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "30_ablation.py"),
        "preflight",
        "--plan",
        str(plan_path),
        "--asset-manifest",
        str(asset_manifest),
        "--output",
        str(round_root / "preflight.json"),
        "--device",
        device,
    )
    if _run_logged(
        preflight_command,
        cwd=PROJECT_ROOT,
        environment=environment,
        log_path=round_root / "preflight.log",
    ):
        _atomic_json(
            round_root / "preflight-summary.json",
            _summary_payload(
                round_name=spec.name,
                layout=layout,
                results=(),
                status="PREFLIGHT_BLOCKED",
                plan_sha256=plan_sha256,
                dry_run=args.dry_run,
            ),
        )
        return 3

    if not args.preflight_only and not args.dry_run:
        _require_completed_prerequisites(output_root, spec, catalog, complete)

    if device == "npu" and not args.skip_device_check:
        try:
            probe = verify_npu_devices(device_ids)
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "checked_at": _utc_now(),
                "status": "FAILED",
                "requested_device_ids": list(device_ids),
                "error": f"{type(exc).__name__}: {exc}",
            }
            _atomic_json(round_root / "device-preflight.json", failure)
            print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return 3
        probe["status"] = "READY"
        _atomic_json(round_root / "device-preflight.json", probe)
        print(
            json.dumps(
                {"status": "NPU_READY", "device_ids": list(device_ids)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.preflight_only:
        _atomic_json(
            round_root / "preflight-summary.json",
            _summary_payload(
                round_name=spec.name,
                layout=layout,
                results=(),
                status="PREFLIGHT_READY",
                plan_sha256=plan_sha256,
                dry_run=args.dry_run,
            ),
        )
        print(json.dumps({"status": "PREFLIGHT_READY", "round": spec.name}), flush=True)
        return 0

    executor = SubprocessShardExecutor(
        project_root=PROJECT_ROOT,
        round_root=round_root,
        environment=environment,
        plan_path=plan_path,
        state_root=state_root,
        asset_manifest=asset_manifest,
        layout=layout,
        python_executable=sys.executable,
        device=device,
        dry_run=args.dry_run,
    )
    received_signal = 0

    def terminate_on_signal(signum, frame) -> None:
        del frame
        nonlocal received_signal
        received_signal = int(signum)
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, terminate_on_signal)
    try:
        results = run_shard_pool(
            shard_count=layout.shard_count,
            device_ids=device_ids,
            worker=executor,
            stagger_seconds=args.launch_stagger_seconds,
        )
    except KeyboardInterrupt:
        executor.terminate_all()
        _atomic_json(
            round_root / "pool-summary.json",
            _summary_payload(
                round_name=spec.name,
                layout=layout,
                results=(),
                status="INTERRUPTED",
                plan_sha256=plan_sha256,
                dry_run=args.dry_run,
            ),
        )
        return 128 + (received_signal or int(signal.SIGINT))
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    pool_ready = (
        len(results) == layout.shard_count
        and all(result.returncode == 0 and not result.error for result in results)
    )
    if not pool_ready:
        payload = _summary_payload(
            round_name=spec.name,
            layout=layout,
            results=results,
            status="SHARD_FAILED",
            plan_sha256=plan_sha256,
            dry_run=args.dry_run,
        )
        _atomic_json(round_root / "pool-summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 3

    status_path = round_root / ("dry-run-status.json" if args.dry_run else "status.json")
    status_command = (
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "30_ablation.py"),
        "status",
        "--plan",
        str(plan_path),
        "--state-root",
        str(state_root),
        "--output",
        str(status_path),
    )
    if _run_logged(
        status_command,
        cwd=PROJECT_ROOT,
        environment=environment,
        log_path=round_root / "status.log",
    ):
        _atomic_json(
            round_root / "pool-summary.json",
            _summary_payload(
                round_name=spec.name,
                layout=layout,
                results=results,
                status="STATUS_FAILED",
                plan_sha256=plan_sha256,
                dry_run=args.dry_run,
            ),
        )
        return 3

    rows = json.loads(status_path.read_text(encoding="utf-8"))["cells"]
    acceptable = {"READY", "COMPLETED"} if args.dry_run else {"COMPLETED"}
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("state", "UNKNOWN"))
        counts[state] = counts.get(state, 0) + 1
    final_ready = len(rows) == len(plan.cells) and set(counts).issubset(acceptable)
    final_payload = _summary_payload(
        round_name=spec.name,
        layout=layout,
        results=results,
        status="READY" if final_ready else "INCOMPLETE",
        plan_sha256=plan_sha256,
        dry_run=args.dry_run,
    )
    final_payload["cell_states"] = counts
    final_payload["expected_cells"] = len(plan.cells)
    _atomic_json(round_root / "pool-summary.json", final_payload)
    print(json.dumps(final_payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if final_ready else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
