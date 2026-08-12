from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if float(self.temperature) < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be in (0,1]")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")


@dataclass(frozen=True)
class BenchmarkRequest:
    name: str
    asset_path: Path
    decode: DecodeConfig = DecodeConfig()


@dataclass(frozen=True)
class BenchmarkPreflight:
    status: str
    benchmark: str
    asset_path: str
    decode: DecodeConfig
    issues: tuple[dict, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["decode"] = asdict(self.decode)
        return payload


def preflight_benchmark(request: BenchmarkRequest) -> BenchmarkPreflight:
    path = Path(request.asset_path)
    issues: list[dict] = []
    if not path.exists():
        issues.append(
            {
                "code": "BENCHMARK_ASSET_MISSING",
                "message": f"benchmark asset is missing: {path}",
                "suggestion": "prepare the benchmark under persistent SAFETY_DATA_ROOT",
            }
        )
    return BenchmarkPreflight(
        status="READY" if not issues else "BLOCKED",
        benchmark=str(request.name),
        asset_path=str(path),
        decode=request.decode,
        issues=tuple(issues),
    )


def decode_cli_args(decode: DecodeConfig) -> tuple[str, ...]:
    return (
        "--temperature",
        str(float(decode.temperature)),
        "--top-p",
        str(float(decode.top_p)),
        "--max-new-tokens",
        str(int(decode.max_new_tokens)),
    )
