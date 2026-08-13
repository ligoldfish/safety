from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.manual_audit import build_blind_packet
from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a stratified, method-blind manual audit packet.")
    parser.add_argument("--input", required=True, help="JSONL with sample_id, method, stratum, prompt, response.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-stratum", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2042)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet, key = build_blind_packet(read_jsonl(args.input), per_stratum=args.per_stratum, seed=args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "packet.jsonl", packet)
    write_json(output / "blind_key.json", {"seed": args.seed, "mapping": key})
    print(json.dumps({"packet": str(output / "packet.jsonl"), "key": str(output / "blind_key.json"), "samples": len(packet)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
