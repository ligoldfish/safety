from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .template_qwen import build_qwen_messages
from src.utils.io import ensure_dir, write_json, write_jsonl


PAN_REQUIRED_FILES = [
    "toxicity.csv",
    "safety.csv",
    "add_moderation.csv",
    "sr_moderation.csv",
]


def _slugify(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def truncate_pan_prompt(text: str, max_length: int) -> str:
    if max_length <= 0 or len(text) <= max_length:
        return text
    half_length = max_length // 2
    return text[:half_length] + text[-half_length:]


def _split_train_test(
    df: pd.DataFrame,
    test_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:test_size].copy()
    train_df = shuffled.iloc[test_size:].copy()
    return train_df, test_df


def _copy_pan_sources(pan_repo_dir: Path, raw_dir: Path) -> Dict[str, str]:
    source_dir = pan_repo_dir / "data"
    if not source_dir.exists():
        raise FileNotFoundError(f"Pan data directory not found: {source_dir}")

    ensure_dir(raw_dir)
    copied: Dict[str, str] = {}
    for filename in PAN_REQUIRED_FILES:
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Required Pan source file not found: {source_path}")
        target_path = raw_dir / filename
        shutil.copy2(source_path, target_path)
        copied[filename] = str(target_path)
    return copied


def _frame_to_records(
    df: pd.DataFrame,
    pan_split: str,
    label: str,
    target_column: str,
    include_system_prompt: bool,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for local_idx, row in df.reset_index(drop=True).iterrows():
        source_dataset = str(row["source_dataset"])
        source_row = int(row["source_row"])
        sample_id = f"{pan_split}_{_slugify(source_dataset)}_{source_row:05d}_{local_idx:05d}"
        user_text = str(row["user_text"])
        record = {
            "id": sample_id,
            "pan_split": pan_split,
            "source_dataset": source_dataset,
            "source_row": source_row,
            "label": label,
            "user_text": user_text,
            "target_response": str(row[target_column]),
            "accept_response": str(row["accept_response"]),
            "rejected_response": str(row["rejected_response"]),
            "messages": build_qwen_messages(
                user_text=user_text,
                system_prompt=system_prompt,
                include_system_prompt=include_system_prompt,
            ),
        }
        if "idx_org" in row and pd.notna(row["idx_org"]):
            record["idx_org"] = int(row["idx_org"])
        if "method" in row and pd.notna(row["method"]):
            record["method"] = str(row["method"])
        if "category" in row and pd.notna(row["category"]):
            record["category"] = str(row["category"])
        if "source" in row and pd.notna(row["source"]):
            record["source"] = str(row["source"])
        if "source_prompt_origin" in row and pd.notna(row["source_prompt_origin"]):
            record["source_prompt_origin"] = str(row["source_prompt_origin"])
        if "forbidden_prompt" in row and pd.notna(row["forbidden_prompt"]):
            record["forbidden_prompt"] = str(row["forbidden_prompt"])
        records.append(record)
    return records


def build_pan_train_test_records(
    raw_dir: str | Path,
    exposure_size: int = 60,
    pan_test_size_per_type: int = 60,
    pan_train_size: int = 2600,
    max_prompt_chars: int = 2048,
    seed: int = 42,
    system_prompt: str = "You are a helpful assistant.",
    include_system_prompt: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    raw_path = Path(raw_dir)
    toxicity_df = pd.read_csv(raw_path / "toxicity.csv")
    safety_df = pd.read_csv(raw_path / "safety.csv")
    add_moderation_df = pd.read_csv(raw_path / "add_moderation.csv")
    sr_moderation_df = pd.read_csv(raw_path / "sr_moderation.csv")

    toxicity_df = toxicity_df.rename(columns={"Unnamed: 0": "source_row"})
    safety_df = safety_df.rename(columns={"Unnamed: 0": "source_row"})
    add_moderation_df = add_moderation_df.rename(columns={"Unnamed: 0": "source_row"})
    sr_moderation_df = sr_moderation_df.rename(columns={"Unnamed: 0": "idx_org"})

    toxicity_df["source_dataset"] = toxicity_df["method"].astype(str)
    safety_df["source_dataset"] = safety_df["method"].astype(str)
    add_moderation_df["source_dataset"] = add_moderation_df["method"].astype(str)
    safety_df["source_prompt_origin"] = "or-bench"
    add_moderation_df["source_prompt_origin"] = add_moderation_df["source_dataset"].map(
        {
            "data/sr_moderation.csv": "strongreject",
            "data/adv_moderation.csv": "adv_moderation",
        }
    ).fillna(add_moderation_df["source_dataset"])
    toxicity_df["source_prompt_origin"] = "strongreject"

    sr_meta = sr_moderation_df[["idx_org", "category", "source"]].copy()
    toxicity_df = toxicity_df.merge(sr_meta, on="idx_org", how="left")
    add_moderation_df = add_moderation_df.merge(sr_meta, on="idx_org", how="left")

    harmful_size = pan_train_size // 2
    safe_size = pan_train_size - harmful_size
    harmful_methods = [str(x) for x in pd.unique(toxicity_df["method"].dropna())]
    harmful_remaining = harmful_size - exposure_size * len(harmful_methods)
    if harmful_remaining < 0:
        raise ValueError("Not enough harmful slots left after exposure selection.")
    if harmful_remaining > len(add_moderation_df):
        raise ValueError("add_moderation.csv does not have enough rows for the requested train size.")

    harmful_train_frames: Dict[str, pd.DataFrame] = {}
    harmful_test_frames: Dict[str, pd.DataFrame] = {}
    for method in harmful_methods:
        method_df = toxicity_df[toxicity_df["method"] == method].copy()
        train_df, test_df = _split_train_test(method_df, pan_test_size_per_type, seed)
        harmful_train_frames[method] = train_df
        harmful_test_frames[method] = test_df

    train_toxic_df = pd.concat(
        [harmful_train_frames[method].iloc[:exposure_size].copy() for method in harmful_methods]
        + [add_moderation_df.iloc[:harmful_remaining].copy()],
        ignore_index=True,
    )
    train_toxic_df["user_text"] = train_toxic_df["jailbroken_prompt"].astype(str)
    train_toxic_df["accept_response"] = train_toxic_df["accept"].astype(str)
    train_toxic_df["rejected_response"] = train_toxic_df["rejected"].astype(str)

    train_safe_df = safety_df.iloc[:safe_size].copy()
    train_safe_df["user_text"] = train_safe_df["jailbroken_prompt"].astype(str)
    train_safe_df["accept_response"] = train_safe_df["accept"].astype(str)
    train_safe_df["rejected_response"] = train_safe_df["rejected"].astype(str)

    test_toxic_df = pd.concat(
        [harmful_test_frames[method].copy() for method in harmful_methods],
        ignore_index=True,
    )
    test_toxic_df["user_text"] = test_toxic_df["jailbroken_prompt"].astype(str)
    test_toxic_df["accept_response"] = test_toxic_df["accept"].astype(str)
    test_toxic_df["rejected_response"] = test_toxic_df["rejected"].astype(str)

    test_safe_df = safety_df.iloc[-len(test_toxic_df) :].copy()
    test_safe_df["user_text"] = test_safe_df["jailbroken_prompt"].astype(str)
    test_safe_df["accept_response"] = test_safe_df["accept"].astype(str)
    test_safe_df["rejected_response"] = test_safe_df["rejected"].astype(str)

    for frame in [train_toxic_df, train_safe_df, test_toxic_df, test_safe_df]:
        frame["user_text"] = frame["user_text"].astype(str).apply(
            lambda x: truncate_pan_prompt(x, max_prompt_chars)
        )

    train_records = _frame_to_records(
        df=train_toxic_df,
        pan_split="pan_train",
        label="harmful",
        target_column="rejected_response",
        include_system_prompt=include_system_prompt,
        system_prompt=system_prompt,
    ) + _frame_to_records(
        df=train_safe_df,
        pan_split="pan_train",
        label="harmless",
        target_column="accept_response",
        include_system_prompt=include_system_prompt,
        system_prompt=system_prompt,
    )
    test_records = _frame_to_records(
        df=test_toxic_df,
        pan_split="pan_test",
        label="harmful",
        target_column="rejected_response",
        include_system_prompt=include_system_prompt,
        system_prompt=system_prompt,
    ) + _frame_to_records(
        df=test_safe_df,
        pan_split="pan_test",
        label="harmless",
        target_column="accept_response",
        include_system_prompt=include_system_prompt,
        system_prompt=system_prompt,
    )

    metadata = {
        "pan_reconstruction": {
            "exposure_size": exposure_size,
            "pan_test_size_per_type": pan_test_size_per_type,
            "pan_train_size": pan_train_size,
            "max_prompt_chars": max_prompt_chars,
            "harmful_size": harmful_size,
            "safe_size": safe_size,
            "harmful_remaining_from_add_moderation": harmful_remaining,
            "harmful_methods": harmful_methods,
            "train_source_counts": pd.Series(
                [record["source_dataset"] for record in train_records]
            ).value_counts().to_dict(),
            "test_source_counts": pd.Series(
                [record["source_dataset"] for record in test_records]
            ).value_counts().to_dict(),
        }
    }
    return train_records, test_records, metadata


def split_alignment_and_validation(
    train_records: List[Dict[str, Any]],
    alignment_size: int,
    analysis_val_size: int,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if alignment_size + analysis_val_size != len(train_records):
        raise ValueError(
            "alignment_size + analysis_val_size must equal the reconstructed Pan train size."
        )

    df = pd.DataFrame(train_records)
    total_size = len(df)
    val_ratio = analysis_val_size / total_size
    df["stratify_key"] = (
        df["label"].astype(str)
        + "::"
        + df["source_dataset"].astype(str)
        + "::"
        + df.get("category", pd.Series([None] * len(df))).fillna("NA").astype(str)
    )

    val_counts: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    assigned = 0
    for stratify_key, group_df in df.groupby("stratify_key", sort=True):
        raw_count = len(group_df) * val_ratio
        base_count = int(raw_count)
        val_counts[str(stratify_key)] = base_count
        remainders.append((raw_count - base_count, str(stratify_key)))
        assigned += base_count

    remaining = analysis_val_size - assigned
    for _, stratify_key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        val_counts[stratify_key] += 1

    alignment_records: List[Dict[str, Any]] = []
    analysis_val_records: List[Dict[str, Any]] = []
    for stratify_key, group_df in df.groupby("stratify_key", sort=True):
        shuffled = group_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_count = val_counts[str(stratify_key)]
        analysis_val_records.extend(
            shuffled.iloc[:val_count].drop(columns=["stratify_key"]).to_dict("records")
        )
        alignment_records.extend(
            shuffled.iloc[val_count:].drop(columns=["stratify_key"]).to_dict("records")
        )

    summary = {
        "alignment_size": len(alignment_records),
        "analysis_val_size": len(analysis_val_records),
        "alignment_label_counts": pd.Series(
            [record["label"] for record in alignment_records]
        ).value_counts().to_dict(),
        "analysis_val_label_counts": pd.Series(
            [record["label"] for record in analysis_val_records]
        ).value_counts().to_dict(),
        "alignment_source_counts": pd.Series(
            [record["source_dataset"] for record in alignment_records]
        ).value_counts().to_dict(),
        "analysis_val_source_counts": pd.Series(
            [record["source_dataset"] for record in analysis_val_records]
        ).value_counts().to_dict(),
    }
    return alignment_records, analysis_val_records, summary


def build_sanity_test_records(
    test_records: List[Dict[str, Any]],
    *,
    size_per_label: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if size_per_label <= 0:
        raise ValueError("size_per_label must be positive.")

    df = pd.DataFrame(test_records)
    sanity_frames: List[pd.DataFrame] = []
    for label in ["harmful", "harmless"]:
        label_df = df[df["label"] == label].copy()
        if len(label_df) < size_per_label:
            raise ValueError(
                f"Not enough {label} samples to build sanity_test_set.jsonl: "
                f"requested {size_per_label}, found {len(label_df)}."
            )
        sampled = label_df.sample(n=size_per_label, random_state=seed).reset_index(drop=True)
        sampled["pan_split"] = "sanity_test"
        sampled["id"] = [
            str(sample_id).replace("pan_test_", "sanity_test_", 1)
            for sample_id in sampled["id"].tolist()
        ]
        sanity_frames.append(sampled)

    sanity_df = pd.concat(sanity_frames, ignore_index=True)
    sanity_records = sanity_df.to_dict("records")
    metadata = {
        "sanity_test_size": len(sanity_records),
        "sanity_test_size_per_label": size_per_label,
        "sanity_test_label_counts": pd.Series(
            [record["label"] for record in sanity_records]
        ).value_counts().to_dict(),
        "sanity_test_source_counts": pd.Series(
            [record["source_dataset"] for record in sanity_records]
        ).value_counts().to_dict(),
    }
    return sanity_records, metadata


def prepare_phase1_datasets(
    pan_repo_dir: str | Path,
    raw_dir: str | Path,
    processed_dir: str | Path,
    metadata_dir: str | Path,
    exposure_size: int,
    pan_test_size_per_type: int,
    pan_train_size: int,
    alignment_size: int,
    analysis_val_size: int,
    sanity_test_size_per_label: int,
    max_prompt_chars: int,
    seed: int,
    system_prompt: str,
    include_system_prompt: bool,
) -> Dict[str, Any]:
    pan_repo_path = Path(pan_repo_dir)
    raw_path = Path(raw_dir)
    processed_path = ensure_dir(processed_dir)
    metadata_path = ensure_dir(metadata_dir)

    copied_files = _copy_pan_sources(pan_repo_path, raw_path)
    train_records, test_records, pan_meta = build_pan_train_test_records(
        raw_dir=raw_path,
        exposure_size=exposure_size,
        pan_test_size_per_type=pan_test_size_per_type,
        pan_train_size=pan_train_size,
        max_prompt_chars=max_prompt_chars,
        seed=seed,
        system_prompt=system_prompt,
        include_system_prompt=include_system_prompt,
    )
    alignment_records, analysis_val_records, split_meta = split_alignment_and_validation(
        train_records=train_records,
        alignment_size=alignment_size,
        analysis_val_size=analysis_val_size,
        seed=seed,
    )
    sanity_test_records, sanity_meta = build_sanity_test_records(
        test_records=test_records,
        size_per_label=sanity_test_size_per_label,
        seed=seed,
    )

    prompts_all = alignment_records + analysis_val_records + test_records + sanity_test_records
    write_jsonl(processed_path / "prompts_all.jsonl", prompts_all)
    write_jsonl(processed_path / "pan_train_set.jsonl", train_records)
    write_jsonl(processed_path / "alignment_set.jsonl", alignment_records)
    write_jsonl(processed_path / "analysis_val_set.jsonl", analysis_val_records)
    write_jsonl(processed_path / "pan_test_set.jsonl", test_records)
    write_jsonl(processed_path / "sanity_test_set.jsonl", sanity_test_records)

    split_info = {
        "seed": seed,
        "copied_files": copied_files,
        "pan_train_size": len(train_records),
        "pan_test_size": len(test_records),
        **pan_meta,
        **split_meta,
        **sanity_meta,
    }
    prompt_template_info = {
        "family": "Qwen3.5",
        "system_prompt": system_prompt,
        "include_system_prompt": include_system_prompt,
        "rendering_rule": "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "feature_position": "last non-padding token of the generation prompt, corresponding to the first assistant token prediction position",
    }
    write_json(metadata_path / "split_info.json", split_info)
    write_json(metadata_path / "prompt_template_info.json", prompt_template_info)

    return {
        "prompts_all_path": str(processed_path / "prompts_all.jsonl"),
        "pan_train_path": str(processed_path / "pan_train_set.jsonl"),
        "alignment_path": str(processed_path / "alignment_set.jsonl"),
        "analysis_val_path": str(processed_path / "analysis_val_set.jsonl"),
        "pan_test_path": str(processed_path / "pan_test_set.jsonl"),
        "sanity_test_path": str(processed_path / "sanity_test_set.jsonl"),
        "split_info_path": str(metadata_path / "split_info.json"),
        "prompt_template_info_path": str(metadata_path / "prompt_template_info.json"),
    }
