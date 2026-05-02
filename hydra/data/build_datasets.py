#!/usr/bin/env python3
"""Build local prompt JSON files for hydra.log_heads sanity runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "hydra" / "data" / "sanity"


DATASET_SPECS = {
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "validation",
    },
    "litbench_train": {
        "path": "SAA-Lab/LitBench-Train",
        "name": None,
        "split": "train",
    },
    "litbench_test": {
        "path": "SAA-Lab/LitBench-Test",
        "name": None,
        "split": "train",
    },
}


def _one_based_choice_label(i: int) -> str:
    return chr(ord("A") + i)


def _format_arc(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    labels = list(sample["choices"]["label"])
    texts = list(sample["choices"]["text"])
    choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    instruction = (
        "Answer the multiple-choice science question.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Give the correct choice and a brief explanation."
    )
    prompt = f"USER: {instruction}\nASSISTANT:"
    return prompt, {
        "id": sample.get("id"),
        "answer": sample.get("answerKey"),
        "choices": dict(zip(labels, texts)),
    }


def _format_piqa(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = {
        "A": sample["sol1"],
        "B": sample["sol2"],
    }
    instruction = (
        "Pick the more plausible solution for the goal.\n\n"
        f"Goal: {sample['goal']}\n\n"
        f"A. {sample['sol1']}\n"
        f"B. {sample['sol2']}\n\n"
        "Give the correct choice and a brief explanation."
    )
    prompt = f"USER: {instruction}\nASSISTANT:"
    label = sample.get("label")
    answer = None if label is None else _one_based_choice_label(int(label))
    return prompt, {"answer": answer, "choices": choices}


def _format_litbench(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "prompt" not in sample and {"chosen_comment_id", "rejected_comment_id"}.issubset(sample):
        instruction = (
            "This LitBench test row provides only paired story comment IDs.\n\n"
            f"Chosen comment ID: {sample['chosen_comment_id']}\n"
            f"Rejected comment ID: {sample['rejected_comment_id']}\n\n"
            "Explain what information would be needed to compare the two stories."
        )
        prompt = f"USER: {instruction}\nASSISTANT:"
        return prompt, {
            "chosen_comment_id": sample.get("chosen_comment_id"),
            "rejected_comment_id": sample.get("rejected_comment_id"),
            "text_available": False,
        }

    instruction = (
        "Continue the creative-writing prompt with an engaging story.\n\n"
        f"Prompt: {sample['prompt']}"
    )
    prompt = f"USER: {instruction}\nASSISTANT:"
    return prompt, {
        "chosen_upvotes": sample.get("chosen_upvotes"),
        "rejected_upvotes": sample.get("rejected_upvotes"),
        "chosen_timestamp": str(sample.get("chosen_timestamp")),
        "rejected_timestamp": str(sample.get("rejected_timestamp")),
    }


def _format_generic(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if {"question", "choices"}.issubset(sample):
        return _format_arc(sample)
    if {"goal", "sol1", "sol2"}.issubset(sample):
        return _format_piqa(sample)
    if "prompt" in sample or {"chosen_comment_id", "rejected_comment_id"}.issubset(sample):
        return _format_litbench(sample)
    raise ValueError(f"Unsupported sample schema: {sorted(sample.keys())}")


def _resolve_split(ds: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(ds, Dataset):
        return ds
    if split in ds:
        return ds[split]
    if len(ds) == 1:
        only_split = next(iter(ds.keys()))
        print(f"[warn] split {split!r} not found; using only available split {only_split!r}")
        return ds[only_split]
    available = ", ".join(ds.keys())
    raise KeyError(f"Split {split!r} not found. Available splits: {available}")


def build_dataset(
    dataset_key: str,
    output_dir: Path,
    split: str | None = None,
    max_samples: int | None = None,
    force: bool = False,
) -> Path:
    spec = DATASET_SPECS[dataset_key]
    selected_split = split or spec["split"]
    suffix = "" if dataset_key.startswith("litbench_") else f"_{selected_split}"
    output_path = output_dir / f"{dataset_key}{suffix}.json"

    if output_path.exists() and not force:
        print(f"[info] using existing {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] loading {spec['path']} {spec['name'] or ''} split={selected_split}")
    if spec["name"] is None:
        raw = load_dataset(spec["path"])
    else:
        raw = load_dataset(spec["path"], spec["name"])
    data = _resolve_split(raw, selected_split)
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))

    rows = []
    for i, sample in enumerate(data):
        prompt_text, metadata = _format_generic(sample)
        rows.append(
            {
                "prompt_text": prompt_text,
                "source_dataset": dataset_key,
                "source_split": selected_split,
                "source_idx": i,
                "metadata": metadata,
            }
        )

    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"[info] wrote {len(rows)} prompts to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_SPECS.keys(), "all"],
        required=True,
        help="Dataset to materialize locally.",
    )
    parser.add_argument("--split", default=None, help="Override the default split.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for local JSON prompt files.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_keys = DATASET_SPECS.keys() if args.dataset == "all" else [args.dataset]
    for dataset_key in dataset_keys:
        build_dataset(
            dataset_key=dataset_key,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            force=args.force,
        )


if __name__ == "__main__":
    main()
