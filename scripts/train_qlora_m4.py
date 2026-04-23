"""M4.3 adapter: run train_qlora_qwen.py on data/processed/ Alpaca-style data.

The upstream ``train_qlora_qwen.py`` expects:
  * ``data/train/*.jsonl`` paths (we have ``data/processed/``)
  * Each row has a ``messages`` chat-format field and ``source`` for ablation
    filtering (we have Alpaca ``instruction/input/output`` + ``category``)

This adapter:
  1. Loads ``data/processed/rag_qa_{train,val}.jsonl``
  2. Converts each Alpaca row into the expected messages format, using the
     system prompt from ``configs/qlora_config.json::data.system_prompt``
  3. Maps category → source (single-letter A..H) so ablation still works
  4. Writes the converted data into ``data/train/`` (gitignored via *.jsonl)
  5. Invokes the upstream trainer via its ``main()`` entry

Usage::

    # smoke (200 samples, Qwen-0.5B, 1 epoch)
    python scripts/train_qlora_m4.py --smoke

    # full run (4400 samples, Qwen-1.5B, 3 epochs)
    python scripts/train_qlora_m4.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Also add scripts/ so we can import train_qlora_qwen as a module
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


LOGGER = logging.getLogger("train_qlora_m4")


def convert_alpaca_to_messages(
    row: dict, system_prompt: str
) -> dict:
    """Convert ``{instruction, input, output}`` to the chat ``messages`` schema.

    The resulting row also carries ``source`` (= category letter) so the
    upstream ablation filter works without modification, and preserves
    ``event_id_source`` / ``category`` for downstream analysis.
    """
    user_text = row["instruction"]
    if row.get("input"):
        user_text = f"{user_text}\n\n{row['input']}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": row["output"]},
    ]
    return {
        "messages": messages,
        "source": row.get("category"),  # A / B / C / ... — matches ablation config
        "event_id_source": row.get("event_id_source"),
        "split": row.get("split"),
    }


def convert_file(
    src: Path,
    dst: Path,
    system_prompt: str,
    *,
    sample_n: int | None = None,
    seed: int = 42,
) -> int:
    """Convert an Alpaca-style JSONL into the chat ``messages`` schema.

    If ``sample_n`` is given, the output is a category-stratified random
    subsample of size ``sample_n`` (categories below their proportional
    share are left intact so minority classes like G/H survive).
    """
    import random
    from collections import defaultdict

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    if sample_n is not None and sample_n < len(rows):
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_cat[r.get("category", "?")].append(r)
        rng = random.Random(seed)
        total = len(rows)
        # Proportional target per category; minority classes kept 100%.
        picked: list[dict] = []
        for cat, items in by_cat.items():
            target = max(1, round(sample_n * len(items) / total))
            if target >= len(items):
                picked.extend(items)
            else:
                picked.extend(rng.sample(items, target))
        # If off-target due to rounding, top up / trim randomly.
        if len(picked) > sample_n:
            picked = rng.sample(picked, sample_n)
        elif len(picked) < sample_n:
            remaining = [r for r in rows if r not in picked]
            picked.extend(rng.sample(remaining, sample_n - len(picked)))
        rows = picked

    count = 0
    with dst.open("w", encoding="utf-8") as fout:
        for row in rows:
            converted = convert_alpaca_to_messages(row, system_prompt)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="200 samples × 1 epoch on Qwen-0.5B — verify the pipeline + OOM risk",
    )
    parser.add_argument(
        "--ablation",
        choices=["v1", "v2", "v3"],
        default="v3",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help=(
            "Downsample training set to this many rows (stratified by "
            "category when possible). Useful to fit within a time budget "
            "on RTX 2060S. Leave unset to train on the full set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "qwen_lora_csrc",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load config to extract system prompt
    config_path = PROJECT_ROOT / "configs" / "qlora_config.json"
    with config_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    system_prompt = cfg["data"]["system_prompt"]

    src_train = PROJECT_ROOT / "data" / "processed" / "rag_qa_train.jsonl"
    src_val = PROJECT_ROOT / "data" / "processed" / "rag_qa_val.jsonl"
    dst_train = PROJECT_ROOT / "data" / "train" / "rag_qa_train.jsonl"
    dst_val = PROJECT_ROOT / "data" / "train" / "rag_qa_val.jsonl"

    n_train = convert_file(src_train, dst_train, system_prompt, sample_n=args.sample_n)
    n_val = convert_file(src_val, dst_val, system_prompt)
    LOGGER.info("converted %d train + %d val rows → data/train/", n_train, n_val)

    # Delegate to the upstream trainer by setting sys.argv and calling main()
    import train_qlora_qwen  # type: ignore  # noqa: E402

    sys.argv = [
        "train_qlora_qwen.py",
        "--config", str(config_path),
        "--train", str(dst_train),
        "--val", str(dst_val),
        "--ablation", args.ablation,
        "--output", str(args.output),
    ]
    if args.smoke:
        sys.argv.append("--debug")

    return train_qlora_qwen.main()


if __name__ == "__main__":
    raise SystemExit(main())
