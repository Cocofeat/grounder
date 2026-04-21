"""Generate open-end QA data by removing multiple-choice options from questions and letter prefixes from answers."""

import json
import os
import re
from pathlib import Path
from glob import glob

BASE = Path("/mnt/data/by/project/internvl_2_5")
DATA = Path("/mnt/data/by/data/coco_new/labels_2025_11_29")


def strip_options_from_question(text):
    """Remove multiple-choice options from question text.

    Input:  '...Question: Does this patient have a focal liver lesion? A. ...; B. ...'
    Output: '...Question: Does this patient have a focal liver lesion?'
    """
    # Find the last '?' before ' A.' pattern
    # Pattern: '? A.' or '?\nA.' marks the start of options
    match = re.search(r'\?\s+A\.', text)
    if match:
        return text[:match.start() + 1]  # keep the '?'
    return text


def strip_letter_prefix(answer):
    """Remove letter prefix from answer.

    Input:  'A. There are a focal lesion or multipul focal lesions'
    Output: 'There are a focal lesion or multipul focal lesions'
    """
    return re.sub(r'^[A-E]\.\s*', '', answer)


def transform_train_entry(entry):
    """Transform a training JSONL entry (has conversations field)."""
    entry = dict(entry)  # shallow copy
    convs = list(entry["conversations"])
    # Transform question
    q = dict(convs[0])
    q["value"] = strip_options_from_question(q["value"])
    # Transform answer
    a = dict(convs[1])
    a["value"] = strip_letter_prefix(a["value"])
    entry["conversations"] = [q, a]
    return entry


def transform_test_entry(entry):
    """Transform a test GT JSON entry (has text and label fields)."""
    entry = dict(entry)
    entry["text"] = strip_options_from_question(entry["text"])
    entry["label"] = strip_letter_prefix(entry["label"])
    return entry


def process_jsonl(src, dst):
    """Process a JSONL file (training data)."""
    entries = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(transform_train_entry(json.loads(line)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(entries)


def process_json(src, dst):
    """Process a JSON file (test GT)."""
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    transformed = [transform_test_entry(e) for e in data]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)
    return len(transformed)


def main():
    total_files = 0

    # === Training — Multi-modal ===
    print("=== Training — Multi-modal ===")
    src_dir = DATA / "multimodal_train"
    dst_dir = DATA / "multimodal_open_end_train"
    for src in sorted(src_dir.glob("*QA*.jsonl")):
        dst = dst_dir / src.name
        n = process_jsonl(src, dst)
        print(f"  {src.name} -> {dst.name}  ({n} entries)")
        total_files += 1

    # === Training — Single-modal ===
    print("\n=== Training — Single-modal ===")
    src_dir = DATA / "train"
    dst_dir = DATA / "open_end_train"
    for src in sorted(src_dir.glob("*QA*.jsonl")):
        dst = dst_dir / src.name
        n = process_jsonl(src, dst)
        print(f"  {src.name} -> {dst.name}  ({n} entries)")
        total_files += 1

    # === Test GT — Multi-modal ===
    print("\n=== Test GT — Multi-modal ===")
    src_base = BASE / "evaluation_archive" / "multi_gating"
    dst_base = BASE / "evaluation_archive" / "multi_gating_open_end"
    for ds_dir in sorted(src_base.iterdir()):
        if not ds_dir.is_dir():
            continue
        gt_dir = ds_dir / "gt"
        if not gt_dir.exists():
            continue
        for src in sorted(gt_dir.glob("*QA*.json")):
            dst = dst_base / ds_dir.name / "gt" / src.name
            n = process_json(src, dst)
            print(f"  {ds_dir.name}/gt/{src.name}  ({n} entries)")
            total_files += 1

    # === Test GT — Single-modal ===
    print("\n=== Test GT — Single-modal ===")
    src_base = BASE / "evaluation_archive" / "single"
    dst_base = BASE / "evaluation_archive" / "single_open_end"
    for ds_dir in sorted(src_base.iterdir()):
        if not ds_dir.is_dir():
            continue
        gt_dir = ds_dir / "gt"
        if not gt_dir.exists():
            continue
        for src in sorted(gt_dir.glob("*QA*.json")):
            dst = dst_base / ds_dir.name / "gt" / src.name
            n = process_json(src, dst)
            print(f"  {ds_dir.name}/gt/{src.name}  ({n} entries)")
            total_files += 1

    print(f"\nDone. {total_files} files generated.")


if __name__ == "__main__":
    main()
