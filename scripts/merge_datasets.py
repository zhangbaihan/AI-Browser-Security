#!/usr/bin/env python3
"""
Merge all converted JSONL files from prompts/raw/ into a single
deduplicated dataset at prompts/all_prompts.jsonl.

Prints statistics: total count, count by technique, count by source.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "prompts", "raw")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "prompts", "all_prompts.jsonl")

REQUIRED_FIELDS = ["id", "technique", "context", "payload", "target_action", "source"]

VALID_TECHNIQUES = [
    "naive_override",
    "context_manipulation",
    "role_hijacking",
    "encoding_obfuscation",
    "payload_splitting",
    "indirect_nested",
    "social_engineering",
    "invisible_text",
    "ui_spoofing",
]


def validate_entry(entry, filepath, line_num):
    """Validate that an entry conforms to the project schema. Returns list of warnings."""
    warnings = []

    for field in REQUIRED_FIELDS:
        if field not in entry:
            warnings.append(f"Missing required field '{field}'")

    technique = entry.get("technique", "")
    if technique and technique not in VALID_TECHNIQUES:
        warnings.append(
            f"Unknown technique '{technique}' (valid: {', '.join(VALID_TECHNIQUES)})"
        )

    payload = entry.get("payload", "")
    if not payload or not payload.strip():
        warnings.append("Empty payload")

    return warnings


def load_jsonl_file(filepath):
    """Load entries from a JSONL file with validation."""
    entries = []
    errors = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: {os.path.basename(filepath)}:{line_num} - Invalid JSON: {e}")
                errors += 1
                continue

            warnings = validate_entry(entry, filepath, line_num)
            if any("Missing required field" in w for w in warnings):
                print(
                    f"  WARNING: {os.path.basename(filepath)}:{line_num} - "
                    f"Skipping (missing fields): {warnings}"
                )
                errors += 1
                continue

            if any("Empty payload" in w for w in warnings):
                errors += 1
                continue

            # Print non-critical warnings but still include the entry
            for w in warnings:
                if "Unknown technique" in w:
                    print(f"  NOTE: {os.path.basename(filepath)}:{line_num} - {w}")

            entries.append(entry)

    return entries, errors


def main():
    parser = argparse.ArgumentParser(
        description="Merge all JSONL files from prompts/raw/ into a single deduplicated dataset."
    )
    parser.add_argument(
        "-i", "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing source JSONL files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output merged JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication (keep all entries even with duplicate payloads)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate files without writing output",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        print()
        print("Run the fetch scripts first to populate prompts/raw/:")
        print("  python scripts/fetch_tensortrust.py --download")
        print("  python scripts/fetch_hackaprompt.py")
        print("  python scripts/fetch_injecagent.py --download")
        sys.exit(1)

    # Find all JSONL files
    jsonl_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".jsonl")
    ])

    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        print()
        print("Run the fetch scripts first:")
        print("  python scripts/fetch_tensortrust.py --download")
        print("  python scripts/fetch_hackaprompt.py")
        print("  python scripts/fetch_injecagent.py --download")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL file(s) in {args.input_dir}:")
    for f in jsonl_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # Load all entries
    all_entries = []
    total_errors = 0
    file_counts = {}

    for filepath in jsonl_files:
        fname = os.path.basename(filepath)
        print(f"Loading {fname}...")
        entries, errors = load_jsonl_file(filepath)
        file_counts[fname] = len(entries)
        total_errors += errors
        all_entries.extend(entries)
        print(f"  -> {len(entries)} valid entries ({errors} errors)")

    print()
    print(f"Total loaded: {len(all_entries)} entries ({total_errors} errors across all files)")

    # Deduplicate
    if not args.no_dedup:
        seen_payloads = set()
        seen_ids = set()
        unique_entries = []
        dup_count = 0

        for entry in all_entries:
            payload_key = entry["payload"].strip().lower()
            entry_id = entry.get("id", "")

            if payload_key in seen_payloads:
                dup_count += 1
                continue

            # Handle ID collisions by appending source
            if entry_id in seen_ids:
                entry["id"] = f"{entry_id}-{entry['source']}"

            seen_payloads.add(payload_key)
            seen_ids.add(entry["id"])
            unique_entries.append(entry)

        print(f"After deduplication: {len(unique_entries)} entries ({dup_count} duplicates removed)")
    else:
        unique_entries = all_entries
        print("Deduplication skipped.")

    if args.validate_only:
        print()
        print("Validation complete (no output written).")
    else:
        # Write output
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for entry in unique_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(unique_entries)} entries to {args.output}")

    # Print statistics
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total entries: {len(unique_entries)}")

    # Count by source
    source_counts = {}
    for entry in unique_entries:
        src = entry.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print()
    print("By source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(unique_entries) if unique_entries else 0
        print(f"  {src:30s} {count:6d}  ({pct:.1f}%)")

    # Count by technique
    technique_counts = {}
    for entry in unique_entries:
        t = entry.get("technique", "unknown")
        technique_counts[t] = technique_counts.get(t, 0) + 1

    print()
    print("By technique:")
    for t, count in sorted(technique_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(unique_entries) if unique_entries else 0
        print(f"  {t:30s} {count:6d}  ({pct:.1f}%)")

    # Count by context (top 15)
    context_counts = {}
    for entry in unique_entries:
        ctx = entry.get("context", "unknown")
        context_counts[ctx] = context_counts.get(ctx, 0) + 1

    print()
    print("By context (top 15):")
    for ctx, count in sorted(context_counts.items(), key=lambda x: -x[1])[:15]:
        pct = 100.0 * count / len(unique_entries) if unique_entries else 0
        print(f"  {ctx:30s} {count:6d}  ({pct:.1f}%)")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
