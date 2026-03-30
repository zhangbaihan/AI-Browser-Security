#!/usr/bin/env python3
"""
Download and convert the Tensor Trust dataset to the project's JSONL format.

Source: https://github.com/HumanCompatibleAI/tensor-trust-data

The Tensor Trust dataset contains prompt injection attacks and defenses
collected from the Tensor Trust game. This script parses their data files
and converts them into the project's standardized JSONL schema.
"""

import argparse
import hashlib
import json
import os
import sys

REPO_URL = "https://github.com/HumanCompatibleAI/tensor-trust-data"
RAW_BASE = "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/main"

# Expected data files from the Tensor Trust repository
DATA_FILES = [
    "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
    "benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "prompts", "raw", "tensortrust.jsonl")
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "tensor-trust-data")

# Mapping from Tensor Trust categories to our technique taxonomy
TECHNIQUE_MAP = {
    "hijacking": "role_hijacking",
    "extraction": "context_manipulation",
    "prompt_extraction": "context_manipulation",
    "prompt_hijacking": "role_hijacking",
    "system_prompt_extraction": "context_manipulation",
    "jailbreak": "naive_override",
    "ignore_instructions": "naive_override",
    "role_play": "social_engineering",
    "encoding": "encoding_obfuscation",
    "payload_split": "payload_splitting",
}

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


def classify_technique(attack_text, category=None):
    """Classify an attack into our technique taxonomy based on its text and category."""
    if category:
        cat_lower = category.lower().strip()
        for key, technique in TECHNIQUE_MAP.items():
            if key in cat_lower:
                return technique

    text_lower = attack_text.lower()
    if any(phrase in text_lower for phrase in [
        "ignore previous", "ignore above", "ignore all", "disregard",
        "forget your instructions", "forget everything",
    ]):
        return "naive_override"
    if any(phrase in text_lower for phrase in [
        "you are now", "pretend you are", "act as", "roleplay",
        "you are a", "new role",
    ]):
        return "role_hijacking"
    if any(phrase in text_lower for phrase in [
        "reveal your", "show me your", "print your", "what is your system",
        "repeat your instructions", "output your",
    ]):
        return "context_manipulation"
    if any(phrase in text_lower for phrase in [
        "base64", "rot13", "hex", "encode", "decode", "\\u",
    ]):
        return "encoding_obfuscation"
    if any(phrase in text_lower for phrase in [
        "step 1", "first part", "combine", "split",
    ]):
        return "payload_splitting"
    if any(phrase in text_lower for phrase in [
        "please help", "as a researcher", "for educational", "hypothetically",
    ]):
        return "social_engineering"

    return "naive_override"


def generate_id(source, text, index):
    """Generate a deterministic unique ID."""
    raw = f"{source}:{index}:{text[:200]}"
    return f"tt-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def try_download_files(data_dir):
    """Attempt to download data files from the Tensor Trust GitHub repo."""
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' library is required for downloading.")
        print("Install it with: pip install requests")
        return False

    os.makedirs(data_dir, exist_ok=True)
    success = False

    for rel_path in DATA_FILES:
        url = f"{RAW_BASE}/{rel_path}"
        local_path = os.path.join(data_dir, os.path.basename(rel_path))

        if os.path.exists(local_path):
            print(f"  Already exists: {local_path}")
            success = True
            continue

        print(f"  Downloading: {url}")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"  Saved: {local_path}")
            success = True
        except requests.RequestException as e:
            print(f"  WARNING: Failed to download {url}: {e}")

    return success


def find_data_files(data_dir):
    """Find available data files in the data directory."""
    files = []
    if not os.path.isdir(data_dir):
        return files
    for fname in os.listdir(data_dir):
        if fname.endswith(".jsonl") or fname.endswith(".json"):
            files.append(os.path.join(data_dir, fname))
    return files


def parse_hijacking_entry(entry, index):
    """Parse a hijacking dataset entry."""
    attack_text = entry.get("attack", "")
    if not attack_text:
        return None

    category = "hijacking"
    access_code = entry.get("access_code", "")
    pre_prompt = entry.get("pre_prompt", "")
    post_prompt = entry.get("post_prompt", "")

    notes_parts = []
    if access_code:
        notes_parts.append(f"access_code={access_code}")
    if pre_prompt:
        notes_parts.append(f"defense_pre_prompt={pre_prompt[:100]}")
    if post_prompt:
        notes_parts.append(f"defense_post_prompt={post_prompt[:100]}")
    if "sample_id" in entry:
        notes_parts.append(f"sample_id={entry['sample_id']}")

    return {
        "id": generate_id("tensortrust", attack_text, index),
        "technique": classify_technique(attack_text, category),
        "context": "game_prompt_injection",
        "payload": attack_text,
        "target_action": "hijack_model_output_to_say_access_granted",
        "source": "tensortrust",
        "notes": "; ".join(notes_parts) if notes_parts else "",
    }


def parse_extraction_entry(entry, index):
    """Parse a prompt extraction dataset entry."""
    attack_text = entry.get("attack", "")
    if not attack_text:
        return None

    pre_prompt = entry.get("pre_prompt", "")
    post_prompt = entry.get("post_prompt", "")

    notes_parts = []
    if pre_prompt:
        notes_parts.append(f"defense_pre_prompt={pre_prompt[:100]}")
    if post_prompt:
        notes_parts.append(f"defense_post_prompt={post_prompt[:100]}")
    if "sample_id" in entry:
        notes_parts.append(f"sample_id={entry['sample_id']}")

    return {
        "id": generate_id("tensortrust", attack_text, index),
        "technique": classify_technique(attack_text, "extraction"),
        "context": "game_prompt_extraction",
        "payload": attack_text,
        "target_action": "extract_system_prompt_or_access_code",
        "source": "tensortrust",
        "notes": "; ".join(notes_parts) if notes_parts else "",
    }


def process_file(filepath, index_offset=0):
    """Process a single data file and return converted entries."""
    entries = []
    basename = os.path.basename(filepath).lower()
    is_extraction = "extraction" in basename

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            idx = index_offset + i
            if is_extraction:
                result = parse_extraction_entry(entry, idx)
            else:
                result = parse_hijacking_entry(entry, idx)

            if result:
                entries.append(result)

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Tensor Trust dataset to project JSONL format."
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-d", "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing Tensor Trust data files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download data files from GitHub",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to convert (default: all)",
    )
    args = parser.parse_args()

    data_files = find_data_files(args.data_dir)

    if not data_files or args.download:
        print(f"Attempting to download Tensor Trust data files...")
        if try_download_files(args.data_dir):
            data_files = find_data_files(args.data_dir)
        else:
            print()

    if not data_files:
        print("=" * 70)
        print("TENSOR TRUST DATA NOT FOUND")
        print("=" * 70)
        print()
        print("To use this script, do one of the following:")
        print()
        print("Option 1: Auto-download (run with --download):")
        print(f"  python {os.path.basename(__file__)} --download")
        print()
        print("Option 2: Clone the repository manually:")
        print(f"  git clone {REPO_URL}")
        print(f"  Then run: python {os.path.basename(__file__)} -d /path/to/tensor-trust-data")
        print()
        print("Option 3: Download specific files into the data directory:")
        print(f"  mkdir -p {args.data_dir}")
        for rel_path in DATA_FILES:
            fname = os.path.basename(rel_path)
            print(f"  curl -L -o {args.data_dir}/{fname} {RAW_BASE}/{rel_path}")
        print()
        sys.exit(1)

    print(f"Found {len(data_files)} data file(s) in {args.data_dir}")

    all_entries = []
    offset = 0
    for fpath in sorted(data_files):
        print(f"  Processing: {os.path.basename(fpath)}")
        entries = process_file(fpath, index_offset=offset)
        print(f"    -> {len(entries)} entries")
        all_entries.extend(entries)
        offset += len(entries)

    if args.max_entries:
        all_entries = all_entries[: args.max_entries]

    # Deduplicate by payload text
    seen_payloads = set()
    unique_entries = []
    for entry in all_entries:
        payload_key = entry["payload"].strip().lower()
        if payload_key not in seen_payloads:
            seen_payloads.add(payload_key)
            unique_entries.append(entry)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in unique_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(unique_entries)} entries to {args.output}")
    print(f"  (deduplicated from {len(all_entries)} total)")

    # Print technique breakdown
    technique_counts = {}
    for entry in unique_entries:
        t = entry["technique"]
        technique_counts[t] = technique_counts.get(t, 0) + 1
    print("  Technique breakdown:")
    for t, c in sorted(technique_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
