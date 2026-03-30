#!/usr/bin/env python3
"""
Download and convert the InjecAgent benchmark to the project's JSONL format.

Source: https://github.com/uiuc-kang-lab/InjecAgent

The InjecAgent benchmark contains prompt injection attacks targeting
tool-integrated LLM agents. This script downloads their data files
and converts them to the project's standardized JSONL schema.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys

REPO_URL = "https://github.com/uiuc-kang-lab/InjecAgent"
RAW_BASE = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main"

# Known data file paths in the InjecAgent repository
DATA_FILES = [
    "data/test_cases_dh_base.json",
    "data/test_cases_dh_enhanced.json",
    "data/test_cases_ds_base.json",
    "data/test_cases_ds_enhanced.json",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "prompts", "raw", "injecagent.jsonl")
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "injecagent")

# Mapping from InjecAgent attack types to our technique taxonomy
# InjecAgent uses categories like "Direct Harm" (dh) and "Data Stealing" (ds)
ATTACK_TYPE_MAP = {
    "direct": "naive_override",
    "observation": "indirect_nested",
    "indirect": "indirect_nested",
    "memory": "context_manipulation",
    "context": "context_manipulation",
    "tool_result": "indirect_nested",
    "combined": "payload_splitting",
    "physical harm": "indirect_nested",
    "financial harm": "indirect_nested",
    "data stealing": "indirect_nested",
    "data theft": "indirect_nested",
    "intrusion": "indirect_nested",
}

# Mapping from InjecAgent harm categories / Expected Achievements
HARM_CATEGORY_MAP = {
    "data_theft": "exfiltrate_data",
    "data_exfiltration": "exfiltrate_data",
    "steal": "exfiltrate_data",
    "unauthorized_access": "gain_unauthorized_access",
    "access": "gain_unauthorized_access",
    "privilege_escalation": "escalate_privileges",
    "system_compromise": "compromise_system",
    "financial_fraud": "perform_financial_fraud",
    "financial": "perform_financial_fraud",
    "social_engineering": "social_engineer_user",
    "malware": "install_malware",
    "denial_of_service": "deny_service",
    "misinformation": "spread_misinformation",
    "harm": "cause_harm",
    "physical": "cause_harm",
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


def generate_id(text, index):
    """Generate a deterministic unique ID."""
    raw = f"injecagent:{index}:{text[:200]}"
    return f"ia-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def classify_technique(entry):
    """Classify an InjecAgent entry into our technique taxonomy."""
    attack_type = entry.get("attack_type", "").lower().strip()
    for key, technique in ATTACK_TYPE_MAP.items():
        if key in attack_type:
            return technique

    # Fallback: inspect the injected content
    injected = entry.get("injected_content", entry.get("attack_prompt", ""))
    if isinstance(injected, str):
        text_lower = injected.lower()
        if any(p in text_lower for p in ["ignore previous", "ignore above", "disregard"]):
            return "naive_override"
        if any(p in text_lower for p in ["you are now", "pretend you", "act as"]):
            return "role_hijacking"
        if any(p in text_lower for p in ["base64", "rot13", "encode"]):
            return "encoding_obfuscation"

    return "indirect_nested"


def map_target_action(entry):
    """Map the InjecAgent harm/intent to a target_action string."""
    for field in ["Expected Achievements", "Attacker Objective", "attacker_objective", "harm_category", "intent"]:
        value = entry.get(field, "")
        if value:
            # Normalize to snake_case action string
            action = value.strip().lower().replace(" ", "_").replace("-", "_")
            # Check our mapping
            for key, mapped in HARM_CATEGORY_MAP.items():
                if key in action:
                    return mapped
            return action

    return "execute_injected_instructions"


def extract_payload(entry):
    """Extract the actual adversarial payload text from an InjecAgent entry."""
    # Try multiple possible field names (InjecAgent uses "Attacker Instruction")
    for field in [
        "Attacker Instruction",
        "attacker_instruction",
        "injected_content",
        "Injected Prompt",
        "attack_prompt",
        "injected_prompt",
        "attack",
        "injection",
    ]:
        value = entry.get(field, "")
        if value and isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def extract_context(entry):
    """Extract context information about where the injection appears."""
    parts = []

    attack_type = entry.get("Attack Type", entry.get("attack_type", ""))
    if attack_type:
        parts.append(f"attack_type={attack_type}")

    tool = entry.get("User Tool", entry.get("tool", entry.get("Tool", entry.get("user_tool", ""))))
    if tool:
        parts.append(f"tool={tool}")

    if parts:
        return "; ".join(parts)
    return "agent_tool_interaction"


def convert_entry(entry, index):
    """Convert a single InjecAgent entry to our JSONL format."""
    payload = extract_payload(entry)
    if not payload:
        return None

    technique = classify_technique(entry)
    target_action = map_target_action(entry)
    context = extract_context(entry)

    notes_parts = []
    for field in [
        "User Instruction", "user_instruction", "user_task",
        "Expected Achievements", "Attacker Objective", "attacker_objective",
        "User Tool", "Tool", "tool",
        "Attack Type", "attack_type",
    ]:
        value = entry.get(field, "")
        if value and isinstance(value, str) and value.strip():
            notes_parts.append(f"{field}={value.strip()[:120]}")

    return {
        "id": generate_id(payload, index),
        "technique": technique,
        "context": context,
        "payload": payload,
        "target_action": target_action,
        "source": "injecagent",
        "notes": "; ".join(notes_parts) if notes_parts else "",
    }


def try_download_files(data_dir):
    """Attempt to download data files from the InjecAgent GitHub repo."""
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError

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
            req = Request(url, headers={"User-Agent": "fetch_injecagent/1.0"})
            with urlopen(req, timeout=60) as resp:
                content = resp.read().decode("utf-8")
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Saved: {local_path}")
            success = True
        except (URLError, HTTPError) as e:
            print(f"  WARNING: Failed to download {url}: {e}")

    return success


def find_data_files(data_dir):
    """Find available data files in the data directory."""
    files = []
    if not os.path.isdir(data_dir):
        return files
    for fname in os.listdir(data_dir):
        if fname.endswith(".json") or fname.endswith(".jsonl") or fname.endswith(".csv"):
            files.append(os.path.join(data_dir, fname))
    return files


def load_json_file(filepath):
    """Load entries from a JSON or JSONL file."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try as a single JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Could be a dict with a data key
            for key in ["data", "test_cases", "entries", "samples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    # Try as JSONL
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return entries


def process_file(filepath, index_offset=0):
    """Process a single data file and return converted entries."""
    entries = []

    if filepath.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_entries = list(reader)
    else:
        raw_entries = load_json_file(filepath)

    for i, raw in enumerate(raw_entries):
        result = convert_entry(raw, index_offset + i)
        if result:
            entries.append(result)

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert InjecAgent benchmark to project JSONL format."
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-d", "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing InjecAgent data files (default: {DEFAULT_DATA_DIR})",
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
        print("Attempting to download InjecAgent data files...")
        if try_download_files(args.data_dir):
            data_files = find_data_files(args.data_dir)
        else:
            print()

    if not data_files:
        print("=" * 70)
        print("INJECAGENT DATA NOT FOUND")
        print("=" * 70)
        print()
        print("To use this script, do one of the following:")
        print()
        print("Option 1: Auto-download (run with --download):")
        print(f"  python {os.path.basename(__file__)} --download")
        print()
        print("Option 2: Clone the repository manually:")
        print(f"  git clone {REPO_URL}")
        print(f"  Then run: python {os.path.basename(__file__)} -d /path/to/InjecAgent/data")
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
