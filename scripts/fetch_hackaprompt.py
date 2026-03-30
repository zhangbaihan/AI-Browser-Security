#!/usr/bin/env python3
"""
Download and convert HackAPrompt competition data to the project's JSONL format.

Since the official HuggingFace dataset (hackaprompt/hackaprompt-dataset) requires
authentication (gated dataset), this script fetches data from public GitHub
repositories that contain verified HackAPrompt competition solutions:

  1. gstaff/hackaprompt-solutions  — 25+ submission files with solutions for all levels
  2. terjanq/hack-a-prompt         — competition-winning prompts (inputs.json)
  3. myfoostrong/hackaprompt-2023   — additional submission data

As a last resort, it will attempt to load from HuggingFace with trust_remote_code=True.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "prompts", "raw", "hackaprompt.jsonl")

# GitHub sources (public, no auth required)
GITHUB_SOURCES = {
    "gstaff_main": {
        "base_url": "https://raw.githubusercontent.com/gstaff/hackaprompt-solutions/main/main_submissions",
        "files": [f"v{i}.json" for i in range(1, 26)],
        "description": "gstaff/hackaprompt-solutions (main submissions)",
    },
    "gstaff_flan": {
        "base_url": "https://raw.githubusercontent.com/gstaff/hackaprompt-solutions/main/flan_only_submissions",
        "files": [f"v{i}.json" for i in range(1, 6)],
        "description": "gstaff/hackaprompt-solutions (flan-only submissions)",
    },
    "terjanq": {
        "base_url": "https://raw.githubusercontent.com/terjanq/hack-a-prompt/main/flan-bruteforce",
        "files": ["inputs.json"],
        "description": "terjanq/hack-a-prompt (competition-winning inputs)",
    },
    "myfoostrong": {
        "base_url": "https://raw.githubusercontent.com/myfoostrong/hackaprompt-2023/main",
        "files": ["submission.json"],
        "description": "myfoostrong/hackaprompt-2023 (submission)",
    },
}

# HackAPrompt levels and their descriptions — used for technique mapping
LEVEL_INFO = {
    0: {
        "technique": "naive_override",
        "target_action": "make_model_say_specific_phrase",
        "description": "Basic prompt injection: make model say 'I have been PWNED'",
    },
    1: {
        "technique": "naive_override",
        "target_action": "override_system_instructions",
        "description": "Override simple system prompt instructions",
    },
    2: {
        "technique": "context_manipulation",
        "target_action": "bypass_input_validation",
        "description": "Bypass input-dependent system prompt",
    },
    3: {
        "technique": "context_manipulation",
        "target_action": "bypass_sandwich_defense",
        "description": "Bypass sandwich defense (instructions repeated after input)",
    },
    4: {
        "technique": "encoding_obfuscation",
        "target_action": "bypass_input_filtering",
        "description": "Bypass basic input filtering",
    },
    5: {
        "technique": "role_hijacking",
        "target_action": "bypass_llm_evaluator",
        "description": "Bypass LLM-based evaluator defense",
    },
    6: {
        "technique": "payload_splitting",
        "target_action": "bypass_advanced_filtering",
        "description": "Bypass advanced filtering and defense",
    },
    7: {
        "technique": "social_engineering",
        "target_action": "bypass_multi_layer_defense",
        "description": "Bypass multiple defense layers simultaneously",
    },
    8: {
        "technique": "indirect_nested",
        "target_action": "bypass_all_defenses",
        "description": "Bypass all defense mechanisms (hardest level)",
    },
    9: {
        "technique": "indirect_nested",
        "target_action": "bypass_all_defenses",
        "description": "Bypass all defense mechanisms (bonus level)",
    },
    10: {
        "technique": "encoding_obfuscation",
        "target_action": "bypass_all_defenses",
        "description": "Bypass all defense mechanisms (bonus level)",
    },
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


def generate_id(text, level, index):
    """Generate a deterministic unique ID."""
    raw = f"hackaprompt:{level}:{index}:{text[:200]}"
    return f"hap-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def refine_technique(text, default_technique):
    """Refine technique classification based on actual payload content."""
    text_lower = text.lower()

    if any(p in text_lower for p in ["base64", "rot13", "hex(", "\\u00", "unicode"]):
        return "encoding_obfuscation"
    if any(p in text_lower for p in [
        "ignore previous", "ignore above", "ignore all prior", "disregard",
    ]):
        return "naive_override"
    if any(p in text_lower for p in [
        "you are now", "pretend you are", "act as", "new identity",
    ]):
        return "role_hijacking"
    if any(p in text_lower for p in ["step 1", "part 1", "combine the"]):
        return "payload_splitting"

    return default_technique


def parse_level_key(key):
    """Parse a level key like 'level_3' into integer 3."""
    try:
        return int(key.replace("level_", ""))
    except (ValueError, AttributeError):
        return 0


def fetch_json(url):
    """Fetch and parse JSON from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "hackaprompt-fetcher/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode("utf-8"))


def convert_github_entry(prompt, level, model, source_tag):
    """Convert a single GitHub-sourced entry to our JSONL format."""
    if not prompt or not prompt.strip():
        return None

    level_data = LEVEL_INFO.get(level, {
        "technique": "naive_override",
        "target_action": "prompt_injection",
        "description": f"Level {level} attack",
    })

    technique = refine_technique(prompt, level_data["technique"])

    notes_parts = [
        f"level={level}",
        f"model={model}",
        f"level_desc={level_data['description']}",
        f"github_source={source_tag}",
    ]

    return {
        "technique": technique,
        "context": f"hackaprompt_level_{level}",
        "payload": prompt,
        "target_action": level_data["target_action"],
        "source": "hackaprompt",
        "notes": "; ".join(notes_parts),
    }


def load_from_github():
    """Load hackaprompt data from public GitHub repositories."""
    all_entries = []
    total_files = 0
    failed_files = 0

    for source_key, source_info in GITHUB_SOURCES.items():
        base_url = source_info["base_url"]
        desc = source_info["description"]
        print(f"  Fetching from {desc}...")

        for filename in source_info["files"]:
            url = f"{base_url}/{filename}"
            try:
                data = fetch_json(url)
                total_files += 1

                # Each file is a dict with level_N keys
                if isinstance(data, dict):
                    for level_key, info in data.items():
                        if isinstance(info, dict):
                            prompt = info.get("prompt", "")
                            model = info.get("model", "unknown")
                            level = parse_level_key(level_key)
                            source_tag = f"{source_key}/{filename}"
                            entry = convert_github_entry(prompt, level, model, source_tag)
                            if entry:
                                all_entries.append(entry)

            except urllib.error.HTTPError as e:
                failed_files += 1
                if e.code != 404:
                    print(f"    WARNING: HTTP {e.code} fetching {filename}")
            except Exception as e:
                failed_files += 1
                print(f"    WARNING: Error fetching {filename}: {e}")

    print(f"  Fetched {total_files} files ({failed_files} failed)")
    return all_entries


def convert_hf_entry(row, index):
    """Convert a single HuggingFace dataset row to our JSONL format."""
    user_input = row.get("user_input", "")
    if not user_input or not user_input.strip():
        return None

    level = row.get("level", 0)
    model = row.get("model", "unknown")
    expected_completion = row.get("expected_completion", "")
    completion = row.get("completion", "")

    level_data = LEVEL_INFO.get(level, {
        "technique": "naive_override",
        "target_action": "prompt_injection",
        "description": f"Level {level} attack",
    })

    technique = refine_technique(user_input, level_data["technique"])

    notes_parts = [
        f"level={level}",
        f"model={model}",
        f"level_desc={level_data['description']}",
    ]
    if expected_completion:
        notes_parts.append(f"expected={expected_completion[:80]}")
    if completion:
        notes_parts.append(f"completion={completion[:80]}")

    return {
        "technique": technique,
        "context": f"hackaprompt_level_{level}",
        "payload": user_input,
        "target_action": level_data["target_action"],
        "source": "hackaprompt",
        "notes": "; ".join(notes_parts),
    }


def load_from_huggingface():
    """Attempt to load the dataset from HuggingFace (requires auth for gated dataset)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  HuggingFace 'datasets' library not installed, skipping.")
        return None

    hf_name = "hackaprompt/hackaprompt-dataset"
    print(f"  Attempting to load '{hf_name}' from HuggingFace...")

    # Try with trust_remote_code and various options
    for kwargs in [
        {"trust_remote_code": True},
        {"trust_remote_code": True, "token": True},
        {},
    ]:
        try:
            ds = load_dataset(hf_name, **kwargs)
            print(f"  Successfully loaded from HuggingFace.")
            return ds
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "gated" in err_msg.lower() or "authentication" in err_msg.lower():
                print(f"  HuggingFace auth required (gated dataset), skipping.")
                return None
            # Try next set of kwargs
            continue

    print(f"  Could not load from HuggingFace after all attempts.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert HackAPrompt dataset to project JSONL format."
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to convert (default: all)",
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=None,
        help="Only include entries at or above this level",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="Only include entries at or below this level",
    )
    args = parser.parse_args()

    all_entries = []

    # --- Strategy 1: GitHub repos (no auth needed) ---
    print("Strategy 1: Fetching from public GitHub repositories...")
    github_entries = load_from_github()
    if github_entries:
        all_entries.extend(github_entries)
        print(f"  Got {len(github_entries)} entries from GitHub.")
    else:
        print("  No entries from GitHub.")

    # --- Strategy 2: HuggingFace (may need auth) ---
    print()
    print("Strategy 2: Trying HuggingFace datasets library...")
    hf_ds = load_from_huggingface()
    if hf_ds is not None:
        hf_count = 0
        for split_name in hf_ds:
            split = hf_ds[split_name]
            print(f"  Processing split '{split_name}': {len(split)} rows")
            for idx, row in enumerate(split):
                entry = convert_hf_entry(row, idx)
                if entry:
                    all_entries.append(entry)
                    hf_count += 1
        print(f"  Got {hf_count} entries from HuggingFace.")

    if not all_entries:
        print()
        print("ERROR: No data could be fetched from any source.")
        print("You may need to:")
        print("  1. Check your internet connection")
        print("  2. Set up a HuggingFace token: huggingface-cli login")
        sys.exit(1)

    # Apply level filters
    if args.min_level is not None or args.max_level is not None:
        filtered = []
        for entry in all_entries:
            # Extract level from context field
            try:
                level = int(entry["context"].replace("hackaprompt_level_", ""))
            except ValueError:
                level = 0
            if args.min_level is not None and level < args.min_level:
                continue
            if args.max_level is not None and level > args.max_level:
                continue
            filtered.append(entry)
        all_entries = filtered

    if args.max_entries:
        all_entries = all_entries[: args.max_entries]

    # Deduplicate by payload text
    seen_payloads = set()
    unique_entries = []
    global_index = 0
    for entry in all_entries:
        payload_key = entry["payload"].strip().lower()
        if payload_key not in seen_payloads:
            seen_payloads.add(payload_key)
            # Assign stable ID now
            level = int(entry["context"].replace("hackaprompt_level_", ""))
            entry["id"] = generate_id(entry["payload"], level, global_index)
            unique_entries.append(entry)
            global_index += 1

    # Reorder fields for consistent output
    ordered_entries = []
    for entry in unique_entries:
        ordered_entries.append({
            "id": entry["id"],
            "technique": entry["technique"],
            "context": entry["context"],
            "payload": entry["payload"],
            "target_action": entry["target_action"],
            "source": entry["source"],
            "notes": entry["notes"],
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in ordered_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(ordered_entries)} entries to {args.output}")
    print(f"  (deduplicated from {len(all_entries)} total)")

    # Print technique breakdown
    technique_counts = {}
    for entry in ordered_entries:
        t = entry["technique"]
        technique_counts[t] = technique_counts.get(t, 0) + 1
    print("  Technique breakdown:")
    for t, c in sorted(technique_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    # Print level breakdown
    level_counts = {}
    for entry in ordered_entries:
        ctx = entry["context"]
        level_counts[ctx] = level_counts.get(ctx, 0) + 1
    print("  Level breakdown:")
    for ctx, c in sorted(level_counts.items()):
        print(f"    {ctx}: {c}")


if __name__ == "__main__":
    main()
