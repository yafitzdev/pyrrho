"""push_to_hub.py — Upload a pyrrho release directory to HuggingFace.

The release directory should contain:
- model.safetensors + config.json + tokenizer files (transformers checkpoint)
- model.onnx + model_quantized.onnx (from scripts/export_onnx.py)
- README.md (from scripts/build_model_card.py)

Requires `huggingface_hub` installed (pyproject extra `[hub]`) and the user
authenticated via `hf auth login` (one-time).

Run from project root:
    python scripts/push_to_hub.py --release-dir models/pyrrho-v2-nano-g1 --dry-run
    python scripts/push_to_hub.py --release-dir models/pyrrho-v2-nano-g1 --repo-id yafitzdev/pyrrho-v2-nano-g1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")


REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "README.md")
OPTIONAL_FILES = ("model.safetensors", "model.onnx", "model_quantized.onnx", "special_tokens_map.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release-dir", type=Path, required=True, help="Directory containing model + README + ONNX")
    p.add_argument("--repo-id", type=str, default="yafitzdev/pyrrho-v2-nano-g1")
    p.add_argument("--private", action="store_true", help="Create the repo private (default: public)")
    p.add_argument("--dry-run", action="store_true", help="List what would be uploaded but don't push")
    p.add_argument("--commit-message", type=str, default="Release: pyrrho-v2-nano-g1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    release = args.release_dir.resolve()
    if not release.exists():
        print(f"ERROR: release dir not found: {release}", file=sys.stderr)
        return 1

    # File checks
    files = list(release.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Release dir   : {release}")
    print(f"Total files   : {len(files)}")
    by_name = {f.name for f in files}
    missing = [r for r in REQUIRED_FILES if r not in by_name]
    if missing:
        print(f"ERROR: missing required files in release dir: {missing}", file=sys.stderr)
        print("  Make sure both build_model_card.py and export_onnx.py have been run "
              "and that the transformers checkpoint files are present.", file=sys.stderr)
        return 1

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"Total size    : {total_bytes / 1e6:.1f} MB")
    print(f"Target repo   : {args.repo_id} ({'private' if args.private else 'public'})")
    print(f"Commit message: {args.commit_message}")
    print()

    # Show file list, sorted by size
    print("Files to upload:")
    for f in sorted(files, key=lambda p: -p.stat().st_size):
        rel = f.relative_to(release)
        size_mb = f.stat().st_size / 1e6
        print(f"  {size_mb:>8.2f} MB  {rel}")

    if args.dry_run:
        print("\n--dry-run set, not pushing. Re-run without --dry-run to upload.")
        return 0

    print("\nImporting huggingface_hub...")
    from huggingface_hub import HfApi, create_repo

    print("Creating repo (no-op if it already exists)...")
    create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    print("Uploading folder...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(release),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    print(f"\nDONE. Live at: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
