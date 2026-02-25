# src/apply_patches.py
#
# Applies local patches to the installed gemini-webapi package at startup.
# This ensures our fixes (curl-cffi TLS impersonation, 3.1-pro model hash,
# per-request sessions) survive fresh `poetry install` runs on any machine.
#
# Called at the top of run.py before any other imports touch gemini_webapi.

import importlib.util
import shutil
from pathlib import Path


PATCHES_DIR = Path(__file__).parent / "patches" / "gemini_webapi"

PATCH_FILES = [
    "constants.py",
    "client.py",
    "utils/get_access_token.py",
    "utils/rotate_1psidts.py",
    "utils/decorators.py",
]


def apply():
    spec = importlib.util.find_spec("gemini_webapi")
    if spec is None or spec.origin is None:
        print("[patches] WARNING: gemini_webapi not found, skipping patches.")
        return

    gemini_root = Path(spec.origin).parent

    applied = []
    for rel in PATCH_FILES:
        src = PATCHES_DIR / rel
        dst = gemini_root / rel
        if not src.exists():
            print(f"[patches] WARNING: patch file missing: {src}")
            continue
        shutil.copy2(src, dst)
        applied.append(rel)

    if applied:
        print(f"[patches] Applied {len(applied)} gemini_webapi patch(es): {', '.join(applied)}")


apply()
