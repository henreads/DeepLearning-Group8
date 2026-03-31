#!/usr/bin/env python3
"""Diagnose and fix VAE x224 training issues."""

import subprocess
import sys
from pathlib import Path


def check_modal_credentials():
    """Check if Modal credentials are configured."""
    print("Checking Modal credentials...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "modal", "whoami"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"[OK] Modal authenticated as: {result.stdout.strip()}")
            return True
        else:
            print(f"[ERROR] Modal not authenticated")
            print("  Run: modal setup")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking Modal: {e}")
        return False


def check_local_data():
    """Check if required data exists locally."""
    print("\nChecking local data...")
    repo_root = Path(__file__).parent.parent

    required_files = [
        repo_root / "data" / "processed" / "x224" / "wm811k" / "metadata_50k_5pct.csv",
        repo_root / "data" / "processed" / "x224" / "wm811k" / "arrays" / "wafer_0000000.npy",
    ]

    all_exist = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {file_path.relative_to(repo_root)}")
        all_exist = all_exist and exists

    return all_exist


def launch_modal_training():
    """Launch training via Modal."""
    print("\nLaunching VAE x224 training via Modal...")
    print("This will run on Modal's A10G GPU and may take 2-3 hours.")
    print()

    repo_root = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "modal", "run", "--detach",
                "modal_apps/vae_x224_main/app.py::main",
                "--no-sync-back"
            ],
            cwd=repo_root,
            capture_output=False,
            text=True,
        )

        if result.returncode == 0:
            print("\n[OK] Job submitted successfully!")
            print("Monitor progress at: https://modal.com/apps")
            print("Download artifacts with: modal run modal_apps/vae_x224_main/app.py::download_artifacts")
        else:
            print(f"\n[ERROR] Failed to submit job (exit code: {result.returncode})")

        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Error launching job: {e}")
        return False


def main():
    """Run diagnostics."""
    print("=" * 60)
    print("VAE x224 Training Diagnostics")
    print("=" * 60)

    # Check prerequisites
    modal_ok = check_modal_credentials()
    data_ok = check_local_data()

    if not modal_ok:
        print("\n[ERROR] Modal not authenticated. Run: modal setup")
        return 1

    if not data_ok:
        print("\n[ERROR] Required data files missing!")
        return 1

    print("\n" + "=" * 60)
    print("All checks passed. Ready to launch training.")
    print("=" * 60)

    # Ask user for confirmation
    response = input("\nLaunch training on Modal? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Cancelled.")
        return 0

    success = launch_modal_training()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
