"""Auto-imported by Python's site initialization for any process whose sys.path
contains this directory. When VR_PROBE=1, it loads the behavior-preserving
verify-rollback accept-step probe (verify_rollback_patch) into the vLLM server
tree. No-op otherwise. See verify_rollback_patch.py and paper_notes.md."""
import os

if os.environ.get("VR_PROBE") == "1":
    try:
        import verify_rollback_patch  # noqa: F401  (self-installs via VR_PROBE)
    except Exception:
        pass
