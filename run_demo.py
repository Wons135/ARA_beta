"""One-command CPU demo. Bootstrap only its own isolated Python 3.11 environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use installed dependencies and cached model/lexicon only.")
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 17), metavar="1..16")
    parser.add_argument("--environment-ready", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    started = time.perf_counter()
    if sys.version_info[:2] != (3, 11):
        parser.error("Python 3.11 is required. Install it, then use py -3.11 run_demo.py (Windows) or python3.11 run_demo.py.")
    os.chdir(ROOT)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # Set BEFORE importing any numerical library, including on CUDA workstations.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if not args.environment_ready:
        env_dir = ROOT / ".venv-demo"
        executable = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not executable.exists():
            print("Creating isolated CPU demo environment (.venv-demo)...", flush=True)
            venv.EnvBuilder(with_pip=True).create(env_dir)
        check = subprocess.run([str(executable), "-c", "from src.demo.environment import verify; verify()"],
                               cwd=ROOT, capture_output=True, text=True)
        if check.returncode:
            if args.offline:
                print("Offline setup needs a complete .venv-demo. Run once online first.\n" + check.stderr, file=sys.stderr)
                return 1
            print("Installing pinned CPU requirements (first setup may take several minutes)...", flush=True)
            install = subprocess.run([str(executable), "-m", "pip", "install", "--disable-pip-version-check",
                                      "-r", str(ROOT / "requirements-demo.txt")], cwd=ROOT)
            if install.returncode:
                print("Dependency installation failed. Check internet access and retry the same command.", file=sys.stderr)
                return install.returncode
        command = [str(executable), str(Path(__file__).resolve()), "--environment-ready", "--threads", str(args.threads)]
        if args.offline:
            command.append("--offline")
        result = subprocess.run(command, cwd=ROOT)
        print(f"Total command time including setup: {time.perf_counter() - started:.1f} s", flush=True)
        return result.returncode

    from src.demo.environment import verify
    verify()
    from src.demo.pipeline import run
    try:
        run(threads=args.threads, offline=args.offline)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Demo stopped: {exc}\nRetry the same command. First run needs internet for DistilBERT and VADER."
              " See outputs/demo/run.log for completed stages.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
