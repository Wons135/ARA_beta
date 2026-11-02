"""Check staged files for unsafe content and export an archive without Git history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".venv-demo", "venv", "__pycache__", ".ipynb_checkpoints",
            ".claude", ".idea", ".vscode", "outputs", "_to_review", ".cache", "node_modules"}
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{25,})\b"),
    re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}\b"),
    re.compile(rb'''(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*["']?\s*[:=]\s*["'][A-Za-z0-9_/+=.-]{12,}["']'''),
]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    files = []
    problems = []
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        info, raw_path = entry.split(b"\t", 1)
        mode, blob, stage = info.decode().split()
        name = raw_path.decode("utf-8")
        path = Path(name)
        if stage != "0":
            problems.append(f"Unmerged index: {name}")
            continue
        if mode != "100644" and mode != "100755":
            problems.append(f"Unsupported file type in index: {name}")
        if path.is_absolute() or ".." in path.parts or EXCLUDED.intersection(path.parts):
            problems.append(f"Excluded path staged: {name}")
        if name.startswith("datasets/") and not name.startswith("datasets/sample/"):
            problems.append(f"Non-sample data staged: {name}")
        if path.suffix.lower() in {".pyc", ".pth", ".pt", ".safetensors", ".pem", ".key", ".p12", ".pfx"} or path.name.startswith(".env"):
            problems.append(f"Artifact or credential file staged: {name}")
        if path.suffix.lower() in {".pdf", ".tex", ".ipynb"}:
            problems.append(f"Manuscript/generated/notebook material needs review: {name}")
        content = git("cat-file", "blob", blob)
        if len(content) > 5_000_000:
            problems.append(f"Large staged file (>5 MB): {name}")
        for pattern in SECRET_PATTERNS:
            match = pattern.search(content)
            if match:
                line = content[:match.start()].count(b"\n") + 1
                problems.append(f"Possible credential: {name}:{line} (value withheld)")
        attribute = git("check-attr", "--cached", "export-ignore", "--", name).decode().strip()
        if not attribute.endswith(": set"):
            files.append((name, content, mode))
    if problems:
        raise SystemExit("Submission check failed:\n" + "\n".join(problems))
    print(f"PASS: checked staged file paths, sizes, and credential patterns; {len(files)} exportable files.")
    if args.check_only:
        return
    folder = ROOT / "outputs/submission"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "ARA-thesis-submission.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content, mode in files:
            info = zipfile.ZipInfo("ARA-thesis-submission/" + name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if mode == "100755" else 0o100644) << 16
            archive.writestr(info, content)
    report = {"file": target.name, "bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "source": "staged Git index; no commit or history included",
              "files": [name for name, _, _ in files]}
    (folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
