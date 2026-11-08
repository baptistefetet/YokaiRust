"""Build frozen revisions, then run interleaved measurements on an idle GPU.

Usage: python3 docs/experiments/2026-11-08/measure.py build|run
The ignored data directory retains binaries and all six raw observations.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

experiment = Path(__file__).resolve().parent
root = experiment.parents[2]
os.chdir(root)
protocol = json.loads((experiment / "performance-protocol.json").read_text())
output = root / "data/alpha-zero-from-scratch-20261108/performance"
environment = dict(os.environ, CARGO_TARGET_DIR=str(root / "target"))


def build(source, arguments, target):
    # Archives retain old mtimes; invalidate local crate fingerprints while
    # reusing unchanged dependency builds in the shared target directory.
    for path in source.rglob("*.rs"):
        path.touch()
    command = ["cargo", *arguments, "--message-format=json-render-diagnostics"]
    result = subprocess.run(command, cwd=source, env=environment, text=True, stdout=subprocess.PIPE, check=True)
    for line in result.stdout.splitlines():
        event = json.loads(line)
        if event.get("reason") == "compiler-artifact" and event.get("executable") and event["target"]["name"] == target:
            return Path(event["executable"])
    raise RuntimeError("Cargo did not return the requested executable")


if sys.argv[1] == "build":
    with tempfile.TemporaryDirectory(prefix="yokai-performance-") as temporary:
        for label in ["before", "after"]:
            source = Path(temporary) / label
            source.mkdir()
            archive = subprocess.check_output(["git", "archive", protocol[label]])
            with tarfile.open(fileobj=io.BytesIO(archive)) as files:
                files.extractall(source)
            (source / "examples").mkdir(exist_ok=True)
            shutil.copyfile(experiment / "measurement.rs", source / "examples/measurement.rs")
            for target, arguments in [
                ("measurement", ["build", "--release", "--example", "measurement"]),
                ("engine", ["bench", "--no-default-features", "--bench", "engine", "--no-run"]),
            ]:
                binary = build(source, arguments, target)
                shutil.copy2(binary, output / f"{label}-{target}")
                print(f"Built {label}-{target}", flush=True)
elif sys.argv[1] == "run":
    fixture = root / protocol["fixed_buffer"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == protocol["fixed_buffer_sha256"]
    for index, label in enumerate(protocol["order"], 1):
        prefix = output / f"{index:02}-{label}"
        with prefix.with_suffix(".engine.log").open("x") as log:
            subprocess.run([str(output / f"{label}-engine"), "all"], stdout=log, stderr=subprocess.STDOUT, check=True)
        with prefix.with_suffix(".metal.log").open("x") as log:
            subprocess.run([
                "/usr/bin/time", "-l", "caffeinate", "-i", str(output / f"{label}-measurement"),
                str(root / "config/retrain-20261108.toml"),
                str(root / "models/references/pre-cleanup-20261108"),
                str(fixture), str(prefix.with_suffix(".json")),
            ], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished {prefix.name}", flush=True)
else:
    raise SystemExit("expected build or run")
