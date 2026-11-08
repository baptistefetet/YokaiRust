"""Continue this experiment one complete attempt at a time until a promotion.

Run only after the original 20-attempt command and its two-attempt extension
have exited. This never evaluates the independent reference or changes settings.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
configuration = Path(protocol["configuration"])
assert hashlib.sha256(configuration.read_bytes()).hexdigest() == protocol["configuration_sha256"]
data = Path("data") / protocol["experiment"]
models = Path("models") / protocol["experiment"]
logs = data / "logs"
assert json.loads((logs / "training-21-22-process.json").read_text())["exit_code"] == 0
source_generation = int(sys.argv[1]) if len(sys.argv) > 1 else 20
assert int((models / "latest").read_text()) == source_generation

while int((models / "latest").read_text()) == source_generation:
    reports = sorted((data / "reports").glob("generation-*.json"))
    previous = json.loads(reports[-1].read_text())
    attempt = previous["candidate_generation"] + 1
    assert attempt >= 23
    stem = logs / f"training-{attempt:02}"
    command = ["caffeinate", "-i", "cargo", "run", "--release", "--", "train", "--config", str(configuration), "--generations", "1"]
    record = {"command": command, "started_at": datetime.now(timezone.utc).isoformat()}
    process_path = stem.with_suffix(".process.json")
    with process_path.open("x") as stream:
        json.dump(record, stream, indent=2)
    start = time.monotonic()
    with stem.with_suffix(".log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-start,
                  finished_at=datetime.now(timezone.utc).isoformat())
    process_path.write_text(json.dumps(record, indent=2)+"\n")
    print(f"Attempt {attempt} exited with {result.returncode}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((data / "reports" / f"generation-{attempt:06}.json").read_text())
    assert report["source_generation"] == source_generation
    promoted = all(report["promotion"].values())
    assert int((models / "latest").read_text()) == (attempt if promoted else source_generation)

champion = int((models / "latest").read_text())
assert champion > source_generation
record = {"accepted_generation": champion, "attempt_boundary": champion,
          "recorded_at": datetime.now(timezone.utc).isoformat()}
with (logs / f"promotion-after-{source_generation}-completed.json").open("x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)
