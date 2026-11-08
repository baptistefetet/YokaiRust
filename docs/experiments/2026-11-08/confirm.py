"""Run one of the preregistered arenas, retaining selection and file identities."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


protocol_path = Path(__file__).with_name("protocol.json")
protocol = json.loads(protocol_path.read_text())
boundary = int(sys.argv[1])
confirmation = protocol["confirmation"]
index = confirmation["attempt_boundaries"].index(boundary)
seed = confirmation["seeds"][index]
config_path = Path(protocol["configuration"])
assert digest(config_path) == protocol["configuration_sha256"]
reference = protocol["reference"]
reference_checkpoint = Path(reference["root"]) / f'generation-{reference["generation"]:06}'
for name, expected in reference["files_sha256"].items():
    assert digest(reference_checkpoint / name) == expected, name

models = Path("models") / protocol["experiment"]
data = Path("data") / protocol["experiment"]
reports = data / "reports"
assert len(list(reports.glob("generation-*.json"))) == boundary
champion = 0
for attempt in range(1, boundary + 1):
    report = json.loads((reports / f"generation-{attempt:06}.json").read_text())
    assert report["source_generation"] == champion
    assert report["candidate_generation"] == attempt
    if all(report["promotion"].values()):
        champion = attempt
assert int((models / "latest").read_text()) == champion
# The pipeline deliberately retains the completed attempt's journal.
journal = json.loads((data / "pending-generation.json").read_text())
assert journal["candidate_generation"] == boundary
assert journal["source_generation"] == report["source_generation"]
checkpoint = models / f"generation-{champion:06}"
candidate_hashes = {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()}
assert candidate_hashes["model.safetensors"] != reference["files_sha256"]["model.safetensors"]

stem = data / "confirmation-arenas" / f"after-{boundary:02}-attempts"
plan = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "protocol_sha256": digest(protocol_path),
    "attempt_boundary": boundary,
    "candidate_generation": champion,
    "candidate_files_sha256": candidate_hashes,
    "reference": reference,
    "seed": seed,
    "confirmation": confirmation,
}
write_new(stem.with_suffix(".plan.json"), plan)
environment = dict(os.environ, **{
    "YOKAI_CANDIDATE_MODELS": str(models),
    "YOKAI_CANDIDATE_GENERATION": str(champion),
    "YOKAI_REFERENCE_MODELS": reference["root"],
    "YOKAI_REFERENCE_GENERATION": str(reference["generation"]),
    "YOKAI_ARENA_GAMES": str(confirmation["games"]),
    "YOKAI_ARENA_SEED": str(seed),
    "YOKAI_ARENA_REPORT": str(stem.with_suffix(".json")),
})
with stem.with_suffix(".log").open("x") as log:
    result = subprocess.run(
        ["caffeinate", "-i", "cargo", "bench", "--bench", "performance", "--", "compare_saved_generations"],
        env=environment, stdout=log, stderr=subprocess.STDOUT,
    )
if result.returncode:
    raise SystemExit(result.returncode)
result = json.loads(stem.with_suffix(".json").read_text())
arena = result["arena"]
assert arena["candidate_wins"] + arena["reference_wins"] + arena["draws"] == confirmation["games"]
decision = {
    "candidate_generation": champion,
    "score": arena["score"],
    "p_value": arena["improvement_p_value"],
    "confirmed": arena["score"] >= confirmation["minimum_score"]
    and arena["improvement_p_value"] <= confirmation["maximum_one_sided_p"],
}
write_new(stem.with_suffix(".decision.json"), decision)
print(json.dumps(decision, indent=2))
