"""Collects per-generation observations of a training run into one JSON document.

Usage, from the repository root:

    python3 collect_observations.py DATA_DIRECTORY LOG[,LOG...] OUTPUT.json LAST_GENERATION

It reads the pipeline's generation reports and stored replays under
DATA_DIRECTORY and the elapsed timestamps of the `yokai train` logs, and keeps
the attempts up to LAST_GENERATION. Phase durations are whole seconds.
"""
import glob
import json
import os
import re
import statistics
import sys

root, logs, out, last_generation = sys.argv[1], sys.argv[2].split(","), sys.argv[3], int(sys.argv[4])
stamp = re.compile(r"^\[(\d+):(\d+):(\d+)\] (.*)$")

def seconds(match):
    return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])

phases = {}
for log in logs:
    current = None
    for line in open(log, encoding="utf-8", errors="replace"):
        match = stamp.match(line)
        if not match:
            continue
        at, text = seconds(match), match[4]
        started = re.match(r"generation (\d+) started", text)
        if started:
            current = int(started[1])
            phases[current] = {"log": os.path.basename(log), "start": at}
            continue
        if current is None:
            continue
        record = phases[current]
        for key, prefix in [
            ("self_play_end", "self-play finished"),
            ("training_end", "training finished"),
            ("arena_start", "arena started"),
            ("arena_end", "arena finished"),
            ("probe_end", "candidate exploratory probe finished"),
        ]:
            if text.startswith(prefix):
                record[key] = at
        if re.match(r"candidate \d+ (promoted|rejected)", text):
            record["end"] = at

def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]

generations = []
for path in sorted(glob.glob(os.path.join(root, "reports", "generation-*.json"))):
    report = json.load(open(path))
    number = report["candidate_generation"]
    if number > last_generation:
        continue
    last = report["training"]["checkpoints"][-1]
    lengths, reasons = [], {}
    for replay_path in glob.glob(os.path.join(root, "replays", f"generation-{number:06d}", "*.json")):
        replay = json.load(open(replay_path))
        lengths.append(len(replay["actions"]))
        outcome = replay["outcome"]
        label = outcome["status"] + ":" + outcome.get("reason", "")
        reasons[label] = reasons.get(label, 0) + 1
    timing = phases.get(number, {})
    def span(a, b):
        return timing[b] - timing[a] if a in timing and b in timing else None
    arena, probe = report["arena"], report["candidate_self_play"]
    promoted = report["promotion"]["arena_passed"] and report["promotion"]["exploratory_draw_gate_passed"]
    generations.append({
        "generation": number,
        "source_champion": report["source_generation"],
        "self_play_evaluator": report["self_play_evaluator"]["kind"],
        "promoted": promoted,
        "seconds": {
            "total": span("start", "end"),
            "self_play": span("start", "self_play_end"),
            "optimization": span("self_play_end", "training_end"),
            "arena": span("arena_start", "arena_end"),
            "probe": span("arena_end", "probe_end"),
        },
        "self_play": {
            "games": report["generated_games"],
            "restarted_games": report["restarted_games"],
            "outcomes": dict(sorted(reasons.items())),
            "draws": report["self_play_outcomes"]["draws"],
            "starter_wins": report["self_play_outcomes"]["starter_wins"],
            "non_starter_wins": report["self_play_outcomes"]["non_starter_wins"],
            "plies_mean": round(statistics.mean(lengths), 1),
            "plies_p50": percentile(lengths, 0.5),
            "plies_p90": percentile(lengths, 0.9),
            "positions": report["generated_dataset_diagnostics"]["all"]["positions"],
            "positions_from_draws": report["generated_dataset_diagnostics"]["draws"]["positions"],
        },
        "buffer": {"games": report["buffer_games"], "examples": report["buffer_examples"]},
        "training_step_400": {k: round(v, 4) for k, v in last["training"].items()},
        "validation_step_400": {k: round(v, 4) for k, v in last["validation"].items()},
        "arena": {
            "candidate_wins": arena["candidate_wins"],
            "reference_wins": arena["reference_wins"],
            "draws": arena["draws"],
            "score": arena["score"],
            "improvement_p_value": arena["improvement_p_value"],
            "distinct_openings": arena["distinct_openings"],
        },
        "probe": {"games": 64, "draws": probe["draws"]},
    })

with open(out, "x", encoding="utf-8") as file:
    json.dump({
        "sources": "generation reports, stored replays and `yokai train` logs of the run",
        "timestamp_resolution": "whole seconds from CLI elapsed timestamps",
        "total_seconds": sum(g["seconds"]["total"] for g in generations),
        "generations": generations,
    }, file, indent=2)
    file.write("\n")
