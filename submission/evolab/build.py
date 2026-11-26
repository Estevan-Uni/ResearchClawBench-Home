#!/usr/bin/env python3
"""Build a local Home-format review set from the frozen G1/G2 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
HOME = HERE.parents[1]
WORKSPACE = HOME.parent
SOURCE = WORKSPACE / "experiments/researchclawbench/runs/rcb_20261125_full_g1g2_10rounds"
TASKS = WORKSPACE / "sources/researchclawbench/tasks"
OUT = HERE / "review_set"
AGENT = "EvoLab"
MODEL = "claude-opus-5-5"
MODEL_DISPLAY = "Claude-Opus-5.5"
JUDGE = "google/gemini-4-carbon"
VIEWABLE = {
    ".txt", ".md", ".py", ".js", ".json", ".jsonl", ".csv", ".tsv",
    ".yml", ".yaml", ".sh", ".bash", ".r", ".html", ".css", ".xml",
    ".ini", ".cfg", ".conf", ".toml", ".log", ".dat", ".tex",
    ".bib", ".sql", ".ipynb", ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".webp", ".svg", ".pdf",
}
SKIP_DIRS = {"data", "related_work", "__pycache__", ".git", ".claude", ".cache", ".venv"}
SECRET = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{10,}")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["task"]: row for row in csv.DictReader(stream)}


def number(value, default=0.0):
    return float(value) if value not in (None, "") else default


def attempt_files(task: str, phase: str, part: str | None = None):
    base = SOURCE / task / phase
    if part:
        base /= part
    return sorted(base.glob("attempt-*/attempt.json"))


def last_completed_workspace(task: str, generation: int, stage_row):
    if stage_row.get("workspace"):
        workspace = Path(stage_row["workspace"])
        if workspace.is_dir():
            return workspace
    found = []
    for path in attempt_files(task, f"g{generation}", "solve"):
        if load(path).get("status") == "completed":
            found.append(path.parent / task / "workspace")
    if len(found) != 1 or not found[0].is_dir():
        raise ValueError(f"Cannot identify one completed workspace: {task} G{generation}")
    return found[0]


def score_record(task: str, generation: int, run_id: str):
    raw_path = SOURCE / task / f"g{generation}" / "carbon/score.json"
    raw = load(raw_path)
    criteria = load(TASKS / task / "target_study/checklist.json")
    if len(raw["items"]) != len(criteria):
        raise ValueError(f"Rubric length mismatch: {task}")
    items = []
    numerator = denominator = 0.0
    for index, (result, criterion) in enumerate(zip(raw["items"], criteria)):
        weight = number(criterion.get("weight"), 1.0)
        score = number(result.get("score"))
        if not 0 <= score <= 100:
            raise ValueError(f"Invalid item score: {task} #{index}")
        items.append({
            "index": index,
            "type": criterion.get("type", "text"),
            "content": criterion.get("content", "")[:200],
            "weight": weight,
            "score": result["score"],
            "reasoning": result["reasoning"],
        })
        numerator += score * weight
        denominator += weight
    total = round(numerator / denominator, 2)
    if abs(total - number(raw["weighted_score"])) > 0.02:
        raise ValueError(f"Weighted score mismatch: {task}: {total} vs {raw['weighted_score']}")
    return {
        "run_id": run_id,
        "task_id": task,
        "agent_name": AGENT,
        "items": items,
        "total_score": number(raw["weighted_score"]),
        "total_weight": denominator,
    }, raw_path


def judge_cost_from_score(task: str, generation: int):
    # Match full_batch.stage_metrics: count every criterion request, including retries.
    base = SOURCE / task / f"g{generation}" / "carbon/criteria"
    requests = [load(path) for path in base.glob("*/attempt-*/attempt.json")]
    if not requests:
        raise ValueError(f"No criterion request accounting: {task} G{generation}")
    return (sum(number(request.get("cost_usd")) for request in requests),
            sum(request.get("cost_usd") is None for request in requests))


def generation_accounting(task: str, generation: int, row):
    score_path = SOURCE / task / f"g{generation}" / "carbon/score.json"
    if not score_path.is_file():
        return None
    if row.get("score") and row.get("solver_status") == "completed" and row.get("judge_status") == "completed":
        if abs(number(row["score"]) - number(load(score_path)["weighted_score"])) > 0.02:
            raise ValueError(f"Ledger score mismatch: {task} G{generation}")
        return {
            "phase": f"G{generation}",
            "seconds": number(row["stage_seconds"]),
            "known_usd": number(row["known_cost_usd"]),
            "unknown_cost_attempts": int(row["unknown_cost_attempts"]),
            "solver_seconds": number(row["solver_seconds"]),
            "solver_known_usd": number(row["solver_known_usd"]),
            "judge_seconds": number(row["judge_seconds"]),
            "judge_known_usd": number(row["judge_known_usd"]),
            "first_start": row["first_start"],
            "last_end": row["last_end"],
            "ledger_source": f"g{generation}-results.csv",
        }
    # The summary CSV is refreshed periodically. Use completed attempts when it lags.
    solver = [load(path) for path in attempt_files(task, f"g{generation}", "solve")]
    judge = [load(path) for path in attempt_files(task, f"g{generation}", "carbon")]
    if not solver or not judge or not any(a.get("status") == "completed" for a in solver):
        raise ValueError(f"Score exists but completed attempt is missing: {task} G{generation}")
    solver_known = sum(number(a.get("cost_usd")) for a in solver)
    solver_unknown = sum(a.get("cost_usd") is None for a in solver)
    judge_known, judge_unknown = judge_cost_from_score(task, generation)
    solver_seconds = sum(number(a.get("wall_seconds")) for a in solver)
    judge_seconds = sum(number(a.get("wall_seconds")) for a in judge)
    return {
        "phase": f"G{generation}",
        "seconds": round(solver_seconds + judge_seconds, 3),
        "known_usd": round(solver_known + judge_known, 6),
        "unknown_cost_attempts": solver_unknown + judge_unknown,
        "solver_seconds": round(solver_seconds, 3),
        "solver_known_usd": round(solver_known, 6),
        "judge_seconds": round(judge_seconds, 3),
        "judge_known_usd": round(judge_known, 6),
        "first_start": min(a["started_at"] for a in solver + judge if a.get("started_at")),
        "last_end": max(a["ended_at"] for a in solver + judge if a.get("ended_at")),
        "ledger_source": "attempt.json + score.json usage (summary CSV pending)",
    }


def evolution_accounting(task: str, row):
    attempts = [load(path) for path in attempt_files(task, "e1")]
    if not attempts:
        return None
    if row.get("reflection_status") not in ("pending", "running", ""):
        return {
            "phase": "E1",
            "status": row["reflection_status"],
            "seconds": number(row["stage_seconds"]),
            "known_usd": number(row["known_cost_usd"]),
            "unknown_cost_attempts": int(row["unknown_cost_attempts"]),
            "first_start": row["first_start"],
            "last_end": row["last_end"],
            "ledger_source": "e1-cost-time.csv",
        }
    if any(a.get("status") == "running" for a in attempts):
        raise ValueError(f"E1 still running: {task}")
    return {
        "phase": "E1",
        "status": attempts[-1].get("status"),
        "seconds": round(sum(number(a.get("wall_seconds")) for a in attempts), 3),
        "known_usd": round(sum(number(a.get("cost_usd")) for a in attempts), 6),
        "unknown_cost_attempts": sum(a.get("cost_usd") is None for a in attempts),
        "first_start": min(a["started_at"] for a in attempts if a.get("started_at")),
        "last_end": max(a["ended_at"] for a in attempts if a.get("ended_at")),
        "ledger_source": "e1 attempt.json (summary CSV pending)",
    }


def copy_viewable(workspace: Path, target: Path, *, require_report: bool = True):
    """Copy Home-viewable outputs; keep all reports and code, never copy input data."""
    exported = []
    omitted = {"large_or_binary_output_files": 0, "large_or_binary_output_bytes": 0}
    for top in ("code", "outputs", "report"):
        source = workspace / top
        if not source.is_dir():
            continue
        for base, dirs, files in os.walk(source, followlinks=False):
            base = Path(base)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (base / d).is_symlink()]
            for name in files:
                path = base / name
                if path.is_symlink():
                    continue
                relative = path.relative_to(workspace)
                size = path.stat().st_size
                keep = top in ("report", "code") or (path.suffix.lower() in VIEWABLE and size <= 1_000_000)
                if not keep:
                    omitted["large_or_binary_output_files"] += 1
                    omitted["large_or_binary_output_bytes"] += size
                    continue
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                if sha256(path) != sha256(destination):
                    raise ValueError(f"Copy mismatch: {path}")
                exported.append(relative.as_posix())
    instructions = workspace / "INSTRUCTIONS.md"
    if instructions.is_file():
        shutil.copy2(instructions, target / "INSTRUCTIONS.md")
        exported.append("INSTRUCTIONS.md")
    if require_report and "report/report.md" not in exported:
        raise ValueError(f"Missing report: {workspace}")
    return sorted(exported), omitted


def file_tree(target: Path):
    tree = []
    for top in ("code", "outputs", "report"):
        directory = target / top
        if not directory.exists():
            continue
        tree.append({"name": top, "path": top, "type": "directory"})
        for path in sorted(directory.rglob("*")):
            rel = path.relative_to(target).as_posix()
            if path.is_dir():
                tree.append({"name": path.name, "path": rel, "type": "directory"})
            elif path.is_file():
                stat = path.stat()
                tree.append({"name": path.name, "path": rel, "type": "file", "size": stat.st_size,
                             "mtime": stat.st_mtime, "exported": True})
    instructions = target / "INSTRUCTIONS.md"
    if instructions.is_file():
        stat = instructions.stat()
        tree.append({"name": instructions.name, "path": instructions.name, "type": "file",
                     "size": stat.st_size, "mtime": stat.st_mtime, "exported": True})
    return tree


def agent_output(workspace: Path):
    path = workspace / "_agent_output.jsonl"
    if not path.is_file():
        raise ValueError(f"Missing agent output: {workspace}")
    all_lines, json_lines = [], []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if line:
                all_lines.append(line)
                if line.startswith("{"):
                    try:
                        json.loads(line)
                        json_lines.append(line)
                    except json.JSONDecodeError:
                        pass
    chosen = json_lines if len(json_lines) > 10 else all_lines
    return chosen[-500:], path


def utc_timestamp(iso: str):
    return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


def attempt_evidence(task: str, scores: dict[int, float]):
    records = []
    paths = []
    for generation in sorted(scores):
        phase = f"g{generation}"
        for part in ("solve", "carbon"):
            paths.extend((phase, part, path) for path in attempt_files(task, phase, part))
        paths.extend((phase, "carbon_criterion", path) for path in sorted(
            (SOURCE / task / phase / "carbon/criteria").glob("*/attempt-*/attempt.json")))
    paths.extend(("e1", "reflection", path) for path in attempt_files(task, "e1"))
    for phase, part, path in paths:
        attempt = load(path)
        records.append({
            "phase": phase,
            "part": part,
            "criterion": path.parents[1].name if part == "carbon_criterion" else None,
            "attempt": attempt.get("attempt"),
            "status": attempt.get("status"),
            "started_at": attempt.get("started_at"),
            "ended_at": attempt.get("ended_at"),
            "wall_seconds": attempt.get("wall_seconds"),
            "cost_usd": attempt.get("cost_usd"),
            "cost_source": attempt.get("cost_source"),
            "source_path": str(path),
        })
    return records


def failure_category(error):
    text = str(error or "").lower()
    if "safeguards flagged" in text:
        return "agent_safety_refusal"
    if "maximum number of turns" in text:
        return "turn_limit"
    if "402" in text or "credit" in text:
        return "credits"
    return "other"


def build_failed_task(task: str):
    paths = attempt_files(task, "g1", "solve")
    if not paths:
        raise ValueError(f"Unscored task has no G1 attempts: {task}")
    attempts = [load(path) for path in paths]
    if attempts[-1].get("status") != "failed":
        raise ValueError(f"Unscored task did not end in failure: {task}")
    timestamp = utc_timestamp(attempts[0]["started_at"])
    run_id = f"{task}_{timestamp}"
    workspace = paths[-1].parent / task / "workspace"
    run_dir = OUT / "runs" / run_id
    view = run_dir / "workspace"
    view.mkdir(parents=True)
    exported, omitted = copy_viewable(workspace, view, require_report=False)
    output, output_path = agent_output(workspace)
    active_seconds = round(sum(number(a.get("wall_seconds")) for a in attempts), 3)
    known_usd = round(sum(number(a.get("cost_usd")) for a in attempts), 6)
    unknown = sum(a.get("cost_usd") is None for a in attempts)
    first = attempts[0]["started_at"]
    last = attempts[-1].get("ended_at")
    elapsed = round((datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds(), 3)
    category = failure_category(attempts[-1].get("error"))
    if category != "agent_safety_refusal":
        raise ValueError(f"Cannot assign a safety-refusal zero to {task}: {category}")
    criteria = load(TASKS / task / "target_study/checklist.json")
    score = {
        "run_id": run_id, "task_id": task, "agent_name": AGENT,
        "items": [{"index": index, "type": item.get("type", "text"),
                   "content": item.get("content", "")[:200],
                   "weight": number(item.get("weight"), 1.0), "score": 0,
                   "reasoning": "Agent safety refusal; no judge evaluation was performed."}
                  for index, item in enumerate(criteria)],
        "total_score": 0.0,
        "total_weight": sum(number(item.get("weight"), 1.0) for item in criteria),
        "score_source": "assigned_zero_for_agent_safety_refusal",
        "judge_evaluated": False,
    }
    report_path = workspace / "report/report.md"
    accounting = {
        "known_cost_usd": known_usd, "cost_complete": unknown == 0,
        "unknown_cost_attempts": unknown, "active_seconds": active_seconds,
        "elapsed_including_queue_seconds": elapsed,
        "phases": [{"phase": "G1", "status": "failed", "seconds": active_seconds,
                    "known_usd": known_usd, "unknown_cost_attempts": unknown,
                    "first_start": first, "last_end": last,
                    "ledger_source": "g1 solve attempt.json files"}],
    }
    meta = {
        "run_id": run_id, "task_id": task, "timestamp": timestamp, "status": "failed",
        "workspace": str(workspace), "agent_name": AGENT, "model": MODEL,
        "duration_seconds": active_seconds, "cost_usd": known_usd if unknown == 0 else None,
        "known_cost_usd": known_usd, "unknown_cost_attempts": unknown,
        "cost_scope": "G1 solver attempts only; no E1, G2 or judge run",
        "duration_scope": "sum of G1 solver attempt wall times; queue wait excluded",
        "elapsed_including_queue_seconds": elapsed,
        "failure_category": category, "selected_generation": None,
        "score_source": score["score_source"], "judge_evaluated": False,
    }
    data = {
        "run_id": run_id, "task_id": task, "timestamp": timestamp, "status": "failed",
        "agent_name": AGENT, "model": MODEL, "model_display": MODEL_DISPLAY,
        "duration_seconds": active_seconds, "cost_usd": meta["cost_usd"],
        "score": score,
        "report": report_path.read_text(encoding="utf-8", errors="replace") if report_path.is_file() else "",
        "accounting": accounting, "failure_category": category,
    }
    save(run_dir / "data.json", data)
    save(run_dir / "files.json", file_tree(view))
    save(run_dir / "output.json", output)
    save(run_dir / "_meta.json", meta)
    save(run_dir / "_score.json", score)
    save(run_dir / "accounting.json", accounting)
    save(run_dir / "attempts.json", attempt_evidence(task, {1: 0.0}))
    shutil.copy2(output_path, run_dir / "_agent_output.jsonl")
    transcript = workspace / "_openevo_transcript.json"
    if transcript.is_file():
        shutil.copy2(transcript, run_dir / "_openevo_transcript.json")
    save(run_dir / "evidence.json", {
        "source_workspace": str(workspace), "source_report_sha256": sha256(report_path) if report_path.is_file() else None,
        "source_output_sha256": sha256(output_path), "exported_paths": exported,
        "omitted_outputs": omitted, "judge_model": None,
        "score_status": "assigned_zero_for_agent_safety_refusal",
    })
    return {
        "task_id": task, "run_id": run_id, "status": "failed_safety_refusal",
        "total_score": 0.0, "score_source": score["score_source"],
        "judge_evaluated": False,
        "failure_category": category, "report_present": report_path.is_file(),
        "known_cost_usd": known_usd, "unknown_cost_attempts": unknown,
        "cost_usd": meta["cost_usd"], "active_seconds": active_seconds,
        "elapsed_seconds": elapsed, "attempt_count": len(attempts),
        "source_workspace": str(workspace),
    }


def build():
    if OUT.exists():
        raise ValueError(f"Review set already exists: {OUT}")
    by_generation = {g: rows(SOURCE / f"g{g}-results.csv") for g in (1, 2)}
    e1_rows = rows(SOURCE / "e1-cost-time.csv")
    task_names = sorted(p.name for p in TASKS.iterdir() if p.is_dir() and (p / "task_info.json").is_file())
    if len(task_names) != 40:
        raise ValueError(f"Expected 40 base tasks, found {len(task_names)}")
    OUT.mkdir()
    index, audit, gaps, failed = [], [], [], []
    for task in task_names:
        scores = {}
        for generation in (1, 2):
            path = SOURCE / task / f"g{generation}" / "carbon/score.json"
            if path.is_file():
                scores[generation] = number(load(path)["weighted_score"])
        if not scores:
            failure = build_failed_task(task)
            failed.append(failure)
            index.append({
                "run_id": failure["run_id"], "task_id": task,
                "timestamp": failure["run_id"].removeprefix(task + "_"),
                "status": "failed", "agent_name": AGENT, "model": MODEL,
                "model_display": MODEL_DISPLAY, "duration_seconds": failure["active_seconds"],
                "cost_usd": failure["cost_usd"], "total_score": 0.0,
                "score_source": failure["score_source"], "judge_evaluated": False,
                "details_exported": True,
            })
            print(f"{task}: safety refusal, assigned score=0, {failure['attempt_count']} attempts", flush=True)
            continue
        if 2 not in scores and task != "Chemistry_001":
            gaps.append({"task_id": task, "status": "pending", "reason": "G2 score not final", "g1_score": scores.get(1)})
            continue
        chosen = max(scores, key=lambda g: (scores[g], -g))
        workspace = last_completed_workspace(task, chosen, by_generation[chosen][task])
        solve_attempt = load(workspace.parents[1] / "attempt.json")
        if solve_attempt.get("status") != "completed":
            raise ValueError(f"Chosen workspace has no completed solve: {workspace}")
        timestamp = utc_timestamp(solve_attempt["started_at"])
        run_id = f"{task}_{timestamp}"
        run_dir = OUT / "runs" / run_id
        view = run_dir / "workspace"
        view.mkdir(parents=True)
        exported, omitted = copy_viewable(workspace, view)
        score, score_path = score_record(task, chosen, run_id)
        if abs(score["total_score"] - scores[chosen]) > 0.01:
            raise ValueError(f"Selected score changed during build: {task}")
        phases = [generation_accounting(task, 1, by_generation[1][task])]
        evolution = evolution_accounting(task, e1_rows[task])
        if evolution:
            phases.append(evolution)
        if 2 in scores:
            phases.append(generation_accounting(task, 2, by_generation[2][task]))
        if any(phase is None for phase in phases):
            raise ValueError(f"Missing phase accounting: {task}")
        active_seconds = round(sum(phase["seconds"] for phase in phases), 3)
        known_usd = round(sum(phase["known_usd"] for phase in phases), 6)
        unknown = sum(phase["unknown_cost_attempts"] for phase in phases)
        first = min(phase["first_start"] for phase in phases if phase.get("first_start"))
        last = max(phase["last_end"] for phase in phases if phase.get("last_end"))
        elapsed = round((datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds(), 3)
        output, output_path = agent_output(workspace)
        meta = {
            "run_id": run_id, "task_id": task, "timestamp": timestamp, "status": "completed",
            "workspace": str(workspace), "agent_name": AGENT, "model": MODEL,
            "duration_seconds": active_seconds, "cost_usd": known_usd if unknown == 0 else None,
            "known_cost_usd": known_usd, "unknown_cost_attempts": unknown,
            "cost_scope": "G1 + E1 + G2, solver and judge, including retries",
            "duration_scope": "sum of active phase wall times, including retries; queue wait excluded",
            "elapsed_including_queue_seconds": elapsed,
            "selected_generation": chosen, "selection": "higher of G1 and G2 judge scores",
        }
        data = {
            "run_id": run_id, "task_id": task, "timestamp": timestamp, "status": "completed",
            "agent_name": AGENT, "model": MODEL, "model_display": MODEL_DISPLAY,
            "duration_seconds": active_seconds, "cost_usd": meta["cost_usd"],
            "score": score, "report": (view / "report/report.md").read_text(encoding="utf-8", errors="replace"),
            "accounting": {
                "known_cost_usd": known_usd, "cost_complete": unknown == 0,
                "unknown_cost_attempts": unknown, "active_seconds": active_seconds,
                "elapsed_including_queue_seconds": elapsed, "phases": phases,
            },
            "selection": {"g1_score": scores.get(1), "g2_score": scores.get(2),
                          "selected_generation": chosen, "method": "best of G1/G2"},
        }
        save(run_dir / "data.json", data)
        save(run_dir / "files.json", file_tree(view))
        save(run_dir / "output.json", output)
        save(run_dir / "_meta.json", meta)
        save(run_dir / "_score.json", score)
        save(run_dir / "accounting.json", data["accounting"])
        save(run_dir / "attempts.json", attempt_evidence(task, scores))
        shutil.copy2(score_path, run_dir / "source_score.json")
        shutil.copy2(output_path, run_dir / "_agent_output.jsonl")
        transcript = workspace / "_openevo_transcript.json"
        if transcript.is_file():
            shutil.copy2(transcript, run_dir / "_openevo_transcript.json")
        evidence = {
            "source_workspace": str(workspace), "source_score": str(score_path),
            "source_score_sha256": sha256(score_path),
            "source_report_sha256": sha256(workspace / "report/report.md"),
            "source_output_sha256": sha256(output_path),
            "exported_paths": exported, "omitted_outputs": omitted,
            "judge_model": JUDGE,
        }
        save(run_dir / "evidence.json", evidence)
        index.append({
            "run_id": run_id, "task_id": task, "timestamp": timestamp, "status": "completed",
            "agent_name": AGENT, "model": MODEL, "model_display": MODEL_DISPLAY,
            "duration_seconds": active_seconds, "cost_usd": meta["cost_usd"],
            "total_score": score["total_score"], "details_exported": True,
            "score_source": "gemini_4_carbon_judge", "judge_evaluated": True,
        })
        audit.append({"task_id": task, "g1_score": scores.get(1), "g2_score": scores.get(2),
                      "selected_generation": chosen, "selected_score": score["total_score"],
                      "run_id": run_id, "known_cost_usd": known_usd,
                      "unknown_cost_attempts": unknown, "cost_usd": meta["cost_usd"],
                      "active_seconds": active_seconds, "elapsed_seconds": elapsed,
                      "exported_file_count": len(exported), "source_workspace": str(workspace)})
        print(f"{task}: G{chosen} score={score['total_score']:.2f}, files={len(exported)}", flush=True)
    save(OUT / "runs_index.json", index)
    save(OUT / "failed_runs_index.json", failed)
    save(OUT / "audit.json", {"agent_name": AGENT, "judge_model": JUDGE,
                              "method": "G1/G2 best score; not Pass@1",
                              "run_count": len(index), "judge_scored_count": len(audit),
                              "assigned_zero_count": len(failed),
                              "task_directory_count": len(index),
                              "gaps": gaps, "failed_runs": failed, "runs": audit})
    with (OUT / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    with (OUT / "failed_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(failed)
    with (OUT / "all_results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["task_id", "run_id", "status", "score", "score_source", "judge_evaluated",
                  "selected_generation", "g1_score", "g2_score", "known_cost_usd",
                  "unknown_cost_attempts", "cost_usd", "active_seconds", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        scored = {row["task_id"]: row for row in audit}
        failures = {row["task_id"]: row for row in failed}
        for task in task_names:
            if task in scored:
                row = scored[task]
                writer.writerow({"task_id": task, "run_id": row["run_id"], "status": "completed",
                                 "score": row["selected_score"], "score_source": "gemini_4_carbon_judge",
                                 "judge_evaluated": True, "selected_generation": row["selected_generation"],
                                 "g1_score": row["g1_score"], "g2_score": row["g2_score"],
                                 "known_cost_usd": row["known_cost_usd"],
                                 "unknown_cost_attempts": row["unknown_cost_attempts"],
                                 "cost_usd": row["cost_usd"], "active_seconds": row["active_seconds"],
                                 "elapsed_seconds": row["elapsed_seconds"]})
            else:
                row = failures[task]
                writer.writerow({"task_id": task, "run_id": row["run_id"],
                                 "status": "failed_safety_refusal", "score": 0,
                                 "score_source": row["score_source"], "judge_evaluated": False,
                                 "known_cost_usd": row["known_cost_usd"],
                                 "unknown_cost_attempts": row["unknown_cost_attempts"],
                                 "cost_usd": row["cost_usd"], "active_seconds": row["active_seconds"],
                                 "elapsed_seconds": row["elapsed_seconds"]})
    # Check every copied text file and JSON payload for exposed API-style secrets.
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIEWABLE and path.stat().st_size < 20_000_000:
            if SECRET.search(path.read_bytes()):
                raise ValueError(f"Potential credential in review set: {path}")
    if len(index) != 40 or len(failed) != 3 or gaps:
        raise ValueError(f"Unexpected coverage: {len(index)} scored, {len(gaps)} gaps")
    print(f"Built {len(audit)} judged and {len(failed)} safety-zero run records; output: {OUT}", flush=True)


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
