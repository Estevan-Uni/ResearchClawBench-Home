#!/usr/bin/env python3
"""Verify local Home-format review records against source files and ledgers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "review_set"
SOURCE = HERE.parents[2] / "experiments/researchclawbench/runs/rcb_20261125_full_g1g2_10rounds"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    audit = load(ROOT / "audit.json")
    index = load(ROOT / "runs_index.json")
    failed = load(ROOT / "failed_runs_index.json")
    assert audit["run_count"] == len(index) == 40
    assert audit["judge_scored_count"] == len(audit["runs"]) == 37
    assert audit["assigned_zero_count"] == len(failed) == 3
    assert audit["failed_runs"] == failed
    assert audit["task_directory_count"] == len(index) == 40
    assert not audit["gaps"]
    assert len({row["task_id"] for row in index}) == len(index)
    assert len({row["run_id"] for row in index}) == len(index)
    assert len({row["task_id"] for row in index}) == 40
    assert len({row["run_id"] for row in index}) == 40
    assert len(list(csv.DictReader((ROOT / "audit.csv").open(newline="", encoding="utf-8")))) == len(audit["runs"])
    assert len(list(csv.DictReader((ROOT / "failed_runs.csv").open(newline="", encoding="utf-8")))) == len(failed)
    all_results = list(csv.DictReader((ROOT / "all_results.csv").open(newline="", encoding="utf-8")))
    assert len(all_results) == 40
    assert {row["task_id"] for row in all_results} == {row["task_id"] for row in index}
    totals = {row["task"]: row for row in csv.DictReader((SOURCE / "task-totals.csv").open(newline="", encoding="utf-8"))}
    copied_files = 0
    for row in audit["runs"]:
        run = ROOT / "runs" / row["run_id"]
        data = load(run / "data.json")
        meta = load(run / "_meta.json")
        score = load(run / "_score.json")
        evidence = load(run / "evidence.json")
        accounting = load(run / "accounting.json")
        attempts = load(run / "attempts.json")
        files = load(run / "files.json")
        assert data["run_id"] == meta["run_id"] == score["run_id"] == row["run_id"]
        assert data["task_id"] == meta["task_id"] == score["task_id"] == row["task_id"]
        assert data["agent_name"] == meta["agent_name"] == score["agent_name"] == "EvoLab"
        assert data["score"] == score and score["total_score"] == row["selected_score"]
        assert next(item for item in index if item["task_id"] == row["task_id"])["total_score"] == score["total_score"]
        result = next(item for item in all_results if item["task_id"] == row["task_id"])
        assert result["status"] == "completed" and result["judge_evaluated"] == "True"
        assert float(result["score"]) == score["total_score"]
        assert load(run / "source_score.json")["weighted_score"] == score["total_score"]
        assert data["report"] == (run / "workspace/report/report.md").read_text(encoding="utf-8", errors="replace")
        assert data["accounting"] == accounting
        assert meta["duration_seconds"] == row["active_seconds"]
        assert meta["cost_usd"] == row["cost_usd"]
        assert (row["cost_usd"] is None) == (row["unknown_cost_attempts"] > 0)
        assert abs(sum(phase["seconds"] for phase in accounting["phases"]) - row["active_seconds"]) < 0.002
        assert abs(sum(phase["known_usd"] for phase in accounting["phases"]) - row["known_cost_usd"]) < 0.00001
        assert len(load(run / "output.json")) <= 500
        assert attempts and all(Path(item["source_path"]).is_file() for item in attempts)
        assert sha256(Path(evidence["source_score"])) == evidence["source_score_sha256"]
        assert sha256(Path(evidence["source_workspace"]) / "report/report.md") == evidence["source_report_sha256"]
        assert sha256(run / "source_score.json") == evidence["source_score_sha256"]
        assert sha256(run / "_agent_output.jsonl") == evidence["source_output_sha256"]
        for file in files:
            if file["type"] == "file" and file.get("exported"):
                assert (run / "workspace" / file["path"]).is_file()
                copied_files += 1
        source_total = totals[row["task_id"]]
        if row["task_id"] not in ("Earth_003", "Information_000", "Neuroscience_001"):
            if source_total["gated_score"]:
                assert abs(float(source_total["gated_score"]) - row["selected_score"]) < 0.02
            assert abs(float(source_total["known_cost_usd"]) - row["known_cost_usd"]) < 0.0001
            assert abs(float(source_total["stage_seconds"]) - row["active_seconds"]) < 0.002
    for row in failed:
        run = ROOT / "runs" / row["run_id"]
        data = load(run / "data.json")
        meta = load(run / "_meta.json")
        evidence = load(run / "evidence.json")
        accounting = load(run / "accounting.json")
        attempts = load(run / "attempts.json")
        files = load(run / "files.json")
        score = load(run / "_score.json")
        assert data["score"] == score and data["status"] == meta["status"] == "failed"
        assert score["total_score"] == row["total_score"] == 0.0
        assert score["score_source"] == row["score_source"] == "assigned_zero_for_agent_safety_refusal"
        assert score["judge_evaluated"] is row["judge_evaluated"] is False
        assert not (run / "source_score.json").exists()
        assert all(item["score"] == 0 for item in score["items"])
        indexed = next(item for item in index if item["task_id"] == row["task_id"])
        assert indexed["run_id"] == row["run_id"] and indexed["total_score"] == 0.0
        result = next(item for item in all_results if item["task_id"] == row["task_id"])
        assert result["status"] == "failed_safety_refusal" and result["judge_evaluated"] == "False"
        assert float(result["score"]) == 0.0
        assert data["run_id"] == meta["run_id"] == row["run_id"]
        assert data["task_id"] == meta["task_id"] == row["task_id"]
        assert data["agent_name"] == meta["agent_name"] == "EvoLab"
        assert data["failure_category"] == meta["failure_category"] == row["failure_category"]
        assert data["accounting"] == accounting
        assert meta["cost_usd"] == row["cost_usd"]
        assert meta["duration_seconds"] == row["active_seconds"]
        assert len(attempts) == row["attempt_count"]
        assert all(Path(item["source_path"]).is_file() for item in attempts)
        assert sha256(run / "_agent_output.jsonl") == evidence["source_output_sha256"]
        report = Path(evidence["source_workspace"]) / "report/report.md"
        assert row["report_present"] == report.is_file()
        if report.is_file():
            assert sha256(report) == evidence["source_report_sha256"]
            assert data["report"] == (run / "workspace/report/report.md").read_text(encoding="utf-8", errors="replace")
        else:
            assert evidence["source_report_sha256"] is None and data["report"] == ""
        for file in files:
            if file["type"] == "file" and file.get("exported"):
                assert (run / "workspace" / file["path"]).is_file()
                copied_files += 1
        source_total = totals[row["task_id"]]
        assert abs(float(source_total["known_cost_usd"]) - row["known_cost_usd"]) < 0.0001
        assert abs(float(source_total["stage_seconds"]) - row["active_seconds"]) < 0.002
    assert {path.name for path in (ROOT / "runs").iterdir() if path.is_dir()} == {
        row["run_id"] for row in index
    }
    print(f"Verified {len(audit['runs'])} judged and {len(failed)} safety-zero runs, "
          f"{copied_files} viewable files, and source accounting")


if __name__ == "__main__":
    main()
