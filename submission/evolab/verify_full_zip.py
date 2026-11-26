#!/usr/bin/env python3
"""Verify the full 40-task ZIP against its per-run SHA-256 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REVIEW = HERE / "review_set"
DEFAULT_ZIP = Path("/dev/shm/ResearchClawBench-EvoLab-G1G2-40tasks-full.zip")


def read_json(zf: zipfile.ZipFile, name: str):
    return json.loads(zf.read(name))


def verify(path: Path):
    index = json.loads((REVIEW / "runs_index.json").read_text(encoding="utf-8"))
    assert len(index) == 40 and len({row["task_id"] for row in index}) == 40
    total_files = total_bytes = 0
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert len(names) == len(set(names)), "Duplicate ZIP entries"
        actual = set(names)
        expected_runs = {row["run_id"] for row in index}
        assert {name.split("/", 1)[0] for name in names} == expected_runs
        for row in index:
            prefix = row["run_id"] + "/"
            assert prefix in actual and prefix + "workspace/" in actual
            for item in ("_meta.json", "_score.json", "data.json", "accounting.json",
                         "attempts.json", "files.json", "archive_manifest.json"):
                assert prefix + item in actual, (row["task_id"], item)
            meta = read_json(zf, prefix + "_meta.json")
            score = read_json(zf, prefix + "_score.json")
            data = read_json(zf, prefix + "data.json")
            manifest = read_json(zf, prefix + "archive_manifest.json")
            files = read_json(zf, prefix + "files.json")
            assert meta["run_id"] == score["run_id"] == data["run_id"] == manifest["run_id"] == row["run_id"]
            assert meta["task_id"] == score["task_id"] == manifest["task_id"] == row["task_id"]
            assert score["total_score"] == row["total_score"]
            assert data["score"] == score
            if row["status"] == "failed":
                assert score["score_source"] == "assigned_zero_for_agent_safety_refusal"
                assert score["judge_evaluated"] is False and score["total_score"] == 0
                assert prefix + "source_score.json" not in actual
            else:
                assert meta["status"] == "completed" and prefix + "source_score.json" in actual
            entries = manifest["files"]
            assert len(entries) == manifest["included_file_count"] == len(files)
            assert sum(item["size"] for item in entries) == manifest["included_file_bytes"]
            expected_workspace = {prefix + "workspace/" + item["path"] for item in entries}
            actual_workspace = {name for name in names if name.startswith(prefix + "workspace/") and not name.endswith("/")}
            assert expected_workspace == actual_workspace
            assert not any(name.startswith(prefix + "workspace/data/") for name in names)
            assert {item["path"] for item in entries} == {item["path"] for item in files}
            for item in entries:
                name = prefix + "workspace/" + item["path"]
                digest = hashlib.sha256()
                size = 0
                with zf.open(name) as stream:
                    while chunk := stream.read(4 * 1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                assert size == item["size"] and digest.hexdigest() == item["sha256"], name
                total_files += 1
                total_bytes += size
            print(f"Verified {row['task_id']}: {len(entries)} files", flush=True)
    print(f"Verified 40 runs, {total_files} workspace files, {total_bytes / 1e9:.2f} GB uncompressed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, nargs="?", default=DEFAULT_ZIP)
    verify(parser.parse_args().zip_path)
