#!/usr/bin/env python3
"""Stream all 40 selected run workspaces into one ZIP, omitting only workspace/data/."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REVIEW = HERE / "review_set"
DEFAULT_OUTPUT = Path("/dev/shm/ResearchClawBench-EvoLab-G1G2-40tasks-full.zip")
SECRET = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{10,}")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def workspace_files(workspace: Path):
    result = []
    for base, dirs, files in os.walk(workspace, followlinks=False):
        base = Path(base)
        if base == workspace and "data" in dirs:
            dirs.remove("data")
        for name in dirs + files:
            if (base / name).is_symlink():
                raise ValueError(f"Symlink in run workspace: {base / name}")
        for name in files:
            path = base / name
            rel = path.relative_to(workspace).as_posix()
            result.append((rel, path))
    return sorted(result)


def write_source(zf: zipfile.ZipFile, archive_name: str, path: Path):
    before = path.stat()
    stamp = datetime.fromtimestamp(before.st_mtime).timetuple()[:6]
    info = zipfile.ZipInfo(archive_name, stamp if stamp[0] >= 1980 else (1980, 1, 1, 0, 0, 0))
    # Large numeric arrays and model weights dominate CPU time. Store them in
    # the ZIP without recompression; all other files use DEFLATE.
    info.compress_type = (zipfile.ZIP_STORED if before.st_size >= 128_000_000
                          else zipfile.ZIP_DEFLATED)
    info.external_attr = (before.st_mode & 0o777) << 16
    info.file_size = before.st_size
    digest = hashlib.sha256()
    carry = b""
    size = 0
    with path.open("rb") as src, zf.open(info, "w", force_zip64=True) as dst:
        while chunk := src.read(4 * 1024 * 1024):
            if SECRET.search(carry + chunk):
                raise ValueError(f"Potential credential in {path}")
            carry = chunk[-64:]
            digest.update(chunk)
            dst.write(chunk)
            size += len(chunk)
    after = path.stat()
    if size != before.st_size or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError(f"Source changed while archiving: {path}")
    return {"path": archive_name.rsplit("/workspace/", 1)[1], "size": size,
            "sha256": digest.hexdigest()}


def build(output: Path):
    audit = load(REVIEW / "audit.json")
    index = load(REVIEW / "runs_index.json")
    records = audit["runs"] + audit["failed_runs"]
    by_task = {item["task_id"]: item for item in records}
    if len(index) != 40 or len(by_task) != 40:
        raise ValueError("Expected exactly 40 distinct tasks")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Output or partial ZIP already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_lists = {}
    source_bytes = 0
    for row in index:
        task = row["task_id"]
        workspace = Path(by_task[task]["source_workspace"])
        if not workspace.is_dir():
            raise FileNotFoundError(workspace)
        files = workspace_files(workspace)
        source_lists[task] = (workspace, files)
        source_bytes += sum(path.stat().st_size for _, path in files)
    free = shutil.disk_usage(output.parent).free
    if free < source_bytes + 1_000_000_000:
        raise OSError(f"Insufficient guaranteed ZIP space: {free} free, {source_bytes} source bytes")
    print(f"Archiving 40 runs, {source_bytes / 1e9:.2f} GB before compression", flush=True)
    partial = output.with_name(output.name + ".partial")
    total_files = 0
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True, strict_timestamps=False) as zf:
        for row in index:
            task, run_id = row["task_id"], row["run_id"]
            review_run = REVIEW / "runs" / run_id
            workspace, files = source_lists[task]
            meta, score = load(review_run / "_meta.json"), load(review_run / "_score.json")
            if meta["task_id"] != task or meta["run_id"] != run_id:
                raise ValueError(f"Run identity mismatch: {task}")
            if score["total_score"] != row["total_score"]:
                raise ValueError(f"Score mismatch: {task}")
            if row["status"] == "failed":
                if score.get("score_source") != "assigned_zero_for_agent_safety_refusal" or score.get("judge_evaluated") is not False:
                    raise ValueError(f"Failed task missing safety-zero provenance: {task}")
            prefix = run_id + "/"
            zf.writestr(prefix, b"")
            zf.writestr(prefix + "workspace/", b"")
            for path in sorted(review_run.iterdir()):
                if path.is_file() and path.name not in {"files.json", "evidence.json"}:
                    zf.write(path, prefix + path.name)
            entries, file_tree = [], []
            for rel, path in files:
                entry = write_source(zf, prefix + "workspace/" + rel, path)
                entries.append(entry)
                file_tree.append({"name": path.name, "path": rel, "type": "file",
                                  "size": entry["size"], "mtime": path.stat().st_mtime,
                                  "exported": True})
            zf.writestr(prefix + "files.json", json_bytes(file_tree))
            data_dir = workspace / "data"
            zf.writestr(prefix + "archive_manifest.json", json_bytes({
                "run_id": run_id, "task_id": task, "source_workspace": str(workspace),
                "omitted_workspace_data": data_dir.is_dir(),
                "included_file_count": len(entries),
                "included_file_bytes": sum(item["size"] for item in entries),
                "files": entries,
            }))
            total_files += len(entries)
            print(f"{task}: {len(entries)} files, {sum(item['size'] for item in entries) / 1e9:.2f} GB", flush=True)
    os.replace(partial, output)
    print(f"Created {output} ({output.stat().st_size / 1e9:.2f} GB); {total_files} workspace files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
