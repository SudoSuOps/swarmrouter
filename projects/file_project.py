#!/usr/bin/env python3
"""
file_project.py — File a signed project OM into R2 + Supabase.

Usage:
    python3 projects/file_project.py --project PRJ-001-SWARMROUTER --sign
    python3 projects/file_project.py --project PRJ-001-SWARMROUTER --status
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECTS_DIR = Path(__file__).parent
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

PROJECT_REGISTRY = {
    "PRJ-001-SWARMROUTER": {
        "project_name": "Project:SwarmRouter",
        "description": "Intelligent routing infrastructure for the Swarm & Bee platform. "
                       "Seven purpose-trained models. Every request routed to the correct slot.",
        "model_lineup": [
            "router-4b", "research-8b", "med-14b",
            "swarmpharma-35b", "swarmcre-35b", "swarmjudge-27b", "swarmresearch-32b"
        ],
        "infrastructure": {
            "compute": ["swarmrails-blackwell-96gb", "swarmrails-3090ti-24gb", "whale-3090-24gb"],
            "r2_buckets": ["sb-projects", "sb-builds", "sb-models"],
            "supabase_tables": ["projects", "build_phases", "model_builds"],
        },
        "phases": [
            {"id": "PRJ-001-PHASE-1", "name": "SwarmRouter-4B-0 Block Zero",     "number": 1},
            {"id": "PRJ-001-PHASE-2", "name": "SwarmJudge-27B v1",               "number": 2},
            {"id": "PRJ-001-PHASE-3", "name": "SwarmPharma-35B v2",              "number": 3},
            {"id": "PRJ-001-PHASE-4", "name": "SwarmCRE-35B v3",                 "number": 4},
            {"id": "PRJ-001-PHASE-5", "name": "SwarmMed-14B v2",                 "number": 5},
            {"id": "PRJ-001-PHASE-6", "name": "SwarmResearch-32B v2",            "number": 6},
        ],
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_to_r2(local_path: Path, r2_key: str, bucket: str = "sb-projects") -> bool:
    cmd = ["npx", "wrangler", "r2", "object", "put",
           f"{bucket}/{r2_key}", "--file", str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def supabase_upsert(table: str, record: dict) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print(f"  [warn] SUPABASE_URL/KEY not set — skipping {table}")
        return False
    import urllib.request, urllib.error
    data = json.dumps(record).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/{table}",
        data=data,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 201)
    except urllib.error.HTTPError as e:
        print(f"  [warn] Supabase {table}: {e.code} {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"  [warn] Supabase error: {e}")
        return False


def file_project(project_id: str, sign: bool = False):
    if project_id not in PROJECT_REGISTRY:
        print(f"Unknown project: {project_id}")
        sys.exit(1)

    meta = PROJECT_REGISTRY[project_id]
    project_dir = PROJECTS_DIR / project_id
    om_path = project_dir / f"OM_{project_id}.md"

    if not om_path.exists():
        print(f"OM not found: {om_path}")
        sys.exit(1)

    sha = sha256_file(om_path)
    now = datetime.utcnow().isoformat() + "Z"
    r2_key = f"{project_id}/OM_{project_id}_v1.0.md"

    print(f"{'='*60}")
    print(f"  Project:  {meta['project_name']}")
    print(f"  ID:       {project_id}")
    print(f"  OM:       {om_path.name}")
    print(f"  SHA256:   {sha}")
    print(f"{'='*60}")

    if not sign:
        print("\n  [dry-run] Pass --sign to file officially.")
        return

    # Upload OM to R2
    print(f"\nUploading OM to sb-projects/{r2_key}...")
    ok = upload_to_r2(om_path, r2_key)
    print(f"  {'✓' if ok else '✗'} R2 upload")

    # Write SHA256SUM
    sha_path = project_dir / f"OM_{project_id}_SHA256.txt"
    with open(sha_path, "w") as f:
        f.write(f"{sha}  {om_path.name}\n")
    upload_to_r2(sha_path, f"{project_id}/OM_{project_id}_SHA256.txt")

    # Register in Supabase projects table
    project_record = {
        "project_id":   project_id,
        "project_name": meta["project_name"],
        "status":       "active",
        "phase":        "PRJ-001-PHASE-1",
        "description":  meta["description"],
        "om_r2_path":   f"sb-projects/{r2_key}",
        "om_signed_at": now,
        "om_signed_by": "operator",
        "phases":       meta["phases"],
        "deliverables": {},
        "model_lineup": meta["model_lineup"],
        "infrastructure": meta["infrastructure"],
    }
    print("Registering in Supabase projects...")
    ok = supabase_upsert("projects", project_record)
    print(f"  {'✓' if ok else '✗'} Supabase projects")

    # Register all phases
    print("Registering build phases...")
    for phase in meta["phases"]:
        phase_record = {
            "project_id":   project_id,
            "phase_id":     phase["id"],
            "phase_name":   phase["name"],
            "phase_number": phase["number"],
            "status":       "active" if phase["number"] == 1 else "pending",
        }
        ok = supabase_upsert("build_phases", phase_record)
        status = "active" if phase["number"] == 1 else "pending"
        print(f"  {'✓' if ok else '✗'} {phase['id']} ({status})")

    print(f"\n{'='*60}")
    print(f"  FILED: {project_id}")
    print(f"  R2:    sb-projects/{project_id}/")
    print(f"  Phase 1 is ACTIVE — ready to train")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="File a project OM into R2 + Supabase")
    parser.add_argument("--project", required=True, help="Project ID e.g. PRJ-001-SWARMROUTER")
    parser.add_argument("--sign", action="store_true", help="Sign and file officially")
    parser.add_argument("--status", action="store_true", help="Show project status")
    args = parser.parse_args()

    if args.status:
        print(f"Project: {args.project}")
        print(f"Known projects: {list(PROJECT_REGISTRY.keys())}")
        return

    file_project(args.project, sign=args.sign)


if __name__ == "__main__":
    main()
