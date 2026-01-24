import json
import os
from datetime import datetime


def log_run(
    run_id: str,
    prompt_version: str,
    sample_size: int,
    notes: str = "",
):
    os.makedirs("runs", exist_ok=True)

    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_version": prompt_version,
        "sample_size": sample_size,
        "notes": notes,
    }

    path = f"runs/{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return path
