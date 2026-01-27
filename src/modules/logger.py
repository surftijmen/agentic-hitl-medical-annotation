import json
import os
from datetime import datetime
from typing import Dict
from pprint import pprint


class EventLogger:
    def __init__(self, run_id: str, base_dir="logs/events", debug=True):
        self.run_id = run_id
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f"{run_id}.json")
        self.debug = debug

    def log(self, event_type: str, payload: Dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            "payload": payload,
        }

        if self.debug:
            print(f"\n===== {event_type} =====")
            pprint(payload)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_run(self,
        prompt_version: str,
        sample_size: int,
        notes: str = "",
    ):
        os.makedirs("logs/runs", exist_ok=True)

        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "prompt_version": prompt_version,
            "sample_size": sample_size,
            "sampler": "random",  # or "hard_cases_v1"
            "model": "gemini-2.5-flash",
            "reviewers": 2,
            "notes": notes,
        }

        path = f"logs/runs/{self.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return path


