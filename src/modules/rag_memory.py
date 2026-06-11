"""Wrapper around Gemini File Search Store providing persistent per-experiment
RAG memory. Uploads validated and corrected cases at end of each run;
retrieval is server-side and surfaces via types.Tool(file_search=...)
passed to the Annotator.
"""
import time
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
import toml

class RAGMemory:
    def __init__(self, store_name: str = None, display_name: str = "medical-annotation-rag"):
        secrets = toml.load("./secrets.toml")
        self.client = genai.Client(api_key=secrets["api"]["key"])
        self.display_name = display_name

        if store_name:
            self.store = self.client.file_search_stores.get(name=store_name)
        else:
            self.store = self.client.file_search_stores.create(
                config={"display_name": display_name}
            )
            print(f"  Created new File Search Store: {self.store.name}")

        # Cache existing doc names once — avoids one API call per add_case
        self._existing: set[str] = set()
        self._existing_lock = threading.Lock()
        try:
            docs = self.client.file_search_stores.documents.list(parent=self.store.name)
            self._existing = {doc.display_name for doc in docs}
        except Exception:
            pass

    @property
    def store_name(self):
        return self.store.name

    # Short poll only for logging — the file is already in Google's system once
    # upload_to_file_search_store returns. Indexing happens asynchronously; we
    # don't need it complete before moving on, because:
    #   - Uploads happen at the END of a pipeline run, after all annotations.
    #   - The next run that uses this store rebuilds `self._existing` from the
    #     live API, so whatever has indexed by then is used.
    # So "fire, briefly peek, move on" is correct and orders of magnitude faster.
    _UPLOAD_POLL_SECS    = 1.0   # poll interval during the brief wait
    _UPLOAD_POLL_ATTEMPTS = 8    # 8 × 1 s = 8 s max inline wait per case

    def _upload_one(self, retrieval_text: str, full_record: dict) -> str:
        """Upload a single case. Returns display_name once Google has accepted
        the file. Does NOT block until indexing completes — the file is queued
        for background indexing and will be retrievable on subsequent runs."""
        subject_id = full_record.get("subject_id", "unknown")
        hadm_id = full_record.get("hadm_id", "unknown")
        display_name = f"case-{subject_id}-{hadm_id}"

        content = f"{retrieval_text}\n\n---METADATA---\n{json.dumps(full_record, indent=2)}"
        tmp_path = f"/tmp/rag_case_{subject_id}_{hadm_id}.txt"
        with open(tmp_path, "w") as f:
            f.write(content)

        try:
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=tmp_path,
                file_search_store_name=self.store.name,
                config={"display_name": display_name},
            )
            # Brief peek — catches fast-indexing cases so logs reflect reality.
            # If still pending, leave it queued; indexing continues server-side.
            for _ in range(self._UPLOAD_POLL_ATTEMPTS):
                if operation.done:
                    break
                time.sleep(self._UPLOAD_POLL_SECS)
                try:
                    operation = self.client.operations.get(operation)
                except Exception:
                    # Transient check failure — the upload itself already landed.
                    break
        finally:
            os.remove(tmp_path)

        return display_name

    def add_case(self, retrieval_text: str, full_record: dict):
        """Add a single case (dedup via in-memory cache, no extra API call)."""
        subject_id = full_record.get("subject_id", "unknown")
        hadm_id = full_record.get("hadm_id", "unknown")
        display_name = f"case-{subject_id}-{hadm_id}"

        with self._existing_lock:
            if display_name in self._existing:
                print(f"    RAG: {display_name} already exists. Skipping.")
                return
            self._existing.add(display_name)  # reserve slot before upload

        name = self._upload_one(retrieval_text, full_record)
        print(f"    RAG: {name} indexed")

    def add_cases_parallel(self, cases: list[tuple[str, dict]], max_workers: int = 12):
        """Upload multiple cases concurrently. cases = [(retrieval_text, full_record), ...]"""
        new_cases = []
        with self._existing_lock:
            for retrieval_text, full_record in cases:
                subject_id = full_record.get("subject_id", "unknown")
                hadm_id = full_record.get("hadm_id", "unknown")
                display_name = f"case-{subject_id}-{hadm_id}"
                if display_name in self._existing:
                    print(f"    RAG: {display_name} already exists. Skipping.")
                else:
                    self._existing.add(display_name)
                    new_cases.append((retrieval_text, full_record))

        if not new_cases:
            return

        print(f"    RAG: uploading {len(new_cases)} new cases in parallel...")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._upload_one, text, record): record
                for text, record in new_cases
            }
            for future in as_completed(futures):
                try:
                    name = future.result()
                    print(f"    RAG: {name} indexed")
                except Exception as e:
                    record = futures[future]
                    print(f"    RAG: upload failed for {record.get('subject_id')}/{record.get('hadm_id')}: {e}")

    def get_tool(self) -> types.Tool:
        """Return the Gemini tool config to pass at generation time."""
        return types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[self.store.name]
            )
        )
    
    def list_cases(self):
        """Print all documents in the RAG store."""
        try:
            # Documents are nested under file_search_stores
            # 'parent' must be the full resource name: 'fileSearchStores/...'
            files = self.client.file_search_stores.documents.list(
                parent=self.store.name
            )
        except Exception as e:
            print(f"Error fetching files: {e}")
            return

        if not files:
            print("No cases found in the store.")
            return

        print(f"Cases in store '{self.display_name}':")
        for f in files:
            # f.display_name is the "case-subject-hadm" label you set
            print(f"  - {f.display_name} (ID: {f.name})")

if __name__ == "__main__":
    # Path to where you saved the store ID
    log_path = "logs/rag_store_name.txt"
    
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            # Read the ID and strip any whitespace/newlines
            saved_store_id = f.read().strip()
        
        print(f"Loading existing store: {saved_store_id}")
        rag = RAGMemory(store_name=saved_store_id)
    else:
        # If the log doesn't exist, create a new one
        print("No log found, creating new store...")
        rag = RAGMemory()
        
        # Save the new ID for next time
        os.makedirs("logs", exist_ok=True)
        with open(log_path, "w") as f:
            f.write(rag.store_name)
            
    rag.list_cases()

    
    # ... (your existing logic to load/create the store) ...
    
    # 1. Add a test case to see if it works
    print("Adding a test case...")
    rag.add_case(
        retrieval_text="Patient presents with acute respiratory distress.",
        full_record={"subject_id": "TEST_SUB_01", "hadm_id": "TEST_HADM_01", "note": "Test medical note"}
    )
    
    # 2. Now list them
    print("\nRefreshing list...")
    rag.list_cases()