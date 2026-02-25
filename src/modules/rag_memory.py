# rag_memory.py
import time
import json
import os
from google import genai
from google.genai import types
import toml

class RAGMemory:
    def __init__(self, store_name: str = None, display_name: str = "medical-annotation-rag"):
        secrets = toml.load("./secrets.toml")
        self.client = genai.Client(api_key=secrets["api"]["key"])
        self.display_name = display_name

        if store_name:
            # Resume existing store
            self.store = self.client.file_search_stores.get(store_name)
        else:
            # Create new store
            self.store = self.client.file_search_stores.create(
                config={"display_name": display_name}
            )
            print(f"  Created new File Search Store: {self.store.name}")

    @property
    def store_name(self):
        return self.store.name

    def add_case(self, retrieval_text: str, full_record: dict):
        """Upload a validated case as a searchable text file."""
        subject_id = full_record.get("subject_id", "unknown")
        hadm_id = full_record.get("hadm_id", "unknown")

        # Combine retrieval text + structured metadata into one document
        content = f"{retrieval_text}\n\n---METADATA---\n{json.dumps(full_record, indent=2)}"

        # Write to a temp file and upload
        tmp_path = f"/tmp/rag_case_{subject_id}_{hadm_id}.txt"
        with open(tmp_path, "w") as f:
            f.write(content)

        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=self.store.name,
            config={"display_name": f"case-{subject_id}-{hadm_id}"},
        )

        # Wait for indexing
        while not operation.done:
            time.sleep(3)
            operation = self.client.operations.get(operation)

        os.remove(tmp_path)
        print(f"    RAG: case {subject_id}/{hadm_id} indexed")

    def get_tool(self) -> types.Tool:
        """Return the Gemini tool config to pass at generation time."""
        return types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[self.store.name]
            )
        )