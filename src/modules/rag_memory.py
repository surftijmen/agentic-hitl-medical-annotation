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
            self.store = self.client.file_search_stores.get(name=store_name)
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
        subject_id = full_record.get("subject_id", "unknown")
        hadm_id = full_record.get("hadm_id", "unknown")
        target_display_name = f"case-{subject_id}-{hadm_id}"

        # 1. CHECK FOR DUPLICATES FIRST
        try:
            existing_docs = self.client.file_search_stores.documents.list(parent=self.store.name)
            if any(doc.display_name == target_display_name for doc in existing_docs):
                print(f"    RAG: {target_display_name} already exists. Skipping.")
                return
        except Exception as e:
            print(f"    RAG: Error checking existing cases: {e}")

        # 2. PROCEED ONLY IF NEW
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