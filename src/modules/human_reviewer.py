from typing import Dict, Optional
import tkinter as tk
from tkinter import ttk


FAILURE_MODES = [
    ("unsupported_inference", "Unsupported inference"),
    ("symptom_diagnosis_confusion", "Symptom–diagnosis confusion"),
    ("hallucination", "Hallucinated diagnosis"),
    ("missed_entity", "Missed diagnosis"),
    ("terminology_gap", "Terminology / naming issue"),
    ("ambiguous_case", "Ambiguous / unclear case"),
]


class HumanReviewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Human-in-the-Loop Medical Annotation Review")
        self.root.geometry("900x650")

        self.result = None
        self.current_correct = None
        self._build_ui()

    def _build_ui(self):
        # ---- Left: Medical note ----
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(left_frame, text="Medical Note", font=("Arial", 11, "bold")).pack(anchor="w")

        self.note_box = tk.Text(left_frame, wrap=tk.WORD)
        self.note_box.pack(fill=tk.BOTH, expand=True)

        note_scroll = ttk.Scrollbar(left_frame, command=self.note_box.yview)
        note_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.note_box.config(yscrollcommand=note_scroll.set)
        self.note_box.config(state=tk.DISABLED)

        # ---- Right: Annotation + controls ----
        right_frame = ttk.Frame(self.root, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(right_frame, text="Model Output", font=("Arial", 11, "bold")).pack(anchor="w")

        self.diagnosis_label = ttk.Label(right_frame, text="")
        self.diagnosis_label.pack(anchor="w", pady=(5, 0))

        self.confidence_label = ttk.Label(right_frame, text="")
        self.confidence_label.pack(anchor="w")

        self.gold_label = ttk.Label(right_frame, text="", foreground="gray")
        self.gold_label.pack(anchor="w", pady=(5, 0))

        # ---- Failure reason (hidden by default) ----
        self.failure_reason_var = tk.StringVar(value="")

        self.failure_frame = ttk.Frame(right_frame)

        ttk.Label(
            self.failure_frame,
            text="Why is it incorrect?",
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        self.failure_dropdown = ttk.Combobox(
            self.failure_frame,
            textvariable=self.failure_reason_var,
            state="readonly",
            values=[label for _, label in FAILURE_MODES],
            width=32
        )
        self.failure_dropdown.pack(anchor="w", pady=5)

        self.failure_frame.pack_forget()

        # ---- Optional feedback ----
        ttk.Label(right_frame, text="Optional comment:").pack(anchor="w", pady=(10, 0))
        self.feedback_box = tk.Text(right_frame, height=6, width=35)
        self.feedback_box.pack()

        # ---- Confidence slider ----
        ttk.Label(right_frame, text="Your confidence:").pack(anchor="w", pady=(10, 0))
        self.confidence_var = tk.IntVar(value=80)

        self.confidence_slider = ttk.Scale(
            right_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var
        )
        self.confidence_slider.pack(fill=tk.X)

        # ---- Buttons ----
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(pady=15)

        ttk.Button(
            btn_frame, text="✅ Correct", command=self._mark_correct
        ).grid(row=0, column=0, padx=10)

        ttk.Button(
            btn_frame, text="❌ Incorrect", command=self._mark_incorrect
        ).grid(row=0, column=1, padx=10)

        ttk.Button(
            btn_frame, text="Submit", command=self._submit
        ).grid(row=1, column=0, columnspan=2, pady=(10, 0))

    # ---- Button logic ----

    def _mark_correct(self):
        self.current_correct = True
        self.failure_frame.pack_forget()
        self.failure_reason_var.set("")

    def _mark_incorrect(self):
        self.current_correct = False
        self.failure_frame.pack(anchor="w", pady=(15, 0), fill=tk.X)

    def _submit(self):
        if self.current_correct is None:
            return  # nothing selected yet

        failure_mode = None

        if self.current_correct is False:
            label = self.failure_reason_var.get()
            if not label:
                return  # must select a failure reason

            for key, lbl in FAILURE_MODES:
                if lbl == label:
                    failure_mode = key
                    break

        self.result = {
            "correct": self.current_correct,
            "failure_mode": failure_mode,
            "comment": self.feedback_box.get("1.0", tk.END).strip() or None,
            "confidence": int(self.confidence_var.get()),
        }

        self.root.quit()

    # ---- Review entrypoint ----

    def review(
        self,
        annotation: Dict,
        medical_note: str,
        gold: Optional[str] = None,
    ) -> Dict:
        # Reset UI state
        self.result = None
        self.current_correct = None
        self.failure_reason_var.set("")
        self.failure_frame.pack_forget()
        self.feedback_box.delete("1.0", tk.END)

        # Update note
        self.note_box.config(state=tk.NORMAL)
        self.note_box.delete("1.0", tk.END)
        self.note_box.insert(tk.END, medical_note)
        self.note_box.config(state=tk.DISABLED)

        # Update labels
        self.diagnosis_label.config(
            text=f"Diagnosis: {annotation.get('diagnosis')}"
        )
        self.confidence_label.config(
            text=f"Model confidence: {annotation.get('confidence_level')}"
        )

        if gold is not None:
            self.gold_label.config(text=f"Gold reference: {gold}")
        else:
            self.gold_label.config(text="")

        self.root.mainloop()
        self.root.update()

        return {
            **self.result,
            "model_confidence": annotation.get("confidence_level", 0),
        }

    def close(self):
        self.root.destroy()
