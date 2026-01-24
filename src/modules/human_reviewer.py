from typing import Dict, Optional
import tkinter as tk
from tkinter import ttk


class HumanReviewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Human-in-the-Loop Medical Annotation Review")
        self.root.geometry("900x650")

        self.result = None
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

        ttk.Label(right_frame, text="Feedback (optional):").pack(anchor="w", pady=(15, 0))
        self.feedback_box = tk.Text(right_frame, height=6, width=35)
        self.feedback_box.pack()

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(pady=15)

        ttk.Button(
            btn_frame, text="✅ Correct", command=lambda: self._submit(True)
        ).grid(row=0, column=0, padx=10)

        ttk.Button(
            btn_frame, text="❌ Incorrect", command=lambda: self._submit(False)
        ).grid(row=0, column=1, padx=10)

    def _submit(self, correct: bool):
        self.result = {
            "correct": correct,
            "reason": self.feedback_box.get("1.0", tk.END).strip() or None,
        }
        self.root.quit()  # exits mainloop but does NOT destroy window

    def review(
        self,
        annotation: Dict,
        medical_note: str,
        gold: Optional[str] = None,
    ) -> Dict:
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
            text=f"Confidence: {annotation.get('confidence_level')}"
        )

        if gold is not None:
            self.gold_label.config(text=f"Real diagnosis: {gold}")
        else:
            self.gold_label.config(text="")

        self.feedback_box.delete("1.0", tk.END)
        self.result = None

        self.root.mainloop()   # waits for button click
        self.root.update()     # keep window alive

        return {
            **self.result,
            "model_confidence": annotation.get("confidence_level", 0),
        }

    def close(self):
        self.root.destroy()
