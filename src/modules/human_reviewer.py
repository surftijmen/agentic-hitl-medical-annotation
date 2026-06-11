"""Tkinter GUI for the human reviewer. Captures verdict (correct/incorrect),
failure mode, free-text comment, and confidence. Returns the same dict
shape as AutoReviewer.review() so run_pipeline() is mode-agnostic.
"""
from typing import Dict, Optional
import tkinter as tk
from tkinter import ttk


FAILURE_MODES = [
    ("unsupported_inference",       "Unsupported inference"),
    ("symptom_diagnosis_confusion", "Symptom–diagnosis confusion"),
    ("hallucination",               "Hallucinated diagnosis"),
    ("missed_entity",               "Missed diagnosis"),
    ("terminology_gap",             "Terminology / naming issue"),
    ("ambiguous_case",              "Ambiguous / unclear case"),
]

# ── Color palette ─────────────────────────────────────────────────────────────
# Uses tk.Button (not ttk.Button) so colors work on macOS Aqua theme.

C_BG          = "#f5f5f5"    # window background
C_PANEL       = "#ffffff"    # right panel background
C_CORRECT_ON  = "#2a7d46"    # green — selected state
C_CORRECT_OFF = "#d4edda"    # pale green — idle
C_WRONG_ON    = "#b83232"    # red — selected state
C_WRONG_OFF   = "#f8d7da"    # pale red — idle
C_SUBMIT      = "#0a7aff"    # macOS system blue
C_SUBMIT_HOV  = "#0060d0"
C_TEXT_LIGHT  = "#6c757d"
C_TEXT_DARK   = "#1a1a1a"
C_WARN        = "#d97706"    # amber for validation warnings
C_DIVIDER     = "#dde0e4"


class HumanReviewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HITL Medical Annotation Review")
        self.root.geometry("1100x720")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        self.result = None
        self.current_correct = None
        self._case_num = 0
        self._total_cases = 0

        self._build_ui()
        self._bind_keys()

    # ── Keyboard shortcuts ────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<c>", lambda _: self._mark_correct())
        self.root.bind("<C>", lambda _: self._mark_correct())
        self.root.bind("<i>", lambda _: self._mark_incorrect())
        self.root.bind("<I>", lambda _: self._mark_incorrect())
        self.root.bind("<Return>", lambda _: self._submit())

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        # ── Progress bar (top strip) ──────────────────────────────
        top_bar = tk.Frame(self.root, bg=C_BG, pady=6)
        top_bar.pack(fill=tk.X, padx=14)

        self.progress_label = tk.Label(
            top_bar, text="", font=("SF Pro Text", 11), bg=C_BG, fg=C_TEXT_LIGHT
        )
        self.progress_label.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(top_bar, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(side=tk.RIGHT, padx=(8, 0))

        # ── Main content area ─────────────────────────────────────
        content = tk.Frame(self.root, bg=C_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        # ── Left: Medical note ────────────────────────────────────
        left = tk.Frame(content, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        tk.Label(
            left, text="Medical Note", font=("SF Pro Text", 12, "bold"),
            bg=C_BG, fg=C_TEXT_DARK, anchor="w"
        ).pack(fill=tk.X, pady=(0, 4))

        note_container = tk.Frame(left, bg=C_DIVIDER, bd=1, relief=tk.FLAT)
        note_container.pack(fill=tk.BOTH, expand=True)

        self.note_box = tk.Text(
            note_container, wrap=tk.WORD, relief=tk.FLAT,
            font=("Menlo", 11), bg=C_PANEL, fg=C_TEXT_DARK,
            padx=10, pady=10, spacing1=2,
        )
        note_scroll = tk.Scrollbar(note_container, command=self.note_box.yview)
        note_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.note_box.config(yscrollcommand=note_scroll.set, state=tk.DISABLED)
        self.note_box.pack(fill=tk.BOTH, expand=True)

        # ── Right: Controls ───────────────────────────────────────
        right = tk.Frame(content, bg=C_PANEL, bd=1, relief=tk.FLAT, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        inner = tk.Frame(right, bg=C_PANEL, padx=16, pady=12)
        inner.pack(fill=tk.BOTH, expand=True)

        # ── Model output section ──────────────────────────────────
        tk.Label(
            inner, text="Model Output", font=("SF Pro Text", 12, "bold"),
            bg=C_PANEL, fg=C_TEXT_DARK, anchor="w"
        ).pack(fill=tk.X)

        self._divider(inner)

        self.diagnosis_box = tk.Text(
            inner, height=3, wrap=tk.WORD,
            font=("SF Pro Text", 13, "bold"),
            bg=C_PANEL, fg=C_TEXT_DARK,
            borderwidth=0, highlightthickness=0,
            padx=0, pady=0, cursor="ibeam",
        )
        self.diagnosis_box.pack(fill=tk.X, pady=(6, 2))
        self.diagnosis_box.config(state=tk.DISABLED)

        self.confidence_label = tk.Label(
            inner, text="", font=("SF Pro Text", 11),
            bg=C_PANEL, fg=C_TEXT_LIGHT, anchor="w"
        )
        self.confidence_label.pack(fill=tk.X)

        self.cot_label = tk.Label(
            inner, text="", font=("SF Pro Text", 10), fg=C_TEXT_LIGHT,
            bg=C_PANEL, anchor="w", wraplength=270, justify=tk.LEFT
        )
        self.cot_label.pack(fill=tk.X, pady=(2, 0))

        self.gold_box = tk.Text(
            inner, height=4, wrap=tk.WORD,
            font=("SF Pro Text", 11, "italic"),
            bg=C_PANEL, fg=C_TEXT_LIGHT,
            borderwidth=0, highlightthickness=0,
            padx=0, pady=0, cursor="ibeam",
        )
        self.gold_box.pack(fill=tk.X, pady=(4, 0))
        self.gold_box.config(state=tk.DISABLED)

        self._divider(inner)

        # ── Verdict buttons ───────────────────────────────────────
        tk.Label(
            inner, text="Your verdict  (c / i)", font=("SF Pro Text", 11, "bold"),
            bg=C_PANEL, fg=C_TEXT_DARK, anchor="w"
        ).pack(fill=tk.X, pady=(8, 6))

        verdict_row = tk.Frame(inner, bg=C_PANEL)
        verdict_row.pack(fill=tk.X)

        # tk.Button so bg/fg work on macOS
        self.btn_correct = tk.Button(
            verdict_row, text="Correct",
            font=("SF Pro Text", 12, "bold"),
            bg=C_CORRECT_OFF, fg=C_TEXT_DARK, activebackground=C_CORRECT_ON,
            activeforeground="white", relief=tk.FLAT, bd=0,
            padx=14, pady=10, cursor="hand2",
            command=self._mark_correct,
        )
        self.btn_correct.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_incorrect = tk.Button(
            verdict_row, text="Incorrect",
            font=("SF Pro Text", 12, "bold"),
            bg=C_WRONG_OFF, fg=C_TEXT_DARK, activebackground=C_WRONG_ON,
            activeforeground="white", relief=tk.FLAT, bd=0,
            padx=14, pady=10, cursor="hand2",
            command=self._mark_incorrect,
        )
        self.btn_incorrect.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Selected-state indicator
        self.verdict_status = tk.Label(
            inner, text="No verdict selected", font=("SF Pro Text", 10),
            bg=C_PANEL, fg=C_TEXT_LIGHT, anchor="w"
        )
        self.verdict_status.pack(fill=tk.X, pady=(4, 0))

        # ── Failure mode (hidden until incorrect is clicked) ──────
        self.failure_reason_var = tk.StringVar(value="")
        self.failure_frame = tk.Frame(inner, bg=C_PANEL)

        tk.Label(
            self.failure_frame, text="Why is it incorrect?",
            font=("SF Pro Text", 11, "bold"), bg=C_PANEL, fg=C_TEXT_DARK, anchor="w"
        ).pack(fill=tk.X, pady=(10, 4))

        self.failure_dropdown = ttk.Combobox(
            self.failure_frame,
            textvariable=self.failure_reason_var,
            state="readonly",
            values=[label for _, label in FAILURE_MODES],
            font=("SF Pro Text", 11),
            width=30,
        )
        self.failure_dropdown.pack(fill=tk.X)
        self.failure_frame.pack_forget()

        self._divider(inner)

        # ── Comment box ───────────────────────────────────────────
        tk.Label(
            inner, text="Comment (optional)",
            font=("SF Pro Text", 11, "bold"), bg=C_PANEL, fg=C_TEXT_DARK, anchor="w"
        ).pack(fill=tk.X, pady=(8, 4))

        self.feedback_box = tk.Text(
            inner, height=4, relief=tk.FLAT,
            font=("SF Pro Text", 11), bg="#f0f0f0", fg=C_TEXT_DARK,
            padx=6, pady=6, bd=1, highlightthickness=1,
            highlightbackground=C_DIVIDER,
        )
        self.feedback_box.pack(fill=tk.X)

        # ── Confidence slider ─────────────────────────────────────
        conf_header = tk.Frame(inner, bg=C_PANEL)
        conf_header.pack(fill=tk.X, pady=(10, 2))

        tk.Label(
            conf_header, text="Your confidence",
            font=("SF Pro Text", 11, "bold"), bg=C_PANEL, fg=C_TEXT_DARK
        ).pack(side=tk.LEFT)

        self.confidence_var = tk.IntVar(value=80)
        self.conf_value_label = tk.Label(
            conf_header, text="80%", font=("SF Pro Text", 11),
            bg=C_PANEL, fg=C_TEXT_LIGHT
        )
        self.conf_value_label.pack(side=tk.RIGHT)

        self.confidence_slider = ttk.Scale(
            inner, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            command=lambda v: self.conf_value_label.config(
                text=f"{int(float(v))}%"
            ),
        )
        self.confidence_slider.pack(fill=tk.X)

        self._divider(inner)

        # ── Validation warning ────────────────────────────────────
        self.warn_label = tk.Label(
            inner, text="", font=("SF Pro Text", 10),
            bg=C_PANEL, fg=C_WARN, anchor="w", wraplength=270,
        )
        self.warn_label.pack(fill=tk.X, pady=(6, 0))

        # ── Submit button ─────────────────────────────────────────
        self.btn_submit = tk.Button(
            inner, text="Submit  ↵",
            font=("SF Pro Text", 13, "bold"),
            bg=C_SUBMIT, fg="white",
            activebackground=C_SUBMIT_HOV, activeforeground="white",
            relief=tk.FLAT, bd=0, padx=0, pady=12, cursor="hand2",
            command=self._submit,
        )
        self.btn_submit.pack(fill=tk.X, pady=(10, 0))

    def _divider(self, parent):
        tk.Frame(parent, bg=C_DIVIDER, height=1).pack(fill=tk.X, pady=6)

    # ── Verdict logic ─────────────────────────────────────────────

    def _mark_correct(self):
        self.current_correct = True
        self.btn_correct.config(bg=C_CORRECT_ON, fg="white")
        self.btn_incorrect.config(bg=C_WRONG_OFF, fg=C_TEXT_DARK)
        self.verdict_status.config(text="Selected: Correct", fg=C_CORRECT_ON)
        self.failure_frame.pack_forget()
        self.failure_reason_var.set("")
        self.warn_label.config(text="")

    def _mark_incorrect(self):
        self.current_correct = False
        self.btn_incorrect.config(bg=C_WRONG_ON, fg="white")
        self.btn_correct.config(bg=C_CORRECT_OFF, fg=C_TEXT_DARK)
        self.verdict_status.config(text="Selected: Incorrect", fg=C_WRONG_ON)
        self.failure_frame.pack(fill=tk.X, pady=(4, 0))

    def _submit(self):
        if self.current_correct is None:
            self.warn_label.config(text="Please select Correct or Incorrect first.")
            return

        failure_mode = None
        if self.current_correct is False:
            label = self.failure_reason_var.get()
            if not label:
                self.warn_label.config(text="Please select a failure mode.")
                return
            for key, lbl in FAILURE_MODES:
                if lbl == label:
                    failure_mode = key
                    break

        self.warn_label.config(text="")
        self.result = {
            "correct": self.current_correct,
            "failure_mode": failure_mode,
            "comment": self.feedback_box.get("1.0", tk.END).strip() or None,
            "confidence": int(self.confidence_var.get()),
        }
        self.root.quit()

    # ── Review entrypoint ─────────────────────────────────────────

    def review(
        self,
        annotation: Dict,
        medical_note: str,
        gold=None,
        case_num: int = None,
        total_cases: int = None,
    ) -> Dict:
        # ── Reset state ───────────────────────────────────────────
        self.result = None
        self.current_correct = None
        self.failure_reason_var.set("")
        self.failure_frame.pack_forget()
        self.feedback_box.delete("1.0", tk.END)
        self.warn_label.config(text="")
        self.btn_correct.config(bg=C_CORRECT_OFF, fg=C_TEXT_DARK)
        self.btn_incorrect.config(bg=C_WRONG_OFF, fg=C_TEXT_DARK)
        self.verdict_status.config(text="No verdict selected", fg=C_TEXT_LIGHT)
        self.confidence_var.set(80)
        self.conf_value_label.config(text="80%")

        # ── Progress ──────────────────────────────────────────────
        if case_num and total_cases:
            self.progress_label.config(
                text=f"Case {case_num} of {total_cases}"
            )
            self.progress_bar["maximum"] = total_cases
            self.progress_bar["value"] = case_num - 1
        else:
            self.progress_label.config(text="")

        # ── Medical note ──────────────────────────────────────────
        self.note_box.config(state=tk.NORMAL)
        self.note_box.delete("1.0", tk.END)
        self.note_box.insert(tk.END, medical_note)
        self.note_box.config(state=tk.DISABLED)

        # ── Model output ──────────────────────────────────────────
        self.diagnosis_box.config(state=tk.NORMAL)
        self.diagnosis_box.delete("1.0", tk.END)
        dx_text = f"Diagnosis: {annotation.get('diagnosis', 'N/A')}"
        auto_correct = annotation.get("_auto_correct")
        if auto_correct is not None:
            dx_text += f"\nAuto-judge: {'✓ correct' if auto_correct else '✗ incorrect'}"
        self.diagnosis_box.insert(tk.END, dx_text)
        self.diagnosis_box.config(state=tk.DISABLED)
        conf = annotation.get("confidence_level", "N/A")
        cons = annotation.get("consistency_score")
        if cons is not None:
            self.confidence_label.config(
                text=f"Confidence: {conf}%   Consistency: {cons:.0%}"
            )
        else:
            self.confidence_label.config(text=f"Confidence: {conf}%")

        # Show CoT reasoning snippet if present
        reasoning = annotation.get("reasoning", "")
        if reasoning:
            snippet = reasoning[:160].replace("\n", " ")
            self.cot_label.config(text=f"Reasoning: {snippet}…")
        else:
            self.cot_label.config(text="")

        if isinstance(gold, dict):
            gold_text = (
                f"Gold reference: {gold.get('long_title', '')} "
                f"(ICD-9 {gold.get('icd9_code', '?')})"
            )
        elif gold:
            gold_text = f"Gold reference: {gold}"
        else:
            gold_text = ""
        self.gold_box.config(state=tk.NORMAL)
        self.gold_box.delete("1.0", tk.END)
        self.gold_box.insert(tk.END, gold_text)
        self.gold_box.config(state=tk.DISABLED)

        self.root.mainloop()
        self.root.update()

        return {
            **self.result,
            "model_confidence": annotation.get("confidence_level", 0),
        }

    def store_validated_case(rag, note, model_output, human_feedback):
        final_diagnosis = (
            model_output["diagnosis"]
            if human_feedback["correct"]
            else human_feedback.get("corrected_diagnosis")
        )
        retrieval_text = f"""
        Patient case with diagnosis {final_diagnosis}.
        {note[:500]}
        """
        record = {
            "retrieval_text": retrieval_text,
            "note": note,
            "final_diagnosis": final_diagnosis,
            "model_diagnosis": model_output["diagnosis"],
            "correct": human_feedback["correct"],
            "failure_mode": human_feedback.get("failure_mode"),
            "doctor_confidence": human_feedback["confidence"],
            "model_confidence": model_output.get("confidence_level", 0),
            "comment": human_feedback.get("comment"),
        }
        rag.add_case(retrieval_text, record)

    def close(self):
        self.root.destroy()
