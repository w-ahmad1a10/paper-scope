"""Graphical user interface for whole-paper semantic search."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .build import add_paper, build_index
from .config import DEFAULT_MODEL, INDEX_DIR
from .index_store import IndexError
from .search import Searcher


class PaperScopeGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PaperScope - Semantic Paper Search")
        self.root.geometry("1100x750")

        self.searcher: Searcher | None = None
        self.model_name = DEFAULT_MODEL

        self._build_ui()
        self._load_index_async()

    # ------------------------------------------------------------------
    # User interface layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Add Paper", command=self._open_add_dialog)
        file_menu.add_command(label="Rebuild Index", command=self._rebuild_index)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: paper list
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="Papers in Collection").pack(anchor=tk.W, pady=(0, 5))
        self.paper_list = tk.Listbox(left)
        self.paper_list.pack(fill=tk.BOTH, expand=True)

        # Right panel: question and results
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        q_frame = ttk.LabelFrame(right, text="Ask a Question")
        q_frame.pack(fill=tk.X, pady=(0, 10))

        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(q_frame, textvariable=self.question_var, font=("Segoe UI", 11))
        self.question_entry.pack(fill=tk.X, padx=10, pady=10)
        self.question_entry.bind("<Return>", lambda _e: self._ask_question())

        self.ask_button = ttk.Button(q_frame, text="Ask", command=self._ask_question)
        self.ask_button.pack(padx=10, pady=(0, 10))

        close_frame = ttk.LabelFrame(right, text="Closest Match")
        close_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.close_text = tk.Text(close_frame, height=9, wrap=tk.WORD, font=("Segoe UI", 10))
        self.close_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.close_text.config(state=tk.DISABLED)

        scores_frame = ttk.LabelFrame(right, text="All Paper Scores")
        scores_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.scores_text = tk.Text(scores_frame, height=10, wrap=tk.WORD, font=("Consolas", 9))
        self.scores_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.scores_text.config(state=tk.DISABLED)

        self.status_var = tk.StringVar(value="Loading index...")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def _load_index_async(self) -> None:
        def work() -> None:
            try:
                self.searcher = Searcher()
                self.root.after(0, self._index_loaded)
            except IndexError:
                self.root.after(0, lambda: self.status_var.set("No index yet. Add a paper to begin."))
            except Exception as exc:  # noqa: BLE001 - surface startup errors to user.
                self.root.after(0, lambda e=exc: self.status_var.set(f"Index error: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _index_loaded(self) -> None:
        count = self.searcher.index.manifest.get("paper_count", 0) if self.searcher else 0
        self.status_var.set(f"Ready - {count} papers")
        self._refresh_list()
        self.ask_button.config(state=tk.NORMAL)

    def _refresh_list(self) -> None:
        self.paper_list.delete(0, tk.END)
        if self.searcher:
            for paper in self.searcher.index.papers:
                self.paper_list.insert(tk.END, f"{paper['key']}  -  {paper['title']}")

    # ------------------------------------------------------------------
    # Add paper
    # ------------------------------------------------------------------
    def _open_add_dialog(self) -> None:
        pdf_path = filedialog.askopenfilename(
            title="Select a PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not pdf_path:
            return

        name = filedialog.askstring("Paper Name", "Enter a name for this paper:")
        if not name:
            return

        key = Path(pdf_path).stem.replace(" ", "_").replace("-", "_")

        self.status_var.set("Adding paper... this may take a minute.")
        self.ask_button.config(state=tk.DISABLED)

        def work() -> None:
            try:
                manifest = add_paper(
                    key=key,
                    title=name,
                    source_path=Path(pdf_path),
                    model_name=self.model_name,
                    index_dir=INDEX_DIR,
                )
                self.root.after(0, lambda: self._paper_added(manifest))
            except Exception as exc:  # noqa: BLE001 - surface add errors to user.
                self.root.after(0, lambda e=exc: self._fail(f"Failed to add paper: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _paper_added(self, manifest: dict) -> None:
        self.status_var.set(f"Paper added. Collection now has {manifest['paper_count']} papers.")
        self.ask_button.config(state=tk.NORMAL)
        self._refresh_list()
        messagebox.showinfo(
            "Paper Added",
            f"Collection now has {manifest['paper_count']} papers.",
        )

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        if not self.searcher:
            messagebox.showwarning("Warning", "No index to rebuild.")
            return

        self.status_var.set("Rebuilding index...")
        self.ask_button.config(state=tk.DISABLED)

        def work() -> None:
            try:
                build_index(model_name=self.model_name, index_dir=INDEX_DIR)
                self.root.after(0, self._rebuilt)
            except Exception as exc:  # noqa: BLE001 - surface rebuild errors.
                self.root.after(0, lambda e=exc: self._fail(f"Rebuild failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _rebuilt(self) -> None:
        try:
            self.searcher = Searcher()
            count = self.searcher.index.manifest.get("paper_count", 0)
            self.status_var.set(f"Index rebuilt - {count} papers")
            self.ask_button.config(state=tk.NORMAL)
            self._refresh_list()
        except Exception as exc:  # noqa: BLE001 - surface reload errors.
            self._fail(f"Failed to reload index: {exc}")

    # ------------------------------------------------------------------
    # Ask question
    # ------------------------------------------------------------------
    def _ask_question(self) -> None:
        if not self.searcher:
            messagebox.showwarning("Warning", "No index loaded. Add papers first.")
            return

        question = self.question_var.get().strip()
        if not question:
            messagebox.showwarning("Warning", "Please enter a question.")
            return

        self.status_var.set("Searching...")
        self.ask_button.config(state=tk.DISABLED)
        self._set_text(self.close_text, "Searching...")
        self._set_text(self.scores_text, "Searching...")

        def work() -> None:
            try:
                hits = self.searcher.search_all(question)
                self.root.after(0, lambda: self._search_done(hits))
            except Exception as exc:  # noqa: BLE001 - surface search errors.
                self.root.after(0, lambda e=exc: self._fail(f"Search failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _search_done(self, hits) -> None:
        count = len(hits)
        self.status_var.set(f"Search complete - {count} papers ranked.")
        self.ask_button.config(state=tk.NORMAL)

        top = hits[0]
        paper = top.paper

        lines = [
            f"Title: {paper['title']}",
            f"Key: {paper['key']}",
            f"Similarity: {top.score:.4f}",
        ]
        if paper.get("year"):
            lines.append(f"Year: {paper['year']}")
        if paper.get("arxiv_id"):
            lines.append(f"arXiv: {paper['arxiv_id']}")
        if paper.get("truncated"):
            lines.append(
                f"Truncated: yes (rate {paper.get('truncation_rate', 0):.4f}, "
                f"tokens {paper.get('token_estimate', 0)})"
            )
        lines.append("")
        lines.append("Text preview:")

        text_path = paper.get("text_path", "")
        if text_path:
            try:
                full = Path(text_path).read_text(encoding="utf-8")
                preview = full[:1500]
                lines.append(preview)
                if len(full) > 1500:
                    lines.append("\n... (preview truncated)")
            except OSError:
                lines.append("(Could not load text preview)")
        else:
            lines.append("(No text available)")

        self._set_text(self.close_text, "\n".join(lines))

        # All scores
        score_lines = [f"{'Rank':<6} {'Score':<8} {'Key':<18} {'Title'}"]
        score_lines.append("-" * 90)
        for rank, hit in enumerate(hits, start=1):
            p = hit.paper
            score_lines.append(f"{rank:<6} {hit.score:.4f}    {p['key']:<18} {p['title']}")
        self._set_text(self.scores_text, "\n".join(score_lines))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.config(state=tk.DISABLED)

    def _fail(self, message: str) -> None:
        self.status_var.set(message)
        self.ask_button.config(state=tk.NORMAL)
        messagebox.showerror("Error", message)


def launch_gui() -> None:
    root = tk.Tk()
    PaperScopeGUI(root)
    root.mainloop()
