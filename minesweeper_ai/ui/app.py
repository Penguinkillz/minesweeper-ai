"""Minesweeper window.

Launch it from the repository root:

    python3 -m minesweeper_ai.ui.app

tkinter and a display are required. The play rules live in ``session.py``.
The comparison tab reads the saved beginner summary. This file does not
choose a cell and does not count a win.
"""

from __future__ import annotations

from minesweeper_ai.ui.dashboard import load_phase1_dashboard
from minesweeper_ai.ui.session import (
    DIFFICULTIES,
    MANUAL,
    PLAYER_LABELS,
    PLAYERS,
    STEP_FAILURES,
    PlaySession,
    cell_label,
    describe_decision,
    describe_status,
    difficulty_label,
    run_step,
    window_commands,
)

MISSING_TK_MESSAGE = (
    "A display and the Python tkinter module are required to open the window. "
    "Install tkinter for this Python, then from the repository root run: "
    "python3 -m minesweeper_ai.ui.app"
)

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]


_CLUE_FG = {
    0: "#404040",
    1: "#0000ff",
    2: "#008000",
    3: "#ff0000",
    4: "#000080",
    5: "#800000",
    6: "#008080",
    7: "#000000",
    8: "#808080",
}
_HIDDEN_BG = "#8fb4d4"
_REVEALED_BG = "#e6e6e6"
_FLAG_BG = "#f3c1c1"
_MINE_BG = "#c94040"
_INTENDED_BG = "#f0c14a"
_MINE_FG = "#ffffff"
_FLAG_FG = "#8b0000"


class MinesweeperWindow:
    """Start, reset, step, and the comparison tab.

    The Step button calls ``run_step`` on ``window_commands``. That calls
    ``PlaySession.step``, which calls ``agent.act`` on the visible board.
    """

    def __init__(self, root: object, session: PlaySession | None = None) -> None:
        if tk is None or ttk is None:
            raise RuntimeError(MISSING_TK_MESSAGE)
        self.root = root
        self.session = session or PlaySession()
        self.commands = window_commands(self.session)
        self._buttons: list[list[object]] = []
        self._shape: tuple[int, int] | None = None

        root.title("Minesweeper")  # type: ignore[attr-defined]
        root.minsize(420, 520)  # type: ignore[attr-defined]
        root.geometry("920x740")  # type: ignore[attr-defined]

        self.difficulty_var = tk.StringVar(value=self.session.difficulty)
        self.player_var = tk.StringVar(value=self.session.player)
        self.seed_var = tk.StringVar(value=str(self.session.seed))
        self.status_var = tk.StringVar(value="")

        controls = ttk.Frame(root, padding=(8, 8, 8, 0))
        controls.pack(fill="x")
        for name in DIFFICULTIES:
            ttk.Radiobutton(
                controls,
                text=difficulty_label(name),
                value=name,
                variable=self.difficulty_var,
            ).pack(side="left", padx=(0, 12))

        players = ttk.Frame(root, padding=(8, 4, 8, 0))
        players.pack(fill="x")
        ttk.Label(players, text="Player").pack(side="left")
        for name in PLAYERS:
            ttk.Radiobutton(
                players,
                text=PLAYER_LABELS[name],
                value=name,
                variable=self.player_var,
                command=self._on_player,
            ).pack(side="left", padx=(8, 0))
        ttk.Label(players, text="Seed").pack(side="left", padx=(16, 4))
        ttk.Entry(players, textvariable=self.seed_var, width=8).pack(side="left")
        ttk.Button(players, text="Start", command=self._on_start).pack(side="left", padx=(12, 0))
        ttk.Button(players, text="Reset", command=self._on_reset).pack(side="left", padx=(4, 0))
        self.step_button = ttk.Button(players, text="Step", command=self._on_step)
        self.step_button.pack(side="left", padx=(4, 0))

        hint = ttk.Label(
            root,
            padding=(8, 4, 8, 0),
            text=(
                "Start and Reset apply the difficulty and the seed. "
                "The first Step is the environment's safe opening reveal, for every agent. "
                "Later Steps ask the selected agent for one move. "
                "Hybrid takes a CLP proof, otherwise one MCMC regime. "
                "Manual play: click to reveal, right-click to flag."
            ),
        )
        hint.pack(fill="x")
        ttk.Label(root, textvariable=self.status_var, padding=(8, 4, 8, 4)).pack(fill="x")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        play = ttk.Frame(notebook)
        comparison = ttk.Frame(notebook)
        notebook.add(play, text="Play")
        notebook.add(comparison, text="Comparison")

        board_wrap = ttk.Frame(play)
        board_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(board_wrap, highlightthickness=0, height=360)
        xscroll = ttk.Scrollbar(board_wrap, orient="horizontal", command=self.canvas.xview)
        yscroll = ttk.Scrollbar(board_wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.board_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.board_frame, anchor="nw")
        self.board_frame.bind("<Configure>", self._fit_board)

        reasoning_wrap = ttk.Frame(play, padding=(0, 4, 0, 0))
        reasoning_wrap.pack(fill="both", expand=True)
        self.reasoning = tk.Text(reasoning_wrap, height=9, wrap="word", font="TkFixedFont")
        reasoning_scroll = ttk.Scrollbar(reasoning_wrap, command=self.reasoning.yview)
        self.reasoning.configure(yscrollcommand=reasoning_scroll.set)
        reasoning_scroll.pack(side="right", fill="y")
        self.reasoning.pack(side="left", fill="both", expand=True)

        self.comparison = tk.Text(comparison, wrap="word", font="TkFixedFont")
        comparison_scroll = ttk.Scrollbar(comparison, command=self.comparison.yview)
        self.comparison.configure(yscrollcommand=comparison_scroll.set)
        comparison_scroll.pack(side="right", fill="y")
        self.comparison.pack(side="left", fill="both", expand=True)
        self._load_comparison()
        self.refresh()

    def refresh(self) -> None:
        """Redraw the board, the reasoning, and the step button from the session."""
        snapshot = self.session.snapshot()
        self.status_var.set(describe_status(snapshot))
        self._draw_board(snapshot)
        self._set_text(self.reasoning, describe_decision(snapshot))
        stepping = (
            snapshot.started
            and snapshot.status == "ongoing"
            and not snapshot.halted
            and self.player_var.get() != MANUAL
        )
        self.step_button.configure(state="normal" if stepping else "disabled")

    def _on_start(self) -> None:
        self._apply_settings(self.commands["start"])

    def _on_reset(self) -> None:
        self._apply_settings(self.commands["reset"])

    def _on_step(self) -> None:
        try:
            self.session.select_player(self.player_var.get())
        except ValueError as exc:
            self.session.note = str(exc)
            self.refresh()
            return
        self.status_var.set("Deciding...")
        self.root.update_idletasks()  # type: ignore[attr-defined]
        try:
            run_step(self.commands)
        except STEP_FAILURES:
            pass
        except RuntimeError:
            pass
        self.refresh()

    def _on_player(self) -> None:
        try:
            self.session.select_player(self.player_var.get())
        except ValueError as exc:
            self.session.note = str(exc)
        self.refresh()

    def _on_reveal(self, row: int, column: int) -> None:
        self._manual(self.commands["reveal"], row, column)

    def _on_flag(self, row: int, column: int) -> str:
        self._manual(self.commands["flag"], row, column)
        return "break"

    def _apply_settings(self, command: object) -> None:
        try:
            self.session.configure(
                difficulty=self.difficulty_var.get(),
                player=self.player_var.get(),
                seed=_parse_seed(self.seed_var.get()),
            )
        except ValueError as exc:
            self.session.note = str(exc)
            self.refresh()
            return
        command()  # type: ignore[operator]
        self.refresh()

    def _manual(self, command: object, row: int, column: int) -> None:
        try:
            self.session.select_player(self.player_var.get())
            command(row, column)  # type: ignore[operator]
        except (RuntimeError, ValueError):
            pass
        self.refresh()

    def _draw_board(self, snapshot: object) -> None:
        if not snapshot.started:  # type: ignore[attr-defined]
            self._clear_board()
            return
        rows = snapshot.rows  # type: ignore[attr-defined]
        cols = snapshot.cols  # type: ignore[attr-defined]
        if self._shape != (rows, cols):
            self._build_buttons(rows, cols)
        for cell in snapshot.cells:  # type: ignore[attr-defined]
            button = self._buttons[cell.row][cell.column]
            bg, fg = _cell_colours(cell)
            button.configure(  # type: ignore[attr-defined]
                text=cell_label(cell),
                bg=bg,
                fg=fg,
                activebackground=bg,
                relief="sunken" if cell.state == "revealed" or cell.is_losing_mine else "raised",
                highlightthickness=2 if cell.intended else 0,
                highlightbackground="#8a5a00" if cell.intended else bg,
            )

    def _build_buttons(self, rows: int, cols: int) -> None:
        self._clear_board()
        built: list[list[object]] = []
        for row in range(rows):
            row_buttons: list[object] = []
            for column in range(cols):
                button = tk.Button(
                    self.board_frame,
                    width=2,
                    font=("TkFixedFont", 12, "bold"),
                    command=lambda r=row, c=column: self._on_reveal(r, c),
                )
                button.bind("<Button-3>", lambda _event, r=row, c=column: self._on_flag(r, c))
                button.bind("<Button-2>", lambda _event, r=row, c=column: self._on_flag(r, c))
                button.grid(row=row, column=column, padx=1, pady=1)
                row_buttons.append(button)
            built.append(row_buttons)
        self._buttons = built
        self._shape = (rows, cols)

    def _clear_board(self) -> None:
        for child in self.board_frame.winfo_children():
            child.destroy()
        self._buttons = []
        self._shape = None

    def _fit_board(self, _event: object) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _load_comparison(self) -> None:
        try:
            text = "\n".join(load_phase1_dashboard().lines())
        except (OSError, ValueError) as exc:
            text = "The comparison view did not load.\n" + str(exc)
        self._set_text(self.comparison, text)

    @staticmethod
    def _set_text(widget: object, text: str) -> None:
        widget.configure(state="normal")  # type: ignore[attr-defined]
        widget.delete("1.0", "end")  # type: ignore[attr-defined]
        widget.insert("1.0", text)  # type: ignore[attr-defined]
        widget.configure(state="disabled")  # type: ignore[attr-defined]


def _cell_colours(cell: object) -> tuple[str, str]:
    if cell.is_losing_mine:  # type: ignore[attr-defined]
        return _MINE_BG, _MINE_FG
    if cell.intended:  # type: ignore[attr-defined]
        return _INTENDED_BG, "#000000"
    if cell.state == "flagged":  # type: ignore[attr-defined]
        return _FLAG_BG, _FLAG_FG
    if cell.state == "revealed":  # type: ignore[attr-defined]
        clue = cell.clue  # type: ignore[attr-defined]
        return _REVEALED_BG, _CLUE_FG.get(clue, "#000000")
    return _HIDDEN_BG, "#000000"


def _parse_seed(text: str) -> int:
    stripped = text.strip()
    if not stripped or stripped[0] == "+" or not _is_integer_text(stripped):
        raise ValueError("Seed must be an integer.")
    return int(stripped)


def _is_integer_text(text: str) -> bool:
    if text[0] == "-":
        return text[1:].isdigit() and text != "-"
    return text.isdigit()


def main() -> None:
    """Open the window. Exits if tkinter is not installed."""
    if tk is None or ttk is None:
        raise SystemExit(MISSING_TK_MESSAGE)
    root = tk.Tk()
    MinesweeperWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
