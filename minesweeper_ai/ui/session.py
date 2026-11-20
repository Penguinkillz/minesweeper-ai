"""One game the window can start, reset, and step.

Nothing here chooses a cell. An agent step calls ``agent.act`` on the
visible board and then asks the environment to reveal or flag. That is the
same ``act`` the Phase 1 harness calls. Manual play calls ``Game.reveal``
and ``Game.flag`` directly. The first reveal still places the mines and
excludes the clicked cell.

This module does not import tkinter. The tests exercise it without a display.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from minesweeper_ai.agents.base import Action, ActionKind, Reasoning
from minesweeper_ai.agents.clp import CLPAgent, UnsatisfiablePosition
from minesweeper_ai.agents.decision_tree import DecisionTreeAgent
from minesweeper_ai.agents.mcmc import FrontierSearchTruncated, MCMCAgent
from minesweeper_ai.env.board import (
    BEGINNER_COLS,
    BEGINNER_MINES,
    BEGINNER_ROWS,
    CellState,
)
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.env.view import VisibleState

MANUAL = "manual"
PLAYER_CLP = "CLP"
PLAYER_MCMC = "MCMC"
PLAYER_TREE = "decision_tree"
PLAYERS: tuple[str, ...] = (MANUAL, PLAYER_CLP, PLAYER_MCMC, PLAYER_TREE)

# Architecture section 1. The environment already accepts these arguments.
# The harness refuses them. The window offers them for play only.
INTERMEDIATE_ROWS = 16
INTERMEDIATE_COLS = 16
INTERMEDIATE_MINES = 40

DIFFICULTIES: dict[str, tuple[int, int, int]] = {
    "beginner": (BEGINNER_ROWS, BEGINNER_COLS, BEGINNER_MINES),
    "intermediate": (INTERMEDIATE_ROWS, INTERMEDIATE_COLS, INTERMEDIATE_MINES),
}

PLAYER_LABELS: dict[str, str] = {
    MANUAL: "Manual",
    PLAYER_CLP: "CLP",
    PLAYER_MCMC: "MCMC",
    PLAYER_TREE: "Decision tree",
}

AgentBuilder = Callable[[str], object]

# Exceptions the harness treats as "do not invent a click".
STEP_FAILURES = (UnsatisfiablePosition, FrontierSearchTruncated, ValueError)


def phase1_agent(name: str) -> object:
    """Build the agent the harness builds for this name.

    CLP and MCMC have no node cap. MCMC uses sampler seed 0. The decision
    tree is loaded from the saved Stage 4 file and is not fit again.
    """
    if name == PLAYER_CLP:
        return CLPAgent()
    if name == PLAYER_MCMC:
        return MCMCAgent(sampler_seed=0)
    if name == PLAYER_TREE:
        return DecisionTreeAgent.from_saved()
    raise ValueError(f"unknown agent {name}")


def _difficulty_name(rows: int, cols: int, mines: int) -> str:
    """Name of a board size the window can start."""
    for name, spec in DIFFICULTIES.items():
        if spec == (rows, cols, mines):
            return name
    raise ValueError(f"no difficulty matches {rows}x{cols}, {mines} mines")


def difficulty_label(name: str) -> str:
    """Button text for a size the environment can construct."""
    rows, cols, mines = DIFFICULTIES[name]
    if name == "intermediate":
        return f"Intermediate ({rows}×{cols}, {mines} mines, play only)"
    return f"Beginner ({rows}×{cols}, {mines} mines)"


@dataclass(frozen=True)
class MoveView:
    """What the window prints after one click or one agent decision.

    ``status`` is one of the five agent statuses, or ``None`` for a manual
    click. A manual click is not given a status the agent did not return.
    """

    source: str
    action: str
    cell: tuple[int, int]
    status: str | None
    uncertain: bool | None
    explanation: str
    agent_name: str | None
    hit_mine: bool
    applied_cell: tuple[int, int]


@dataclass(frozen=True)
class ShownCell:
    """One cell drawn from the visible grid.

    ``is_losing_mine`` is the cell whose reveal ended the game. The hidden
    mine set is not a field here.
    """

    row: int
    column: int
    state: str
    clue: int | None
    intended: bool
    is_losing_mine: bool


@dataclass(frozen=True)
class BoardSnapshot:
    """The texts and cells the window redraws. Built without tkinter."""

    started: bool
    status: str
    player: str
    difficulty: str
    seed: int
    rows: int
    cols: int
    mine_count: int
    mines_unaccounted: int | None
    halted: bool
    note: str
    last: MoveView | None
    cells: tuple[ShownCell, ...]


def shown_cells(
    state: VisibleState,
    intended: tuple[int, int] | None = None,
    losing_cell: tuple[int, int] | None = None,
) -> tuple[ShownCell, ...]:
    """Flags, clues, and hidden cells from a visible state.

    The losing cell is the reveal that already lost the game. Other mines
    stay hidden, because the environment does not open them.
    """
    cells: list[ShownCell] = []
    for row in range(state.height):
        for column in range(state.width):
            visible = state.cells[row][column]
            losing = losing_cell == (row, column)
            cells.append(
                ShownCell(
                    row=row,
                    column=column,
                    state=visible.state.value,
                    clue=None if losing else visible.number,
                    intended=intended == (row, column),
                    is_losing_mine=losing,
                )
            )
    return tuple(cells)


def cell_label(cell: ShownCell) -> str:
    """The character drawn on one square."""
    if cell.is_losing_mine:
        return "*"
    if cell.state == CellState.FLAGGED.value:
        return "F"
    if cell.state == CellState.REVEALED.value and cell.clue is not None:
        return str(cell.clue)
    return ""


class PlaySession:
    """The board, the selected player, and the last decision."""

    def __init__(self, agent_builder: AgentBuilder = phase1_agent) -> None:
        self.agent_builder = agent_builder
        self.difficulty = "beginner"
        self.player = MANUAL
        self.seed = 0
        self.game: Game | None = None
        self.agent: object | None = None
        self.last: MoveView | None = None
        self.note = ""
        self.halted = False
        self.losing_cell: tuple[int, int] | None = None

    def configure(
        self,
        *,
        difficulty: str | None = None,
        player: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Store the next start settings. An in-progress board is left as it is.

        ``player`` is applied immediately when a game is already going, so
        the next step uses that agent on the current visible board.
        """
        if difficulty is not None:
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"unknown difficulty {difficulty}")
            self.difficulty = difficulty
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            self.seed = seed
        if player is not None:
            self.select_player(player)

    def select_player(self, name: str) -> None:
        """Choose manual play or one of the three harness agents.

        Selecting the player that is already active keeps the current agent.
        A different player replaces it and leaves the board as it is.
        """
        if name not in PLAYERS:
            raise ValueError(f"unknown player {name}")
        changed = name != self.player or self.agent is None
        self.player = name
        if self.game is None:
            self.agent = None
            return
        if not changed:
            return
        self.agent = None if name == MANUAL else self.agent_builder(name)

    def start(self) -> None:
        """Begin a new game with the stored size, seed, and player."""
        rows, cols, mines = DIFFICULTIES[self.difficulty]
        self.game = Game(rows=rows, cols=cols, mine_count=mines, seed=self.seed)
        self.agent = None if self.player == MANUAL else self.agent_builder(self.player)
        self.last = None
        self.note = ""
        self.halted = False
        self.losing_cell = None

    def reset(self) -> None:
        """Clear the board and keep the seed, the size, and the player.

        A new game is built so a difficulty or seed edit takes effect. The
        environment places mines again on the next first reveal.
        """
        self.start()

    def step(self) -> MoveView:
        """Ask the selected agent for one move and apply it.

        The argument to ``act`` is ``Game.visible_state()``. The harness
        calls that same method. A failure that the harness treats as an
        abort is stored and no other cell is clicked.
        """
        game = self._require_started()
        if self.player == MANUAL or self.agent is None:
            self.note = "Select CLP, MCMC, or the decision tree before stepping."
            raise RuntimeError(self.note)
        if self.halted or game.status is not GameStatus.ONGOING:
            self.note = "The game is not waiting for a move."
            raise RuntimeError(self.note)
        state = game.visible_state()
        try:
            action, reasoning = self.agent.act(state)  # type: ignore[attr-defined]
        except STEP_FAILURES as exc:
            self.halted = True
            self.note = f"{type(exc).__name__}: {exc}"
            raise
        if not isinstance(action, Action) or not isinstance(reasoning, Reasoning):
            self.halted = True
            self.note = "The agent did not return an action and a reasoning record."
            raise RuntimeError(self.note)
        return self._apply_agent(action, reasoning)

    def reveal(self, row: int, column: int) -> None:
        """Manual reveal. The environment handles the first click and flood-fill."""
        game = self._require_manual_cell(row, column)
        if game.board.states[row][column] is not CellState.HIDDEN:
            self.note = "That cell is not hidden."
            return
        first = not game.board.mines_placed
        game.reveal(row, column)
        self._record_manual("reveal", (row, column), first)

    def flag(self, row: int, column: int) -> None:
        """Manual flag toggle, using the environment rule."""
        game = self._require_manual_cell(row, column)
        state = game.board.states[row][column]
        if state is CellState.REVEALED:
            self.note = "A revealed cell cannot be flagged."
            return
        game.flag(row, column)
        self.last = MoveView(
            source="manual",
            action="flag",
            cell=(row, column),
            status=None,
            uncertain=None,
            explanation="Manual flag. The environment toggled the flag. No agent status was returned.",
            agent_name=None,
            hit_mine=False,
            applied_cell=(row, column),
        )
        self.note = ""

    def snapshot(self) -> BoardSnapshot:
        """The current board as the window should draw it."""
        if self.game is None:
            rows, cols, mines = DIFFICULTIES[self.difficulty]
            return BoardSnapshot(
                started=False,
                status="not started",
                player=self.player,
                difficulty=self.difficulty,
                seed=self.seed,
                rows=rows,
                cols=cols,
                mine_count=mines,
                mines_unaccounted=None,
                halted=self.halted,
                note=self.note,
                last=self.last,
                cells=(),
            )
        state = self.game.visible_state()
        intended = self.last.cell if self.last is not None else None
        return BoardSnapshot(
            started=True,
            status="stopped" if self.halted and self.game.status is GameStatus.ONGOING else self.game.status.value,
            player=self.player,
            difficulty=_difficulty_name(self.game.board.rows, self.game.board.cols, self.game.board.mine_count),
            seed=self.game.board.seed,
            rows=self.game.board.rows,
            cols=self.game.board.cols,
            mine_count=self.game.board.mine_count,
            mines_unaccounted=state.mines_unaccounted,
            halted=self.halted,
            note=self.note,
            last=self.last,
            cells=shown_cells(state, intended, self.losing_cell),
        )

    def _apply_agent(self, action: Action, reasoning: Reasoning) -> MoveView:
        game = self.game
        if game is None:
            raise RuntimeError("Start a game first.")
        first = not game.board.mines_placed
        hit_mine = False
        applied = False
        try:
            if action.kind is ActionKind.REVEAL:
                game.reveal(action.row, action.column)
                applied = True
                hit_mine = game.status is GameStatus.LOST
                if hit_mine:
                    self.losing_cell = (action.row, action.column)
            elif action.kind is ActionKind.FLAG:
                game.flag(action.row, action.column)
                applied = True
            else:
                self.halted = True
                self.note = f"unknown action {action.kind}"
                raise RuntimeError(self.note)
        except ValueError as exc:
            self.halted = True
            self.note = f"ValueError: {exc}"
        self.last = MoveView(
            source="agent",
            action=action.kind.value,
            cell=reasoning.cell,
            status=reasoning.status.value,
            uncertain=reasoning.uncertain,
            explanation=reasoning.explanation,
            agent_name=reasoning.agent_name,
            hit_mine=hit_mine,
            applied_cell=(action.row, action.column),
        )
        if self.halted and not applied:
            return self.last
        if first and game.board.mines_placed and not hit_mine:
            self.note = "First reveal. The environment placed the mines and excluded this cell."
        else:
            self.note = _outcome_note(game, first=first)
        return self.last

    def _record_manual(self, action: str, cell: tuple[int, int], first: bool) -> None:
        game = self.game
        if game is None:
            raise RuntimeError("Start a game first.")
        hit_mine = game.status is GameStatus.LOST
        if hit_mine:
            self.losing_cell = cell
        self.last = MoveView(
            source="manual",
            action=action,
            cell=cell,
            status=None,
            uncertain=None,
            explanation="Manual click. The environment applied it. No agent status was returned.",
            agent_name=None,
            hit_mine=hit_mine,
            applied_cell=cell,
        )
        if first and game.board.mines_placed and not hit_mine:
            self.note = "First reveal. The environment placed the mines and excluded this cell."
        else:
            self.note = _outcome_note(game, first=first)

    def _require_started(self) -> Game:
        if self.game is None:
            self.note = "Start a game first."
            raise RuntimeError(self.note)
        return self.game

    def _require_manual_cell(self, row: int, column: int) -> Game:
        game = self._require_started()
        if self.player != MANUAL:
            self.note = "This game is stepped by the agent. Switch to manual to click."
            raise RuntimeError(self.note)
        if self.halted or game.status is not GameStatus.ONGOING:
            self.note = "The game is not waiting for a move."
            raise RuntimeError(self.note)
        game.board.require_in_bounds(row, column)
        return game


def window_commands(session: PlaySession) -> dict[str, Callable[..., object]]:
    """The callables the window binds to Start, Reset, and Step.

    Step is ``PlaySession.step``. That method calls ``agent.act`` on the
    visible state, which is the call the harness makes inside ``play_game``.
    """
    return {
        "start": session.start,
        "reset": session.reset,
        "step": session.step,
        "reveal": session.reveal,
        "flag": session.flag,
    }


def run_step(commands: dict[str, Callable[..., object]]) -> object:
    """The Step button. It calls the session step in ``commands``."""
    return commands["step"]()


def describe_status(snapshot: BoardSnapshot) -> str:
    """The one-line status the window shows above the board."""
    player = PLAYER_LABELS.get(snapshot.player, snapshot.player)
    text = (
        f"{difficulty_label(snapshot.difficulty)} · seed {snapshot.seed} · "
        f"{player} · {snapshot.status}"
    )
    if snapshot.mines_unaccounted is not None:
        text += f" · mines unaccounted {snapshot.mines_unaccounted}"
    return text


def describe_decision(snapshot: BoardSnapshot) -> str:
    """The reasoning panel. Status text is the agent's, or none for a manual click."""
    lines: list[str] = []
    last = snapshot.last
    if last is None:
        lines.append("No move yet.")
    else:
        if last.agent_name:
            lines.append(f"Agent: {last.agent_name}")
        lines.append(f"Intended cell: ({last.cell[0]}, {last.cell[1]})")
        lines.append(f"Action: {last.action}")
        if last.applied_cell != last.cell:
            lines.append(f"Applied cell: ({last.applied_cell[0]}, {last.applied_cell[1]})")
        if last.status is None:
            lines.append("Status: none")
            lines.append("This click was manual, so there is no agent status.")
        else:
            lines.append(f"Status: {last.status}")
            lines.append(f"Uncertain: {'yes' if last.uncertain else 'no'}")
        lines.append("")
        lines.append(last.explanation)
    if snapshot.note:
        lines.append("")
        lines.append(snapshot.note)
    return "\n".join(lines)


def _outcome_note(game: Game, first: bool) -> str:
    if game.status is GameStatus.LOST:
        return "A mine was revealed. The game is lost."
    if game.status is GameStatus.WON:
        return "Every safe cell is open. The game is won."
    if first and game.board.mines_placed:
        return "First reveal. The environment placed the mines and excluded this cell."
    return ""
