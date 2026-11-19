"""Shared decision interface for every agent.

Section 3 of the architecture plan: visible state goes in, and one action
plus a reasoning record comes out. Later agents use the same types. This
module has no solver.
"""

import enum
from typing import Protocol

from minesweeper_ai.env.view import VisibleState


class ActionKind(enum.Enum):
    """A move is a reveal or a flag. The environment applies it."""

    REVEAL = "reveal"
    FLAG = "flag"


class DecisionStatus(enum.Enum):
    """How the agent classifies the move.

    ``proved_safe`` and ``proved_mine`` are finished proofs. The other three
    statuses are uncertain decisions. A guess, a sample frequency, or a tree
    prediction must not be reported as a proof.
    """

    PROVED_SAFE = "proved_safe"
    PROVED_MINE = "proved_mine"
    ESTIMATED = "estimated"
    MODEL_PREDICTION = "model_prediction"
    FALLBACK = "fallback"


class Action:
    """One cell and the operation to perform on it."""

    def __init__(self, kind: ActionKind, row: int, column: int) -> None:
        self.kind = kind
        self.row = row
        self.column = column

    @classmethod
    def reveal(cls, row: int, column: int) -> "Action":
        return cls(ActionKind.REVEAL, row, column)

    @classmethod
    def flag(cls, row: int, column: int) -> "Action":
        return cls(ActionKind.FLAG, row, column)

    def cell(self) -> tuple[int, int]:
        return (self.row, self.column)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.kind is other.kind and self.row == other.row and self.column == other.column

    def __hash__(self) -> int:
        return hash((self.kind, self.row, self.column))

    def __repr__(self) -> str:
        return f"Action({self.kind.value}, {self.row}, {self.column})"


class Reasoning:
    """The record the UI and the evaluator both read.

    A proved status is a finished proof, so ``uncertain`` is false.
    ``estimated``, ``model_prediction``, and ``fallback`` are not proofs,
    so ``uncertain`` is true. Section 4.4 says an uncertain CLP reveal
    "stays uncertain". The shared status list has no sixth value, so that
    reveal uses ``fallback``.
    """

    def __init__(
        self,
        agent_name: str,
        cell: tuple[int, int],
        status: DecisionStatus,
        explanation: str,
        uncertain: bool,
    ) -> None:
        proved = status in (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE)
        if proved and uncertain:
            raise ValueError("a proved status is a finished proof and cannot be uncertain")
        if not proved and not uncertain:
            raise ValueError("a status that is not a finished proof is uncertain")
        self.agent_name = agent_name
        self.cell = cell
        self.status = status
        self.explanation = explanation
        self.uncertain = uncertain

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reasoning):
            return NotImplemented
        return (
            self.agent_name == other.agent_name
            and self.cell == other.cell
            and self.status is other.status
            and self.explanation == other.explanation
            and self.uncertain == other.uncertain
        )

    def __repr__(self) -> str:
        return (
            f"Reasoning({self.agent_name}, {self.cell}, {self.status.value}, "
            f"uncertain={self.uncertain})"
        )


class Agent(Protocol):
    """``visible state → action + reasoning``. The state has no mine layer."""

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        """Choose one move from the visible grid alone."""
