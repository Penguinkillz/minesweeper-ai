"""Hybrid agent: a CLP proof is kept, and a guess uses one MCMC regime.

Regime A marginal 0 is a finished count, so it is a proof. A Regime B
frequency of 0 is a walk average, so it is not. The two are not averaged.
The harness factories stay the three original agents.
"""

import unittest
from fractions import Fraction

from minesweeper_ai.agents.base import Action, ActionKind, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import CLPAgent
from minesweeper_ai.agents.hybrid import AGENT_NAME, HybridAgent
from minesweeper_ai.agents.mcmc import REGIME_EXACT, REGIME_LOCAL, MCMCAgent
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.view import VisibleCell, VisibleState
from minesweeper_ai.evaluation.harness import (
    LOG_PATH,
    PHASE2_LOG_PATH,
    phase1_agent_factories,
)
from minesweeper_ai.ui.session import PLAYERS, PlaySession, phase1_agent

Cell = tuple[int, int]


def visible(
    rows: int,
    cols: int,
    mine_count: int,
    revealed: dict[Cell, int],
) -> VisibleState:
    """A visible grid. This helper has no mine layer."""
    grid = []
    for row in range(rows):
        line = []
        for column in range(cols):
            if (row, column) in revealed:
                line.append(VisibleCell(state=CellState.REVEALED, number=revealed[(row, column)]))
            else:
                line.append(VisibleCell(state=CellState.HIDDEN, number=None))
        grid.append(tuple(line))
    return VisibleState(
        width=cols,
        height=rows,
        mine_count=mine_count,
        mines_unaccounted=mine_count,
        cells=tuple(grid),
    )


def two_cell() -> VisibleState:
    """Clue 1 with hidden neighbours A and B. Neither cell is proved."""
    return visible(1, 3, 1, {(0, 1): 1})


def frontier_over_16() -> VisibleState:
    """Eighteen hidden cells, each touching a revealed 0. K = 0, so only the zero layout is legal."""
    revealed = {(row, column): 0 for row in range(6) for column in (0, 2, 4)}
    return visible(6, 6, 0, revealed)


class _Boom:
    """Fails the test if the hybrid asks MCMC after a CLP proof."""

    def analyse(self, state: VisibleState) -> object:
        raise AssertionError("MCMC was asked after CLP returned a proof")


class _NoProof:
    """A CLP stand-in that returns a guess, never a proof."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        return Action.reveal(*self.cell), Reasoning(
            agent_name="CLP",
            cell=self.cell,
            status=DecisionStatus.FALLBACK,
            explanation="CLP has no proof on this test double.",
            uncertain=True,
        )


class _Script:
    def __init__(self, regime: str, probabilities: dict[Cell, Fraction], frontier_size: int) -> None:
        self.regime = regime
        self.probabilities = probabilities
        self.frontier_size = frontier_size

    def analyse(self, state: VisibleState) -> "_Script":
        return self


class ProofTests(unittest.TestCase):
    def test_clp_proof_is_taken_unchanged(self) -> None:
        cases = (
            visible(1, 3, 0, {(0, 1): 0}),
            visible(1, 2, 1, {(0, 0): 1}),
        )
        for state in cases:
            clp_action, clp_reasoning = CLPAgent().act(state)
            self.assertIn(clp_reasoning.status, (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE))
            hybrid = HybridAgent(clp_agent=CLPAgent(), mcmc_agent=_Boom())
            action, reasoning = hybrid.act(state)
            self.assertEqual(action, clp_action)
            self.assertIs(reasoning.status, clp_reasoning.status)
            self.assertEqual(reasoning.cell, clp_reasoning.cell)
            self.assertEqual(reasoning.explanation, clp_reasoning.explanation)
            self.assertIs(reasoning.uncertain, False)
            self.assertEqual(reasoning.agent_name, AGENT_NAME)
            self.assertEqual(hybrid.last_source, "clp")
            self.assertIsNone(hybrid.last_mcmc)

    def test_exact_marginal_zero_is_a_proof(self) -> None:
        # The same clue-0 board. CLP would prove (0, 0) safe. This double
        # refuses, so the guess uses Regime A. Marginal 0 is still a proof.
        state = visible(1, 3, 0, {(0, 1): 0})
        hybrid = HybridAgent(clp_agent=_NoProof((0, 2)), mcmc_agent=MCMCAgent())
        action, reasoning = hybrid.act(state)
        self.assertEqual(hybrid.last_source, REGIME_EXACT)
        assert hybrid.last_mcmc is not None
        self.assertEqual(hybrid.last_mcmc.probabilities[(0, 0)], 0)  # type: ignore[attr-defined]
        self.assertEqual(hybrid.last_mcmc.probabilities[(0, 2)], 0)  # type: ignore[attr-defined]
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertEqual(action.cell(), (0, 0))
        self.assertIs(reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertIs(reasoning.uncertain, False)
        self.assertIn("exact marginal enumeration", reasoning.explanation)
        self.assertIn("Exact marginal 0 is proved safe", reasoning.explanation)
        self.assertNotIn("local block Gibbs", reasoning.explanation)

        mine = visible(1, 2, 1, {(0, 0): 1})
        flagged = HybridAgent(clp_agent=_NoProof((0, 0)), mcmc_agent=MCMCAgent()).act(mine)
        mine_action, mine_reasoning = flagged
        self.assertEqual(mine_action.cell(), (0, 1))
        self.assertIs(mine_action.kind, ActionKind.FLAG)
        self.assertIs(mine_reasoning.status, DecisionStatus.PROVED_MINE)
        self.assertIn("Exact marginal 1 is proved mine", mine_reasoning.explanation)

    def test_local_frequency_zero_is_not_a_proof(self) -> None:
        # |F| = 18, so the unchanged MCMC agent stays in Regime B even with
        # exact marginals enabled. K = 0, so every retained frequency is 0.
        # One retained sweep is enough to show that 0. The status rule does
        # not depend on the sweep count.
        state = frontier_over_16()
        hybrid = HybridAgent(
            clp_agent=_NoProof((0, 5)),
            mcmc_agent=MCMCAgent(sampler_seed=0, burn_in_sweeps=0, retained_sweeps=1),
        )
        action, reasoning = hybrid.act(state)
        self.assertEqual(hybrid.last_source, REGIME_LOCAL)
        assert hybrid.last_mcmc is not None
        self.assertGreater(hybrid.last_mcmc.frontier_size, 16)  # type: ignore[attr-defined]
        self.assertEqual(hybrid.last_mcmc.probabilities[(0, 1)], 0)  # type: ignore[attr-defined]
        self.assertEqual(action.cell(), (0, 1))
        self.assertNotEqual(action.cell(), (0, 5))
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertIs(reasoning.uncertain, True)
        self.assertNotIn(reasoning.status, (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE))
        self.assertIn("local block Gibbs only", reasoning.explanation)
        self.assertIn("sample frequency 0", reasoning.explanation)
        self.assertNotIn("exact marginal enumeration", reasoning.explanation)
        self.assertNotIn("proved_safe", reasoning.explanation)
        self.assertNotIn("proved_mine", reasoning.explanation)


class GuessRuleTests(unittest.TestCase):
    def test_a_real_clp_guess_uses_exact_marginals_and_not_the_other_regime(self) -> None:
        state = two_cell()
        _clp_action, clp_reasoning = CLPAgent().act(state)
        self.assertIs(clp_reasoning.status, DecisionStatus.FALLBACK)
        hybrid = HybridAgent()
        action, reasoning = hybrid.act(state)
        self.assertEqual(hybrid.last_source, REGIME_EXACT)
        self.assertEqual(action.cell(), (0, 0))
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertIs(reasoning.uncertain, True)
        assert hybrid.last_mcmc is not None
        self.assertEqual(hybrid.last_mcmc.probabilities[(0, 0)], Fraction(1, 2))  # type: ignore[attr-defined]
        self.assertIn("exact marginal enumeration", reasoning.explanation)
        self.assertNotIn("local block Gibbs", reasoning.explanation)

    def test_local_guess_prefers_the_frontier_cell_farthest_from_one(self) -> None:
        # Interior (0, 5) has frequency 0. That is farther from 1 than any
        # frontier cell, and it is still not the guess. Frontier (0, 2) at
        # 2/5 beats frontier cells closer to 1. The 0 is not a proof.
        state = visible(1, 7, 2, {(0, 1): 1, (0, 3): 1})
        probabilities = {
            (0, 0): Fraction(9, 10),
            (0, 2): Fraction(2, 5),
            (0, 4): Fraction(7, 10),
            (0, 5): Fraction(0, 1),
            (0, 6): Fraction(2, 5),
        }
        hybrid = HybridAgent(
            clp_agent=_NoProof((0, 0)),
            mcmc_agent=_Script(REGIME_LOCAL, probabilities, frontier_size=3),
        )
        action, reasoning = hybrid.act(state)
        self.assertEqual(action.cell(), (0, 2))
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertIs(reasoning.uncertain, True)
        self.assertNotIn("proved", reasoning.explanation)

    def test_local_regime_without_a_frontier_cell_says_fallback(self) -> None:
        state = visible(1, 3, 1, {})
        probabilities = {(0, 0): Fraction(0, 1), (0, 1): Fraction(0, 1), (0, 2): Fraction(1, 1)}
        hybrid = HybridAgent(
            clp_agent=_NoProof((0, 2)),
            mcmc_agent=_Script(REGIME_LOCAL, probabilities, frontier_size=0),
        )
        action, reasoning = hybrid.act(state)
        self.assertEqual(action.cell(), (0, 0))
        self.assertNotEqual(action.cell(), (0, 2))
        self.assertIs(reasoning.status, DecisionStatus.FALLBACK)
        self.assertIs(reasoning.uncertain, True)
        self.assertIn("Status fallback", reasoning.explanation)
        self.assertNotIn("proved_safe", reasoning.explanation)

    def test_the_same_zero_is_a_proof_only_in_regime_a(self) -> None:
        # One probability, two regimes. The hybrid does not blend them into
        # a third number, and it does not copy the proof label across.
        state = visible(1, 3, 0, {(0, 1): 0})
        probabilities = {(0, 0): Fraction(0, 1), (0, 2): Fraction(0, 1)}
        exact = HybridAgent(
            clp_agent=_NoProof((0, 2)),
            mcmc_agent=_Script(REGIME_EXACT, probabilities, frontier_size=2),
        ).act(state)
        local = HybridAgent(
            clp_agent=_NoProof((0, 2)),
            mcmc_agent=_Script(REGIME_LOCAL, probabilities, frontier_size=2),
        ).act(state)
        self.assertIs(exact[1].status, DecisionStatus.PROVED_SAFE)
        self.assertIs(local[1].status, DecisionStatus.ESTIMATED)
        self.assertEqual(exact[0].cell(), local[0].cell())
        self.assertIn("exact marginal enumeration", exact[1].explanation)
        self.assertNotIn("local block Gibbs", exact[1].explanation)
        self.assertIn("local block Gibbs only", local[1].explanation)
        self.assertNotIn("exact marginal enumeration", local[1].explanation)


class WindowTests(unittest.TestCase):
    def test_the_window_lists_hybrid_beside_the_three_agents(self) -> None:
        self.assertEqual(PLAYERS, ("manual", "CLP", "MCMC", "decision_tree", "hybrid"))
        self.assertEqual([name for name, _factory in phase1_agent_factories()], ["CLP", "MCMC", "decision_tree"])
        agent = phase1_agent("hybrid")
        self.assertIsInstance(agent, HybridAgent)
        self.assertIsNone(agent.clp_agent.node_limit)  # type: ignore[attr-defined]
        self.assertEqual(agent.mcmc_agent.sampler_seed, 0)  # type: ignore[attr-defined]
        self.assertIsNone(agent.mcmc_agent.node_limit)  # type: ignore[attr-defined]
        self.assertTrue(agent.mcmc_agent.exact_marginals)  # type: ignore[attr-defined]

    def test_one_smoke_step_selects_hybrid_and_leaves_the_logs_alone(self) -> None:
        phase1 = LOG_PATH.read_bytes()
        phase2 = PHASE2_LOG_PATH.read_bytes()
        session = PlaySession()
        session.configure(player="hybrid", seed=0, difficulty="beginner")
        session.start()
        self.assertIsInstance(session.agent, HybridAgent)
        record = session.step()
        self.assertEqual(record.agent_name, AGENT_NAME)
        self.assertEqual(record.cell, (0, 0))
        self.assertEqual(record.status, "estimated")
        self.assertEqual(record.action, "reveal")
        self.assertIn("exact marginal enumeration", record.explanation)
        game = session.game
        assert game is not None
        self.assertNotIn((0, 0), game.board.mines)
        self.assertEqual(LOG_PATH.read_bytes(), phase1)
        self.assertEqual(PHASE2_LOG_PATH.read_bytes(), phase2)
