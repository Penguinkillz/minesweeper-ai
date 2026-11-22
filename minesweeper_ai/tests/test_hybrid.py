"""Hybrid agent: a CLP proof is kept, and a guess uses one MCMC regime.

Regime A marginal 0 is a finished count, so it is a proof. A Regime B
frequency of 0 is a walk average, so it is not. The two are not averaged.
The harness factories stay the three original agents.
"""

import unittest
from fractions import Fraction
from pathlib import Path

from minesweeper_ai.agents.base import Action, ActionKind, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import CLPAgent, QueryResult, query
from minesweeper_ai.agents.hybrid import AGENT_NAME, HybridAgent
from minesweeper_ai.agents.mcmc import REGIME_EXACT, REGIME_LOCAL, MCMCAgent
from minesweeper_ai.env.board import PUBLISHED_FIRST_CLICK, CellState
from minesweeper_ai.env.view import VisibleCell, VisibleState
from minesweeper_ai.evaluation.harness import (
    LOG_PATH,
    PHASE2_LOG_PATH,
    phase1_agent_factories,
)
from minesweeper_ai.ui.session import PLAYERS, PlaySession, describe_decision, phase1_agent


def _bytes_if_present(path: Path) -> bytes | None:
    if path.is_file():
        return path.read_bytes()
    return None

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

    def analyse(self, state: VisibleState, enumeration: QueryResult | None = None) -> object:
        raise AssertionError("MCMC was asked after CLP returned a proof")


class _Cleared:
    """A real frontier with the proof sets emptied.

    The guess tests use this so a board CLP would solve still reaches MCMC.
    It is not a ``QueryResult``, so MCMC does its own query, as it does when
    nothing is passed in.
    """

    def __init__(self, found: QueryResult) -> None:
        self.frontier = found.frontier
        self.interior = found.interior
        self.proved_safe = frozenset()
        self.proved_mine = frozenset()
        self.unsatisfiable = False
        self.finished = found.finished
        self.solutions = ()


class _NoProof:
    """A CLP stand-in whose query reports no proved cell."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell

    def analyse(self, state: VisibleState) -> _Cleared:
        return _Cleared(query(state))


class _Script:
    def __init__(self, regime: str, probabilities: dict[Cell, Fraction], frontier_size: int) -> None:
        self.regime = regime
        self.probabilities = probabilities
        self.frontier_size = frontier_size
        self.enumeration: QueryResult | None = None

    def analyse(self, state: VisibleState, enumeration: QueryResult | None = None) -> "_Script":
        self.enumeration = enumeration
        return self


def proved_mine_and_open_frontier() -> VisibleState:
    """(0, 0) is a proved mine and the first frontier cell. Two frontier cells stay open.

    CLP's own action is a fallback reveal. The proved mine is not that action.
    """
    return visible(
        2,
        5,
        3,
        {(0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1},
    )


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


class ProvedMineTests(unittest.TestCase):
    def test_regime_b_tie_does_not_reveal_the_clp_proved_mine(self) -> None:
        # CLP's selected action is a fallback, so the proved mine is not the
        # move act() returns. Several frontier cells share frequency 1.
        # Row-major order of that tie is the proved mine. The hybrid must not
        # reveal it.
        state = proved_mine_and_open_frontier()
        found = query(state)
        self.assertIn((0, 0), found.proved_mine)
        self.assertTrue(any(cell not in found.proved_mine for cell in found.frontier))
        self.assertFalse(found.proved_safe)
        self.assertEqual(found.frontier[0], (0, 0))
        clp_action, clp_reasoning = CLPAgent().act(state)
        self.assertIs(clp_reasoning.status, DecisionStatus.FALLBACK)
        self.assertIs(clp_action.kind, ActionKind.REVEAL)
        self.assertNotIn(clp_action.cell(), found.proved_mine)

        probabilities = {cell: Fraction(1, 1) for cell in found.frontier}
        tied = min(probabilities, key=lambda cell: (-abs(probabilities[cell] - 1), cell[0], cell[1]))
        self.assertEqual(tied, (0, 0))
        script = _Script(REGIME_LOCAL, probabilities, frontier_size=len(found.frontier))
        action, reasoning = HybridAgent(mcmc_agent=script).act(state)
        assert script.enumeration is not None
        self.assertEqual(script.enumeration.proved_mine, found.proved_mine)
        self.assertIn((0, 0), script.enumeration.proved_mine)
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertNotIn(action.cell(), found.proved_mine)
        self.assertIn(action.cell(), found.frontier)
        self.assertEqual(action.cell(), (0, 3))
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertIs(reasoning.uncertain, True)
        self.assertNotIn(reasoning.status, (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE))

    def test_a_proved_mine_is_excluded_even_when_its_frequency_is_farthest_from_one(self) -> None:
        state = proved_mine_and_open_frontier()
        found = query(state)
        probabilities = {cell: Fraction(9, 10) for cell in found.frontier}
        for cell in found.proved_mine:
            probabilities[cell] = Fraction(0, 1)
        action, reasoning = HybridAgent(
            mcmc_agent=_Script(REGIME_LOCAL, probabilities, frontier_size=len(found.frontier)),
        ).act(state)
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertNotIn(action.cell(), found.proved_mine)
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertNotEqual(reasoning.status, DecisionStatus.PROVED_MINE)

    def test_proved_safe_wins_over_a_proved_mine(self) -> None:
        state = visible(2, 3, 2, {(1, 0): 1, (1, 1): 2, (1, 2): 1})
        found = query(state)
        self.assertIn((0, 1), found.proved_safe)
        self.assertTrue(found.proved_mine)
        clp_action, clp_reasoning = CLPAgent().act(state)
        action, reasoning = HybridAgent(mcmc_agent=_Boom()).act(state)
        self.assertEqual(action, clp_action)
        self.assertEqual(action.cell(), (0, 1))
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertEqual(reasoning.explanation, clp_reasoning.explanation)
        self.assertIs(reasoning.uncertain, False)

    def test_no_eligible_reveal_flags_the_proved_mine(self) -> None:
        state = visible(1, 3, 2, {(0, 1): 2})
        found = query(state)
        self.assertEqual(found.proved_mine, frozenset({(0, 0), (0, 2)}))
        self.assertFalse(found.proved_safe)
        clp_action, clp_reasoning = CLPAgent().act(state)
        action, reasoning = HybridAgent(mcmc_agent=_Boom()).act(state)
        self.assertEqual(action, clp_action)
        self.assertIs(action.kind, ActionKind.FLAG)
        self.assertEqual(action.cell(), (0, 0))
        self.assertIs(reasoning.status, DecisionStatus.PROVED_MINE)
        self.assertEqual(reasoning.explanation, clp_reasoning.explanation)
        self.assertIs(reasoning.uncertain, False)

    def test_an_open_cell_is_revealed_when_the_frontier_is_only_proved_mines(self) -> None:
        # The first hidden cell is a proved mine. Regime B has no remaining
        # frontier cell to rank. The reveal is the next hidden cell.
        state = visible(1, 4, 2, {(0, 0): 1})
        found = query(state)
        self.assertEqual(found.proved_mine, frozenset({(0, 1)}))
        self.assertIn((0, 2), found.interior)
        probabilities = {
            (0, 1): Fraction(1, 1),
            (0, 2): Fraction(1, 1),
            (0, 3): Fraction(1, 1),
        }
        action, reasoning = HybridAgent(
            mcmc_agent=_Script(REGIME_LOCAL, probabilities, frontier_size=len(found.frontier)),
        ).act(state)
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertEqual(action.cell(), (0, 2))
        self.assertNotIn(action.cell(), found.proved_mine)
        self.assertIs(reasoning.status, DecisionStatus.FALLBACK)
        self.assertIs(reasoning.uncertain, True)

    def test_real_local_frequency_zero_or_one_stays_estimated(self) -> None:
        # The first accepted frontier is (0, 1) on the two hidden cells, so a
        # chain that stays there records frequency 0 and frequency 1. Neither
        # frequency is a proof. CLP did not prove either cell.
        state = two_cell()
        found = query(state)
        self.assertFalse(found.proved_safe)
        self.assertFalse(found.proved_mine)
        hybrid = HybridAgent(
            mcmc_agent=MCMCAgent(
                exact_marginals=False,
                burn_in_sweeps=0,
                retained_sweeps=1,
                self_loop_probability=1,
            ),
        )
        action, reasoning = hybrid.act(state)
        self.assertEqual(hybrid.last_source, REGIME_LOCAL)
        assert hybrid.last_mcmc is not None
        self.assertEqual(hybrid.last_mcmc.probabilities[(0, 0)], 0)  # type: ignore[attr-defined]
        self.assertEqual(hybrid.last_mcmc.probabilities[(0, 2)], 1)  # type: ignore[attr-defined]
        self.assertEqual(action.cell(), (0, 0))
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertIs(reasoning.uncertain, True)
        self.assertNotIn(reasoning.status, (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE))
        self.assertIn("sample frequency 0", reasoning.explanation)
        self.assertNotIn("proved_safe", reasoning.explanation)
        self.assertNotIn("proved_mine", reasoning.explanation)


class ReuseTests(unittest.TestCase):
    def test_standalone_clp_action_is_unchanged_on_the_mixed_board(self) -> None:
        state = proved_mine_and_open_frontier()
        action, reasoning = CLPAgent().act(state)
        found = CLPAgent().analyse(state)
        self.assertIs(reasoning.status, DecisionStatus.FALLBACK)
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertEqual(action.cell(), (0, 3))
        self.assertIn((0, 0), found.proved_mine)
        self.assertIn((0, 3), found.frontier)
        flagged = CLPAgent().proof(state, found, (0, 0))
        self.assertIs(flagged[0].kind, ActionKind.FLAG)
        self.assertIs(flagged[1].status, DecisionStatus.PROVED_MINE)
        self.assertIn("mine", flagged[1].explanation)

    def test_mcmc_with_the_saved_query_matches_mcmc_alone(self) -> None:
        state = two_cell()
        enumerated = query(state)
        local = dict(
            exact_marginals=False,
            sampler_seed=0,
            burn_in_sweeps=0,
            retained_sweeps=1,
            self_loop_probability=1,
        )
        direct = MCMCAgent(**local).analyse(state)
        reused = MCMCAgent(**local).analyse(state, enumeration=enumerated)
        self.assertEqual(direct.regime, reused.regime)
        self.assertEqual(direct.regime, REGIME_LOCAL)
        self.assertEqual(direct.probabilities, reused.probabilities)
        self.assertEqual(direct.action, reused.action)
        self.assertEqual(direct.initial_assignment, reused.initial_assignment)
        self.assertEqual(direct.reasoning.status, reused.reasoning.status)
        self.assertEqual(direct.reasoning.explanation, reused.reasoning.explanation)

        exact = MCMCAgent().analyse(state)
        exact_reused = MCMCAgent().analyse(state, enumeration=enumerated)
        self.assertEqual(exact.regime, REGIME_EXACT)
        self.assertEqual(exact_reused.regime, REGIME_EXACT)
        self.assertEqual(exact.probabilities, exact_reused.probabilities)
        self.assertEqual(exact.action, exact_reused.action)
        self.assertEqual(exact.reasoning.explanation, exact_reused.reasoning.explanation)
        self.assertEqual(exact.frontier_weights, exact_reused.frontier_weights)

        mixed = proved_mine_and_open_frontier()
        mixed_query = query(mixed)
        self.assertTrue(mixed_query.interior)
        mixed_direct = MCMCAgent(**local).analyse(mixed)
        mixed_reused = MCMCAgent(**local).analyse(mixed, enumeration=mixed_query)
        self.assertEqual(mixed_direct.initial_assignment, mixed_reused.initial_assignment)
        self.assertEqual(mixed_direct.probabilities, mixed_reused.probabilities)
        self.assertEqual(mixed_direct.action, mixed_reused.action)

    def test_a_finished_query_is_not_enumerated_again(self) -> None:
        import minesweeper_ai.agents.clp as clp_module

        state = two_cell()
        builds = {"n": 0}
        queries = {"n": 0}
        original_build = clp_module._build_model
        original_query = clp_module.query

        def _build(visible_state: VisibleState):
            builds["n"] += 1
            return original_build(visible_state)

        def _query(visible_state: VisibleState, node_limit: int | None = None) -> QueryResult:
            queries["n"] += 1
            return original_query(visible_state, node_limit)

        clp_module._build_model = _build  # type: ignore[assignment]
        clp_module.query = _query  # type: ignore[method-assign]
        try:
            for exact in (True, False):
                builds["n"] = 0
                queries["n"] = 0
                agent = MCMCAgent(
                    exact_marginals=exact,
                    burn_in_sweeps=0,
                    retained_sweeps=1,
                    self_loop_probability=1,
                )
                HybridAgent(mcmc_agent=agent).act(state)
                self.assertEqual(queries["n"], 1, msg=f"exact={exact}")
                self.assertEqual(builds["n"], 1, msg=f"exact={exact}")
        finally:
            clp_module._build_model = original_build  # type: ignore[assignment]
            clp_module.query = original_query  # type: ignore[method-assign]

    def test_a_truncated_query_keeps_no_start_so_the_legal_search_still_runs(self) -> None:
        # query() drops partial solutions. The saved result has no legal start,
        # so Regime B still searches under the caller's cap. It does not run
        # the full query again.
        import minesweeper_ai.agents.clp as clp_module

        state = two_cell()
        enumerated = query(state, node_limit=1)
        self.assertFalse(enumerated.finished)
        self.assertEqual(enumerated.solutions, ())
        queries = {"n": 0}
        original_query = clp_module.query

        def _query(visible_state: VisibleState, node_limit: int | None = None) -> QueryResult:
            queries["n"] += 1
            return original_query(visible_state, node_limit)

        clp_module.query = _query  # type: ignore[method-assign]
        try:
            direct = MCMCAgent(node_limit=1, burn_in_sweeps=0, retained_sweeps=1).analyse(state)
            reused = MCMCAgent(node_limit=1, burn_in_sweeps=0, retained_sweeps=1).analyse(
                state,
                enumeration=enumerated,
            )
        finally:
            clp_module.query = original_query  # type: ignore[method-assign]
        self.assertEqual(queries["n"], 1)
        self.assertEqual(direct.regime, REGIME_LOCAL)
        self.assertEqual(reused.regime, REGIME_LOCAL)
        self.assertEqual(direct.probabilities, reused.probabilities)
        self.assertEqual(direct.action, reused.action)


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

    def test_opening_click_is_the_environment_before_hybrid_act(self) -> None:
        phase1 = _bytes_if_present(LOG_PATH)
        phase2 = _bytes_if_present(PHASE2_LOG_PATH)
        asked: list[VisibleState] = []
        real = PlaySession()
        real.configure(player="hybrid", seed=0, difficulty="beginner")
        real.start()
        self.assertIsInstance(real.agent, HybridAgent)
        original = real.agent.act  # type: ignore[union-attr]

        def _watch(state: VisibleState) -> tuple[Action, Reasoning]:
            asked.append(state)
            return original(state)

        real.agent.act = _watch  # type: ignore[union-attr]
        opening = real.step()
        self.assertEqual(asked, [])
        self.assertEqual(opening.source, "environment")
        self.assertIsNone(opening.status)
        self.assertIsNone(opening.uncertain)
        self.assertIsNone(opening.agent_name)
        self.assertEqual(opening.cell, PUBLISHED_FIRST_CLICK)
        self.assertEqual(opening.action, "reveal")
        self.assertNotIn("estimated", opening.explanation)
        panel = describe_decision(real.snapshot())
        self.assertIn("Status: none", panel)
        self.assertNotIn("Status: estimated", panel)
        self.assertNotIn("Agent: hybrid", panel)
        game = real.game
        assert game is not None
        self.assertTrue(game.board.mines_placed)
        self.assertNotIn(PUBLISHED_FIRST_CLICK, game.board.mines)
        self.assertIs(game.board.states[0][0], CellState.REVEALED)

        for name in ("CLP", "MCMC", "decision_tree", "hybrid"):
            seen: list[VisibleState] = []

            def builder(player: str, seen: list[VisibleState] = seen) -> object:
                class _Agent:
                    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
                        seen.append(state)
                        return Action.reveal(1, 1), Reasoning(
                            agent_name=AGENT_NAME,
                            cell=(1, 1),
                            status=DecisionStatus.FALLBACK,
                            explanation="The opening is already done.",
                            uncertain=True,
                        )

                return _Agent()

            session = PlaySession(agent_builder=builder)
            session.configure(player=name, seed=0, difficulty="beginner")
            session.start()
            first = session.step()
            self.assertEqual(seen, [], msg=name)
            self.assertEqual(first.source, "environment")
            self.assertIsNone(first.status)
            self.assertIsNone(first.agent_name)
            self.assertEqual(first.cell, (0, 0))
            second = session.step()
            self.assertEqual(len(seen), 1, msg=name)
            self.assertIs(seen[0].cells[0][0].state, CellState.REVEALED)
            self.assertEqual(second.cell, (1, 1))
            self.assertEqual(second.status, "fallback")
        self.assertEqual(_bytes_if_present(LOG_PATH), phase1)
        self.assertEqual(_bytes_if_present(PHASE2_LOG_PATH), phase2)
