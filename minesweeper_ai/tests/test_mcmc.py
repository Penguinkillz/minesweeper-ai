"""MCMC agent: block Gibbs, exact marginals, and the local-only walk.

The exact marginals in these tests are counted by enumerating legal layouts
in this file. That count is not the agent's weight function. Regime B is
compared with the exact marginal enumeration switched off. A large gap on those fixtures
is stored and does not fail the test. The two-cell tolerance is the one
exception, and it is not a convergence claim.
"""

import inspect
import math
import unittest
from fractions import Fraction

from minesweeper_ai.agents.base import ActionKind, DecisionStatus
from minesweeper_ai.agents.clp import UnsatisfiablePosition, query
from minesweeper_ai.agents.mcmc import (
    BURN_IN_SWEEPS,
    FRONTIER_EXACT_LIMIT,
    RETAINED_SWEEPS,
    FrontierSearchTruncated,
    MCMCAgent,
    legal_block_assignments,
)
from minesweeper_ai.env.board import CellState, neighbours
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.env.view import VisibleCell, VisibleState

Cell = tuple[int, int]

# Fixed sampler seed for the two-cell Regime B check. The tolerance is the
# Stage 0 sample size (burn-in 50, retain 200), not a tuned win rate.
TWO_CELL_SEED = 0


def visible(
    rows: int,
    cols: int,
    mine_count: int,
    revealed: dict[Cell, int],
    flagged: frozenset[Cell] | None = None,
) -> VisibleState:
    """A visible grid. This helper has no mine layer."""
    flagged = flagged or frozenset()
    grid = []
    for row in range(rows):
        line = []
        for column in range(cols):
            if (row, column) in revealed:
                line.append(VisibleCell(state=CellState.REVEALED, number=revealed[(row, column)]))
            elif (row, column) in flagged:
                line.append(VisibleCell(state=CellState.FLAGGED, number=None))
            else:
                line.append(VisibleCell(state=CellState.HIDDEN, number=None))
        grid.append(tuple(line))
    return VisibleState(
        width=cols,
        height=rows,
        mine_count=mine_count,
        mines_unaccounted=mine_count - len(flagged),
        cells=tuple(grid),
    )


def two_cell() -> VisibleState:
    """Clue 1 with hidden neighbours A and B only. Each legal layout has probability 1/2."""
    return visible(1, 3, 1, {(0, 1): 1})


def three_cell_sum_one() -> VisibleState:
    """One clue of 1 whose three hidden neighbours sum to 1. Each marginal is 1/3."""
    return visible(2, 2, 1, {(0, 0): 1})


def different_residuals() -> VisibleState:
    """Two frontiers with different residuals, so equal frontier weights are wrong.

    Columns are A, clue 1, B, clue 1, C, U1, U2. K = 2.
    (A, B, C) = (0, 1, 0) leaves r = 1 and weight C(2, 1) = 2.
    (A, B, C) = (1, 0, 1) leaves r = 0 and weight C(2, 0) = 1.
    """
    return visible(1, 7, 2, {(0, 1): 1, (0, 3): 1})


def frontier_over_16() -> VisibleState:
    """Eighteen hidden cells, each touching a revealed 0. K = 0, so only the zero layout is legal."""
    revealed = {(row, column): 0 for row in range(6) for column in (0, 2, 4)}
    return visible(6, 6, 0, revealed)


def pattern_121() -> VisibleState:
    """The overlapping 1-2-1. The only layout is mine, safe, mine on row 0."""
    return visible(2, 3, 2, {(1, 0): 1, (1, 1): 2, (1, 2): 1})


def hidden_cells(state: VisibleState) -> tuple[Cell, ...]:
    cells = []
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.HIDDEN:
                cells.append((row, column))
    return tuple(cells)


def independently_legal(state: VisibleState, assignment: dict[Cell, int]) -> bool:
    """Clues and K, written here so the oracle does not call the sampler."""
    flagged = 0
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.FLAGGED:
                flagged += 1
    if sum(assignment.values()) + flagged != state.mine_count:
        return False
    for row in range(state.height):
        for column in range(state.width):
            shown = state.cells[row][column]
            if shown.state is not CellState.REVEALED:
                continue
            total = 0
            for neighbour in neighbours(state.height, state.width, row, column):
                neighbour_state = state.cells[neighbour[0]][neighbour[1]].state
                if neighbour_state is CellState.FLAGGED:
                    total += 1
                elif neighbour_state is CellState.HIDDEN:
                    total += assignment[neighbour]
            if total != shown.number:
                return False
    return True


def enumerate_marginals(state: VisibleState) -> dict[Cell, Fraction]:
    """Exact P(mine) by listing every legal assignment of the hidden cells."""
    hidden = hidden_cells(state)
    if len(hidden) > 16:
        raise ValueError("this oracle is for the small fixtures only")
    counts = {cell: 0 for cell in hidden}
    layouts = 0
    for mask in range(1 << len(hidden)):
        assignment = {cell: (mask >> index) & 1 for index, cell in enumerate(hidden)}
        if not independently_legal(state, assignment):
            continue
        layouts += 1
        for cell in hidden:
            counts[cell] += assignment[cell]
    if layouts == 0:
        raise AssertionError("the fixture has no legal layout")
    return {cell: Fraction(counts[cell], layouts) for cell in hidden}


def equal_frontier_marginals(state: VisibleState) -> dict[Cell, Fraction]:
    """The wrong distribution: each accepted frontier has weight 1, ignoring C(|U|, r)."""
    enumerated = query(state)
    total = len(enumerated.solutions)
    frontier_index = {cell: index for index, cell in enumerate(enumerated.frontier)}
    domains = dict(enumerated.propagated_domains)
    probabilities: dict[Cell, Fraction] = {}
    unfixed: list[Cell] = []
    for cell in hidden_cells(state):
        if cell in frontier_index:
            column = frontier_index[cell]
            hits = sum(1 for solution in enumerated.solutions if solution.values[column] == 1)
            probabilities[cell] = Fraction(hits, total)
        elif domains[cell] == frozenset((0, 1)):
            unfixed.append(cell)
        elif domains[cell] == frozenset((0,)):
            probabilities[cell] = Fraction(0, 1)
        else:
            probabilities[cell] = Fraction(1, 1)
    if unfixed:
        shared = sum(Fraction(solution.residual, solution.unfixed_interior) for solution in enumerated.solutions)
        shared /= total
        for cell in unfixed:
            probabilities[cell] = shared
    return probabilities


def _asserts_no_convergence_claim(test: unittest.TestCase, explanation: str) -> None:
    text = explanation.lower()
    test.assertNotIn("converg", text)
    test.assertNotIn("unique", text)
    test.assertIn("local block gibbs only", text)
    test.assertIn("stochastic gibbs sampling", text)
    test.assertNotIn("exact marginal enumeration", text)
    test.assertNotIn("proved", text)


class SingleSiteTests(unittest.TestCase):
    def test_single_site_gibbs_does_not_leave_its_start_state(self) -> None:
        # S = {(1, 0), (0, 1)}. The states differ in both cells. The full
        # conditional of either cell, given the other, is the current value.
        state = two_cell()
        start = {(0, 0): 1, (0, 2): 0}
        assignment = dict(start)
        for _ in range(100):
            for cell in ((0, 0), (0, 2)):
                legal = legal_block_assignments(state, assignment, (cell,))
                self.assertEqual(legal, ((assignment[cell],),))
                assignment[cell] = legal[0][0]
        self.assertEqual(assignment, start)
        # The same conditional on the pair does offer the other state. That
        # is why the agent updates a block and does not ship this chain.
        pair = legal_block_assignments(state, start, ((0, 0), (0, 2)))
        self.assertEqual(pair, ((1, 0), (0, 1)))


class ExactMarginalTests(unittest.TestCase):
    def test_two_cell_three_cell_and_different_residuals_match_enumeration(self) -> None:
        fixtures = {
            "two_cell": (two_cell(), {(0, 0): Fraction(1, 2), (0, 2): Fraction(1, 2)}),
            "three_cell": (
                three_cell_sum_one(),
                {(0, 1): Fraction(1, 3), (1, 0): Fraction(1, 3), (1, 1): Fraction(1, 3)},
            ),
            "different_residuals": (
                different_residuals(),
                {
                    (0, 0): Fraction(1, 3),
                    (0, 2): Fraction(2, 3),
                    (0, 4): Fraction(1, 3),
                    (0, 5): Fraction(1, 3),
                    (0, 6): Fraction(1, 3),
                },
            ),
        }
        for name, (state, expected) in fixtures.items():
            with self.subTest(fixture=name):
                exact = enumerate_marginals(state)
                self.assertEqual(exact, expected)
                result = MCMCAgent().analyse(state)
                self.assertEqual(result.regime, "exact_marginals")
                self.assertLessEqual(result.frontier_size, FRONTIER_EXACT_LIMIT)
                self.assertEqual(result.recorded_states, 0)
                self.assertEqual(result.discarded_sweeps, 0)
                self.assertEqual(result.kernels, ())
                self.assertIsNone(result.initial_assignment)
                self.assertIn("exact marginal enumeration", result.reasoning.explanation)
                self.assertIn("No full assignment was drawn", result.reasoning.explanation)
                self.assertNotIn("stochastic Gibbs", result.reasoning.explanation)
                self.assertIn("|F|", result.reasoning.explanation)
                self.assertIn("Total weight", result.reasoning.explanation)
                self.assertNotIn("local block Gibbs", result.reasoning.explanation)
                for cell, marginal in exact.items():
                    gap = abs(float(result.probabilities[cell] - marginal))
                    self.assertLessEqual(gap, 1e-9)
                    self.assertEqual(result.probabilities[cell], marginal)

    def test_binomial_fixture_fails_if_frontiers_are_weighted_equally(self) -> None:
        state = different_residuals()
        exact = enumerate_marginals(state)
        flat = equal_frontier_marginals(state)
        self.assertNotEqual(flat[(0, 0)], exact[(0, 0)])
        self.assertEqual(flat[(0, 0)], Fraction(1, 2))
        self.assertEqual(exact[(0, 0)], Fraction(1, 3))
        result = MCMCAgent().analyse(state)
        self.assertEqual(sorted(result.frontier_weights or ()), [1, 2])
        self.assertEqual(result.total_weight, 3)
        self.assertEqual(result.probabilities[(0, 0)], exact[(0, 0)])
        self.assertNotEqual(result.probabilities[(0, 0)], flat[(0, 0)])
        self.assertEqual(result.probabilities[(0, 2)], Fraction(2, 3))
        self.assertEqual(result.probabilities[(0, 5)], Fraction(1, 3))

    def test_opening_board_uses_the_binomial_residual_and_no_burn_in(self) -> None:
        state = Game(seed=0).visible_state()
        result = MCMCAgent().analyse(state)
        self.assertEqual(result.regime, "exact_marginals")
        self.assertEqual(result.frontier_size, 0)
        self.assertEqual(result.frontier_weights, (math.comb(81, 10),))
        self.assertEqual(result.probabilities[(0, 0)], Fraction(10, 81))
        self.assertEqual(result.probabilities[(8, 8)], Fraction(10, 81))
        self.assertEqual(result.action.cell(), (0, 0))
        self.assertIs(result.action.kind, ActionKind.REVEAL)
        self.assertIs(result.reasoning.status, DecisionStatus.ESTIMATED)
        self.assertTrue(result.reasoning.uncertain)
        self.assertEqual(result.recorded_states, 0)

    def test_marginal_zero_is_proved_safe_and_marginal_one_is_a_flag(self) -> None:
        safe = MCMCAgent().analyse(visible(1, 3, 0, {(0, 1): 0}))
        self.assertEqual(safe.probabilities[(0, 0)], 0)
        self.assertIs(safe.reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertFalse(safe.reasoning.uncertain)
        self.assertIs(safe.action.kind, ActionKind.REVEAL)
        self.assertEqual(safe.action.cell(), (0, 0))

        mine = MCMCAgent().analyse(visible(1, 2, 1, {(0, 0): 1}))
        self.assertEqual(mine.probabilities[(0, 1)], 1)
        self.assertIs(mine.reasoning.status, DecisionStatus.PROVED_MINE)
        self.assertFalse(mine.reasoning.uncertain)
        self.assertIs(mine.action.kind, ActionKind.FLAG)
        self.assertEqual(mine.action.cell(), (0, 1))

        overlap = MCMCAgent().analyse(pattern_121())
        self.assertEqual(overlap.probabilities, {(0, 0): Fraction(1, 1), (0, 1): Fraction(0, 1), (0, 2): Fraction(1, 1)})
        # The safe cell is the lowest marginal, so it is revealed before either mine is flagged.
        self.assertEqual(overlap.action.cell(), (0, 1))
        self.assertIs(overlap.reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertIs(overlap.action.kind, ActionKind.REVEAL)

    def test_a_finished_forced_cell_stays_regime_a_under_a_zero_node_cap(self) -> None:
        # Propagation grounds the only neighbour before labeling, so the cap
        # is never hit. A cap that is not hit is still Regime A.
        result = MCMCAgent(node_limit=0).analyse(visible(1, 2, 1, {(0, 0): 1}))
        self.assertEqual(result.regime, "exact_marginals")
        self.assertIs(result.reasoning.status, DecisionStatus.PROVED_MINE)

    def test_a_flag_counts_as_a_mine_inside_the_exact_marginal(self) -> None:
        state = visible(1, 3, 1, {(0, 1): 1}, flagged=frozenset({(0, 0)}))
        result = MCMCAgent().analyse(state)
        self.assertEqual(result.probabilities[(0, 2)], 0)
        self.assertEqual(enumerate_marginals(state)[(0, 2)], 0)
        self.assertIs(result.reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertEqual(result.action.cell(), (0, 2))


class RegimeBTests(unittest.TestCase):
    def test_two_cell_walk_is_within_the_stage_0_tolerance(self) -> None:
        # The pair kernel connects the two legal states, so this one fixture
        # is irreducible by hand. The 0.05 tolerance applies here only.
        state = two_cell()
        exact = enumerate_marginals(state)
        agent = MCMCAgent(sampler_seed=TWO_CELL_SEED, exact_marginals=False, keep_states=True)
        result = agent.analyse(state)
        self.assertEqual(result.regime, "local_only")
        self.assertEqual(result.retained_sweeps, RETAINED_SWEEPS)
        self.assertEqual(result.discarded_sweeps, BURN_IN_SWEEPS)
        self.assertEqual(result.recorded_states, RETAINED_SWEEPS)
        self.assertEqual(result.states_recorded_during_burn_in, 0)
        self.assertFalse(result.initial_state_retained)
        self.assertIsNotNone(result.retained_states)
        self.assertEqual(len(result.retained_states or ()), RETAINED_SWEEPS)
        self.assertEqual(result.kernels, ("pair", "clue-neighbourhood"))
        self.assertIsNone(result.frontier_weights)
        self.assertIs(result.reasoning.status, DecisionStatus.ESTIMATED)
        self.assertTrue(result.reasoning.uncertain)
        _asserts_no_convergence_claim(self, result.reasoning.explanation)
        self.assertIn("Retained sweeps 200", result.reasoning.explanation)
        self.assertIn("Kernels: pair, clue-neighbourhood", result.reasoning.explanation)
        self.assertIn("Self-loop probability 0.05", result.reasoning.explanation)
        for cell, marginal in exact.items():
            gap = abs(float(result.probabilities[cell] - marginal))
            self.assertLessEqual(gap, 0.05, msg=f"{cell} gap {gap}")
        assert result.retained_states is not None
        for bits in result.retained_states:
            assignment = dict(zip(result.hidden_cells, bits))
            self.assertTrue(independently_legal(state, assignment))
            self.assertEqual(sum(assignment.values()), 1)
        repeated = MCMCAgent(sampler_seed=TWO_CELL_SEED, exact_marginals=False).analyse(state)
        self.assertEqual(repeated.probabilities, result.probabilities)

    def test_other_fixtures_store_the_gap_and_do_not_claim_convergence(self) -> None:
        # The gap is computed for every hidden cell. This test fails if that
        # comparison is missing. It does not fail because a gap is large.
        stored: dict[str, dict[Cell, float]] = {}
        fixtures = {
            "three_cell": three_cell_sum_one(),
            "different_residuals": different_residuals(),
        }
        for name, state in fixtures.items():
            result = MCMCAgent(sampler_seed=TWO_CELL_SEED, exact_marginals=False).analyse(state)
            exact = enumerate_marginals(state)
            gaps = {cell: abs(float(result.probabilities[cell] - exact[cell])) for cell in exact}
            self.assertEqual(set(gaps), set(hidden_cells(state)))
            stored[name] = gaps
            _asserts_no_convergence_claim(self, result.reasoning.explanation)
            self.assertIs(result.reasoning.status, DecisionStatus.ESTIMATED)
            self.assertTrue(result.reasoning.uncertain)
            self.assertIs(result.action.kind, ActionKind.REVEAL)
        self.assertEqual(set(stored), {"three_cell", "different_residuals"})
        for name, gaps in stored.items():
            self.assertGreater(len(gaps), 0, msg=name)

    def test_initial_legal_state_matches_the_clp_frontier_and_is_not_retained(self) -> None:
        state = different_residuals()
        result = MCMCAgent(exact_marginals=False, keep_states=True).analyse(state)
        enumerated = query(state)
        first = enumerated.solutions[0]
        self.assertIsNotNone(result.initial_assignment)
        assert result.initial_assignment is not None
        for cell, value in zip(enumerated.frontier, first.values):
            self.assertEqual(result.initial_assignment[cell], value)
        # One interior witness: the row-major first r unfixed interior cells.
        interior = [cell for cell in enumerated.interior if cell in result.initial_assignment]
        mines = [cell for cell in interior if result.initial_assignment[cell] == 1]
        self.assertEqual(len(mines), first.residual)
        self.assertEqual(mines, list(interior[: first.residual]))
        self.assertTrue(independently_legal(state, result.initial_assignment))
        assert result.retained_states is not None
        for bits in result.retained_states:
            self.assertTrue(independently_legal(state, dict(zip(result.hidden_cells, bits))))
        self.assertEqual(result.recorded_states, RETAINED_SWEEPS)
        self.assertEqual(len(result.retained_states or ()), RETAINED_SWEEPS)
        self.assertEqual(result.states_recorded_during_burn_in, 0)
        self.assertFalse(result.initial_state_retained)

    def test_a_frequency_of_zero_is_not_proved_safe(self) -> None:
        state = frontier_over_16()
        result = MCMCAgent().analyse(state)
        self.assertGreater(result.frontier_size, FRONTIER_EXACT_LIMIT)
        self.assertEqual(result.regime, "local_only")
        self.assertEqual(result.probabilities[(0, 1)], 0)
        self.assertIs(result.reasoning.status, DecisionStatus.ESTIMATED)
        self.assertTrue(result.reasoning.uncertain)
        self.assertIs(result.action.kind, ActionKind.REVEAL)
        self.assertEqual(result.action.cell(), (0, 1))
        _asserts_no_convergence_claim(self, result.reasoning.explanation)

    def test_one_record_does_not_mix_the_two_regimes(self) -> None:
        exact = MCMCAgent().analyse(two_cell())
        local = MCMCAgent(exact_marginals=False).analyse(two_cell())
        self.assertIn("exact marginal enumeration", exact.reasoning.explanation)
        self.assertIn("No full assignment was drawn", exact.reasoning.explanation)
        self.assertNotIn("stochastic Gibbs", exact.reasoning.explanation)
        self.assertNotIn("local block Gibbs", exact.reasoning.explanation)
        self.assertEqual(exact.recorded_states, 0)
        self.assertIsNotNone(exact.frontier_weights)
        _asserts_no_convergence_claim(self, local.reasoning.explanation)
        self.assertIsNone(local.frontier_weights)
        self.assertEqual(local.recorded_states, RETAINED_SWEEPS)
        mixed = "exact marginal enumeration" in local.reasoning.explanation and "local block Gibbs" in local.reasoning.explanation
        self.assertFalse(mixed)


class RegimeSelectionTests(unittest.TestCase):
    def test_truncated_enumeration_is_regime_b_when_the_start_is_inside_the_cap(self) -> None:
        # The two-cell clue accepts its first frontier on node 1 and needs a
        # later node for the second solution. Cap 1 discards the enumeration
        # and still leaves the legal-start search a witness.
        state = two_cell()
        capped = query(state, node_limit=1)
        self.assertFalse(capped.finished)
        self.assertEqual(capped.solutions, ())
        self.assertLessEqual(len(capped.frontier), FRONTIER_EXACT_LIMIT)
        result = MCMCAgent(node_limit=1).analyse(state)
        self.assertEqual(result.regime, "local_only")
        self.assertIs(result.reasoning.status, DecisionStatus.ESTIMATED)
        self.assertNotIn("exact marginal enumeration", result.reasoning.explanation)
        _asserts_no_convergence_claim(self, result.reasoning.explanation)

    def test_frontier_above_16_is_regime_b_even_though_enumeration_would_finish(self) -> None:
        state = frontier_over_16()
        enumerated = query(state)
        self.assertTrue(enumerated.finished)
        self.assertGreater(len(enumerated.frontier), FRONTIER_EXACT_LIMIT)
        result = MCMCAgent().analyse(state)
        self.assertEqual(result.regime, "local_only")
        self.assertGreater(result.frontier_size, 16)
        self.assertIsNone(result.frontier_weights)
        self.assertIsNone(result.total_weight)
        _asserts_no_convergence_claim(self, result.reasoning.explanation)


class LegalStartLimitTests(unittest.TestCase):
    def test_a_cap_that_stops_before_the_first_witness_is_not_unsatisfiable(self) -> None:
        with self.assertRaises(FrontierSearchTruncated) as caught:
            MCMCAgent(node_limit=0).act(two_cell())
        self.assertFalse(issubclass(FrontierSearchTruncated, UnsatisfiablePosition))
        self.assertNotIsInstance(caught.exception, UnsatisfiablePosition)
        self.assertEqual(caught.exception.node_limit, 0)
        self.assertGreater(caught.exception.nodes, caught.exception.node_limit)
        text = str(caught.exception).lower()
        self.assertIn("truncated", text)
        self.assertNotIn("no legal layout", text)

    def test_stopping_early_on_an_empty_tree_does_not_prove_it_empty(self) -> None:
        # Two separated clues of 1, with one interior cell, and K = 1.
        # Propagation does not fail. The finished frontier search accepts
        # nothing (two nodes). A cap that expires on the first or second
        # attempt has not seen that, so it must not be called unsatisfiable.
        state = visible(1, 7, 1, {(0, 1): 1, (0, 5): 1})
        for limit in (None, 2):
            with self.assertRaises(UnsatisfiablePosition) as finished:
                MCMCAgent(node_limit=limit).act(state)
            self.assertNotIsInstance(finished.exception, FrontierSearchTruncated)
            self.assertIn("no legal layout", str(finished.exception).lower())
        for limit in (0, 1):
            with self.assertRaises(FrontierSearchTruncated) as truncated:
                MCMCAgent(node_limit=limit).act(state)
            self.assertNotIsInstance(truncated.exception, UnsatisfiablePosition)
            self.assertGreater(truncated.exception.nodes, truncated.exception.node_limit)
            self.assertIn("truncated", str(truncated.exception).lower())
            self.assertNotIn("no legal layout", str(truncated.exception).lower())

    def test_propagation_failure_stays_unsatisfiable_under_a_zero_cap(self) -> None:
        state = visible(1, 2, 1, {(0, 0): 0}, flagged=frozenset({(0, 1)}))
        with self.assertRaises(UnsatisfiablePosition) as caught:
            MCMCAgent(node_limit=0).act(state)
        self.assertNotIsInstance(caught.exception, FrontierSearchTruncated)

    def test_a_zero_cap_still_starts_when_propagation_accepts_the_frontier(self) -> None:
        result = MCMCAgent(node_limit=0).analyse(frontier_over_16())
        self.assertEqual(result.regime, "local_only")
        self.assertIsNotNone(result.initial_assignment)
        assert result.initial_assignment is not None
        self.assertTrue(all(bit == 0 for bit in result.initial_assignment.values()))

    def test_a_large_frontier_that_propagation_rejects_is_not_a_truncation(self) -> None:
        revealed = {(row, column): 0 for row in range(6) for column in (0, 2, 4)}
        state = visible(6, 6, 1, revealed)
        with self.assertRaises(UnsatisfiablePosition) as caught:
            MCMCAgent(node_limit=0).act(state)
        self.assertNotIsInstance(caught.exception, FrontierSearchTruncated)


class LeakageTests(unittest.TestCase):
    def test_agent_source_does_not_read_the_mine_layer(self) -> None:
        import minesweeper_ai.agents.mcmc as mcmc

        source = inspect.getsource(mcmc)
        self.assertNotIn(".mines", source)
        self.assertNotIn("place_mines", source)
        signature = inspect.signature(MCMCAgent.act)
        self.assertEqual(list(signature.parameters), ["self", "state"])

    def test_samples_ignore_a_hidden_map_edited_after_the_view_is_built(self) -> None:
        game = Game(seed=3)
        game.reveal(4, 4)
        state = game.visible_state()
        agent = MCMCAgent(sampler_seed=1)
        first = agent.analyse(state)
        self.assertGreater(len(game.board.mines), 0)
        game.board.mines.clear()
        game.board.mines.add((0, 0))
        second = agent.analyse(state)
        self.assertEqual(first.action, second.action)
        self.assertEqual(first.reasoning, second.reasoning)
        self.assertEqual(first.probabilities, second.probabilities)
        self.assertEqual(first.regime, second.regime)
        self.assertFalse(hasattr(state, "mines"))

    def test_a_hand_built_view_never_had_a_mine_set(self) -> None:
        state = two_cell()
        result = MCMCAgent().analyse(state)
        self.assertFalse(hasattr(state, "mines"))
        self.assertEqual(result.probabilities[(0, 0)], Fraction(1, 2))

    def test_unsatisfiable_visible_grid_does_not_invent_a_click(self) -> None:
        state = visible(1, 2, 1, {(0, 0): 0}, flagged=frozenset({(0, 1)}))
        with self.assertRaises(UnsatisfiablePosition):
            MCMCAgent().act(state)


class PlayTests(unittest.TestCase):
    def test_first_click_is_the_environment_safe_click(self) -> None:
        game = Game(seed=0)
        self.assertEqual(game.board.mines, set())
        action, reasoning = MCMCAgent().act(game.visible_state())
        self.assertEqual(action.cell(), (0, 0))
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertTrue(reasoning.uncertain)
        self.assertIsNot(reasoning.status, DecisionStatus.PROVED_SAFE)
        game.reveal(action.row, action.column)
        self.assertNotIn((0, 0), game.board.mines)
        self.assertIsNot(game.status, GameStatus.LOST)

    def test_beginner_game_plays_to_the_end_and_proved_moves_match_the_board(self) -> None:
        game = Game(seed=0)
        agent = MCMCAgent()
        steps = 0
        saw_exact = False
        while game.status is GameStatus.ONGOING and steps < 200:
            action, reasoning = agent.act(game.visible_state())
            steps += 1
            result = agent.last_result
            self.assertIsNotNone(result)
            assert result is not None
            if result.regime == "exact_marginals":
                saw_exact = True
                self.assertNotIn("local block Gibbs", reasoning.explanation)
            else:
                self.assertEqual(result.regime, "local_only")
                _asserts_no_convergence_claim(self, reasoning.explanation)
                self.assertTrue(reasoning.uncertain)
                self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
            if reasoning.status is DecisionStatus.PROVED_SAFE:
                self.assertFalse(reasoning.uncertain)
                self.assertIs(action.kind, ActionKind.REVEAL)
                game.reveal(action.row, action.column)
                self.assertIsNot(game.status, GameStatus.LOST)
                self.assertNotIn((action.row, action.column), game.board.mines)
            elif reasoning.status is DecisionStatus.PROVED_MINE:
                self.assertFalse(reasoning.uncertain)
                self.assertIs(action.kind, ActionKind.FLAG)
                self.assertIn((action.row, action.column), game.board.mines)
                game.flag(action.row, action.column)
            else:
                self.assertTrue(reasoning.uncertain)
                self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
                self.assertIs(action.kind, ActionKind.REVEAL)
                game.reveal(action.row, action.column)
        self.assertIsNot(game.status, GameStatus.ONGOING)
        self.assertLess(steps, 200)
        self.assertTrue(saw_exact)

    def test_decisions_are_reproducible_for_a_seed(self) -> None:
        self.assertEqual(_transcript(0), _transcript(0))


def _transcript(seed: int) -> tuple[tuple[object, ...], ...]:
    game = Game(seed=seed)
    agent = MCMCAgent(sampler_seed=seed)
    played = []
    while game.status is GameStatus.ONGOING and len(played) < 200:
        action, reasoning = agent.act(game.visible_state())
        regime = agent.last_result.regime if agent.last_result is not None else None
        played.append((action.kind, action.row, action.column, reasoning.status, reasoning.uncertain, regime))
        if action.kind is ActionKind.REVEAL:
            game.reveal(action.row, action.column)
        else:
            game.flag(action.row, action.column)
    return tuple(played)


if __name__ == "__main__":
    unittest.main()
