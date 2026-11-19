"""MCMC Minesweeper agent: block Gibbs in two regimes that are never averaged.

Slide mapping (``16-AI-MCMC.pdf``):

| Slide idea | Where it is in this file |
| --- | --- |
| Monte Carlo estimate of an expectation | the retained frequency of each cell |
| Gibbs as Metropolis–Hastings with ratio 1 | ``_resample_block`` |
| Full conditional of a block | ``legal_block_assignments`` |
| Detailed balance for that conditional | the Regime B explanation |
| Aperiodicity by a self-loop mixture | ``SELF_LOOP_PROBABILITY`` inside ``_sweep`` |

The target is uniform on the legal mine layouts. A layout is legal when it
meets every visible clue and the mine count K. The chain never stores an
illegal layout, and it never reads the hidden mine map.

Single-site Gibbs is not this agent. On the two-cell clue it cannot leave
its start state. The unit test runs that chain. The player uses a block.

Regime A computes exact posterior marginals from a finished frontier
enumeration and the binomial residual weights. It does not draw a layout.
It runs only when ``|F| <= 16`` and the enumeration finishes. Marginal 0 is
proved safe. Marginal 1 is proved mine. There is no burn-in and no local chain.

Regime B is stochastic block Gibbs: the pair kernel and the clue-neighbourhood
kernel, mixed with a self-loop. It runs when Regime A does not, and only from
a legal start found within the same node cap. Every reveal is estimated. The
text does not claim convergence and does not claim a unique stationary
distribution.
"""

import math
import random
from collections.abc import Mapping, Sequence
from fractions import Fraction

from minesweeper_ai.agents import clp
from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import QueryResult, UnsatisfiablePosition
from minesweeper_ai.env.board import CellState, neighbours
from minesweeper_ai.env.view import VisibleState

AGENT_NAME = "MCMC"
Cell = tuple[int, int]

# Section 5.3 and 5.5. These are the Stage 0 choices, not a fitted win rate.
FRONTIER_EXACT_LIMIT = 16
BURN_IN_SWEEPS = 50
RETAINED_SWEEPS = 200
SELF_LOOP_PROBABILITY = 0.05

REGIME_EXACT = "exact_marginals"
REGIME_LOCAL = "local_only"
EXACT_MARGINAL_TEXT = "exact marginal enumeration"
LOCAL_ONLY_TEXT = "local block Gibbs only"
LOCAL_KERNELS = ("pair", "clue-neighbourhood")


class MCMCResult:
    """One decision, produced by exactly one regime.

    ``probabilities`` covers every hidden cell. Regime A stores the binomial
    weights of the finished frontier enumeration. Regime B stores a walk
    average and leaves the weights empty. ``initial_state_retained`` is false:
    the legal start is not a term in the retained mean.
    """

    def __init__(
        self,
        regime: str,
        frontier_size: int,
        probabilities: dict[Cell, Fraction],
        total_weight: int | None,
        frontier_weights: tuple[int, ...] | None,
        retained_sweeps: int,
        discarded_sweeps: int,
        recorded_states: int,
        states_recorded_during_burn_in: int,
        initial_state_retained: bool,
        kernels: tuple[str, ...],
        action: Action,
        reasoning: Reasoning,
        hidden_cells: tuple[Cell, ...],
        retained_states: tuple[tuple[int, ...], ...] | None,
        initial_assignment: dict[Cell, int] | None,
    ) -> None:
        if regime not in (REGIME_EXACT, REGIME_LOCAL):
            raise ValueError(f"unknown regime {regime}")
        self.regime = regime
        self.frontier_size = frontier_size
        self.probabilities = probabilities
        self.total_weight = total_weight
        self.frontier_weights = frontier_weights
        self.retained_sweeps = retained_sweeps
        self.discarded_sweeps = discarded_sweeps
        self.recorded_states = recorded_states
        self.states_recorded_during_burn_in = states_recorded_during_burn_in
        self.initial_state_retained = initial_state_retained
        self.kernels = kernels
        self.action = action
        self.reasoning = reasoning
        self.hidden_cells = hidden_cells
        self.retained_states = retained_states
        self.initial_assignment = initial_assignment


class FrontierSearchTruncated(Exception):
    """The legal-start search hit its node cap before accepting a frontier.

    The clues may still have a legal layout. This is a truncated search, not
    ``UnsatisfiablePosition``, and no click is invented from it.
    """

    def __init__(self, node_limit: int, nodes: int) -> None:
        self.node_limit = node_limit
        self.nodes = nodes
        super().__init__(
            f"Legal-start frontier search stopped at node limit {node_limit} "
            f"after {nodes} frontier attempts, before any frontier was accepted. "
            "The search was truncated. No click is invented."
        )


class MCMCAgent:
    """Play from ``VisibleState`` only.

    ``exact_marginals`` switches the weighted enumeration off, so a small
    frontier can still be scored as a stochastic Gibbs walk.
    ``node_limit`` caps frontier value attempts. The enumeration and the
    legal-start search each have their own counter, and both use this cap.
    ``None`` means neither search is capped.
    """

    def __init__(
        self,
        sampler_seed: int = 0,
        node_limit: int | None = None,
        *,
        exact_marginals: bool = True,
        burn_in_sweeps: int = BURN_IN_SWEEPS,
        retained_sweeps: int = RETAINED_SWEEPS,
        self_loop_probability: float = SELF_LOOP_PROBABILITY,
        keep_states: bool = False,
    ) -> None:
        if burn_in_sweeps < 0 or retained_sweeps < 1:
            raise ValueError("Regime B discards a non-negative burn-in and retains at least one sweep")
        if not 0 <= self_loop_probability <= 1:
            raise ValueError("self-loop probability must lie in [0, 1]")
        self.sampler_seed = sampler_seed
        self.node_limit = node_limit
        self.exact_marginals = exact_marginals
        self.burn_in_sweeps = burn_in_sweeps
        self.retained_sweeps = retained_sweeps
        self.self_loop_probability = self_loop_probability
        self.keep_states = keep_states
        self.last_result: MCMCResult | None = None

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        result = self.analyse(state)
        self.last_result = result
        return result.action, result.reasoning

    def analyse(self, state: VisibleState) -> MCMCResult:
        """Choose the regime, then the cell. The two regimes are not averaged."""
        _domains, frontier, _interior, _constraints, _watches = clp._build_model(state)
        frontier_size = len(frontier)
        # Regime A is the finished full-block conditional. A larger frontier,
        # a truncated search, or an explicit switch falls through to Regime B
        # without using any partial weights.
        if self.exact_marginals and frontier_size <= FRONTIER_EXACT_LIMIT:
            enumerated = clp.query(state, self.node_limit)
            if enumerated.unsatisfiable:
                _raise_unsatisfiable()
            if enumerated.finished:
                return _regime_a(state, enumerated)
        # Separate counter, same cap. This search stops at the first accepted
        # frontier, so it can succeed after the full enumeration was truncated.
        start = _first_legal_assignment(state, self.node_limit)
        if start.status == "truncated":
            if self.node_limit is None:
                raise RuntimeError("an uncapped legal-start search was marked truncated")
            raise FrontierSearchTruncated(self.node_limit, start.nodes)
        if start.assignment is None:
            _raise_unsatisfiable()
        return _regime_b(
            state,
            frontier_size,
            start.assignment,
            random.Random(self.sampler_seed),
            self.burn_in_sweeps,
            self.retained_sweeps,
            self.self_loop_probability,
            self.keep_states,
        )


def legal_block_assignments(
    state: VisibleState,
    assignment: Mapping[Cell, int],
    block: tuple[Cell, ...],
) -> tuple[tuple[int, ...], ...]:
    """Legal settings of ``block``, given the cells outside it.

    For the uniform distribution on legal layouts, the full conditional is
    uniform on this tuple. The current setting is included when ``assignment``
    itself is legal. Order is the integer masks of ``block``, low bit first.
    """
    position = _Position(state)
    bits = _bits_from_assignment(position, assignment)
    return tuple(_legal_settings(position, bits, tuple(position.index[cell] for cell in block)))


class _Clue:
    """One revealed clue, reduced to the hidden neighbours still free."""

    def __init__(self, required: int, cells: tuple[int, ...]) -> None:
        self.required = required
        self.cells = cells


class _Position:
    """Clues and K, indexed over the hidden cells. Built from the visible grid."""

    def __init__(self, state: VisibleState) -> None:
        self.hidden = _hidden_cells(state)
        self.index = {cell: index for index, cell in enumerate(self.hidden)}
        flagged = _flagged_cells(state)
        self.mine_quota = state.mine_count - len(flagged)
        clues: list[_Clue] = []
        self.consistent = 0 <= self.mine_quota <= len(self.hidden)
        for row in range(state.height):
            for column in range(state.width):
                visible = state.cells[row][column]
                if visible.state is not CellState.REVEALED:
                    continue
                if visible.number is None:
                    raise ValueError(f"revealed cell {(row, column)} has no clue")
                required = visible.number
                free: list[int] = []
                for neighbour in neighbours(state.height, state.width, row, column):
                    cell_state = state.cells[neighbour[0]][neighbour[1]].state
                    if cell_state is CellState.FLAGGED:
                        required -= 1
                    elif cell_state is CellState.HIDDEN:
                        free.append(self.index[neighbour])
                if not free:
                    if required != 0:
                        self.consistent = False
                    continue
                clues.append(_Clue(required, tuple(free)))
        self.clues = tuple(clues)
        # Revealed clues that still have a hidden neighbour, in row-major order.
        # The neighbourhood kernel draws one of these blocks.
        blocks: list[tuple[int, ...]] = []
        for row in range(state.height):
            for column in range(state.width):
                visible = state.cells[row][column]
                if visible.state is not CellState.REVEALED:
                    continue
                block = tuple(
                    self.index[neighbour]
                    for neighbour in neighbours(state.height, state.width, row, column)
                    if neighbour in self.index
                )
                if block:
                    blocks.append(block)
        self.neighbourhoods = tuple(blocks)

    def mine_count_ok(self, bits: Sequence[int]) -> bool:
        return sum(bits) == self.mine_quota

    def clues_ok(self, bits: Sequence[int]) -> bool:
        for clue in self.clues:
            total = 0
            for index in clue.cells:
                total += bits[index]
            if total != clue.required:
                return False
        return True

    def is_legal(self, bits: Sequence[int]) -> bool:
        if not self.consistent:
            return False
        return self.mine_count_ok(bits) and self.clues_ok(bits)


def _legal_settings(position: _Position, bits: Sequence[int], block: tuple[int, ...]) -> list[tuple[int, ...]]:
    """List the block assignments that keep the whole layout legal.

    Cells outside ``block`` stay at ``bits``. At most ``2 ** len(block)``
    candidates. The binomial residual is not applied: each surviving setting
    is one full layout, which is what a local conditional counts.
    """
    if not block:
        return [()]
    outside_mines = 0
    block_set = set(block)
    for index, bit in enumerate(bits):
        if index not in block_set:
            outside_mines += bit
    local_of = {global_index: local for local, global_index in enumerate(block)}
    reduced: list[tuple[int, tuple[int, ...]]] = []
    for clue in position.clues:
        touch = tuple(local_of[index] for index in clue.cells if index in block_set)
        if not touch:
            continue
        outside = 0
        for index in clue.cells:
            if index not in block_set:
                outside += bits[index]
        reduced.append((clue.required - outside, touch))
    found: list[tuple[int, ...]] = []
    width = len(block)
    for mask in range(1 << width):
        local = tuple((mask >> bit) & 1 for bit in range(width))
        mines = outside_mines + sum(local)
        if mines != position.mine_quota:
            continue
        if all(sum(local[index] for index in touch) == need for need, touch in reduced):
            found.append(local)
    return found


def _resample_block(
    position: _Position,
    bits: list[int],
    block: tuple[int, ...],
    rng: random.Random,
) -> None:
    """Draw the block from its full conditional and write it back.

    The proposal is that conditional, the cells outside the block are
    unchanged, and ``p(x_B | x_{-B}) p(x_{-B}) = p(x)``. The Metropolis–Hastings
    ratio is therefore 1, and the draw is always accepted.
    """
    legal = _legal_settings(position, bits, block)
    if not legal:
        raise RuntimeError("a legal layout has an empty block conditional")
    chosen = legal[rng.randrange(len(legal))]
    for index, bit in zip(block, chosen):
        bits[index] = bit


def _sweep(
    position: _Position,
    bits: list[int],
    rng: random.Random,
    self_loop_probability: float,
) -> None:
    """One sweep: one attempted local update per hidden cell.

    Each attempt stays put with the self-loop probability, and otherwise
    picks the pair kernel or the clue-neighbourhood kernel with equal
    probability. The self-loop is an identity kernel. Mixed with either
    Gibbs kernel, the uniform distribution on legal layouts stays stationary.
    The mixture does not establish irreducibility.
    """
    hidden_count = len(position.hidden)
    for _ in range(hidden_count):
        if rng.random() < self_loop_probability:
            continue
        if rng.random() < 0.5:
            _pair_update(position, bits, rng)
        else:
            _neighbourhood_update(position, bits, rng)


def _pair_update(position: _Position, bits: list[int], rng: random.Random) -> None:
    """Resample two hidden cells. This is the move that connects (1, 0) to (0, 1)."""
    if len(position.hidden) < 2:
        return
    first, second = rng.sample(range(len(position.hidden)), 2)
    _resample_block(position, bits, (first, second), rng)


def _neighbourhood_update(position: _Position, bits: list[int], rng: random.Random) -> None:
    """Resample the hidden neighbourhood of one revealed clue. Size at most 8."""
    if not position.neighbourhoods:
        return
    block = position.neighbourhoods[rng.randrange(len(position.neighbourhoods))]
    _resample_block(position, bits, block, rng)


def _regime_a(state: VisibleState, enumerated: QueryResult) -> MCMCResult:
    """Exact marginals from the finished weights. No samples and no burn-in."""
    probabilities, weights, total = _exact_marginals(state, enumerated)
    hidden = _hidden_cells(state)
    cell = _lowest(probabilities)
    status, action = _exact_status(cell, probabilities[cell])
    explanation = _explain_exact(cell, probabilities[cell], weights, total, len(enumerated.frontier), status)
    reasoning = Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=status,
        explanation=explanation,
        uncertain=status is DecisionStatus.ESTIMATED,
    )
    return MCMCResult(
        regime=REGIME_EXACT,
        frontier_size=len(enumerated.frontier),
        probabilities=probabilities,
        total_weight=total,
        frontier_weights=weights,
        retained_sweeps=0,
        discarded_sweeps=0,
        recorded_states=0,
        states_recorded_during_burn_in=0,
        initial_state_retained=False,
        kernels=(),
        action=action,
        reasoning=reasoning,
        hidden_cells=hidden,
        retained_states=None,
        initial_assignment=None,
    )


def _exact_marginals(
    state: VisibleState,
    enumerated: QueryResult,
) -> tuple[dict[Cell, Fraction], tuple[int, ...], int]:
    """Normalised binomial weights. Interior cells share ``r / |U_unfixed|``.

    A frontier assignment ``f`` extends in ``C(|U_unfixed|, r)`` layouts and
    in no layouts when the CLP search has already rejected the residual. The
    search only returns the accepted frontiers, so each weight is that
    binomial coefficient. Weighting frontiers equally would be a different
    distribution whenever two frontiers leave different residuals.
    """
    weights = tuple(math.comb(solution.unfixed_interior, solution.residual) for solution in enumerated.solutions)
    total = sum(weights)
    if total <= 0:
        _raise_unsatisfiable()
    domains = dict(enumerated.propagated_domains)
    frontier_index = {cell: index for index, cell in enumerate(enumerated.frontier)}
    interior = set(enumerated.interior)
    probabilities: dict[Cell, Fraction] = {}
    unfixed_interior: list[Cell] = []
    for cell in _hidden_cells(state):
        if cell in frontier_index:
            column = frontier_index[cell]
            mine_weight = sum(
                weight
                for solution, weight in zip(enumerated.solutions, weights)
                if solution.values[column] == 1
            )
            probabilities[cell] = Fraction(mine_weight, total)
            continue
        if cell not in interior:
            raise RuntimeError(f"hidden cell {cell} is neither frontier nor interior")
        domain = domains[cell]
        if domain == frozenset((0,)):
            probabilities[cell] = Fraction(0, 1)
        elif domain == frozenset((1,)):
            probabilities[cell] = Fraction(1, 1)
        elif domain == frozenset((0, 1)):
            unfixed_interior.append(cell)
        else:
            raise RuntimeError(f"domain of {cell} is not a subset of {{0, 1}}")
    if unfixed_interior:
        shared = Fraction(0, 1)
        for solution, weight in zip(enumerated.solutions, weights):
            if solution.unfixed_interior == 0:
                raise RuntimeError("an unfixed interior cell has no residual width")
            shared += Fraction(solution.residual, solution.unfixed_interior) * weight
        shared /= total
        for cell in unfixed_interior:
            probabilities[cell] = shared
    return probabilities, weights, total


def _regime_b(
    state: VisibleState,
    frontier_size: int,
    assignment: dict[Cell, int],
    rng: random.Random,
    burn_in_sweeps: int,
    retained_sweeps: int,
    self_loop_probability: float,
    keep_states: bool,
) -> MCMCResult:
    """Local kernels only. The start state is discarded with the burn-in."""
    position = _Position(state)
    bits = _bits_from_assignment(position, assignment)
    if not position.is_legal(bits):
        raise RuntimeError("the CLP initialiser returned a layout the clues reject")
    counts = [0 for _ in position.hidden]
    recorded_during_burn_in = 0
    for _ in range(burn_in_sweeps):
        _sweep(position, bits, rng, self_loop_probability)
    recorded = 0
    kept: list[tuple[int, ...]] = []
    for _ in range(retained_sweeps):
        _sweep(position, bits, rng, self_loop_probability)
        recorded += 1
        for index, bit in enumerate(bits):
            counts[index] += bit
        if keep_states:
            kept.append(tuple(bits))
    probabilities = {
        cell: Fraction(counts[position.index[cell]], recorded) for cell in position.hidden
    }
    cell = _lowest(probabilities)
    # A sample frequency of 0 is still a walk average. It is not a proof.
    action = Action.reveal(*cell)
    explanation = _explain_local(
        cell,
        probabilities[cell],
        frontier_size,
        retained_sweeps,
        burn_in_sweeps,
        self_loop_probability,
    )
    reasoning = Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=DecisionStatus.ESTIMATED,
        explanation=explanation,
        uncertain=True,
    )
    return MCMCResult(
        regime=REGIME_LOCAL,
        frontier_size=frontier_size,
        probabilities=probabilities,
        total_weight=None,
        frontier_weights=None,
        retained_sweeps=retained_sweeps,
        discarded_sweeps=burn_in_sweeps,
        recorded_states=recorded,
        states_recorded_during_burn_in=recorded_during_burn_in,
        initial_state_retained=False,
        kernels=LOCAL_KERNELS,
        action=action,
        reasoning=reasoning,
        hidden_cells=position.hidden,
        retained_states=tuple(kept) if keep_states else None,
        initial_assignment=dict(assignment),
    )


class _LegalStart:
    """Outcome of one capped search for a single legal layout.

    ``found`` means a frontier was accepted at or before the cap.
    ``unsatisfiable`` means the tree was exhausted, or propagation failed,
    and nothing was accepted. ``truncated`` means the next frontier attempt
    would have passed the cap. Truncation is not an empty solution set.
    """

    def __init__(self, assignment: dict[Cell, int] | None, status: str, nodes: int) -> None:
        self.assignment = assignment
        self.status = status
        self.nodes = nodes


def _first_legal_assignment(state: VisibleState, node_limit: int | None) -> _LegalStart:
    """One legal layout from the CLP frontier search, under ``node_limit``.

    Labeling uses the same variable order and value order as ``clp.query``:
    unfixed frontier cells, row-major, trying 0 and then 1. A frontier value
    attempt is one node, and a cell propagation has already fixed is not.
    ``None`` runs until the first accepted frontier or the tree is exhausted.

    The search stops at the first frontier the residual accepts. It does not
    continue, so this cap is not a second unlimited enumeration. The interior
    witness places those ``r`` mines on the row-major first unfixed interior
    cells. It is one placement, not a tour of ``C(|U|, r)``, and Regime B
    does not report it as a posterior draw.

    Propagation that fails before any attempt is ``unsatisfiable`` even when
    the cap is 0: no frontier node was required. Hitting the cap before an
    accept is ``truncated``.
    """
    domains, frontier, interior, constraints, watches = clp._build_model(state)
    if not clp.propagate(domains, constraints, watches):
        return _LegalStart(None, "unsatisfiable", 0)
    propagated = {cell: frozenset(values) for cell, values in domains.items()}
    order = tuple(cell for cell in frontier if len(domains[cell]) > 1)
    flexible = tuple(cell for cell in interior if propagated[cell] == frozenset((0, 1)))
    fixed_interior_mines = sum(1 for cell in interior if propagated[cell] == frozenset((1,)))
    found: dict[Cell, int] | None = None
    nodes = 0
    truncated = False

    def accept() -> dict[Cell, int] | None:
        frontier_mines = 0
        values: list[int] = []
        for cell in frontier:
            domain = domains[cell]
            if len(domain) != 1:
                raise RuntimeError(f"frontier cell {cell} was not ground")
            value = next(iter(domain))
            values.append(value)
            frontier_mines += value
        residual = state.mine_count - frontier_mines - fixed_interior_mines
        if residual < 0 or residual > len(flexible):
            return None
        assignment: dict[Cell, int] = {}
        for cell, value in zip(frontier, values):
            if _is_hidden(state, cell):
                assignment[cell] = value
        placed = 0
        for cell in flexible:
            bit = 1 if placed < residual else 0
            placed += bit
            assignment[cell] = bit
        for cell in interior:
            if cell in assignment or not _is_hidden(state, cell):
                continue
            domain = propagated[cell]
            if domain == frozenset((1,)):
                assignment[cell] = 1
            elif domain == frozenset((0,)):
                assignment[cell] = 0
            else:
                raise RuntimeError(f"interior cell {cell} was left unfixed")
        return assignment

    def search(index: int) -> None:
        nonlocal found, nodes, truncated
        if truncated or found is not None:
            return
        if index == len(order):
            found = accept()
            return
        cell = order[index]
        if not domains[cell]:
            return
        if len(domains[cell]) == 1:
            search(index + 1)
            return
        for value in (0, 1):
            if truncated or found is not None:
                return
            if value not in domains[cell]:
                continue
            nodes += 1
            if node_limit is not None and nodes > node_limit:
                truncated = True
                return
            saved = {item: set(values) for item, values in domains.items()}
            domains[cell] = {value}
            if clp.propagate(domains, constraints, watches):
                search(index + 1)
            domains.clear()
            domains.update({item: set(values) for item, values in saved.items()})

    search(0)
    if found is not None:
        return _LegalStart(found, "found", nodes)
    if truncated:
        return _LegalStart(None, "truncated", nodes)
    return _LegalStart(None, "unsatisfiable", nodes)


def _exact_status(cell: Cell, marginal: Fraction) -> tuple[DecisionStatus, Action]:
    """Marginal 0 is a reveal. Marginal 1 is a flag. Anything else is estimated."""
    if marginal == 0:
        return DecisionStatus.PROVED_SAFE, Action.reveal(*cell)
    if marginal == 1:
        return DecisionStatus.PROVED_MINE, Action.flag(*cell)
    return DecisionStatus.ESTIMATED, Action.reveal(*cell)


def _lowest(probabilities: Mapping[Cell, Fraction]) -> Cell:
    """Smallest mine probability, then smallest row, then smallest column."""
    if not probabilities:
        raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
    return min(probabilities, key=lambda cell: (probabilities[cell], cell[0], cell[1]))


def _explain_exact(
    cell: Cell,
    marginal: Fraction,
    weights: tuple[int, ...],
    total: int,
    frontier_size: int,
    status: DecisionStatus,
) -> str:
    listed = ", ".join(str(weight) for weight in weights)
    if status is DecisionStatus.PROVED_SAFE:
        proof = "Exact marginal 0 is proved safe. This is a finished count."
    elif status is DecisionStatus.PROVED_MINE:
        proof = "Exact marginal 1 is proved mine. This is a finished count."
    else:
        proof = "The exact marginal is neither 0 nor 1, so the reveal is estimated and uncertain."
    return (
        f"{EXACT_MARGINAL_TEXT}. |F| = {frontier_size}. "
        f"Binomial weights {listed}. Total weight {total}. "
        f"Exact marginal of {_fmt(cell)} is {marginal}. "
        f"No full assignment was drawn. {proof}"
    )


def _explain_local(
    cell: Cell,
    frequency: Fraction,
    frontier_size: int,
    retained_sweeps: int,
    discarded_sweeps: int,
    self_loop_probability: float,
) -> str:
    # The wording is the walk average only. It must not say the chain
    # converged, and it must not say the stationary distribution is unique.
    kernels = ", ".join(LOCAL_KERNELS)
    return (
        f"{LOCAL_ONLY_TEXT}. |F| = {frontier_size}. "
        f"Retained sweeps {retained_sweeps}. Discarded sweeps {discarded_sweeps}. "
        f"Kernels: {kernels}. Self-loop probability {self_loop_probability:.2f}. "
        f"Sample frequency of {_fmt(cell)} is {frequency}. "
        "Stochastic Gibbs sampling. This is a walk average. "
        "Each kernel satisfies detailed balance for the uniform distribution on the legal layouts, "
        "and that distribution is a stationary distribution of the walk. "
        "Status estimated."
    )


def _bits_from_assignment(position: _Position, assignment: Mapping[Cell, int]) -> list[int]:
    missing = [cell for cell in position.hidden if cell not in assignment]
    if missing:
        raise ValueError(f"assignment is missing hidden cells {missing}")
    return [assignment[cell] for cell in position.hidden]


def _hidden_cells(state: VisibleState) -> tuple[Cell, ...]:
    cells = []
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.HIDDEN:
                cells.append((row, column))
    return tuple(cells)


def _flagged_cells(state: VisibleState) -> tuple[Cell, ...]:
    cells = []
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.FLAGGED:
                cells.append((row, column))
    return tuple(cells)


def _is_hidden(state: VisibleState, cell: Cell) -> bool:
    return state.cells[cell[0]][cell[1]].state is CellState.HIDDEN


def _fmt(cell: Cell) -> str:
    return f"({cell[0]},{cell[1]})"


def _raise_unsatisfiable() -> None:
    raise UnsatisfiablePosition(
        "The visible clues and K have no legal layout. "
        "No cell is proved safe or proved mine, and no click is invented "
        "from an empty solution set."
    )


def _print_trace(title: str, state: VisibleState, agent: MCMCAgent) -> None:
    result = agent.analyse(state)
    print(title)
    print(f"  regime: {result.regime}")
    print(f"  action: {result.action.kind.value} {result.action.cell()}")
    print(f"  status: {result.reasoning.status.value} uncertain: {result.reasoning.uncertain}")
    print(f"  {result.reasoning.explanation}")
    for cell in result.hidden_cells:
        print(f"  P({_fmt(cell)}) = {result.probabilities[cell]}")


def main() -> None:
    """Two-cell traces, then one other fixture in both regimes, then a beginner game.

    The first reveal goes through the environment, which places the mines and
    excludes the clicked cell. The agent does not place them.
    Single-site Gibbs is not run here. The test is what shows that chain
    staying on its start state.
    """
    from minesweeper_ai.agents.base import ActionKind
    from minesweeper_ai.env.board import BEGINNER_COLS, BEGINNER_MINES, BEGINNER_ROWS, PUBLISHED_BEGINNER_SEED
    from minesweeper_ai.env.rules import Game, GameStatus
    from minesweeper_ai.env.view import VisibleCell

    def pattern(rows: int, cols: int, mine_count: int, revealed: dict[Cell, int]) -> VisibleState:
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

    two_cell = pattern(1, 3, 1, {(0, 1): 1})
    different_residuals = pattern(1, 7, 2, {(0, 1): 1, (0, 3): 1})
    _print_trace("Two-cell pattern, Regime A.", two_cell, MCMCAgent())
    _print_trace(
        "Two-cell pattern, Regime B, exact marginal enumeration switched off.",
        two_cell,
        MCMCAgent(exact_marginals=False),
    )
    _print_trace("Different residuals, Regime A. Weights are binomial, not equal.", different_residuals, MCMCAgent())
    _print_trace(
        "Different residuals, Regime B, stochastic Gibbs only.",
        different_residuals,
        MCMCAgent(exact_marginals=False),
    )

    game = Game(seed=PUBLISHED_BEGINNER_SEED)
    agent = MCMCAgent()
    print(
        f"Full beginner game: {BEGINNER_ROWS}x{BEGINNER_COLS}, {BEGINNER_MINES} mines, "
        f"seed {PUBLISHED_BEGINNER_SEED}."
    )
    step = 0
    while game.status is GameStatus.ONGOING and step < 200:
        action, reasoning = agent.act(game.visible_state())
        step += 1
        regime = agent.last_result.regime if agent.last_result is not None else "?"
        print(
            f"step {step}: {action.kind.value} {action.cell()} "
            f"regime={regime} status={reasoning.status.value} uncertain={reasoning.uncertain}"
        )
        print(f"  {reasoning.explanation}")
        if step == 1 and action.cell() != (0, 0):
            raise SystemExit(1)
        if action.kind is ActionKind.REVEAL:
            game.reveal(action.row, action.column)
            if reasoning.status is DecisionStatus.PROVED_SAFE and game.status is GameStatus.LOST:
                raise SystemExit(1)
        else:
            game.flag(action.row, action.column)
    print(f"game status: {game.status.value} steps: {step}")
    if game.status is GameStatus.ONGOING:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
