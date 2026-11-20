"""CLP Minesweeper agent: finite domains, delayed sums, labeling on the frontier.

Slide mapping (``12-AI-PROLOG.pdf``), written out so the steps can be walked:

| Slide idea | Where it is in this file |
| --- | --- |
| Finite-domain Boolean, 1 = mine and 0 = safe | ``domains``, each a subset of ``{0, 1}`` |
| Delayed ``#=`` posted before the variable is ground | ``SumConstraint`` stored and woken later |
| Bounds propagation to a fixed point | ``propagate`` / ``_revise`` |
| Empty domain, the fail port | ``_revise`` returns failure; the search in ``query`` backtracks |
| ``labeling(Vars)`` with a chosen order | the search in ``query``, frontier only, row-major, values 0 then 1 |
| The four ports call, exit, redo, fail | the trace strings in that search |
| Yes / No query | ``query`` returns solutions, or an unsatisfiable store |

The interior is not labeled. A finished frontier assignment is accepted when
the residual mine count ``r`` satisfies ``0 ≤ r ≤ |U_unfixed|``. That is the
existence check for an interior arrangement. The arrangements are not built.

A proof is one of: a domain reduced to one value, a finished failed labeling,
or a finished residual test. A search that hits ``node_limit`` is truncated
and proves nothing beyond domains that propagation had already reduced.
"""

from collections.abc import Iterable

from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.view import VisibleState

AGENT_NAME = "CLP"
Cell = tuple[int, int]


class UnsatisfiablePosition(Exception):
    """The visible clues and K have no legal layout.

    Section 4.4: the log records the failure. No click is invented from an
    empty solution set.
    """


class SumConstraint:
    """A delayed sum: the variables in ``cells`` add up to ``target``.

    Posted while those variables may still be ``{0, 1}``. Nothing is solved
    at construction time. ``propagate`` wakes the constraint when a domain
    shrinks. That is the delayed-constraint idea, not an instantiation fault.
    """

    def __init__(self, cells: tuple[Cell, ...], target: int) -> None:
        self.cells = cells
        self.target = target


class FrontierSolution:
    """One accepted frontier assignment.

    ``values`` is only the frontier, in row-major order. Interior cells are
    not assigned here. ``residual`` is how many mines the unfixed interior
    must still hold, and ``unfixed_interior`` is how many such cells there
    are. The assignment is legal exactly when
    ``0 ≤ residual ≤ unfixed_interior``.
    """

    def __init__(self, values: tuple[int, ...], residual: int, unfixed_interior: int) -> None:
        self.values = values
        self.residual = residual
        self.unfixed_interior = unfixed_interior

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrontierSolution):
            return NotImplemented
        return (
            self.values == other.values
            and self.residual == other.residual
            and self.unfixed_interior == other.unfixed_interior
        )

    def __repr__(self) -> str:
        return (
            f"FrontierSolution(values={self.values}, residual={self.residual}, "
            f"unfixed_interior={self.unfixed_interior})"
        )


class QueryResult:
    """The answer to one CLP query on a visible grid."""

    def __init__(
        self,
        frontier: tuple[Cell, ...],
        interior: tuple[Cell, ...],
        labeling_variables: tuple[Cell, ...],
        attempted_cells: tuple[Cell, ...],
        propagated_domains: tuple[tuple[Cell, frozenset[int]], ...],
        solutions: tuple[FrontierSolution, ...],
        nodes: int,
        finished: bool,
        unsatisfiable: bool,
        proved_safe: frozenset[Cell],
        proved_mine: frozenset[Cell],
        trace: tuple[str, ...],
    ) -> None:
        self.frontier = frontier
        self.interior = interior
        self.labeling_variables = labeling_variables
        self.attempted_cells = attempted_cells
        self.propagated_domains = propagated_domains
        self.solutions = solutions
        self.nodes = nodes
        self.finished = finished
        self.unsatisfiable = unsatisfiable
        self.proved_safe = proved_safe
        self.proved_mine = proved_mine
        self.trace = trace

    def domain_of(self, cell: Cell) -> frozenset[int]:
        for found, domain in self.propagated_domains:
            if found == cell:
                return domain
        raise KeyError(cell)


def query(state: VisibleState, node_limit: int | None = None) -> QueryResult:
    """Ask for the frontier assignments consistent with the visible clues and K.

    ``node_limit`` counts frontier value attempts only. ``None`` runs until
    the tree is exhausted. A hit limit sets ``finished`` false and drops any
    partial solutions, so they cannot be read as a complete answer.
    """
    domains, frontier, interior, constraints, watches = _build_model(state)
    trace: list[str] = []
    if not propagate(domains, constraints, watches):
        trace.append("fail propagation")
        return _result(
            state_frontier=frontier,
            interior=interior,
            labeling_variables=(),
            attempted=(),
            domains=domains,
            solutions=(),
            nodes=0,
            finished=True,
            unsatisfiable=True,
            proved_safe=frozenset(),
            proved_mine=frozenset(),
            trace=trace,
        )

    propagated = _freeze(domains)
    # Labeling variables are the unfixed frontier, row-major. Interior cells
    # are not in this list. Cells propagation has already grounded are not
    # choice points; their single value is already in the domain.
    order = tuple(cell for cell in frontier if len(domains[cell]) > 1)
    flexible_interior = tuple(cell for cell in interior if propagated[cell] == frozenset((0, 1)))
    interior_fixed_mines = sum(1 for cell in interior if propagated[cell] == frozenset((1,)))

    solutions: list[FrontierSolution] = []
    attempted: list[Cell] = []
    nodes = 0
    truncated = False

    def search(index: int) -> None:
        nonlocal nodes, truncated
        if truncated:
            return
        if index == len(order):
            accepted = _accept_frontier(
                domains,
                frontier,
                state.mine_count,
                interior_fixed_mines,
                len(flexible_interior),
                trace,
            )
            if accepted is not None:
                solutions.append(accepted)
                trace.append("exit solution")
            return
        cell = order[index]
        current = domains[cell]
        if not current:
            trace.append(f"fail {cell[0]},{cell[1]} empty domain")
            return
        if len(current) == 1:
            # Propagation grounded this frontier variable. Labeling does not
            # branch, and this is not a frontier node.
            only = next(iter(current))
            trace.append(f"exit {cell[0]},{cell[1]}={only} propagation")
            search(index + 1)
            return
        for value in (0, 1):
            if value not in domains[cell]:
                continue
            nodes += 1
            if node_limit is not None and nodes > node_limit:
                truncated = True
                trace.append("fail node limit")
                return
            if cell not in attempted:
                attempted.append(cell)
            saved = _copy_domains(domains)
            # call: try this value, then propagate before going deeper.
            trace.append(f"call {cell[0]},{cell[1]}={value}")
            domains[cell] = {value}
            if not propagate(domains, constraints, watches):
                trace.append(f"fail {cell[0]},{cell[1]}={value}")
                _restore(domains, saved)
                trace.append(f"redo {cell[0]},{cell[1]}")
                continue
            trace.append(f"exit {cell[0]},{cell[1]}={value}")
            search(index + 1)
            _restore(domains, saved)
            if truncated:
                return
            trace.append(f"redo {cell[0]},{cell[1]}")

    search(0)
    finished = not truncated
    # A partial solution list is not the answer to the query.
    if not finished:
        solutions = []
    unsatisfiable = finished and not solutions
    proved_safe, proved_mine = _proofs(
        propagated,
        frontier,
        flexible_interior,
        solutions,
        finished,
        unsatisfiable,
    )
    return _result(
        state_frontier=frontier,
        interior=interior,
        labeling_variables=order,
        attempted=tuple(attempted),
        domains=propagated,
        solutions=tuple(solutions),
        nodes=nodes,
        finished=finished,
        unsatisfiable=unsatisfiable,
        proved_safe=proved_safe,
        proved_mine=proved_mine,
        trace=trace,
    )


def propagate(
    domains: dict[Cell, set[int]],
    constraints: list[SumConstraint],
    watches: dict[Cell, tuple[int, ...]],
) -> bool:
    """Wake delayed sums until no domain changes, or some constraint fails.

    Failure means a sum cannot meet its target inside the current bounds,
    which empties the feasible region. Callers treat that as the fail port.
    """
    queue = list(range(len(constraints)))
    queued = set(queue)
    while queue:
        index = queue.pop()
        queued.discard(index)
        ok, changed = _revise(constraints[index], domains)
        if not ok:
            return False
        for cell in changed:
            for other in watches[cell]:
                if other not in queued:
                    queued.add(other)
                    queue.append(other)
    return True


def _revise(constraint: SumConstraint, domains: dict[Cell, set[int]]) -> tuple[bool, list[Cell]]:
    """Bounds consistency for ``sum(cells) = target`` with each variable in ``{0, 1}``.

    ``lb`` is the sum of lower bounds and ``ub`` the sum of upper bounds.

    - ``lb > target`` or ``ub < target``: the constraint fails.
    - ``lb == target``: every unfixed variable is set to its lower bound.
    - ``ub == target``: every unfixed variable is set to its upper bound.
    """
    lower = 0
    upper = 0
    unfixed: list[Cell] = []
    for cell in constraint.cells:
        domain = domains[cell]
        if not domain:
            return False, []
        lo = min(domain)
        hi = max(domain)
        lower += lo
        upper += hi
        if lo != hi:
            unfixed.append(cell)
    target = constraint.target
    if lower > target or upper < target:
        return False, []
    if not unfixed:
        return True, []
    changed: list[Cell] = []
    if lower == target:
        for cell in unfixed:
            lo = min(domains[cell])
            domains[cell] = {lo}
            changed.append(cell)
    elif upper == target:
        for cell in unfixed:
            hi = max(domains[cell])
            domains[cell] = {hi}
            changed.append(cell)
    return True, changed


def _accept_frontier(
    domains: dict[Cell, set[int]],
    frontier: tuple[Cell, ...],
    mine_count: int,
    interior_fixed_mines: int,
    unfixed_interior: int,
    trace: list[str],
) -> FrontierSolution | None:
    """Accept a fully assigned frontier when ``0 ≤ r ≤ |U_unfixed|``.

    ``r`` is K minus the mines already fixed on the frontier and on interior
    cells whose domains were singletons before labeling. It is the number of
    mines the unfixed interior must still hold. Checking the inequality is
    the whole interior test. ``C(|U_unfixed|, r)`` is not computed, and no
    interior cell is given a value by this function.

    Cells the global sum has already forced are reflected in ``r``: forcing
    every unfixed interior cell to 0 is ``r = 0``, and forcing every one of
    them to 1 is ``r = |U_unfixed|``.
    """
    values: list[int] = []
    frontier_mines = 0
    for cell in frontier:
        domain = domains[cell]
        if len(domain) != 1:
            raise RuntimeError(f"frontier cell {cell} was labeled before it was ground")
        value = next(iter(domain))
        values.append(value)
        frontier_mines += value
    residual = mine_count - frontier_mines - interior_fixed_mines
    if residual < 0 or residual > unfixed_interior:
        trace.append(f"fail residual {residual} outside 0..{unfixed_interior}")
        return None
    return FrontierSolution(tuple(values), residual, unfixed_interior)


def _proofs(
    propagated: dict[Cell, frozenset[int]],
    frontier: tuple[Cell, ...],
    flexible_interior: tuple[Cell, ...],
    solutions: list[FrontierSolution],
    finished: bool,
    unsatisfiable: bool,
) -> tuple[frozenset[Cell], frozenset[Cell]]:
    """Proofs from a reduced domain, a finished labeling, or a finished residual.

    A truncated search contributes only the domains propagation had already
    reduced to one value. Those reductions do not depend on the unfinished
    tree. An unsatisfiable store proves nothing: there is no layout to be
    safe in, and no click is read off the empty set.
    """
    if unsatisfiable:
        return frozenset(), frozenset()
    safe: set[Cell] = set()
    mine: set[Cell] = set()
    for cell, domain in propagated.items():
        if domain == frozenset((0,)):
            safe.add(cell)
        elif domain == frozenset((1,)):
            mine.add(cell)
    if not finished:
        return frozenset(safe), frozenset(mine)
    frontier_index = {cell: index for index, cell in enumerate(frontier)}
    for cell in frontier:
        if cell in safe or cell in mine:
            continue
        index = frontier_index[cell]
        seen = {solution.values[index] for solution in solutions}
        if seen == {0}:
            safe.add(cell)
        elif seen == {1}:
            mine.add(cell)
    if solutions and flexible_interior:
        if all(solution.residual == 0 for solution in solutions):
            safe.update(flexible_interior)
        elif all(solution.residual == solution.unfixed_interior for solution in solutions):
            mine.update(flexible_interior)
    return frozenset(safe), frozenset(mine)


class CLPAgent:
    """Play from ``VisibleState`` only. The hidden mine set is not an argument.

    Decision order, from section 4.4:

    - If any cell is proved safe, reveal the row-major first. Status
      ``proved_safe``.
    - Otherwise reveal the row-major first frontier cell that is not a proved
      mine. Status stays uncertain (``fallback``).
    - Otherwise flag the row-major first proved mine. A proved mine is flagged,
      not revealed.
    - If the frontier has no remaining reveal, reveal the row-major first
      interior cell that is not a proved mine. Uncertain.

    Flagged cells are not reveal candidates. They already store a mine proof,
    so once every frontier cell is flagged the next uncertain reveal is an
    interior cell, which is the plan's empty-frontier case for what is left
    to play. An uncertain reveal is never played as if one accepted frontier
    assignment were the hidden board.
    """

    def __init__(self, node_limit: int | None = None) -> None:
        self.node_limit = node_limit

    def analyse(self, state: VisibleState) -> QueryResult:
        """One query, including every proved safe cell and every proved mine.

        ``act`` decides from this result. A caller that needs the other proved
        cells, not only the one returned action, uses this instead of ``act``.
        """
        return query(state, self.node_limit)

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        """Play one move. The decision order above is unchanged."""
        return self.decide(state, self.analyse(state))

    def decide(self, state: VisibleState, result: QueryResult) -> tuple[Action, Reasoning]:
        """Choose the move from a query that has already been run."""
        if result.unsatisfiable:
            raise UnsatisfiablePosition(
                "The constraint store has no satisfying assignment. "
                "No cell is proved safe or proved mine, and no click is invented "
                "from an empty solution set."
            )
        flagged = _flagged_cells(state)
        safe_cell = _first(cell for cell in _row_major(result.proved_safe) if cell not in flagged)
        if safe_cell is not None:
            return _move(state, result, Action.reveal(*safe_cell), DecisionStatus.PROVED_SAFE, safe_cell)
        frontier_guess = _first(
            cell
            for cell in result.frontier
            if cell not in result.proved_mine and cell not in flagged
        )
        if frontier_guess is not None:
            return _move(state, result, Action.reveal(*frontier_guess), DecisionStatus.FALLBACK, frontier_guess)
        mine_cell = _first(cell for cell in _row_major(result.proved_mine) if cell not in flagged)
        if mine_cell is not None:
            return _move(state, result, Action.flag(*mine_cell), DecisionStatus.PROVED_MINE, mine_cell)
        interior_guess = _first(
            cell
            for cell in result.interior
            if cell not in result.proved_mine and cell not in flagged
        )
        if interior_guess is not None:
            return _move(state, result, Action.reveal(*interior_guess), DecisionStatus.FALLBACK, interior_guess)
        raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")

    def proof(self, state: VisibleState, result: QueryResult, cell: Cell) -> tuple[Action, Reasoning]:
        """Reveal one proved safe cell, or flag one proved mine.

        ``decide`` can return a fallback while other cells stay proved. This
        returns the proof for one of those cells and does not run a query.
        The explanation is the same text ``decide`` uses for that proof.
        """
        if cell in result.proved_safe:
            return _move(state, result, Action.reveal(*cell), DecisionStatus.PROVED_SAFE, cell)
        if cell in result.proved_mine:
            return _move(state, result, Action.flag(*cell), DecisionStatus.PROVED_MINE, cell)
        raise ValueError(f"{cell} is not proved safe or proved mine")


def _move(
    state: VisibleState,
    result: QueryResult,
    action: Action,
    status: DecisionStatus,
    cell: Cell,
) -> tuple[Action, Reasoning]:
    uncertain = status is DecisionStatus.FALLBACK
    return action, Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=status,
        explanation=_explain(state, result, cell, status),
        uncertain=uncertain,
    )


def _explain(state: VisibleState, result: QueryResult, cell: Cell, status: DecisionStatus) -> str:
    domains = _format_domains(result.propagated_domains)
    kind = _proof_kind(result, cell)
    if status is DecisionStatus.PROVED_SAFE and kind == "domain":
        reason = (
            f"Bounds propagation reduced the domain of {_fmt(cell)} to {{0}}. "
            "A singleton domain is a finished proof that the cell is safe."
        )
    elif status is DecisionStatus.PROVED_MINE and kind == "domain":
        reason = (
            f"Bounds propagation reduced the domain of {_fmt(cell)} to {{1}}. "
            "A singleton domain is a finished proof that the cell is a mine."
        )
    elif status is DecisionStatus.PROVED_SAFE and kind == "residual":
        reason = (
            f"Every accepted frontier assignment leaves residual r = 0 on the unfixed interior. "
            f"{_fmt(cell)} is safe in every extension. No interior arrangement was built."
        )
    elif status is DecisionStatus.PROVED_MINE and kind == "residual":
        reason = (
            "Every accepted frontier assignment leaves residual r = |U_unfixed|. "
            f"{_fmt(cell)} is a mine in every extension. No interior arrangement was built."
        )
    elif status is DecisionStatus.PROVED_SAFE:
        reason = (
            f"Finished labeling of the frontier with {_fmt(cell)} fixed to 1 accepts nothing (fail). "
            "The cell is safe. Variable order is row-major on the frontier; value order is 0 then 1."
        )
    elif status is DecisionStatus.PROVED_MINE:
        reason = (
            f"Finished labeling of the frontier with {_fmt(cell)} fixed to 0 accepts nothing (fail). "
            "The cell is a mine. Variable order is row-major on the frontier; value order is 0 then 1."
        )
    elif not result.finished:
        reason = (
            f"Labeling hit the node limit after {result.nodes} frontier nodes (fail). "
            "A truncated search is not a proof, so this reveal stays uncertain."
        )
    elif not result.frontier:
        if result.solutions:
            residual = result.solutions[0].residual
            unfixed = result.solutions[0].unfixed_interior
        else:
            residual = state.mine_count
            unfixed = len(result.interior)
        reason = (
            "No revealed clue, so the frontier is empty and labeling does not enter the interior. "
            f"The global sum leaves residual r = {residual} on |U_unfixed| = {unfixed}. "
            "That is not r = 0 and not r = |U_unfixed|, so no interior cell is proved. "
            "The row-major first interior cell is revealed, and the reveal is uncertain."
        )
    else:
        reason = (
            f"Finished labeling accepts {len(result.solutions)} frontier assignments, "
            f"and {_fmt(cell)} is not the same value in all of them. "
            "Both values remain possible, so this reveal is not a proof. "
            "One accepted assignment is not played as the hidden board. "
            "Ports used on the frontier: call, exit, redo, fail."
        )
    return (
        f"{reason} Frontier size {len(result.frontier)}. "
        f"Labeling nodes {result.nodes}. Finished {result.finished}. "
        f"Domains after propagation, before labeling: {domains}"
    )


def _proof_kind(result: QueryResult, cell: Cell) -> str:
    domain = result.domain_of(cell) if any(found == cell for found, _domain in result.propagated_domains) else None
    if domain is not None and len(domain) == 1:
        return "domain"
    if cell in result.interior:
        return "residual"
    return "labeling"


def _build_model(
    state: VisibleState,
) -> tuple[dict[Cell, set[int]], tuple[Cell, ...], tuple[Cell, ...], list[SumConstraint], dict[Cell, tuple[int, ...]]]:
    """Variables for hidden cells, a delayed sum per clue, and the global sum K.

    Revealed cells are the constant 0 and are not variables. A flag is the
    assignment 1, stored because a previous proof flagged that cell. It is
    not an extra constraint form. ``mines_unaccounted`` is not subtracted
    again: flagged cells already contribute 1 inside the sum that equals K.
    """
    domains: dict[Cell, set[int]] = {}
    for row in range(state.height):
        for column in range(state.width):
            cell_state = state.cells[row][column]
            if cell_state.state is CellState.REVEALED:
                continue
            if cell_state.state is CellState.FLAGGED:
                domains[(row, column)] = {1}
            else:
                domains[(row, column)] = {0, 1}

    constraints: list[SumConstraint] = []
    for row in range(state.height):
        for column in range(state.width):
            visible = state.cells[row][column]
            if visible.state is not CellState.REVEALED:
                continue
            if visible.number is None:
                raise ValueError(f"revealed cell {(row, column)} has no clue")
            hidden: list[Cell] = []
            for neighbour in _neighbours(state.height, state.width, row, column):
                if neighbour in domains:
                    hidden.append(neighbour)
            if not hidden and visible.number == 0:
                continue
            constraints.append(SumConstraint(tuple(hidden), visible.number))

    in_clue: set[Cell] = set()
    for constraint in constraints:
        in_clue.update(constraint.cells)
    frontier = tuple(sorted(cell for cell in domains if cell in in_clue))
    interior = tuple(sorted(cell for cell in domains if cell not in in_clue))
    constraints.append(SumConstraint(tuple(sorted(domains)), state.mine_count))

    watches: dict[Cell, list[int]] = {cell: [] for cell in domains}
    for index, constraint in enumerate(constraints):
        for cell in constraint.cells:
            watches[cell].append(index)
    frozen_watches = {cell: tuple(indexes) for cell, indexes in watches.items()}
    return domains, frontier, interior, constraints, frozen_watches


def _neighbours(height: int, width: int, row: int, column: int) -> list[Cell]:
    found: list[Cell] = []
    for d_row in (-1, 0, 1):
        for d_column in (-1, 0, 1):
            if d_row == 0 and d_column == 0:
                continue
            next_row = row + d_row
            next_column = column + d_column
            if 0 <= next_row < height and 0 <= next_column < width:
                found.append((next_row, next_column))
    return found


def _result(
    state_frontier: tuple[Cell, ...],
    interior: tuple[Cell, ...],
    labeling_variables: tuple[Cell, ...],
    attempted: tuple[Cell, ...],
    domains: dict[Cell, set[int]] | dict[Cell, frozenset[int]],
    solutions: tuple[FrontierSolution, ...],
    nodes: int,
    finished: bool,
    unsatisfiable: bool,
    proved_safe: frozenset[Cell],
    proved_mine: frozenset[Cell],
    trace: list[str],
) -> QueryResult:
    frozen = tuple(sorted((cell, frozenset(values)) for cell, values in domains.items()))
    return QueryResult(
        frontier=state_frontier,
        interior=interior,
        labeling_variables=labeling_variables,
        attempted_cells=attempted,
        propagated_domains=frozen,
        solutions=solutions,
        nodes=nodes,
        finished=finished,
        unsatisfiable=unsatisfiable,
        proved_safe=proved_safe,
        proved_mine=proved_mine,
        trace=tuple(trace),
    )


def _freeze(domains: dict[Cell, set[int]]) -> dict[Cell, frozenset[int]]:
    return {cell: frozenset(values) for cell, values in domains.items()}


def _copy_domains(domains: dict[Cell, set[int]]) -> dict[Cell, set[int]]:
    return {cell: set(values) for cell, values in domains.items()}


def _restore(domains: dict[Cell, set[int]], saved: dict[Cell, set[int]]) -> None:
    domains.clear()
    for cell, values in saved.items():
        domains[cell] = set(values)


def _flagged_cells(state: VisibleState) -> set[Cell]:
    flagged: set[Cell] = set()
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.FLAGGED:
                flagged.add((row, column))
    return flagged


def _row_major(cells: frozenset[Cell]) -> tuple[Cell, ...]:
    return tuple(sorted(cells))


def _first(cells: Iterable[Cell]) -> Cell | None:
    return next(iter(cells), None)


def _format_domains(domains: tuple[tuple[Cell, frozenset[int]], ...]) -> str:
    parts = []
    for (row, column), domain in domains:
        values = ",".join(str(value) for value in sorted(domain))
        parts.append(f"({row},{column}):{{{values}}}")
    return " ".join(parts)


def _fmt(cell: Cell) -> str:
    return f"({cell[0]},{cell[1]})"


def _print_patterns() -> None:
    """The two hand patterns, then one frontier whose interior is only a residual."""
    from minesweeper_ai.env.view import VisibleCell

    def show(title: str, state: VisibleState) -> None:
        result = query(state)
        print(title)
        print(f"  propagated domains: {_format_domains(result.propagated_domains)}")
        print(f"  labeling variables: {result.labeling_variables}")
        print(f"  attempted cells: {result.attempted_cells}")
        print(f"  interior: {result.interior}")
        print(f"  nodes: {result.nodes} finished: {result.finished} unsatisfiable: {result.unsatisfiable}")
        for solution in result.solutions:
            print(
                f"  solution values={solution.values} residual={solution.residual} "
                f"|U_unfixed|={solution.unfixed_interior}"
            )
        print(f"  proved safe: {sorted(result.proved_safe)}")
        print(f"  proved mine: {sorted(result.proved_mine)}")
        ports = [line for line in result.trace if line.split(" ", 1)[0] in {"call", "exit", "redo", "fail"}]
        print(f"  ports: {' | '.join(ports[:12])}")

    single = _pattern(
        VisibleCell,
        rows=1,
        cols=2,
        mine_count=1,
        revealed={(0, 0): 1},
    )
    show("Pattern: clue 1 with one hidden neighbour. Propagation sets that domain to {1}.", single)

    pair = _pattern(
        VisibleCell,
        rows=1,
        cols=3,
        mine_count=1,
        revealed={(0, 1): 1},
    )
    show(
        "Pattern: A + B = 1. Propagation forces neither cell. Exhausted labeling reports both solutions.",
        pair,
    )

    wide = _pattern(
        VisibleCell,
        rows=2,
        cols=6,
        mine_count=2,
        revealed={(0, 0): 1, (1, 0): 1},
    )
    show(
        "Pattern: same A + B = 1 with a large interior. "
        "Accepted frontiers carry residual r, and labeling never assigns the interior.",
        wide,
    )


def _pattern(
    visible_cell: type,
    rows: int,
    cols: int,
    mine_count: int,
    revealed: dict[Cell, int],
) -> VisibleState:
    grid = []
    for row in range(rows):
        line = []
        for column in range(cols):
            if (row, column) in revealed:
                line.append(visible_cell(state=CellState.REVEALED, number=revealed[(row, column)]))
            else:
                line.append(visible_cell(state=CellState.HIDDEN, number=None))
        grid.append(tuple(line))
    return VisibleState(
        width=cols,
        height=rows,
        mine_count=mine_count,
        mines_unaccounted=mine_count,
        cells=tuple(grid),
    )


def main() -> None:
    """Beginner demonstration: the hand patterns, then one full 9×9 game.

    The agent is given ``visible_state()`` only. The first reveal is safe
    because the environment places the mines, not because the agent looked.
    """
    from minesweeper_ai.env.board import BEGINNER_MINES, BEGINNER_COLS, BEGINNER_ROWS, PUBLISHED_BEGINNER_SEED
    from minesweeper_ai.env.rules import Game, GameStatus

    _print_patterns()
    game = Game(seed=PUBLISHED_BEGINNER_SEED)
    agent = CLPAgent()
    print(
        f"Full beginner game: {BEGINNER_ROWS}x{BEGINNER_COLS}, {BEGINNER_MINES} mines, "
        f"seed {PUBLISHED_BEGINNER_SEED}."
    )
    step = 0
    while game.status is GameStatus.ONGOING and step < 200:
        state = game.visible_state()
        action, reasoning = agent.act(state)
        step += 1
        print(
            f"step {step}: {action.kind.value} {action.cell()} "
            f"status={reasoning.status.value} uncertain={reasoning.uncertain}"
        )
        print(f"  {reasoning.explanation}")
        if action.kind.value == "reveal":
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
