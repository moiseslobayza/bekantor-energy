"""One A* loop for scalar and lexicographic nonnegative additive objectives."""
from dataclasses import dataclass
from fractions import Fraction
import heapq
from itertools import count
import math
from time import perf_counter


class ScalarArithmetic:
    zero = 0.

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def better(a, b):
        return b is None or a < b - 1e-12  # Original scalar reference tolerance.

    @staticmethod
    def stale(a, b):
        return a > b + 1e-12

    @staticmethod
    def valid(value):
        return math.isfinite(value) and value >= 0


class LexicographicArithmetic:
    """Exact sums of the represented binary edge weights, no epsilon weighting."""
    zero = (Fraction(0), Fraction(0))

    @staticmethod
    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    @staticmethod
    def better(a, b):
        return b is None or a < b

    @staticmethod
    def stale(a, b):
        return a > b

    @staticmethod
    def valid(value):
        return isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, Fraction) and v >= 0 for v in value)


@dataclass(frozen=True)
class SearchResult:
    actions: tuple
    states: tuple
    value: object
    expanded_nodes: int
    generated_nodes: int
    elapsed_seconds: float


def astar_search(problem, initial_state, settings, limit_error, arithmetic=None):
    """Reopen improved labels. Finite acyclic time graph; first popped goal wins.

    Scalar comparisons retain the reference numerical tolerance. Tuple labels
    are ordered exactly and never traded via a finite scalar penalty.
    """
    ops = arithmetic or ScalarArithmetic()
    started, serial = perf_counter(), count()
    estimate = problem.heuristic(initial_state)
    if not ops.valid(estimate):
        raise ValueError("Heuristic must be finite, nonnegative and match objective units")
    frontier = [(estimate, -initial_state.step, next(serial), ops.zero, initial_state)]
    distance, parents = {initial_state: ops.zero}, {}
    expanded, generated = 0, 1
    while frontier:
        if expanded >= settings.max_expanded or perf_counter() - started > settings.max_seconds:
            raise limit_error("Search resource limit reached; optimality not established")
        _, _, _, cost, state = heapq.heappop(frontier)
        if ops.stale(cost, distance[state]):
            continue
        if problem.is_goal(state):
            actions, states = [], [state]
            while state != initial_state:
                previous, action = parents[state]
                actions.append(action)
                states.append(previous)
                state = previous
            return SearchResult(tuple(reversed(actions)), tuple(reversed(states)), cost,
                                expanded, generated, perf_counter() - started)
        expanded += 1
        for action, successor, edge in problem.successors(state):
            if not ops.valid(edge):
                raise ValueError("Search edge must be finite, nonnegative and match objective units")
            candidate = ops.add(cost, edge)
            if ops.better(candidate, distance.get(successor)):
                distance[successor] = candidate
                parents[successor] = (state, action)
                estimate = problem.heuristic(successor)
                if not ops.valid(estimate):
                    raise ValueError("Heuristic must be finite, nonnegative and match objective units")
                heapq.heappush(frontier, (ops.add(candidate, estimate), -successor.step,
                                         next(serial), candidate, successor))
                generated += 1
    raise ValueError("No feasible plan reaches the terminal requirement")
