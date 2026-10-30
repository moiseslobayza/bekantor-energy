from datetime import datetime, timedelta
import random
import unittest

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.logic import evaluate_action, forward_chain
from bekantor_energy.energy.planner import AStarPlanner, BatterySearchProblem, PlannerConfig
from bekantor_energy.energy.simulator import EPS, PhysicalSimulator


def inputs(demand, solar, prices, config):
    starts = tuple(datetime(2020, 1, 1) + timedelta(hours=i * config.step_hours) for i in range(len(demand)))
    return PlanningInputs(LoadForecast(starts[0], starts, tuple(demand), "fixture"),
                          tuple(solar), config, "logic_fixture")


class OriginalEdges(BatterySearchProblem):
    """Frozen pre-logic edge generator used only as a regression oracle."""

    def physical_successors(self, state):
        if state.step >= self.horizon:
            return []
        c, t = self.inputs.config, state.step
        successors = []
        for target in self.levels:
            delta = target - state.stored_kwh
            action = (EnergyAction(charge_kw=delta / (c.charge_efficiency * c.step_hours)) if delta > 0
                      else EnergyAction(discharge_kw=-delta * c.discharge_efficiency / c.step_hours))
            try:
                result = self.simulator.step(state, action, self.inputs.forecast.demand_kwh[t],
                                             self.inputs.solar_kwh[t], c)
            except ValueError:
                continue
            if abs(result.state.stored_kwh - target) > EPS or result.cost < -EPS:
                raise ArithmeticError("Invalid reference transition")
            successors.append((action, EnergyState(t + 1, target), result))
        return successors


class LogicTests(unittest.TestCase):
    def setUp(self):
        self.config = HomeConfig()
        self.simulator = PhysicalSimulator()

    def test_rules_and_two_stage_inference(self):
        cases = (
            (EnergyState(0, 1), EnergyAction(discharge_kw=1), 2, 0, "R1"),
            (EnergyState(0, 5), EnergyAction(charge_kw=1), 0, 1, "R2"),
            (EnergyState(0, 2), EnergyAction(charge_kw=3), 0, 0, "R3c"),
            (EnergyState(0, 2), EnergyAction(discharge_kw=3), 4, 0, "R3d"),
            (EnergyState(0, 2), EnergyAction(1, 1), 2, 0, "R4"),
            (EnergyState(0, 2), EnergyAction(discharge_kw=1), 0, 0, "R5"),
        )
        for state, action, demand, solar, expected in cases:
            with self.subTest(rule=expected):
                result = evaluate_action(state, action, demand, solar, self.config)
                names = [item.rule.name for item in result.fired]
                self.assertIn(expected, names)
                self.assertIn("R8", names)
                self.assertFalse(result.admissible)
                self.assertIn("discard_from_search", result.facts - result.initial_facts)

    def test_grid_unavailable_and_priority(self):
        result = evaluate_action(EnergyState(0, 2), EnergyAction(charge_kw=1), 0, 0,
                                 self.config, grid_available=False)
        self.assertIn("R6", [item.rule.name for item in result.fired])
        self.assertFalse(result.admissible)
        conflict = forward_chain({"action_admissible", "discharge_exceeds_deficit"})
        self.assertIn("action_forbidden", conflict.facts)
        self.assertFalse(conflict.admissible)  # Deny overrides a conflicting allow fact.

    def test_valid_boundary_actions_and_reproducibility(self):
        state = EnergyState(0, 2)
        action = EnergyAction(discharge_kw=.95)
        first = evaluate_action(state, action, 1, 0, self.config)
        self.assertTrue(first.admissible)
        self.assertEqual(first, evaluate_action(state, action, 1, 0, self.config))
        self.simulator.step(state, action, 1, 0, self.config)
        for state, action, demand, solar in (
            (EnergyState(0, 1), EnergyAction(), 0, 0),
            (EnergyState(0, 5), EnergyAction(), 0, 2),
            (EnergyState(0, 2), EnergyAction(charge_kw=2.5), 0, 0),
        ):
            self.assertTrue(evaluate_action(state, action, demand, solar, self.config).admissible)
            self.simulator.step(state, action, demand, solar, self.config)

    def test_filter_matches_physical_simulator_on_seeded_candidates(self):
        rng = random.Random(51)
        for _ in range(1000):
            c = HomeConfig(step_hours=rng.choice((.25, .5, 1.)),
                           charge_efficiency=rng.uniform(.6, 1),
                           discharge_efficiency=rng.uniform(.6, 1))
            state = EnergyState(0, rng.uniform(c.min_soc * c.battery_capacity_kwh, c.battery_capacity_kwh))
            action = EnergyAction(charge_kw=rng.uniform(0, 4)) if rng.randrange(2) else EnergyAction(discharge_kw=rng.uniform(0, 4))
            demand, solar, grid = rng.uniform(0, 5), rng.uniform(0, 5), bool(rng.randrange(2))
            allowed = evaluate_action(state, action, demand, solar, c, grid_available=grid).admissible
            try:
                self.simulator.step(state, action, demand, solar, c, grid_available=grid)
                physical = True
            except ValueError:
                physical = False
            self.assertEqual(allowed, physical)

    def test_astar_edges_and_plan_match_original(self):
        rng = random.Random(52)
        for _ in range(20):
            c = HomeConfig(horizon_steps=3, battery_capacity_kwh=2, min_soc=.25,
                           initial_soc=.5, max_charge_kw=1, max_discharge_kw=1,
                           charge_efficiency=.8, discharge_efficiency=.9,
                           import_tariff_per_kwh=(.1, .3, .2))
            data = inputs([rng.uniform(0, 2) for _ in range(3)],
                          [rng.uniform(0, 2) for _ in range(3)], c.import_prices, c)
            initial = EnergyState(0, 1.)
            settings = PlannerConfig(storage_step_kwh=.5)
            current = BatterySearchProblem(data, initial, settings)
            original = OriginalEdges(data, initial, settings)
            for t in range(3):
                for level in current.levels:
                    state = EnergyState(t, level)
                    self.assertEqual(current.physical_successors(state), original.physical_successors(state))
            planner = AStarPlanner(settings)
            new = planner.search(current, initial)
            old = planner.search(original, initial)
            self.assertEqual(new.actions, old.actions)
            self.assertEqual(new.states, old.states)
            self.assertEqual(new.predicted_cost, old.predicted_cost)


if __name__ == "__main__":
    unittest.main()
