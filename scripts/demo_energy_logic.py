"""Small CPU-only teaching example; uses synthetic demand, solar and prices."""
from datetime import datetime, timedelta

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.logic import RULES, evaluate_action
from bekantor_energy.energy.planner import AStarPlanner, BatterySearchProblem, PlannerConfig


def main():
    config = HomeConfig(battery_capacity_kwh=2, initial_soc=.25, min_soc=0,
                        charge_efficiency=1, discharge_efficiency=1,
                        max_charge_kw=1, max_discharge_kw=1, horizon_steps=2,
                        import_tariff_per_kwh=(.1, 1.))
    starts = (datetime(2026, 1, 1), datetime(2026, 1, 1) + timedelta(hours=1))
    forecast = LoadForecast(starts[0], starts, (0., 1.), "pronóstico_sintético")
    data = PlanningInputs(forecast, (0., 0.), config, "ejemplo_didáctico")
    initial = EnergyState(0, .5)
    settings = PlannerConfig(storage_step_kwh=.5)
    problem = BatterySearchProblem(data, initial, settings)
    print(f"Estado inicial: hora={initial.step}, batería={initial.stored_kwh:.2f} kWh, SOC={initial.stored_kwh / config.battery_capacity_kwh:.0%}")
    print("Demanda prevista: 0 y 1 kWh; solar: 0 kWh; tarifas simuladas: 0,10 y 1,00 por kWh")
    print(f"Reglas evaluadas: {', '.join(rule.name for rule in RULES)}")
    print("Candidatas de A* en la primera hora:")
    for target in problem.levels:
        delta = target - initial.stored_kwh
        action = EnergyAction(charge_kw=delta) if delta > 0 else EnergyAction(discharge_kw=-delta)
        reasoning = evaluate_action(initial, action, forecast.demand_kwh[0], data.solar_kwh[0], config)
        print(f"  batería objetivo {target:.2f} kWh: {'ACEPTADA' if reasoning.admissible else 'RECHAZADA'}")
        print(f"    hechos iniciales: {', '.join(sorted(reasoning.initial_facts))}")
        print(f"    hechos nuevos: {', '.join(sorted(reasoning.facts - reasoning.initial_facts)) or '(ninguno)'}")
        print(f"    reglas activadas: {', '.join(item.rule.name for item in reasoning.fired) or '(ninguna)'}")
        for item in reasoning.fired:
            print(f"      {item.rule.name}: {item.rule.explanation}")
    plan = AStarPlanner(settings).plan(data, initial)
    first = plan.actions[0]
    print(f"A* elige primero: cargar {first.charge_kw:.2f} kW, descargar {first.discharge_kw:.2f} kW")
    print("Plan completo: " + "; ".join(
        f"hora {i}: carga={action.charge_kw:.2f} kW, descarga={action.discharge_kw:.2f} kW"
        for i, action in enumerate(plan.actions)))
    print(f"Costo previsto del plan: {plan.predicted_cost:.2f} unidades simuladas")
    print("Las acciones se simulan; no controlan hardware real.")


if __name__ == "__main__":
    main()
