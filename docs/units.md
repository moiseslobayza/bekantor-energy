# Unidades y convenciones físicas

| Magnitud | Unidad interna | Significado |
|---|---|---|
| Potencia | kW | Tasa de transferencia de energía; 1 kW = 1000 W |
| Energía de un intervalo | kWh | Integral de potencia sobre horas; 1 kWh = 1000 Wh |
| Potencia pico solar | kWp | Potencia nominal fotovoltaica en condiciones de referencia |
| Capacidad de batería | kWh | Energía nominal almacenable, no potencia máxima |
| Estado de carga SOC | fracción [0,1] | Energía almacenada / capacidad nominal |
| Eficiencia | fracción (0,1] | Energía útil / energía de entrada |
| Tarifa | SIM/kWh por defecto | Moneda simulada explícita, no tarifa comercial vigente |

Para potencia constante: `E_kWh = P_kW * dt_h`. Para potencia media por minuto:
`E_h = sum(P_min_kW) / 60`, sólo si hay 60 minutos disponibles. Una potencia
de 2 kW durante 30 minutos consume 1 kWh; durante 60 minutos consume 2 kWh.
No se debe sumar kW horarios y llamarlos kWh sin explicitar duración.

UCI `Global_active_power` ya está en kW: **no dividir otra vez por 1000**.
Los `Sub_metering_1/2/3` son Wh de cada minuto: sumar y dividir por 1000.
No se suman al consumo global: son componentes del total, no demanda adicional.
La potencia reactiva publicada por UCI se conserva con su nombre original; no
se usa como energía activa ni como objetivo de predicción.

Las horas se etiquetan por su inicio, intervalo `[t, t+1h)`. Se conservan las
etiquetas del reloj original UCI sin inventar una zona horaria. No se consideran
UTC ni se localizan automáticamente a Europe/Paris, debido a ambigüedad de DST.
Se rechazan etiquetas duplicadas para revisión explícita.

PVGIS v6 entrega el vector `power` en W. `peak_power_unit` es kWp y `unit` en la
respuesta v6 describe **ángulos** (por ejemplo `radians`), no el vector de potencia.
Se convierte `pv_kw = power / 1000`. La energía solar horaria es una aproximación
rectangular: `pv_kwh_estimate = pv_kw * 1h`, con timestamps UTC verificados. No se
afirma que sea la integral medida de una hora de generación física. No se vuelve
a aplicar `pv_system_efficiency` después de la respuesta: PVGIS ya la incluyó.

Convención de batería en el bus AC, implementada por `PhysicalSimulator` en Fase 3:

```text
stored_next = stored + eta_charge * charge_kw * dt_h
                     - discharge_kw * dt_h / eta_discharge
grid_import + solar + discharge = demand + charge + grid_export + curtailment
min_soc * capacity <= stored_next <= capacity
0 <= charge_kw <= max_charge_kw
0 <= discharge_kw <= max_discharge_kw
```

`charge` y `discharge` son potencia en el bus; prohibir ambas positivas a la vez.
También se evitará importar y exportar simultáneamente. La reserva es fracción
de capacidad nominal, no una eficiencia adicional. Si ambas eficiencias son
0.95, la de ciclo es `0.95 * 0.95 = 0.9025`; no 0.95 ni 1.90.
El simulador aplica estas ecuaciones y registra las pérdidas de conversión,
energía no suministrada y residuos de balance. En cortes se sustituye `demand`
por `served = demand - unserved` en el balance AC; la reserva sigue siendo dura.
El balance total, incluyendo energía almacenada y pérdidas, también se verifica.
La demostración A* deshabilita exportación y toda remuneración de excedentes.
