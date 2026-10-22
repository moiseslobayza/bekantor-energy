# Fixtures sintéticos

Ningún archivo de esta carpeta contiene mediciones reales UCI ni generación real.
`uci_synthetic.txt` reproduce su esquema con 12 horas artificiales: falta el
timestamp 03:07; los minutos 01:10–01:11 y 06:00–06:11 tienen valores ausentes.
El primer hueco se puede rellenar causalmente; el largo debe dejar energía
horaria desconocida. El archivo PVGIS sólo reproduce el contrato mínimo v6 con
4 valores artificiales. Su campo `unit` corresponde a ángulos, no a potencia.
