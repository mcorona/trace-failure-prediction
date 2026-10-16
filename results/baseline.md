# Baseline: detección de fallas en trazas de agentes (Who&When, Algorithm-Generated)

## Datos
- 126 trazas, todas fallidas (`is_correct = False` en todas). No hay clase negativa nativa.
- Etiqueta por prefijo: `y = 1` si `mistake_step <= t-1`. 1099 filas, 66 % positivas.
- `mistake_step` es base 0 (verificado: 0 fuera de rango; 20 fuera si fuera base 1).
- 3 trazas (gid 95, 106, 109): el agente marcado no coincide con `history[s].name`.
  El paso `s` suele ser la consecuencia visible (salida de terminal, verificación),
  y la acción causal está en `s-1` o antes. Se conservan tal cual.

## Evaluación
- GroupKFold(5) por traza (`gid`).
- AUROC global sobre OOF.
- AUROC **dentro de (fold, t)**: elimina el efecto de posición. El control trivial debe dar 0.500.
- Bootstrap por traza (300 réplicas) para IC95%.
- Celdas con al menos 10 positivos y 10 negativos.

## Resultados

| Modelo | AUROC global (OOF) | AUROC dentro de (fold, t) | IC95% |
|---|---|---|---|
| Control `n_msgs` | 0.798 | 0.500 | [0.500, 0.500] |
| LogReg (7 features) | 0.795 | 0.590 | [0.435, 0.728] |
| GBM (7 features) | 0.793 | 0.544 | [0.419, 0.688] |

Margen GBM − control: IC95% [-0.081, 0.188] (cruza 0).

## Conclusión
El AUROC global (~0.80) se explica por el avance de la traza. Con las 7 features actuales
no hay evidencia de señal más allá de la posición. Hallazgo negativo, reportado tal cual.

## Limitaciones
- 126 trazas y 13 celdas válidas: intervalos anchos.
- Sin trazas exitosas: no es predicción real, es detección del error ya visible.
- Evaluación limitada a t ≤ 8 (los bins tardíos casi no tienen negativos).

## Reproducir
uv run python build_prefixes.py
uv run python baseline.py
uv run python within_fold_t.py

## Iteración 2: features del paso actual
Features: last_exit_fail, last_err, last_is_terminal, sim_prev, same_agent_repeat,
n_err_recent3, last_len. Mismo protocolo (dentro de (fold, t), bootstrap por traza).

Univariado (AUROC dentro de (fold, t)): todas entre 0.47 y 0.50, salvo n_err_recent3 = 0.586.

| Modelo | AUROC dentro de (fold, t) | IC95% |
|---|---|---|
| LogReg (step) | 0.578 | [0.435, 0.694] |
| GBM (step) | 0.593 | [0.485, 0.701] |

Sin mejora significativa frente a la iteración 1 (IC solapados).
Hipótesis: la etiqueta acumulativa (y=1 para todo t >= mistake_step) no mide eventos
del paso actual. Pendiente: etiqueta de evento y trazas exitosas.
