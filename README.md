# trace-failure-prediction

Detección de fallas en trazas de agentes LLM a partir de sus prefijos, usando el
dataset **Who&When** (Zhang et al., ICML 2025), subconjunto `Algorithm-Generated`
(126 trazas, todas fallidas).

## Resultado principal

Con features del paso actual y etiqueta de **evento** (solo el paso `mistake_step`),
el detector supera al control no lineal de posición (one-hot de `t`) por +0.13 a +0.15 AUROC (IC95% excluye 0).
Con la etiqueta **acumulativa** (todo `t >= mistake_step`), el AUROC global de ~0.80
se explica casi por completo por el avance de la traza.

## Datos

- `is_correct = False` en las 126 trazas: no hay clase negativa nativa.
- `mistake_step` es base 0 (verificado).
- 3 trazas (gid 95, 106, 109) tienen el agente marcado distinto de `history[s].name`:
  el paso marcado suele ser la consecuencia visible; la acción causal está antes.
  Se conservan tal cual.

## Etiquetas

- **Acumulativa** `y = 1` si `t-1 >= mistake_step`: 66 % positivas.
- **Evento** `y_ev = 1` si `t-1 == mistake_step`: 11.5 % positivas.
- **Ventana** `y_win = 1` si `t-1 ∈ {mistake_step, mistake_step+1}`: 22.8 % positivas.

## Evaluación

- `GroupKFold(5)` por traza.
- Acumulativa: AUROC dentro de `(fold, t)` (elimina la posición) y AUROC global OOF.
- Evento/ventana: AUROC agrupado OOF contra control `t`, con bootstrap por traza (300 réplicas)
  sobre la diferencia.
- Control trivial dentro de `(fold, t)` = 0.500 exacto (validación del método).

## Resultados

Etiqueta acumulativa (dentro de `(fold, t)`):

| Modelo | AUROC | IC95% |
|---|---|---|
| Control `t` | 0.500 | — |
| LogReg (7 features acumuladas) | 0.590 | [0.435, 0.728] |
| GBM (7 features acumuladas) | 0.544 | [0.419, 0.688] |
| LogReg (7 features del paso) | 0.578 | [0.435, 0.694] |
| GBM (7 features del paso) | 0.593 | [0.485, 0.701] |

Etiquetas de evento y ventana (AUROC agrupado OOF):

| Etiqueta | Modelo | AUROC | Margen vs control, IC95% |
|---|---|---|---|
| Evento | Control `t` | 0.616 | — |
| Evento | LogReg step | 0.774 | [0.104, 0.208] |
| Evento | GBM step | 0.788 | [0.115, 0.222] |
| Ventana | Control `t` | 0.567 | — |
| Ventana | LogReg step | 0.627 | [-0.014, 0.120] |
| Ventana | GBM step | 0.658 | [0.026, 0.151] |

## Limitaciones

- 126 trazas, sin trazas exitosas: no es predicción real, es detección de un error ya visible.
- Evaluación de evento agrupada (no dentro de `(fold, t)`) por falta de positivos por celda.
- Las features podrían codificar en parte la forma en que el anotador marcó el paso (ablación pendiente).
- Las features de error pueden estar alineadas con la forma en que el anotador marcó el paso.
- Evaluación limitada a `t <= 8` en el análisis acumulativo.

## Reproducir

```bash
uv sync
uv run python build_prefixes.py       # prefixes.csv (acumulativa)
uv run python baseline.py             # OOF acumulativa
uv run python within_fold_t.py        # AUROC dentro de (fold, t) + bootstrap
uv run python evaluate_step.py        # step_features.csv + features del paso
uv run python evaluate_event.py       # etiquetas evento y ventana vs control
uv run python control_nonlinear.py    # control no lineal de posición (one-hot de t)
```

Scripts `check_*.py`, `explore.py` y `within_t*.py` son exploratorios y registran los
pasos de verificación y la corrección del método (ver `results/baseline.md`).

## Próximos pasos

1. Ablación de features para identificar cuáles aportan en la etiqueta de evento.
2. Trazas exitosas para tener negativos reales y evaluar predicción, no solo detección.
