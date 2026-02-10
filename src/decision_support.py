"""
Support functions for decision-oriented ranking in forecast and simulation views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _min_max_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    span = values.max() - values.min()
    if np.isnan(span) or np.isclose(span, 0.0):
        return pd.Series(1.0, index=values.index)
    normalized = (values - values.min()) / span
    return normalized if higher_is_better else (1.0 - normalized)


def rank_forecast_models(
    results_df: pd.DataFrame,
    weight_accuracy: float = 0.5,
    weight_rmse: float = 0.3,
    weight_wape: float = 0.2,
) -> pd.DataFrame:
    ranked = results_df.copy()

    weights = np.array([weight_accuracy, weight_rmse, weight_wape], dtype=float)
    if weights.sum() <= 0:
        weights = np.array([0.5, 0.3, 0.2], dtype=float)
    weights = weights / weights.sum()

    ranked["score_accuracy"] = _min_max_score(ranked["Acuracia (%)"], higher_is_better=True)
    ranked["score_rmse"] = _min_max_score(ranked["RMSE"], higher_is_better=False)
    ranked["score_wape"] = _min_max_score(ranked["WAPE (%)"], higher_is_better=False)

    ranked["Score de Decisao"] = (
        ranked["score_accuracy"] * weights[0]
        + ranked["score_rmse"] * weights[1]
        + ranked["score_wape"] * weights[2]
    ) * 100

    ranked = ranked.sort_values("Score de Decisao", ascending=False).reset_index(drop=True)
    return ranked


def rank_simulation_policies(
    results_df: pd.DataFrame,
    min_fill_rate: float,
    weight_cost: float = 0.55,
    weight_service: float = 0.30,
    weight_risk: float = 0.15,
    infeasible_penalty: float = 0.35,
) -> pd.DataFrame:
    ranked = results_df.copy()

    weights = np.array([weight_cost, weight_service, weight_risk], dtype=float)
    if weights.sum() <= 0:
        weights = np.array([0.55, 0.30, 0.15], dtype=float)
    weights = weights / weights.sum()

    ranked["score_cost"] = _min_max_score(ranked["Custo Total (media)"], higher_is_better=False)
    ranked["score_service"] = _min_max_score(ranked["Fill Rate (%)"], higher_is_better=True)
    ranked["score_risk"] = _min_max_score(ranked["Faltas Totais"], higher_is_better=False)
    ranked["Atende Fill Rate"] = ranked["Fill Rate (%)"] >= min_fill_rate

    ranked["Score de Decisao"] = (
        ranked["score_cost"] * weights[0]
        + ranked["score_service"] * weights[1]
        + ranked["score_risk"] * weights[2]
    ) * 100

    ranked["Score Ajustado"] = np.where(
        ranked["Atende Fill Rate"],
        ranked["Score de Decisao"],
        ranked["Score de Decisao"] * (1.0 - infeasible_penalty),
    )

    ranked = ranked.sort_values(
        ["Atende Fill Rate", "Score Ajustado", "Fill Rate (%)"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ranked


def simulation_justification(
    ranked_df: pd.DataFrame,
    min_fill_rate: float,
) -> list[str]:
    if ranked_df.empty:
        return ["Nao ha dados suficientes para gerar justificativa."]

    best = ranked_df.iloc[0]
    avg_cost = ranked_df["Custo Total (media)"].mean()
    avg_fill = ranked_df["Fill Rate (%)"].mean()
    avg_stockout = ranked_df["Faltas Totais"].mean()

    feasibility_text = (
        f"Atende o fill rate minimo de {min_fill_rate:.1f}%."
        if bool(best["Atende Fill Rate"])
        else (
            f"Nenhuma politica atingiu {min_fill_rate:.1f}%; "
            "esta foi a melhor combinacao de custo e servico."
        )
    )

    reasons = [
        f"Maior score ajustado ({best['Score Ajustado']:.1f}) no cenario atual.",
        feasibility_text,
        (
            f"Custo total medio de R$ {best['Custo Total (media)']:,.0f}, "
            f"{((avg_cost - best['Custo Total (media)']) / avg_cost * 100) if avg_cost else 0:.1f}% "
            "melhor que a media."
        ),
        (
            f"Fill rate de {best['Fill Rate (%)']:.1f}% "
            f"(media do cenario: {avg_fill:.1f}%)."
        ),
        (
            f"Faltas medias de {best['Faltas Totais']:.1f} un "
            f"(media do cenario: {avg_stockout:.1f} un)."
        ),
    ]
    return reasons
