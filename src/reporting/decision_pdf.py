"""
PDF export for simulation decision reports.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _currency(value: float) -> str:
    return f"R$ {value:,.2f}"


def _short_policy_name(policy_name: str, max_len: int = 44) -> str:
    if len(policy_name) <= max_len:
        return policy_name
    return f"{policy_name[: max_len - 3]}..."


def build_simulation_decision_pdf(
    store_id: str,
    item_id: str,
    recommended_policy: str,
    reasons: Iterable[str],
    ranked_df: pd.DataFrame,
    scenario_params: dict[str, float | int | str],
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleDecision",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#8c1118"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtle",
            parent=styles["BodyText"],
            fontSize=9,
            textColor=colors.HexColor("#666666"),
        )
    )

    elements = []
    elements.append(Paragraph("Relatorio de Decisao - Simulacao Monte Carlo", styles["TitleDecision"]))
    elements.append(Paragraph(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Subtle"]))
    elements.append(Spacer(1, 0.45 * cm))

    elements.append(Paragraph(f"<b>Loja:</b> {store_id}", styles["BodyText"]))
    elements.append(Paragraph(f"<b>Item:</b> {item_id}", styles["BodyText"]))
    elements.append(Spacer(1, 0.25 * cm))

    elements.append(Paragraph("<b>Cenario Avaliado</b>", styles["Heading3"]))
    scenario_lines = [
        f"Horizonte: {scenario_params.get('horizon_days', '-') } dias",
        f"Replicacoes: {scenario_params.get('n_replications', '-')}",
        f"Lead time medio: {scenario_params.get('lead_time_mean', '-') } dias",
        f"Variabilidade LT: {scenario_params.get('lt_variability_pct', '-') }%",
        f"Custo por pedido (K): {_currency(float(scenario_params.get('ordering_cost', 0.0)))}",
        f"Custo de falta: {_currency(float(scenario_params.get('stockout_cost', 0.0)))}/un",
        f"Fill rate minimo: {float(scenario_params.get('min_fill_rate', 0.0)):.1f}%",
    ]
    for line in scenario_lines:
        elements.append(Paragraph(f"- {line}", styles["BodyText"]))
    elements.append(Spacer(1, 0.35 * cm))

    elements.append(Paragraph("<b>Recomendacao</b>", styles["Heading3"]))
    elements.append(Paragraph(f"Politica recomendada: <b>{recommended_policy}</b>", styles["BodyText"]))
    for reason in reasons:
        elements.append(Paragraph(f"- {reason}", styles["BodyText"]))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("<b>Ranking de Politicas</b>", styles["Heading3"]))
    top_ranked = ranked_df.head(5).copy()

    table_rows = [[
        "Politica",
        "Score",
        "Custo Medio",
        "Fill Rate",
        "Faltas",
        "Status",
    ]]
    for _, row in top_ranked.iterrows():
        table_rows.append([
            _short_policy_name(str(row["Politica"])),
            f"{float(row['Score Ajustado']):.1f}",
            _currency(float(row["Custo Total (media)"])),
            f"{float(row['Fill Rate (%)']):.1f}%",
            f"{float(row['Faltas Totais']):.1f}",
            "Ok" if bool(row["Atende Fill Rate"]) else "Abaixo",
        ])

    table = Table(
        table_rows,
        colWidths=[6.1 * cm, 1.6 * cm, 3.1 * cm, 2.2 * cm, 1.7 * cm, 2.0 * cm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8c1118")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.35 * cm))
    elements.append(
        Paragraph(
            "Observacao: a recomendacao considera custo, nivel de servico e risco de faltas.",
            styles["Subtle"],
        )
    )

    doc.build(elements)
    return buffer.getvalue()
