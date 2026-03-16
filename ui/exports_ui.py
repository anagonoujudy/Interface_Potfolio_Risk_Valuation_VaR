from __future__ import annotations

import os
import tempfile
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ============================================================
# OUTILS INTERNES
# ============================================================

TITLE_SIZE = 14
TEXT_SIZE = 10
LEFT_MARGIN = 0.05
RIGHT_MARGIN = 0.95


def _safe_fmt_pct(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{x:.2%}"
    except Exception:
        return "N/A"


def _safe_fmt_num(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{x:,.2f}"
    except Exception:
        return "N/A"


def _save_figure(fig, path: str, dpi: int = 150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _wrap_text(text: str, width: int = 120) -> str:
    return "\n".join(textwrap.fill(par, width=width) for par in text.split("\n"))


def _draw_wrapped_text(ax, x, y, text, fontsize=TEXT_SIZE, color="#0F172A", weight=None, width=120):
    wrapped = _wrap_text(text, width=width)
    ax.text(
        x, y, wrapped,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        va="top",
        ha="left",
    )


def _select_best_var_method_for_reporting(
    var_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> dict:
    result = {
        "method": "N/A",
        "var_return": None,
        "var_money": None,
        "status": "N/A",
    }

    if backtest_summary is None or backtest_summary.empty:
        return result

    required_cols = ["Method", "Kupiec p-value", "Conditional Coverage p-value"]
    if not all(col in backtest_summary.columns for col in required_cols):
        return result

    tmp = backtest_summary.copy()
    tmp = tmp.dropna(subset=["Kupiec p-value", "Conditional Coverage p-value"])

    if tmp.empty:
        return result

    tmp = tmp.sort_values(
        by=["Kupiec p-value", "Conditional Coverage p-value"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_method = tmp.iloc[0]["Method"]
    result["method"] = best_method

    if var_summary is not None and not var_summary.empty and "Method" in var_summary.columns:
    
        row = var_summary.loc[var_summary["Method"] == best_method]
        if not row.empty:
            if "VaR_Return" in row.columns:
                result["var_return"] = row["VaR_Return"].iloc[0]
            if "VaR_Money" in row.columns:
                result["var_money"] = row["VaR_Money"].iloc[0]
            if "Status" in row.columns:
                result["status"] = row["Status"].iloc[0]
                

    return result


def _build_key_messages(
    performance_metrics: dict,
    best_var_info: dict,
    confidence: float,
    horizon: int,
) -> list[str]:
    messages = []

    ann_ret = performance_metrics.get("Annualized Return")
    ann_vol = performance_metrics.get("Annualized Volatility")
    mdd = performance_metrics.get("Max Drawdown")
    beta = performance_metrics.get("Beta vs Market")

    messages.append(
        f"Le portefeuille présente un rendement annualisé de {_safe_fmt_pct(ann_ret)} "
        f"pour une volatilité annualisée de {_safe_fmt_pct(ann_vol)}."
    )

    messages.append(
        f"La baisse maximale observée sur l’historique ressort à {_safe_fmt_pct(mdd)}."
    )

    if pd.notna(beta):
        messages.append(
            f"La sensibilité du portefeuille au marché est estimée par un bêta de {_safe_fmt_num(beta)}."
        )

    if best_var_info["method"] != "N/A":
        messages.append(
            f"La méthode de VaR la plus robuste au regard du backtesting est {best_var_info['method']}."
        )

        if best_var_info["var_return"] is not None:
            messages.append(
                f"Sur un horizon de {horizon} jour(s), la VaR retenue à {int(confidence * 100)} % "
                f"est estimée à {_safe_fmt_pct(best_var_info['var_return'])}."
            )

        if best_var_info["var_money"] is not None:
            messages.append(
                f"En valeur monétaire, cela correspond à une perte potentielle d’environ "
                f"{_safe_fmt_num(best_var_info['var_money'])} €."
            )

    return messages


def _get_method_explanations() -> dict[str, str]:
    return {
        "Historique": (
            "La méthode historique repose directement sur les pertes observées dans le passé. "
            "Elle ne suppose pas de loi particulière pour les rendements ; elle répond à la question : "
            "si demain ressemblait à l’un des épisodes passés, quelle perte extrême devrait-on anticiper ?"
        ),
        "Variance-Covariance": (
            "La méthode variance-covariance suppose une dynamique plus régulière des rendements et résume le risque "
            "à travers la moyenne et la volatilité. Elle est simple, rapide à mettre en œuvre et souvent utilisée "
            "comme point de comparaison de base."
        ),
        "Cornish-Fisher": (
            "Cornish-Fisher prolonge l’approche paramétrique classique en corrigeant le quantile de la loi normale "
            "pour tenir compte de l’asymétrie et de l’épaisseur des queues. Elle devient utile lorsque les rendements "
            "ne sont pas parfaitement symétriques ni gaussiens."
        ),
        "RiskMetrics": (
            "RiskMetrics introduit une volatilité conditionnelle qui réagit davantage aux chocs récents. "
            "Cette logique est particulièrement adaptée aux marchés financiers, où les périodes calmes et agitées "
            "se succèdent rarement de façon homogène."
        ),
        "GARCH-Student": (
            "Le modèle GARCH-Student cherche à modéliser explicitement l’hétéroscédasticité conditionnelle, c’est-à-dire "
            "les changements de régime de volatilité. Le recours à une loi de Student permet en plus de mieux capter "
            "les événements extrêmes que la loi normale."
        ),
        "EVT-POT": (
            "L’approche EVT-POT se concentre uniquement sur les observations extrêmes au-delà d’un seuil. "
            "Son intérêt est de mieux décrire la queue de distribution, là où se situent précisément les pertes "
            "les plus critiques pour la gestion du risque."
        ),
        "EVT-GARCH": (
            "EVT-GARCH combine deux dimensions complémentaires : une volatilité conditionnelle dynamique via GARCH, "
            "puis une modélisation spécifique de la queue extrême via l’EVT. C’est généralement l’une des approches "
            "les plus riches lorsque l’objectif est de capturer à la fois la persistance de volatilité et les pertes rares."
        ),
    }


def _build_var_table_commentary(
    var_summary: pd.DataFrame,
    best_var_info: dict,
) -> str:
    if var_summary is None or var_summary.empty:
        return (
            "Aucune synthèse de VaR n’est disponible. Cela signifie qu’aucune méthode n’a produit "
            "de résultat exploitable dans la configuration retenue."
        )

    valid = var_summary.dropna(subset=["VaR_Return"]).copy()
    if valid.empty:
        return (
            "Les méthodes ont été lancées, mais aucune estimation finale de VaR n’a pu être retenue "
            "de façon exploitable sur l’échantillon disponible."
        )

    max_row = valid.loc[valid["VaR_Return"].idxmax()]
    min_row = valid.loc[valid["VaR_Return"].idxmin()]

    text = (
        f"La comparaison des estimations montre que la méthode la plus prudente est {max_row['Method']}, "
        f"avec une VaR de {_safe_fmt_pct(max_row['VaR_Return'])}, tandis que l’estimation la moins sévère "
        f"provient de {min_row['Method']}, avec {_safe_fmt_pct(min_row['VaR_Return'])}. "
    )

    if best_var_info["method"] != "N/A":
        text += (
            f"La méthode retenue pour l’interprétation finale n’est pas nécessairement la plus conservatrice ; "
            f"elle est choisie parce qu’elle offre le meilleur compromis entre crédibilité statistique et qualité "
            f"de couverture selon les tests de backtesting. Ici, la méthode sélectionnée est {best_var_info['method']}."
        )

    return text


def _build_backtesting_commentary(
    backtest_summary: pd.DataFrame,
    best_var_info: dict,
) -> str:
    if backtest_summary is None or backtest_summary.empty:
        return (
            "Le tableau de backtesting n’a pas pu être constitué. Dans ce cas, il n’est pas possible de juger "
            "la capacité des modèles à couvrir correctement les pertes observées."
        )

    if best_var_info["method"] == "N/A":
        return (
            "Le backtesting a été exécuté, mais aucun modèle n’a émergé de manière suffisamment nette pour être "
            "retenu comme référence principale."
        )

    row = backtest_summary.loc[backtest_summary["Method"] == best_var_info["method"]]
    if row.empty:
        return (
            f"La méthode retenue est {best_var_info['method']}. Ce choix traduit une meilleure qualité relative "
            f"de couverture au regard des autres modèles testés."
        )

    kupiec = row["Kupiec p-value"].iloc[0] if "Kupiec p-value" in row.columns else None
    cc = row["Conditional Coverage p-value"].iloc[0] if "Conditional Coverage p-value" in row.columns else None

    return (
        f"Le backtesting suggère que {best_var_info['method']} se comporte le mieux parmi les modèles testés. "
        f"Ses p-values de Kupiec et de couverture conditionnelle ressortent respectivement à "
        f"{_safe_fmt_num(kupiec)} et {_safe_fmt_num(cc)}. "
        f"En pratique, cela signifie que ce modèle décrit plus correctement la fréquence des violations "
        f"et leur dynamique dans le temps."
    )


def _build_conclusion_text(
    capital_invested: float,
    performance_metrics: dict,
    best_var_info: dict,
    confidence: float,
    horizon: int,
) -> str:
    ann_ret = performance_metrics.get("Annualized Return")
    ann_vol = performance_metrics.get("Annualized Volatility")
    mdd = performance_metrics.get("Max Drawdown")

    return (
        f"En conclusion, sur la base d’un capital investi de {_safe_fmt_num(capital_invested)} €, "
        f"le portefeuille présente un rendement annualisé attendu de {_safe_fmt_pct(ann_ret)} "
        f"pour une volatilité annualisée de {_safe_fmt_pct(ann_vol)}. "
        f"La baisse maximale observée sur l’historique atteint {_safe_fmt_pct(mdd)}. "
        f"La perte potentielle maximale retenue, au niveau de confiance de {int(confidence * 100)} % "
        f"et sur un horizon de {horizon} jour(s), est estimée à {_safe_fmt_pct(best_var_info['var_return'])} "
        f"soit environ {_safe_fmt_num(best_var_info['var_money'])} €. "
        f"Cette estimation doit être lue comme un repère de gestion prudent, utile pour le pilotage du risque "
        f"et la communication au management, sans être interprétée comme une borne absolue en toutes circonstances."
    )


def _compute_covariance_matrix(asset_returns: pd.DataFrame | None) -> pd.DataFrame:
    if asset_returns is None or asset_returns.empty:
        return pd.DataFrame()
    return asset_returns.dropna(how="any").cov()


def _compute_correlation_matrix(asset_returns: pd.DataFrame | None) -> pd.DataFrame:
    if asset_returns is None or asset_returns.empty:
        return pd.DataFrame()
    return asset_returns.dropna(how="any").corr()


def _add_matrix_sheet(
    workbook,
    writer,
    covariance_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
):
    worksheet = workbook.add_worksheet("Cov_Corr_Matrices")
    writer.sheets["Cov_Corr_Matrices"] = worksheet

    worksheet.set_zoom(90)
    worksheet.hide_gridlines(2)
    worksheet.set_column("A:Z", 15)

    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 15,
        "font_color": "#0B1F3A",
    })

    section_fmt = workbook.add_format({
        "bold": True,
        "font_size": 11,
        "font_color": "white",
        "bg_color": "#0B1F3A",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    })

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#D9E2F3",
        "border": 1,
        "align": "center",
    })

    value_fmt = workbook.add_format({
        "border": 1,
        "num_format": "0.0000",
        "align": "center",
    })

    worksheet.write("B2", "Matrices de covariance et de corrélation", title_fmt)

    # Covariance
    if not covariance_matrix.empty:
        worksheet.write("B4", "Matrice de covariance", section_fmt)

        start_row = 4
        start_col = 1

        for j, col in enumerate(["Actif"] + list(covariance_matrix.columns)):
            worksheet.write(start_row, start_col + j, col, header_fmt)

        for i, idx in enumerate(covariance_matrix.index):
            worksheet.write(start_row + 1 + i, start_col, idx, header_fmt)
            for j, col in enumerate(covariance_matrix.columns):
                worksheet.write(start_row + 1 + i, start_col + 1 + j, covariance_matrix.loc[idx, col], value_fmt)

    # Correlation
    if not correlation_matrix.empty:
        corr_row = 4 + len(covariance_matrix.index) + 5 if not covariance_matrix.empty else 4
        worksheet.write(corr_row, 1, "Matrice de corrélation", section_fmt)

        for j, col in enumerate(["Actif"] + list(correlation_matrix.columns)):
            worksheet.write(corr_row, 1 + j, col, header_fmt)

        for i, idx in enumerate(correlation_matrix.index):
            worksheet.write(corr_row + 1 + i, 1, idx, header_fmt)
            for j, col in enumerate(correlation_matrix.columns):
                worksheet.write(corr_row + 1 + i, 2 + j, correlation_matrix.loc[idx, col], value_fmt)

        n = len(correlation_matrix.index)
        first_row = corr_row + 1
        last_row = corr_row + n
        first_col = 2
        last_col = 1 + n

        worksheet.conditional_format(
            first_row, first_col, last_row, last_col,
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",
                "mid_color": "#FFEB84",
                "max_color": "#63BE7B",
            }
        )


def _add_risk_dashboard_sheet(
    workbook,
    writer,
    portfolio_value: pd.Series,
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame | None,
    var_series_dict: dict,
    market_value: pd.Series | None = None,
    asset_price_levels: pd.DataFrame | None = None,
    capital_invested: float | None = None,
):
    worksheet = workbook.add_worksheet("Risk_Dashboard")
    writer.sheets["Risk_Dashboard"] = worksheet

    worksheet.set_zoom(85)
    worksheet.hide_gridlines(2)
    worksheet.set_default_row(20)
    worksheet.set_column("A:A", 3)
    worksheet.set_column("B:L", 14)
    worksheet.set_column("M:W", 14)
    worksheet.set_column("X:Z", 8)

    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 16,
        "font_color": "#0B1F3A",
    })

    section_fmt = workbook.add_format({
        "bold": True,
        "font_size": 11,
        "font_color": "white",
        "bg_color": "#0B1F3A",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    })

    label_fmt = workbook.add_format({
        "bold": True,
        "border": 1,
        "bg_color": "#F3F7FB",
    })

    value_fmt = workbook.add_format({
        "border": 1,
        "num_format": "0.0000",
    })

    worksheet.write("B2", "Risk Dashboard", title_fmt)

    clean_returns = portfolio_returns.dropna().copy()
    if clean_returns.empty:
        worksheet.write("B6", "Aucune donnée de rendement disponible.")
        return

    worksheet.write("B4", "Statistiques descriptives", section_fmt)

    stats = clean_returns.describe()
    skewness = clean_returns.skew()
    kurt = clean_returns.kurtosis()

    stats_labels = list(stats.index) + ["Skewness", "Kurtosis"]
    stats_values = list(stats.values) + [skewness, kurt]

    row0 = 5
    col0 = 1

    for i, (label, value) in enumerate(zip(stats_labels, stats_values)):
        worksheet.write(row0 + i, col0, label, label_fmt)
        worksheet.write(row0 + i, col0 + 1, float(value), value_fmt)

    tmpdir = tempfile.mkdtemp()
    plt.style.use("ggplot")

    worksheet.write("N4", "Distribution", section_fmt)
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.hist(clean_returns, bins=40, density=True)
    ax.set_title("Distribution des rendements", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    hist_path = os.path.join(tmpdir, "hist_returns.png")
    _save_figure(fig, hist_path)
    worksheet.insert_image("N5", hist_path, {"x_scale": 0.95, "y_scale": 0.95})

    worksheet.write("N21", "Rendements du portefeuille", section_fmt)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    clean_returns.plot(ax=ax, linewidth=0.9)
    ax.set_title("Rendements du portefeuille", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    returns_path = os.path.join(tmpdir, "returns_plot.png")
    _save_figure(fig, returns_path)
    worksheet.insert_image("N22", returns_path, {"x_scale": 0.95, "y_scale": 0.95})

    worksheet.write("B21", "Carrés des rendements", section_fmt)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    (clean_returns ** 2).plot(ax=ax, linewidth=0.9)
    ax.set_title("Carrés des rendements", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    squared_path = os.path.join(tmpdir, "returns_squared.png")
    _save_figure(fig, squared_path)
    worksheet.insert_image("B22", squared_path, {"x_scale": 0.95, "y_scale": 0.95})

    worksheet.write("B38", "ACF rendements au carré", section_fmt)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    plot_acf(clean_returns ** 2, lags=40, ax=ax)
    ax.set_title("ACF - Rendements au carré", fontsize=10)
    acf_sq_path = os.path.join(tmpdir, "acf_squared.png")
    _save_figure(fig, acf_sq_path)
    worksheet.insert_image("B39", acf_sq_path, {"x_scale": 0.95, "y_scale": 0.95})

    worksheet.write("N38", "PACF rendements au carré", section_fmt)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    plot_pacf(clean_returns ** 2, lags=40, ax=ax, method="ywm")
    ax.set_title("PACF - Rendements au carré", fontsize=10)
    pacf_sq_path = os.path.join(tmpdir, "pacf_squared.png")
    _save_figure(fig, pacf_sq_path)
    worksheet.insert_image("N39", pacf_sq_path, {"x_scale": 0.95, "y_scale": 0.95})

    worksheet.write("B56", "Rendements et seuils de VaR", section_fmt)
    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    ax.plot(
        clean_returns.index,
        clean_returns.values,
        label="Portfolio Returns",
        linewidth=1.0,
        color="black",
    )

    method_colors = {
        "Historique": "red",
        "Variance-Covariance": "blue",
        "Cornish-Fisher": "magenta",
        "RiskMetrics": "brown",
        "GARCH-Student": "orange",
        "EVT-POT": "green",
        "EVT-GARCH": "purple",
    }

    for method_name, series in var_series_dict.items():
        if series is None or len(series.dropna()) == 0:
            continue
        aligned = series.reindex(clean_returns.index).ffill().dropna()
        if aligned.empty:
            continue
        ax.plot(
            aligned.index,
            -aligned.values,
            linestyle="--",
            linewidth=1.2,
            label=method_name,
            color=method_colors.get(method_name, None),
        )

    ax.set_title("Returns and VaR", fontsize=10)
    ax.set_ylabel("Rendements")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    returns_var_path = os.path.join(tmpdir, "returns_and_var.png")
    _save_figure(fig, returns_var_path)
    worksheet.insert_image("B57", returns_var_path, {"x_scale": 0.98, "y_scale": 0.98})

    worksheet.write("N56", "Portefeuille, indice et actions", section_fmt)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))

    portfolio_value.plot(ax=ax, linewidth=1.6, color="black", label="Portefeuille")

    if market_value is not None and not market_value.empty:
        market_value.plot(ax=ax, linewidth=1.2, linestyle="--", label="Indice de marché")

    if asset_price_levels is not None and not asset_price_levels.empty and capital_invested is not None:
        aligned_assets = asset_price_levels.copy().dropna(how="all")
        if not aligned_assets.empty:
            normalized_assets = aligned_assets / aligned_assets.iloc[0] * capital_invested
            normalized_assets.plot(ax=ax, linewidth=0.8, alpha=0.9)

    ax.set_title("Evolution comparée", fontsize=10)
    ax.set_ylabel("Valeur normalisée")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    comp_path = os.path.join(tmpdir, "comparative_prices.png")
    _save_figure(fig, comp_path)
    worksheet.insert_image("N57", comp_path, {"x_scale": 0.95, "y_scale": 0.95})


# ============================================================
# EXPORT EXCEL
# ============================================================

def export_risk_excel(
    output_path: str,
    portfolio_name: str,
    selected_assets: list[str],
    selected_weights: dict[str, float],
    capital_invested: float,
    confidence: float,
    horizon: int,
    portfolio_value: pd.Series,
    portfolio_returns: pd.Series,
    portfolio_pnl: pd.Series,
    performance_metrics: dict,
    var_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    asset_contrib: pd.DataFrame,
    mc_quantiles: pd.DataFrame,
    diagnostics: dict,
    var_series_dict: dict,
    asset_returns: pd.DataFrame | None = None,
    market_value: pd.Series | None = None,
    asset_price_levels: pd.DataFrame | None = None,
):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#0B1F3A",
            "font_color": "white",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })

        title_fmt = workbook.add_format({
            "bold": True,
            "font_size": 14,
            "font_color": "#0B1F3A",
        })

        label_fmt = workbook.add_format({
            "bold": True,
            "font_color": "#0B1F3A",
        })

        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        num_fmt = workbook.add_format({"num_format": "#,##0.00"})

        best_var_info = _select_best_var_method_for_reporting(var_summary, backtest_summary)
        key_messages = _build_key_messages(
            performance_metrics=performance_metrics,
            best_var_info=best_var_info,
            confidence=confidence,
            horizon=horizon,
        )

        covariance_matrix = _compute_covariance_matrix(asset_returns)
        correlation_matrix = _compute_correlation_matrix(asset_returns)

        # Executive Summary
        summary_df = pd.DataFrame({
            "Metric": list(performance_metrics.keys()),
            "Value": list(performance_metrics.values())
        })
        summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

        ws = writer.sheets["Executive_Summary"]
        ws.set_zoom(90)
        ws.hide_gridlines(2)
        ws.set_column("A:Z", 18)

        for col_num, value in enumerate(summary_df.columns.values):
            ws.write(0, col_num, value, header_fmt)

        ws.write("E2", "Portfolio Name", title_fmt)
        ws.write("F2", portfolio_name)

        ws.write("E3", "Generated At", label_fmt)
        ws.write("F3", datetime.now().strftime("%d/%m/%Y %H:%M"))

        ws.write("E4", "Capital Invested (€)", label_fmt)
        ws.write("F4", capital_invested, num_fmt)

        ws.write("E5", "Confidence Level", label_fmt)
        ws.write("F5", confidence, pct_fmt)

        ws.write("E6", "Holding Horizon (days)", label_fmt)
        ws.write("F6", horizon)

        ws.write("E8", "Selected Assets", title_fmt)
        row = 8
        for asset in selected_assets:
            ws.write(row, 5, asset)
            row += 1

        ws.write("H8", "Weights", title_fmt)
        row = 8
        for asset, weight in selected_weights.items():
            ws.write(row, 7, asset)
            ws.write(row, 8, weight, pct_fmt)
            row += 1

        ws.write("E15", "Selected VaR Method", title_fmt)
        ws.write("F15", best_var_info["method"])

        ws.write("E16", "Selected VaR Return", label_fmt)
        if best_var_info["var_return"] is not None:
            ws.write("F16", best_var_info["var_return"], pct_fmt)
        else:
            ws.write("F16", "N/A")

        ws.write("E17", "Selected VaR Money", label_fmt)
        if best_var_info["var_money"] is not None:
            ws.write("F17", best_var_info["var_money"], num_fmt)
        else:
            ws.write("F17", "N/A")

        ws.write("E20", "Key Messages", title_fmt)
        row = 20
        for msg in key_messages:
            ws.write(row, 5, f"• {msg}")
            row += 1

        # Portfolio Series
        series_df = pd.concat(
            [
                portfolio_value.rename("Portfolio_Value"),
                portfolio_returns.rename("Portfolio_Log_Return"),
                portfolio_pnl.rename("Portfolio_PnL"),
            ],
            axis=1,
        )
        series_df.to_excel(writer, sheet_name="Portfolio_Series")

        # VaR Summary
        var_summary.to_excel(writer, sheet_name="VaR_Summary", index=False)

        # Backtesting
        backtest_summary.to_excel(writer, sheet_name="Backtesting", index=False)

        # Contributions
        asset_contrib.to_excel(writer, sheet_name="Asset_Contributions", index=False)

        # Monte Carlo
        mc_quantiles.to_excel(writer, sheet_name="MonteCarlo_PnL")

        # Diagnostics
        diag_rows = []
        for method, diag in diagnostics.items():
            row = {"Method": method}
            for k, v in diag.items():
                row[k] = str(v) if isinstance(v, (list, dict)) else v
            diag_rows.append(row)

        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_excel(writer, sheet_name="Diagnostics", index=False)

        # Covariance + Correlation
        _add_matrix_sheet(
            workbook=workbook,
            writer=writer,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
        )

        # Charts
        chart_sheet = workbook.add_worksheet("Charts")
        writer.sheets["Charts"] = chart_sheet
        chart_sheet.hide_gridlines(2)
        chart_sheet.set_zoom(85)

        tmpdir = tempfile.mkdtemp()

        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        portfolio_value.plot(ax=ax, linewidth=1.5)
        ax.set_title("Evolution de la valeur du portefeuille", fontsize=10)
        ax.set_ylabel("Valeur")
        path1 = os.path.join(tmpdir, "portfolio_value.png")
        _save_figure(fig, path1)

        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        portfolio_pnl.plot(ax=ax, linewidth=1.5)
        ax.set_title("Evolution du PnL", fontsize=10)
        ax.set_ylabel("PnL")
        path2 = os.path.join(tmpdir, "portfolio_pnl.png")
        _save_figure(fig, path2)

        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        mc_quantiles.plot(ax=ax, linewidth=1.1)
        ax.set_title("Prévision Monte Carlo du PnL à 16 jours", fontsize=10)
        ax.set_xlabel("Jour")
        ax.set_ylabel("PnL projeté")
        path3 = os.path.join(tmpdir, "mc_pnl.png")
        _save_figure(fig, path3)

        chart_sheet.insert_image("B2", path1, {"x_scale": 0.92, "y_scale": 0.92})
        chart_sheet.insert_image("B20", path2, {"x_scale": 0.92, "y_scale": 0.92})
        chart_sheet.insert_image("B38", path3, {"x_scale": 0.92, "y_scale": 0.92})

        # Risk Dashboard
        _add_risk_dashboard_sheet(
            workbook=workbook,
            writer=writer,
            portfolio_value=portfolio_value,
            portfolio_returns=portfolio_returns,
            asset_returns=asset_returns,
            var_series_dict=var_series_dict,
            market_value=market_value,
            asset_price_levels=asset_price_levels,
            capital_invested=capital_invested,
        )

        for sheet_name in writer.sheets:
            ws_sheet = writer.sheets[sheet_name]
            ws_sheet.set_zoom(90)

        mapping = {
            "VaR_Summary": var_summary,
            "Backtesting": backtest_summary,
            "Asset_Contributions": asset_contrib,
            "Diagnostics": diag_df,
            "MonteCarlo_PnL": mc_quantiles.reset_index(),
        }

        for sheet_name, df_tmp in mapping.items():
            ws_sheet = writer.sheets[sheet_name]
            ws_sheet.hide_gridlines(2)
            ws_sheet.set_column("A:Z", 18)
            for col_num, value in enumerate(df_tmp.columns.values):
                ws_sheet.write(0, col_num, value, header_fmt)


# ============================================================
# EXPORT PDF CORPORATE
# ============================================================

def export_risk_pdf(
    output_path: str,
    portfolio_name: str,
    selected_assets: list[str],
    selected_weights: dict[str, float],
    capital_invested: float,
    confidence: float,
    horizon: int,
    performance_metrics: dict,
    var_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    portfolio_value: pd.Series,
    portfolio_pnl: pd.Series,
    mc_quantiles: pd.DataFrame,
    asset_returns: pd.DataFrame | None = None,
    market_value: pd.Series | None = None,
    asset_price_levels: pd.DataFrame | None = None,
    portfolio_source: str = "custom",
    selection_methodology: str | None = None,
):
    best_var_info = _select_best_var_method_for_reporting(var_summary, backtest_summary)
    key_messages = _build_key_messages(
        performance_metrics=performance_metrics,
        best_var_info=best_var_info,
        confidence=confidence,
        horizon=horizon,
    )

    method_explanations = _get_method_explanations()
    var_commentary = _build_var_table_commentary(var_summary, best_var_info)
    backtesting_commentary = _build_backtesting_commentary(backtest_summary, best_var_info)
    conclusion_text = _build_conclusion_text(
        capital_invested=capital_invested,
        performance_metrics=performance_metrics,
        best_var_info=best_var_info,
        confidence=confidence,
        horizon=horizon,
    )
    covariance_matrix = _compute_covariance_matrix(asset_returns)
    correlation_matrix = _compute_correlation_matrix(asset_returns)

    if selection_methodology is None:
        if portfolio_source.lower() in ["recommandé", "recommended"]:
            selection_methodology = (
                "Le portefeuille a été sélectionné à partir du moteur de recommandation interne, "
                "qui estime les rendements attendus des actifs via le modèle CAPM à partir de l’indice de marché, "
                "puis compare plusieurs combinaisons d’actifs selon un compromis rendement / risque."
            )
        else:
            selection_methodology = (
                "Le portefeuille a été construit par l’utilisateur à partir d’une sélection d’actifs "
                "effectuée par secteur. Les poids ont ensuite été déterminés selon l’une des trois stratégies "
                "suivantes : portefeuille équilibré, portefeuille personnalisé ou portefeuille optimisé selon le CAPM."
            )

    with PdfPages(output_path) as pdf:
        # PAGE 1 - GARDE
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.add_patch(Rectangle((0, 0.87), 1, 0.13, transform=ax.transAxes, color="#0B1F3A"))
        ax.text(0.04, 0.935, "FOJUMMA EQUITY", transform=ax.transAxes,
                fontsize=24, fontweight="bold", color="white", va="center")

        ax.text(LEFT_MARGIN, 0.72, "RAPPORT DE RISQUE DE MARCHÉ",
                transform=ax.transAxes, fontsize=28, fontweight="bold", color="#0B1F3A")

        ax.text(LEFT_MARGIN, 0.63,
                "Synthèse du portefeuille, estimation de la Value at Risk\net validation par backtesting",
                transform=ax.transAxes, fontsize=TEXT_SIZE, color="#334155")

        cover_info = [
            f"Portefeuille : {portfolio_name}",
            f"Date de génération : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            f"Capital investi : {capital_invested:,.2f} €",
            f"Niveau de confiance : {confidence:.0%}",
            f"Horizon de détention : {horizon} jour(s)",
        ]

        ax.text(LEFT_MARGIN, 0.45, "\n".join(cover_info),
                transform=ax.transAxes, fontsize=TEXT_SIZE, color="#0F172A", va="top")

        _draw_wrapped_text(
            ax, LEFT_MARGIN, 0.16,
            "Document préparé pour le Directeur du Département Risque. "
            "Le rapport présente la construction du portefeuille, les hypothèses de calcul, "
            "les estimations de Value at Risk, la validation par backtesting et une projection du PnL.",
            fontsize=TEXT_SIZE - 1,
            color="#475569",
            width=120,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 2 - SYNTHÈSE
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.95, "Synthèse exécutive",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        summary_lines = [
            f"Rendement annualisé : {_safe_fmt_pct(performance_metrics.get('Annualized Return'))}",
            f"Volatilité annualisée : {_safe_fmt_pct(performance_metrics.get('Annualized Volatility'))}",
            f"Sharpe Ratio : {_safe_fmt_num(performance_metrics.get('Sharpe Ratio'))}",
            f"Max Drawdown : {_safe_fmt_pct(performance_metrics.get('Max Drawdown'))}",
            f"Beta vs Marché : {_safe_fmt_num(performance_metrics.get('Beta vs Market'))}",
            f"Méthode de VaR retenue : {best_var_info['method']}",
            f"VaR retenue (en %) : {_safe_fmt_pct(best_var_info['var_return'])}",
            f"VaR retenue (en €) : {_safe_fmt_num(best_var_info['var_money'])} €",
        ]

        ax.text(LEFT_MARGIN, 0.80, "\n".join(summary_lines),
                transform=ax.transAxes, fontsize=TEXT_SIZE, color="#0F172A", va="top")

        box_x = LEFT_MARGIN
        box_y = 0.10
        box_w = RIGHT_MARGIN - LEFT_MARGIN
        box_h = 0.36
        ax.add_patch(Rectangle((box_x, box_y), box_w, box_h,
                               transform=ax.transAxes, color="#EAF2FB", ec="#0B1F3A", lw=1.2))
        ax.text(box_x + 0.02, box_y + box_h - 0.05, "Messages clés",
                transform=ax.transAxes, fontsize=TITLE_SIZE - 2, fontweight="bold",
                color="#0B1F3A", va="top")

        y = box_y + box_h - 0.12
        for msg in key_messages:
            _draw_wrapped_text(
                ax, box_x + 0.03, y, f"• {msg}",
                fontsize=TEXT_SIZE - 1,
                color="#0F172A",
                width=105,
            )
            y -= 0.085

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 3 - CONSTRUCTION
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.95, "Construction du portefeuille",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        _draw_wrapped_text(ax, LEFT_MARGIN, 0.83, selection_methodology, fontsize=TEXT_SIZE, width=120)

        composition_lines = ["Composition et pondérations :"]
        for asset, weight in selected_weights.items():
            composition_lines.append(f"• {asset} : {weight:.2%}")

        ax.text(LEFT_MARGIN, 0.58, "\n".join(composition_lines),
                transform=ax.transAxes, fontsize=TEXT_SIZE, color="#0F172A", va="top")

        _draw_wrapped_text(
            ax, LEFT_MARGIN, 0.20,
            "Une fois la composition figée, le portefeuille est étudié à partir de ses rendements logarithmiques "
            "historiques. Cette base permet d’évaluer à la fois la performance, le risque, la stabilité du comportement "
            "dans le temps et la crédibilité des différentes méthodes de VaR.",
            fontsize=TEXT_SIZE - 1,
            color="#475569",
            width=120,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 4 - PRINCIPES DE CALCUL
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.95, "Principes de calcul et contraintes retenues",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        calc_text = (
            "Les rendements du portefeuille sont calculés à partir des rendements logarithmiques des actifs. "
            "Le cadre de construction retient une somme des poids égale à 100 %, des positions longues uniquement "
            "et une allocation limitée aux actifs sélectionnés. Dans les configurations CAPM, les rendements attendus "
            "sont estimés à partir du bêta de chaque actif vis-à-vis du marché. Les modèles de VaR sont ensuite comparés, "
            "puis validés par backtesting avant d’être retenus pour l’interprétation."
        )

        _draw_wrapped_text(ax, LEFT_MARGIN, 0.84, calc_text, fontsize=TEXT_SIZE, width=120)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 5 - LOGIQUE DES MÉTHODES
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.95, "Logique des méthodes de Value at Risk",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        y = 0.87
        for method, explanation in method_explanations.items():
            ax.text(LEFT_MARGIN, y, method,
                    transform=ax.transAxes, fontsize=TITLE_SIZE - 1, fontweight="bold",
                    color="#0B1F3A", va="top")
            _draw_wrapped_text(
                ax, LEFT_MARGIN, y - 0.04, explanation,
                fontsize=TEXT_SIZE - 1,
                width=120,
            )
            y -= 0.12

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 6 - EVOLUTION COMPARÉE
        fig, ax = plt.subplots(figsize=(9.4, 4.6))

        portfolio_value.plot(ax=ax, linewidth=1.8, label="Portefeuille", color="black")

        if market_value is not None and not market_value.empty:
            market_value.plot(ax=ax, linewidth=1.4, label="Indice de marché", linestyle="--")

        if asset_price_levels is not None and not asset_price_levels.empty:
            aligned_assets = asset_price_levels.copy().dropna(how="all")
            if not aligned_assets.empty:
                normalized_assets = aligned_assets / aligned_assets.iloc[0] * capital_invested
                normalized_assets.plot(ax=ax, linewidth=0.9, alpha=0.9)

        ax.set_title(
            "Evolution comparée du portefeuille, du marché et des actifs",
            fontsize=TITLE_SIZE,
            fontweight="bold",
        )
        ax.set_ylabel("Valeur normalisée")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 7 - MATRICES
        if not covariance_matrix.empty or not correlation_matrix.empty:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")

            ax.text(LEFT_MARGIN, 0.96, "Matrices de covariance et de corrélation",
                    transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold",
                    color="#0B1F3A", va="top")

            if not covariance_matrix.empty:
                cov_ax = fig.add_axes([LEFT_MARGIN, 0.48, 0.40, 0.28])
                cov_ax.axis("off")
                cov_table = cov_ax.table(
                    cellText=covariance_matrix.round(5).values,
                    rowLabels=covariance_matrix.index,
                    colLabels=covariance_matrix.columns,
                    loc="center",
                )
                cov_table.auto_set_font_size(False)
                cov_table.set_fontsize(7)
                cov_table.scale(1, 1.15)

            if not correlation_matrix.empty:
                corr_ax = fig.add_axes([0.55, 0.48, 0.40, 0.28])
                corr_ax.axis("off")
                corr_table = corr_ax.table(
                    cellText=correlation_matrix.round(4).values,
                    rowLabels=correlation_matrix.index,
                    colLabels=correlation_matrix.columns,
                    loc="center",
                )
                corr_table.auto_set_font_size(False)
                corr_table.set_fontsize(7)
                corr_table.scale(1, 1.15)

            _draw_wrapped_text(
                ax, LEFT_MARGIN, 0.28,
                "La matrice de covariance renseigne sur l’intensité conjointe des variations des actifs, "
                "tandis que la matrice de corrélation permet une lecture standardisée des dépendances. "
                "Une corrélation élevée et positive signifie que deux actions ont tendance à évoluer dans le même sens, "
                "ce qui limite les bénéfices de diversification. À l’inverse, des corrélations plus faibles "
                "ou plus hétérogènes améliorent en général la capacité du portefeuille à amortir les chocs.",
                fontsize=TEXT_SIZE - 1,
                width=120,
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # PAGE 8 - TABLEAU VAR + COMMENTAIRE
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.96, "Synthèse des estimations de VaR",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        table_ax = fig.add_axes([LEFT_MARGIN, 0.38, RIGHT_MARGIN - LEFT_MARGIN, 0.40])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=var_summary.round(6).values,
            colLabels=var_summary.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)

        ax.text(LEFT_MARGIN, 0.28, "Lecture du tableau",
                transform=ax.transAxes, fontsize=TITLE_SIZE - 1, fontweight="bold",
                color="#0B1F3A", va="top")
        _draw_wrapped_text(ax, LEFT_MARGIN, 0.23, var_commentary, fontsize=TEXT_SIZE - 1, width=120)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 9 - BACKTESTING + COMMENTAIRE
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.96, "Résultats de backtesting",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        table_ax = fig.add_axes([LEFT_MARGIN, 0.40, RIGHT_MARGIN - LEFT_MARGIN, 0.36])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=backtest_summary.round(6).values,
            colLabels=backtest_summary.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.20)

        ax.text(LEFT_MARGIN, 0.30, "Interprétation",
                transform=ax.transAxes, fontsize=TITLE_SIZE - 1, fontweight="bold",
                color="#0B1F3A", va="top")
        _draw_wrapped_text(ax, LEFT_MARGIN, 0.25, backtesting_commentary, fontsize=TEXT_SIZE - 1, width=120)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PAGE 10 - PNL + MONTE CARLO
        fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.0))

        portfolio_pnl.plot(ax=axes[0], linewidth=1.5)
        axes[0].set_title("Evolution du PnL historique", fontsize=TITLE_SIZE - 1, fontweight="bold")
        axes[0].set_ylabel("PnL")
        axes[0].grid(alpha=0.3)

        mc_quantiles.plot(ax=axes[1], linewidth=1.0)
        axes[1].set_title("Prévision Monte Carlo du PnL à 16 jours", fontsize=TITLE_SIZE - 1, fontweight="bold")
        axes[1].set_xlabel("Jour")
        axes[1].set_ylabel("PnL projeté")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # PAGE 11 - CONCLUSION
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        ax.text(LEFT_MARGIN, 0.95, "Conclusion",
                transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color="#0B1F3A", va="top")

        _draw_wrapped_text(ax, LEFT_MARGIN, 0.82, conclusion_text, fontsize=TEXT_SIZE, width=120)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)