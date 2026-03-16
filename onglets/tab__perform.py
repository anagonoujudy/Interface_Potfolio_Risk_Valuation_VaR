from __future__ import annotations

import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from onglets.module.backtesting import backtest_var_dict, select_best_var_model
from onglets.module.performance import (
    compute_asset_contributions,
    compute_performance_metrics,
    compute_portfolio_market_series,
    monte_carlo_pnl_forecast,
)
from onglets.module.utils import load_price_data
from ui.exports_ui import export_risk_excel, export_risk_pdf
from onglets.module.mailer import send_report_email


DATA_FILE = r"onglets\Data_set.xlsx"
INDEX_NAME = "Cac40"


def _fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"


def _fmt_num(x):
    return f"{x:,.2f}" if pd.notna(x) else "-"


def render_performance_tab():
    st.title("Performance")

    if not st.session_state.get("var_estimation_done", False):
        st.info(
            "Veuillez d’abord valider un portefeuille dans Accueil puis lancer l’estimation de la VaR dans Paramétrage."
        )
        return

    var_results = st.session_state["var_results"]
    selected_assets = st.session_state["selected_portfolio_assets"]
    selected_weights = st.session_state["selected_portfolio_weights"]
    portfolio_name = st.session_state.get("selected_portfolio_name", "Portefeuille")
    portfolio_source = st.session_state.get("selected_portfolio_source", "custom")

    capital_invested = float(st.session_state["capital_invested"])
    confidence = float(st.session_state["var_confidence"])
    horizon = int(st.session_state["var_horizon"])

    price_df = load_price_data(DATA_FILE)
    market_series = price_df[INDEX_NAME] if INDEX_NAME in price_df.columns else None
    asset_price_df = price_df.drop(columns=[INDEX_NAME]) if INDEX_NAME in price_df.columns else price_df.copy()

    perf_data = compute_portfolio_market_series(
        asset_price_df=asset_price_df,
        weights=selected_weights,
        market_series=market_series,
        capital_invested=capital_invested,
    )

    perf_metrics = compute_performance_metrics(
        portfolio_value=perf_data["portfolio_value"],
        portfolio_returns=perf_data["portfolio_returns"],
        market_returns=perf_data["market_returns"],
    )

    asset_contrib = compute_asset_contributions(asset_price_df, selected_weights)

    mc = monte_carlo_pnl_forecast(
        portfolio_returns=perf_data["portfolio_returns"],
        current_portfolio_value=float(perf_data["portfolio_value"].iloc[-1]),
        horizon_days=16,
        n_sims=5000,
    )

    # ======================================================
    # BACKTESTING
    # ======================================================
    backtest_details, backtest_summary = backtest_var_dict(
        returns=var_results["portfolio_returns"],
        var_dict=var_results["var_series_dict"],
        confidence=confidence,
        horizon=horizon,
    )

    best_var_model = None
    try:
        best_var_model = select_best_var_model(backtest_summary)
    except Exception:
        best_var_model = None

    # ======================================================
    # KPI
    # ======================================================
    # ======================================================
    # Détermination de la VaR retenue
    # ======================================================
    var_value = None
    var_money = None

    if best_var_model is not None:
        selected_method = best_var_model["Method"]
        var_row = var_results["var_summary"].loc[
            var_results["var_summary"]["Method"] == selected_method
        ]

        if not var_row.empty and pd.notna(var_row["VaR_Return"].iloc[0]):
            var_value= float(var_row["VaR_Return"].iloc[0])
            var_money= float(var_row["VaR_Money"].iloc[0])

    st.subheader("Indicateurs clés")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement annualisé", _fmt_pct(perf_metrics["Annualized Return"]))
    c2.metric("Volatilité annualisée", _fmt_pct(perf_metrics["Annualized Volatility"]))
    c3.metric(
        "Sharpe Ratio",
        f"{perf_metrics['Sharpe Ratio']:.2f}" if pd.notna(perf_metrics["Sharpe Ratio"]) else "-"
    )
    print(var_value)
    print(type(var_value))
    c4.metric("Perte maximale estimée",_fmt_num(var_money))


    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Beta vs Marché",
        f"{perf_metrics['Beta vs Market']:.2f}" if pd.notna(perf_metrics["Beta vs Market"]) else "-"
    )
    c6.metric(
        "Corrélation vs Marché",
        f"{perf_metrics['Correlation vs Market']:.2f}" if pd.notna(perf_metrics["Correlation vs Market"]) else "-"
    )
    c7.metric("Valeur finale", _fmt_num(perf_data["portfolio_value"].iloc[-1]))
    c8.metric("PnL final", _fmt_num(perf_data["portfolio_pnl"].iloc[-1]))

    # ======================================================
    # EVOLUTION PORTEFEUILLE / INDICE / ACTIONS
    # ======================================================
    st.markdown("---")
    st.subheader("Evolution du cours du portefeuille")

    available_curves = ["Portefeuille"]
    if perf_data["market_value"] is not None:
        available_curves.append("Indice de marché")
    available_curves.extend(selected_assets)

    default_curves = ["Portefeuille"]
    if "Indice de marché" in available_curves:
        default_curves.append("Indice de marché")

    selected_curves = st.multiselect(
        "Choisir les courbes à afficher",
        options=available_curves,
        default=default_curves,
    )

    fig_value = go.Figure()

    if "Portefeuille" in selected_curves:
        fig_value.add_trace(
            go.Scatter(
                x=perf_data["portfolio_value"].index,
                y=perf_data["portfolio_value"].values,
                mode="lines",
                name="Portefeuille",
            )
        )

    if "Indice de marché" in selected_curves and perf_data["market_value"] is not None:
        fig_value.add_trace(
            go.Scatter(
                x=perf_data["market_value"].index,
                y=perf_data["market_value"].values,
                mode="lines",
                name="Indice de marché",
            )
        )

    for asset in selected_assets:
        if asset in selected_curves:
            serie = asset_price_df[asset].loc[perf_data["portfolio_value"].index].dropna()
            serie = serie / serie.iloc[0] * capital_invested
            fig_value.add_trace(
                go.Scatter(
                    x=serie.index,
                    y=serie.values,
                    mode="lines",
                    name=asset,
                )
            )

    fig_value.update_layout(
        title="Evolution comparée du portefeuille, du marché et des actifs",
        xaxis_title="Date",
        yaxis_title="Valeur normalisée",
        height=500,
    )
    st.plotly_chart(fig_value, use_container_width=True)

    # ======================================================
    # PNL HISTORIQUE
    # ======================================================
    st.subheader("Evolution du PnL historique")

    fig_pnl = go.Figure()
    fig_pnl.add_trace(
        go.Scatter(
            x=perf_data["portfolio_pnl"].index,
            y=perf_data["portfolio_pnl"].values,
            mode="lines",
            name="PnL historique",
        )
    )
    fig_pnl.update_layout(
        title="Evolution historique du PnL",
        xaxis_title="Date",
        yaxis_title="PnL (€)",
        height=450,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ======================================================
    # MONTE CARLO 16 JOURS
    # ======================================================
    st.subheader("Prévision du PnL à 16 jours par simulation Monte Carlo")

    fig_mc = go.Figure()
    for col in ["q05", "q25", "q50", "q75", "q95"]:
        fig_mc.add_trace(
            go.Scatter(
                x=mc["forecast_quantiles"].index,
                y=mc["forecast_quantiles"][col],
                mode="lines",
                name=col.upper(),
            )
        )

    fig_mc.update_layout(
        title="Quantiles projetés du PnL sur 16 jours",
        xaxis_title="Jour",
        yaxis_title="PnL projeté (€)",
        height=450,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.info(
        "La prévision Monte Carlo repose sur les rendements logarithmiques historiques du portefeuille "
        "et une dynamique gaussienne stationnaire. Elle constitue une aide prospective à la décision."
    )

    # ======================================================
    # CONTRIBUTIONS + VAR
    # ======================================================
    st.markdown("---")
    st.subheader("Analyse des contributions et estimation de la VaR")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Contributions par actif")
        st.dataframe(asset_contrib, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Tableau comparatif des VaR")
        st.dataframe(var_results["var_summary"], use_container_width=True, hide_index=True)

    # ======================================================
    # BACKTESTING
    # ======================================================
    st.markdown("---")
    st.subheader("Backtesting des modèles de VaR")

    st.dataframe(backtest_summary, use_container_width=True, hide_index=True)

    if best_var_model is not None:
        st.success(
            f"Le modèle de VaR le plus robuste selon les tests de couverture est : "
            f"**{best_var_model['Method']}**."
        )

        selected_method = best_var_model["Method"]
        var_row = var_results["var_summary"].loc[var_results["var_summary"]["Method"] == selected_method]

        if not var_row.empty:
            var_value = var_row["VaR_Return"].iloc[0]
            st.markdown(
                f"""
                **Interprétation économique**

                - Méthode retenue : **{selected_method}**
                - Niveau de confiance : **{int(confidence * 100)} %**
                - Horizon : **{horizon} jour(s)**
                - VaR estimée : **{_fmt_pct(var_value)}**

                Cette mesure indique la perte potentielle maximale attendue sur l’horizon retenu,
                dans des conditions normales de marché, au niveau de confiance choisi.
                """
            )

     # ======================================================
    # REPORTING
    # ======================================================
    st.markdown("---")
    st.subheader("Reporting")



    col_excel, col_pdf = st.columns(2)

    with col_excel:
        if st.button("Générer le reporting Excel", use_container_width=True):
            tmpdir = tempfile.gettempdir()
            excel_path = os.path.join(tmpdir, "RiskManagementResult.xlsx")
            export_risk_excel(
                output_path=excel_path,
                portfolio_name=portfolio_name,
                selected_assets=selected_assets,
                selected_weights=selected_weights,
                capital_invested=capital_invested,
                confidence=confidence,
                horizon=horizon,
                portfolio_value=perf_data["portfolio_value"],
                portfolio_returns=perf_data["portfolio_returns"],
                portfolio_pnl=perf_data["portfolio_pnl"],
                performance_metrics=perf_metrics,
                var_summary=var_results["var_summary"],
                backtest_summary=backtest_summary,
                asset_contrib=asset_contrib,
                mc_quantiles=mc["forecast_quantiles"],
                diagnostics=var_results["diagnostics"],
                var_series_dict=var_results["var_series_dict"],
                asset_returns=var_results.get("asset_log_returns"),
                market_value=perf_data.get("market_value"),
                asset_price_levels=var_results.get("asset_prices"),
            )
            st.session_state["excel_report_path"] = excel_path
            st.success("Reporting Excel généré avec succès.")

    with col_pdf:
        if st.button("Générer le reporting PDF", use_container_width=True):
            tmpdir = tempfile.gettempdir()
            pdf_path = os.path.join(tmpdir, "RiskManagementExecutiveReport.pdf")
            export_risk_pdf(
                output_path=pdf_path,
                portfolio_name=portfolio_name,
                selected_assets=selected_assets,
                selected_weights=selected_weights,
                capital_invested=capital_invested,
                confidence=confidence,
                horizon=horizon,
                performance_metrics=perf_metrics,
                var_summary=var_results["var_summary"],
                backtest_summary=backtest_summary,
                portfolio_value=perf_data["portfolio_value"],
                portfolio_pnl=perf_data["portfolio_pnl"],
                mc_quantiles=mc["forecast_quantiles"],
                asset_returns=var_results.get("asset_log_returns"),
                portfolio_source=portfolio_source,
                market_value=perf_data.get("market_value"),
                asset_price_levels=var_results.get("asset_prices"),
            )
            st.session_state["pdf_report_path"] = pdf_path
            st.success("Reporting PDF généré avec succès.")

    # ======================================================
    # BOUTONS DE TÉLÉCHARGEMENT
    # ======================================================
    dl1, dl2 = st.columns(2)

    with dl1:
        excel_path = st.session_state.get("excel_report_path")
        if excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="Télécharger le reporting Excel",
                    data=f,
                    file_name="RiskManagementResult.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    with dl2:
        pdf_path = st.session_state.get("pdf_report_path")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Télécharger le reporting PDF",
                    data=f,
                    file_name="RiskManagementExecutiveReport.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
    st.markdown("---")
    st.subheader("Envoyer le reporting par email")

    recipient_email = st.text_input(
        "Adresse email du destinataire",
        placeholder="directeur@entreprise.com"
    )

    if st.button("Envoyer le reporting par email"):

        if not recipient_email:
            st.warning("Veuillez saisir une adresse email.")

        else:
            try:

                send_report_email(
                    recipient_email=recipient_email,
                    subject="Reporting Risque de Marché",
                    body="""
    Bonjour,

    Veuillez trouver ci-joint le reporting de risque du portefeuille.

    Cordialement,
    FOJUMMA EQUITY
    """,
                    attachments=[excel_path, pdf_path]
                )

                st.success("Le reporting a été envoyé avec succès.")

            except Exception as e:
                st.error(f"Erreur lors de l'envoi : {e}")