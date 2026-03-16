import streamlit as st
import pandas as pd

from onglets.module.utils import load_price_data
from onglets.module.VarEstimation import compute_all_var_methods


DATA_FILE = "Data_set.xlsx"
INDEX_NAME = "Cac40"


def _format_pct(x: float) -> str:
    return f"{x:.2%}"


def render_settings_tab():
    st.title("Paramétrage")
    st.caption(
        "Définissez les paramètres d’estimation de la Value at Risk du portefeuille validé."
    )

    if not st.session_state.next_step:
        st.warning("Veuillez d’abord valider un portefeuille dans l’onglet Accueil.")
        return

    selected_assets = st.session_state.get("selected_portfolio_assets", [])
    selected_weights = st.session_state.get("selected_portfolio_weights", {})

    if not selected_assets or not selected_weights:
        st.error("Aucun portefeuille valide n’est disponible.")
        return

    st.subheader("Portefeuille retenu")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(f"**Nom du portefeuille :** {st.session_state.get('selected_portfolio_name', 'Portefeuille')}")
        st.write(f"**Source :** {st.session_state.get('selected_portfolio_source', 'N/A')}")
        st.write(f"**Actifs sélectionnés :** {', '.join(selected_assets)}")

    with col2:
        weights_df = pd.DataFrame({
            "Actif": list(selected_weights.keys()),
            "Poids (%)": [round(v * 100, 2) for v in selected_weights.values()]
        })
        st.dataframe(weights_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    st.subheader("Paramètres d’estimation de la VaR")

    c1, c2 = st.columns(2)

    with c1:
        capital = st.number_input(
            "Capital investi (€)",
            min_value=1000.0,
            max_value=100_000_000.0,
            value=float(st.session_state["capital_invested"]),
            step=1000.0,
        )
        st.session_state["capital_invested"] = capital

    with c2:
        confidence = st.selectbox(
            "Niveau de confiance",
            options=[0.95, 0.99],
            index=1 if st.session_state["var_confidence"] == 0.99 else 0,
            format_func=lambda x: f"{int(x * 100)} %",
        )
        st.session_state["var_confidence"] = confidence

    horizon = st.selectbox(
        "Horizon de détention",
        options=[1, 10],
        index=0 if st.session_state["var_horizon"] == 1 else 1,
        format_func=lambda x: f"{x} jour" if x == 1 else f"{x} jours",
    )
    st.session_state["var_horizon"] = horizon

    st.info(
        f"Un niveau de confiance de **{int(confidence * 100)} %** conduit à une VaR plus prudente "
        f"qu’un niveau de 95 %, car il retient un scénario de perte plus extrême. "
        f"En pratique, plus le niveau de confiance est élevé, plus la VaR estimée tend à augmenter."
    )

    st.markdown("---")

    if st.button("Lancer l’estimation de la VaR", use_container_width=True):
        try:
            with st.spinner("Calcul des estimations de VaR en cours..."):
                price_df = load_price_data(DATA_FILE)

                if INDEX_NAME in price_df.columns:
                    asset_price_df = price_df.drop(columns=[INDEX_NAME]).copy()
                else:
                    asset_price_df = price_df.copy()

                portfolio_assets_histo_price = asset_price_df[selected_assets].dropna(how="any").copy()

                results = compute_all_var_methods(
                    portfolio_prices_df=portfolio_assets_histo_price,
                    weights=selected_weights,
                    confidence=confidence,
                    horizon=horizon,
                    capital_invested=capital,
                )

            st.session_state["portfolio_assets_histo_price"] = portfolio_assets_histo_price
            st.session_state["var_results"] = results

            var_summary = results["var_summary"].copy()

            n_success = (var_summary["Status"] == "success").sum()
            n_warning = (var_summary["Status"] == "warning").sum()
            n_error = (var_summary["Status"] == "error").sum()

            if n_success + n_warning == 0:
                st.session_state["var_estimation_done"] = False
                st.error(
                    "Aucune méthode d’estimation de la VaR n’a pu aboutir. "
                    "Veuillez ajuster l’horizon, le portefeuille ou la profondeur historique."
                )
                st.dataframe(var_summary, use_container_width=True, hide_index=True)
                return

            st.session_state["var_estimation_done"] = True

            if n_error == 0 and n_warning == 0:
                st.success(
                    "L’estimation de la VaR a été réalisée avec succès pour l’ensemble des méthodes. "
                    "Veuillez maintenant consulter l’onglet Performance pour analyser les résultats du portefeuille."
                )
            else:
                st.warning(
                    f"L’estimation de la VaR est terminée : {n_success} méthode(s) en succès, "
                    f"{n_warning} en avertissement et {n_error} en échec. "
                    "Les résultats exploitables sont disponibles dans l’onglet Performance."
                )

            st.markdown("### Synthèse des méthodes estimées")
            st.dataframe(var_summary, use_container_width=True, hide_index=True)

            with st.expander("Interprétation méthodologique"):
                st.markdown(
                    """
                    Les méthodes de VaR ne reposent pas sur les mêmes hypothèses statistiques.
                    Certaines approches non paramétriques ou semi-paramétriques, notamment celles fondées sur les valeurs extrêmes,
                    peuvent devenir difficilement estimables si l’historique utile est trop court ou si le nombre d’observations extrêmes
                    est insuffisant sur l’horizon retenu.

                    Dans ce cadre, un statut :
                    - **success** signifie que la méthode a été estimée normalement ;
                    - **warning** signifie que la méthode a été estimée, avec certaines fenêtres non exploitables ignorées ;
                    - **error** signifie que la méthode n’a pas pu être estimée de façon fiable dans la configuration choisie.
                    """
                )

        except Exception as e:
            st.session_state["var_estimation_done"] = False
            st.session_state["var_results"] = None
            st.error(f"Erreur lors de l’estimation de la VaR : {e}")