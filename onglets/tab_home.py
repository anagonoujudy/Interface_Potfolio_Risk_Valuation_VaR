import streamlit as st

from onglets.module.utils import load_price_data, split_asset_and_market_returns
from onglets.module.capm import *


DATA_FILE = "Data_set.xlsx"
INDEX_NAME = "Cac40"
SECTOR_BUCKETS = {
    "Energie": ["BOLL_Price", "ENGIE_Price", "TTEF_Price"],
    "Finance": ["BNPP", "AXA SA"],
    "Industrie": ["AirbusSE", "Safran SA", "Air Liquide SA", "DSY (Dassault)"],
    "Luxe": ["LVMH", "OREP", "PRTP"],
    "Technologie": ["STMPA (STMicroelectronics)", "Capegimini", "CAGR"],
}


def _format_pct(x: float) -> str:
    return f"{x:.2%}" if x is not None else "-"


def _format_weights(weights: dict) -> str:
    return " | ".join([f"{k}: {v:.0%}" for k, v in weights.items()])


def _load_selected_portfolio(row, source):
    st.session_state["selected_portfolio_name"] = row["portfolio_name"]
    st.session_state["selected_portfolio_assets"] = row["assets"]
    st.session_state["selected_portfolio_weights"] = row["weights"]
    st.session_state["selected_portfolio_source"] = source
    st.session_state["portfolio_ready"] = True

def _get_sector_buckets(asset_returns):
    assets_tickers = list(asset_returns.columns)

    return {
        "Luxe": assets_tickers[:3],
        "Industrie": assets_tickers[3:6],
        "Energie": assets_tickers[6:9],
        "Technologie": assets_tickers[9:12],
        "Finance": assets_tickers[12:],
    }


def render_home_tab():
    st.title("FOJUMMA EQUITY")
    st.markdown("### Plateforme de Management du Risque de Marché")
    st.caption(
        "Sélectionnez un portefeuille recommandé par le moteur d’optimisation "
        "ou construisez librement votre portefeuille personnalisé."
    )

    try:
        price_df = load_price_data(DATA_FILE)
        asset_returns, market_returns = split_asset_and_market_returns(price_df, index_col=INDEX_NAME)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Actions disponibles", asset_returns.shape[1])
    col2.metric("Indice de référence", INDEX_NAME)
    col3.metric("Portefeuilles suggérés", 4)
    col4.metric("Taille des suggestions", "5 actifs")

    st.markdown("---")

    st.subheader("Portefeuilles recommandés")
    st.write(
        "Les portefeuilles ci-dessous sont construits à partir des rendements attendus estimés "
        "par le CAPM, avec le CAC 40 comme indice de marché, puis classés selon un compromis "
        "entre rendement attendu et risque."
    )

    with st.spinner("Génération des portefeuilles suggérés..."):
        try:
            top4 = get_top_4_portfolio_suggestions(
                asset_returns=asset_returns,
                market_returns=market_returns
            )
        except Exception as e:
            st.error(f"Impossible de générer les portefeuilles suggérés : {e}")
            return

    cols = st.columns(2)
    for i, (_, row) in enumerate(top4.iterrows()):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {row['portfolio_name']}")
                st.caption(row["profile"])

                st.write(f"**Actions :** {', '.join(row['assets'])}")
                st.write(f"**Poids proposés :** {_format_weights(row['weights'])}")

                c1, c2 = st.columns(2)
                c1.metric("Rendement espéré", _format_pct(row["expected_return"]))
                c2.metric("Volatilité", _format_pct(row["volatility"]))

                c3, c4 = st.columns(2)
                c3.metric("Bêta portefeuille", f"{row['beta']:.2f}")
                c4.metric("Rendement / Risque", f"{row['score']:.2f}")

                st.write(f"**Concentration :** {row['concentration']}")

                if st.button("Utiliser ce portefeuille", key=f"pf_{i}", use_container_width=True):
                    source = "recommandé"
                    _load_selected_portfolio(row, source)
                    st.success(
                        f"{row['portfolio_name']} chargé avec succès. "
                        "Vous pouvez poursuivre dans l’onglet Paramétrage."
                    )
    with st.expander("Méthodologie de sélection"):
        st.markdown(
            f"""
            - les rendements attendus des actions sont estimés par le **CAPM** ;
            - l’indice **{INDEX_NAME}** sert de proxy du marché ;
            - l’indice n’est **pas** inclus dans les portefeuilles ;
            - chaque portefeuille suggéré contient **5 actions** ;
            - la volatilité est estimée à partir de la matrice de covariance historique ;
            - les suggestions sont indicatives et destinées à aider l’utilisateur dans sa sélection.
            """
        )

    st.subheader("Constituer un portefeuille personnalisé")
    st.write(
        "Vous pouvez ignorer ces suggestions et sélectionner librement vos actifs, "
        "puis choisir une stratégie d’allocation."
    )

    if st.button("Créer mon portefeuille manuellement", use_container_width=True):
        st.session_state["custom_mode_active"] = True
        st.session_state["selected_portfolio_source"] = "custom"
        st.session_state["portfolio_ready"] = False

    if st.session_state["custom_mode_active"]:
        sector_buckets = _get_sector_buckets(asset_returns)

        st.markdown("#### Choix des actifs selon le secteur de préférence")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            energy_choice = st.multiselect("**Energie**", sector_buckets["Energie"], key="energy_choice")
        with c2:
            finance_choice = st.multiselect("**Finance**", sector_buckets["Finance"], key="finance_choice")
        with c3:
            industry_choice = st.multiselect("**Industrie**", sector_buckets["Industrie"], key="industry_choice")
        with c4:
            luxury_choice = st.multiselect("**Luxe**", sector_buckets["Luxe"], key="luxury_choice")
        with c5:
            tech_choice = st.multiselect("**Technologie**", sector_buckets["Technologie"], key="tech_choice")

        portfolio_assets = (
            energy_choice
            + finance_choice
            + industry_choice
            + luxury_choice
            + tech_choice
        )

        st.markdown("#### Stratégie de répartition des poids")

        strategy = st.radio(
            "Choisissez une stratégie",
            [
                "Portefeuille équilibré",
                "Portefeuille personnalisé",
                "Portefeuille optimisé CAPM",
            ],
            horizontal=True,
            key="custom_strategy",
        )

        final_weights = None
        summary_metrics = None
#---------------------------------------------------------------------------
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1 : 
            if st.checkbox("Valider"): 
                st.session_state.portfolio_validated = True
        if st.session_state.portfolio_validated == True:
            if strategy == "Portefeuille équilibré":
                st.info("Chaque actif sélectionné reçoit le même poids.")
                try:
                    final_weights = compute_equal_weights(portfolio_assets)
                except Exception as e:
                    st.error(str(e))

            elif strategy == "Portefeuille personnalisé":
                st.info("Saisissez vos pondérations manuellement. La somme doit être égale à 100 %.")
                entered_weights = {}

                if portfolio_assets:
                    cols = st.columns(2)
                    default_w = round(100 / len(portfolio_assets), 2)

                    for i, asset in enumerate(portfolio_assets):
                        with cols[i % 2]:
                            entered_weights[asset] = st.number_input(
                                f"Poids {asset} (%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=default_w,
                                step=0.5,
                                key=f"custom_weight_{asset}",
                            )

                    st.write(f"**Somme des poids saisis :** {sum(entered_weights.values()):.2f} %")

                try:
                    final_weights = compute_custom_weights_from_percent(portfolio_assets, entered_weights)
                except Exception as e:
                    st.warning(str(e))

            elif strategy == "Portefeuille optimisé CAPM":
                objective_type = st.radio(
                    "Optimisation basée sur :",
                    ["Rendement cible", "Volatilité cible"],
                    horizontal=True,
                    key="custom_objective_type",
                )

                if objective_type == "Rendement cible":
                    target_return = st.number_input(
                        "Rendement annuel souhaité",
                        min_value=0.01,
                        max_value=1.00,
                        value=float(st.session_state["custom_target_return"]),
                        step=0.01,
                        format="%.2f",
                        key="custom_target_return",
                    )
                    target_volatility = None
                else:
                    target_volatility = st.number_input(
                        "Volatilité annuelle souhaitée",
                        min_value=0.01,
                        max_value=1.00,
                        value=float(st.session_state["custom_target_volatility"]),
                        step=0.01,
                        format="%.2f",
                        key="custom_target_volatility",
                    )
                    target_return = None

                try:
                    opt_result = compute_capm_optimized_weights(
                        selected_assets=portfolio_assets,
                        asset_returns=asset_returns,
                        market_returns=market_returns,
                        target_return=target_return,
                        target_volatility=target_volatility,
                    )
                    final_weights = opt_result["weights"]
                    summary_metrics = opt_result

                    if opt_result["target_met"]:
                        st.success(opt_result["message"])
                    else:
                        st.warning(opt_result["message"])

                    st.write(
                        f"**Rendement estimé :** {_format_pct(opt_result['expected_return'])} | "
                        f"**Volatilité estimée :** {_format_pct(opt_result['volatility'])} | "
                        f"**Bêta estimé :** {opt_result['beta']:.2f}"
                    )

                except Exception as e:
                    st.error(f"Erreur d’optimisation : {e}")

        if portfolio_assets:
            st.markdown("#### Portefeuille sélectionné")
            st.write(", ".join(portfolio_assets))
        else:
            st.info("Aucun actif sélectionné pour le moment.")

        if final_weights:
            st.markdown("#### Poids retenus")
            weights_df = pd.DataFrame(
                {
                    "Actif": list(final_weights.keys()),
                    "Poids (%)": [round(v * 100, 2) for v in final_weights.values()],
                }
            )
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

        if st.button("Valider le portefeuille personnalisé", use_container_width=True):
            if not portfolio_assets:
                st.error("Veuillez sélectionner au moins un actif.")
            elif not final_weights:
                st.error("Les poids du portefeuille ne sont pas valides.")
            else:
                st.session_state["selected_portfolio_name"] = "Portefeuille personnalisé"
                st.session_state["selected_portfolio_assets"] = portfolio_assets
                st.session_state["selected_portfolio_weights"] = final_weights
                st.session_state["selected_portfolio_source"] = "custom"
                st.session_state["portfolio_ready"] = True

                st.success("Portefeuille personnalisé validé. Vous pouvez poursuivre dans l’onglet Paramétrage.")
        with st.expander("Méthodologie d’optimisation du portefeuille (Modèle CAPM)"):

            st.markdown(
            """
            ### Construction du portefeuille selon le modèle CAPM

            Le portefeuille présenté ci-dessus est construit à partir du modèle théorique 
            du **Capital Asset Pricing Model (CAPM)**, largement utilisé dans la gestion 
            d’actifs et l’analyse du risque de marché.

            ---
            
            ### 1. Estimation des paramètres du modèle

            Pour chaque action sélectionnée, les éléments suivants sont estimés :

            - **Rendements logarithmiques journaliers** calculés à partir des prix historiques.
            - **Bêta de l’actif**, mesurant la sensibilité de l’action aux variations du marché.
            - L’indice **CAC 40** est utilisé comme **proxy du portefeuille de marché**.

            Le bêta est estimé selon la relation :

            βᵢ = Cov(Rᵢ , Rₘ) / Var(Rₘ)

            où :

            - Rᵢ représente le rendement de l’actif
            - Rₘ représente le rendement du marché (CAC 40)

            ---
            
            ### 2. Estimation du rendement attendu des actifs

            Le rendement attendu de chaque action est estimé à l’aide de la formule CAPM :

            E(Rᵢ) = Rf + βᵢ (E(Rₘ) − Rf)

            avec :

            - **Rf** : taux sans risque annuel
            - **E(Rₘ)** : rendement attendu du marché
            - **βᵢ** : sensibilité de l’actif au marché

            Cette estimation permet d’obtenir un rendement attendu cohérent avec le niveau 
            de risque systématique de chaque actif.

            ---
            
            ### 3. Mesure du risque du portefeuille

            Le risque du portefeuille est mesuré par sa **volatilité annuelle**, obtenue 
            à partir de la **matrice de covariance des rendements** :

            σₚ = √( wᵀ Σ w )

            où :

            - **w** : vecteur des poids du portefeuille
            - **Σ** : matrice de covariance des rendements des actifs

            Cette mesure intègre les effets de **diversification entre actifs**.

            ---
            
            ### 4. Détermination d’un intervalle réaliste pour les objectifs

            Avant de procéder à l’optimisation, il est nécessaire de déterminer si les 
            objectifs fixés par l’utilisateur (rendement ou volatilité cible) sont 
            **réalistes au regard des caractéristiques des actifs sélectionnés**.

            #### Intervalle réaliste de rendement

            Dans un portefeuille long-only (sans ventes à découvert), le rendement du 
            portefeuille est une combinaison pondérée des rendements attendus des actifs.

            Ainsi :

            min(E(Rᵢ)) ≤ E(Rₚ) ≤ max(E(Rᵢ))

            où :

            - **E(Rᵢ)** : rendement attendu de chaque actif
            - **E(Rₚ)** : rendement attendu du portefeuille

            Cela signifie que le rendement cible doit nécessairement se situer entre le 
            rendement attendu de l’actif le moins performant et celui de l’actif le plus performant.

            #### Intervalle réaliste de volatilité

            De manière similaire, la volatilité du portefeuille dépend des volatilités 
            individuelles des actifs ainsi que de leurs corrélations.

            Dans une approche simplifiée, la volatilité cible doit se situer dans un intervalle :

            σ_min ≤ σₚ ≤ σ_max

            où :

            - **σ_min** correspond approximativement à la volatilité de l’actif le moins risqué
            - **σ_max** correspond à celle de l’actif le plus risqué

            Si l’objectif fixé sort de cet intervalle, il est considéré comme **incompatible 
            avec les caractéristiques des actifs sélectionnés**.

            ---
            
            ### 5. Optimisation du portefeuille

            L’optimisation consiste à rechercher les pondérations du portefeuille respectant 
            les contraintes suivantes :

            - Somme des poids = 100 %
            - Positions **long-only** (pas de ventes à découvert)
            - Allocation uniquement parmi les actifs sélectionnés

            Selon le mode choisi par l’utilisateur, l’algorithme recherche :

            - un portefeuille dont le **rendement attendu est proche d’un rendement cible**
            - ou un portefeuille dont la **volatilité est proche d’une volatilité cible**

            L’optimisation est réalisée par **simulation d’un grand nombre de portefeuilles 
            admissibles**, puis sélection du portefeuille minimisant l’écart avec l’objectif fixé.

            ---
            
            ### 6. Indicateurs présentés

            Les indicateurs affichés dans l’interface permettent d’évaluer la stratégie proposée :

            - **Rendement attendu du portefeuille**
            - **Volatilité estimée**
            - **Bêta du portefeuille par rapport au marché**
            - **Ratio rendement / risque**

            Ces indicateurs permettent au gestionnaire ou au responsable du risque 
            d’apprécier le compromis entre **performance attendue et niveau de risque**.
            """
            )
    st.markdown("--")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c7 : 
        if st.button("Valider tout"): 
           st.session_state.next_step = True
