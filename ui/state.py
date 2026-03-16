import streamlit as st


def init_session_state():
    defaults = {
        "selected_portfolio_name": None,
        "selected_portfolio_assets": [],
        "selected_portfolio_weights": {},
        "selected_portfolio_source": None,   # recommended | custom
        "portfolio_ready": False,
        "portfolio_validated": False,
        "next_step": False,

        "custom_mode_active": False,
        "custom_strategy": "Portefeuille équilibré",
        "custom_objective_type": "Rendement cible",
        "custom_target_return": 0.12,
        "custom_target_volatility": 0.20,

        # navigation
        "active_tab": "Accueil",

        
        # paramétrage VaR
        "capital_invested": 10000.0,
        "var_confidence": 0.99,
        "var_horizon": 1,

        # résultats VaR
        "var_estimation_done": False,
        "var_results": None,
        "portfolio_assets_histo_price": None,

        # performances / exports
        "performance_ready": False,
        "excel_report_path": None,
        "pdf_report_path": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
