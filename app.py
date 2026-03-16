import streamlit as st
import streamlit.components.v1 as components
from ui.theme import apply_theme

from onglets.tab_home import render_home_tab
from onglets.tab_settings import render_settings_tab
from onglets.tab__perform import render_performance_tab

apply_theme()

from ui.state import init_session_state


def main():
    st.set_page_config(
        page_title="Risk Management Platform",
        layout="wide"
    )

    init_session_state()

    tabs = st.tabs(["Accueil", "Paramétrage", "Performance"])

    with tabs[0]:
        render_home_tab()

    with tabs[1]:
        render_settings_tab()

    with tabs[2]:
        render_performance_tab()


if __name__ == "__main__":
    main()