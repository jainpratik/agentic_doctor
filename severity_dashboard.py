# Generate severity_dashboard.py to visualize qSOFA, CURB-65, A1C, BP using Plotly

import plotly.graph_objects as go
import streamlit as st


def show_score_gauges(scores):
    st.subheader("📈 Severity Score Gauges")

    # qSOFA gauge
    qsofa_val = scores.get("qsofa", 0)
    st.plotly_chart(
        go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=qsofa_val,
                title={"text": "qSOFA Score"},
                gauge={
                    "axis": {"range": [0, 3]},
                    "bar": {
                        "color": (
                            "red"
                            if qsofa_val >= 2
                            else "orange" if qsofa_val == 1 else "green"
                        )
                    },
                },
            )
        ),
        use_container_width=True,
    )

    # CURB-65 gauge
    curb_val = scores.get("curb65", 0)
    st.plotly_chart(
        go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=curb_val,
                title={"text": "CURB-65 Score"},
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {
                        "color": (
                            "red"
                            if curb_val >= 3
                            else "orange" if curb_val == 2 else "green"
                        )
                    },
                },
            )
        ),
        use_container_width=True,
    )

    # A1C gauge
    a1c_val = scores.get("A1C", 0)
    st.plotly_chart(
        go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=a1c_val,
                title={"text": "A1C (%)"},
                gauge={
                    "axis": {"range": [4, 12]},
                    "bar": {
                        "color": (
                            "red"
                            if a1c_val >= 6.5
                            else "orange" if a1c_val >= 5.7 else "green"
                        )
                    },
                },
            )
        ),
        use_container_width=True,
    )

    # BP gauge (Systolic)
    sbp_val = scores.get("SBP", 0)
    st.plotly_chart(
        go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sbp_val,
                title={"text": "Systolic BP (mmHg)"},
                gauge={
                    "axis": {"range": [90, 200]},
                    "bar": {
                        "color": (
                            "red"
                            if sbp_val >= 140
                            else "orange" if sbp_val >= 130 else "green"
                        )
                    },
                },
            )
        ),
        use_container_width=True,
    )
