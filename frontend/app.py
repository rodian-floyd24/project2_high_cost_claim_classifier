from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE_URL = os.environ.get("PREDICTION_API_URL", "http://127.0.0.1:8000")

STATE_OPTIONS = ["CA", "FL", "NY", "TX", "PA", "OH", "IL", "NC", "GA", "MI", "unknown"]
RACE_OPTIONS = ["1", "2", "3", "4", "5", "6", "unknown"]
INTERVENTION_OPTIONS = {
    "none": "No recent intervention",
    "recent_low_touch": "Recent low-touch outreach",
    "recent_intensive": "Recent intensive intervention",
}

PRESETS = {
    "Low-risk routine beneficiary": {
        "age_band": "65_74",
        "sex": "female",
        "race_code": "1",
        "state_code": "FL",
        "enrollment_months_count": 12,
        "chronic_condition_count": 1,
        "inpatient_claim_count": 0,
        "outpatient_claim_count": 2,
        "carrier_claim_count": 4,
        "pde_claim_count": 8,
        "total_claim_days": 3,
        "unique_provider_count": 3,
        "rx_total_cost": 650.0,
        "inpatient_total_cost": 0.0,
        "outpatient_total_cost": 420.0,
        "carrier_total_cost": 310.0,
        "prior_intervention_status": "none",
    },
    "Moderate chronic-care beneficiary": {
        "age_band": "75_84",
        "sex": "male",
        "race_code": "1",
        "state_code": "TX",
        "enrollment_months_count": 12,
        "chronic_condition_count": 4,
        "inpatient_claim_count": 1,
        "outpatient_claim_count": 8,
        "carrier_claim_count": 12,
        "pde_claim_count": 20,
        "total_claim_days": 11,
        "unique_provider_count": 7,
        "rx_total_cost": 3200.0,
        "inpatient_total_cost": 6400.0,
        "outpatient_total_cost": 2800.0,
        "carrier_total_cost": 2400.0,
        "prior_intervention_status": "recent_low_touch",
    },
    "Very-high-risk complex beneficiary": {
        "age_band": "85_plus",
        "sex": "female",
        "race_code": "2",
        "state_code": "NY",
        "enrollment_months_count": 12,
        "chronic_condition_count": 7,
        "inpatient_claim_count": 3,
        "outpatient_claim_count": 14,
        "carrier_claim_count": 25,
        "pde_claim_count": 26,
        "total_claim_days": 29,
        "unique_provider_count": 13,
        "rx_total_cost": 6400.0,
        "inpatient_total_cost": 18800.0,
        "outpatient_total_cost": 5400.0,
        "carrier_total_cost": 4300.0,
        "prior_intervention_status": "none",
    },
    "Recently managed very-high-risk beneficiary": {
        "age_band": "85_plus",
        "sex": "female",
        "race_code": "2",
        "state_code": "NY",
        "enrollment_months_count": 12,
        "chronic_condition_count": 8,
        "inpatient_claim_count": 2,
        "outpatient_claim_count": 12,
        "carrier_claim_count": 22,
        "pde_claim_count": 26,
        "total_claim_days": 24,
        "unique_provider_count": 11,
        "rx_total_cost": 5900.0,
        "inpatient_total_cost": 15100.0,
        "outpatient_total_cost": 5100.0,
        "carrier_total_cost": 4100.0,
        "prior_intervention_status": "recent_intensive",
    },
}


def post_json(endpoint: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="High-Cost Beneficiary Risk Studio",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --page-bg: #f3f7fb;
        --panel-bg: rgba(255,255,255,0.9);
        --panel-border: rgba(15, 23, 42, 0.10);
        --text-strong: #10243e;
        --text-muted: #4b5f79;
        --accent: #0f766e;
        --accent-2: #2457b8;
        --sidebar-bg: #1f2430;
    }
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(15,118,110,0.08), transparent 24%),
            radial-gradient(circle at 100% 0%, rgba(37,99,235,0.08), transparent 26%),
            linear-gradient(180deg, var(--page-bg) 0%, #edf3f9 100%);
        color: var(--text-strong);
    }
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #f5f7fb;
    }
    [data-testid="stSidebar"] .stCaption {
        color: rgba(245,247,251,0.72);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(15,118,110,0.95), rgba(30,64,175,0.92));
        color: white;
        box-shadow: 0 20px 50px rgba(15,23,42,0.18);
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(15,23,42,0.08);
        color: var(--text-strong);
    }
    .section-shell {
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 22px;
        padding: 1.25rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 12px 28px rgba(15,23,42,0.06);
        margin-top: 0.35rem;
    }
    .section-title {
        color: var(--text-strong);
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.85rem 0;
        letter-spacing: -0.03em;
    }
    .subtle-note {
        background: rgba(219,234,254,0.65);
        border: 1px solid rgba(59,130,246,0.16);
        color: var(--text-strong);
        border-radius: 16px;
        padding: 0.95rem 1rem;
    }
    .methodology-block {
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-left: 6px solid var(--accent-2);
        color: var(--text-strong);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin: 0.4rem 0 1rem 0;
    }
    .methodology-block strong {
        display: block;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }
    .stMarkdown h3 {
        color: var(--text-strong);
        margin-top: 0.2rem;
        margin-bottom: 0.65rem;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: var(--text-strong) !important;
        font-weight: 700;
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-muted);
    }
    .stAlert {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">High-Cost Beneficiary Risk Studio</h1>
      <p style="margin:0.6rem 0 0 0; font-size:1.05rem;">
        Interactive actuarial decision support. Enter current-year utilization and cost signals to estimate
        next-year high-cost risk, map the beneficiary into a discrete MDP state, and view a simulated
        intervention recommendation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Scenario presets")
    preset_name = st.selectbox("Load a starting profile", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    st.caption("These presets make the app demo-friendly and help explain the model behavior.")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Beneficiary profile</div>', unsafe_allow_html=True)
    demographic_col, enrollment_col = st.columns(2)
    with demographic_col:
        age_band = st.selectbox("Age band", ["under_65", "65_74", "75_84", "85_plus", "unknown"], index=["under_65", "65_74", "75_84", "85_plus", "unknown"].index(preset["age_band"]))
        sex = st.selectbox("Sex", ["female", "male", "unknown"], index=["female", "male", "unknown"].index(preset["sex"]))
        race_code = st.selectbox("Race code", RACE_OPTIONS, index=RACE_OPTIONS.index(preset["race_code"]))
        state_code = st.selectbox("State code", STATE_OPTIONS, index=STATE_OPTIONS.index(preset["state_code"]))
    with enrollment_col:
        enrollment_months_count = st.slider("Enrollment months", 0, 12, preset["enrollment_months_count"])
        chronic_condition_count = st.slider("Chronic condition count", 0, 11, preset["chronic_condition_count"])
        unique_provider_count = st.slider("Unique providers", 0, 25, preset["unique_provider_count"])
        total_claim_days = st.slider("Total claim days", 0, 60, preset["total_claim_days"])

    prior_intervention_status = st.selectbox(
        "Prior intervention status",
        list(INTERVENTION_OPTIONS.keys()),
        index=list(INTERVENTION_OPTIONS.keys()).index(preset["prior_intervention_status"]),
        format_func=lambda key: INTERVENTION_OPTIONS[key],
    )

    st.subheader("Utilization and cost profile")
    claim_col, cost_col = st.columns(2)
    with claim_col:
        inpatient_claim_count = st.slider("Inpatient claims", 0, 8, preset["inpatient_claim_count"])
        outpatient_claim_count = st.slider("Outpatient claims", 0, 30, preset["outpatient_claim_count"])
        carrier_claim_count = st.slider("Carrier claims", 0, 40, preset["carrier_claim_count"])
        pde_claim_count = st.slider("Prescription events", 0, 40, preset["pde_claim_count"])
    with cost_col:
        inpatient_total_cost = st.number_input("Inpatient cost ($)", min_value=0.0, step=100.0, value=float(preset["inpatient_total_cost"]))
        outpatient_total_cost = st.number_input("Outpatient cost ($)", min_value=0.0, step=100.0, value=float(preset["outpatient_total_cost"]))
        carrier_total_cost = st.number_input("Carrier cost ($)", min_value=0.0, step=100.0, value=float(preset["carrier_total_cost"]))
        rx_total_cost = st.number_input("Prescription cost ($)", min_value=0.0, step=100.0, value=float(preset["rx_total_cost"]))

    payload = {
        "age_band": age_band,
        "sex": sex,
        "race_code": race_code,
        "state_code": state_code,
        "enrollment_months_count": enrollment_months_count,
        "chronic_condition_count": chronic_condition_count,
        "inpatient_claim_count": inpatient_claim_count,
        "outpatient_claim_count": outpatient_claim_count,
        "carrier_claim_count": carrier_claim_count,
        "pde_claim_count": pde_claim_count,
        "total_claim_days": total_claim_days,
        "unique_provider_count": unique_provider_count,
        "rx_total_cost": rx_total_cost,
        "inpatient_total_cost": inpatient_total_cost,
        "outpatient_total_cost": outpatient_total_cost,
        "carrier_total_cost": carrier_total_cost,
        "prior_intervention_status": prior_intervention_status,
    }

    score_clicked = st.button("Score Beneficiary", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision-support summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="methodology-block">
          <strong>Methodology and limitation</strong>
          The risk engine is empirically trained on observed data, while the reinforcement-learning policy layer is a simulated decision prototype built on stylized transition and reward assumptions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if score_clicked:
        result = None
        state_response = None
        recommendation = None
        simulation = None
        section_errors: list[str] = []

        try:
            decision_support = post_json("/decision_support", payload)
            result = decision_support["prediction"]
            state_response = decision_support["state"]
            recommendation = decision_support["recommendation"]
            simulation = decision_support["simulation"]
        except requests.RequestException as exc:
            section_errors.append(f"Unified `/decision_support` endpoint unavailable: {exc}")
            endpoint_map = {
                "Risk prediction": "/predict",
                "Current state": "/state",
                "Recommendation": "/recommend_action",
                "Simulation": "/simulate",
            }
            for section_name, endpoint in endpoint_map.items():
                try:
                    payload_response = post_json(endpoint, payload)
                    if endpoint == "/predict":
                        result = payload_response
                    elif endpoint == "/state":
                        state_response = payload_response
                    elif endpoint == "/recommend_action":
                        recommendation = payload_response
                    elif endpoint == "/simulate":
                        simulation = payload_response
                except requests.RequestException as endpoint_exc:
                    section_errors.append(f"{section_name} unavailable from `{endpoint}`: {endpoint_exc}")

        if section_errors:
            for message in section_errors:
                st.warning(message)

        if result is not None:
            probability = result["risk_probability"]
            annual_cost_proxy = result["annual_claim_cost_proxy"]
            st.markdown("### Risk prediction")
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Risk probability", f"{probability:.1%}")
            metric_b.metric("Risk tier", result["risk_tier"])
            metric_c.metric("Predicted class", "High-cost" if result["predicted_high_cost"] else "Not high-cost")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Next-year high-cost risk"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0f766e"},
                        "steps": [
                            {"range": [0, result["decision_threshold"] * 100], "color": "#dbeafe"},
                            {"range": [result["decision_threshold"] * 100, 100], "color": "#bfdbfe"},
                        ],
                        "threshold": {
                            "line": {"color": "#b45309", "width": 4},
                            "thickness": 0.75,
                            "value": result["decision_threshold"] * 100,
                        },
                    },
                )
            )
            gauge.update_layout(height=300, margin=dict(l=15, r=15, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            cost_df = pd.DataFrame(
                {
                    "category": ["Inpatient", "Outpatient", "Carrier", "Prescription"],
                    "amount": [
                        result["cost_mix"]["inpatient"],
                        result["cost_mix"]["outpatient"],
                        result["cost_mix"]["carrier"],
                        result["cost_mix"]["prescription"],
                    ],
                }
            )
            cost_fig = px.bar(
                cost_df,
                x="amount",
                y="category",
                orientation="h",
                color="category",
                color_discrete_sequence=["#0f766e", "#2563eb", "#0ea5e9", "#b45309"],
                title=f"Current-year cost mix (${annual_cost_proxy:,.0f} total)",
            )
            cost_fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(cost_fig, use_container_width=True)
        else:
            st.warning("Risk prediction is unavailable right now.")

        current_state = None
        if recommendation is not None:
            current_state = recommendation["current_state"]
        elif state_response is not None:
            current_state = state_response["current_state"]

        if current_state is not None:
            st.markdown("### Current MDP state")
            state_col_a, state_col_b = st.columns(2)
            state_col_a.markdown(
                f"""
                <div class="metric-card">
                  <strong>Discrete state</strong><br/>
                  {current_state["label"]}<br/><br/>
                  <strong>State ID:</strong> {current_state["state_id"]}<br/>
                  <strong>Prior intervention:</strong> {INTERVENTION_OPTIONS[current_state["prior_intervention_status"]]}
                </div>
                """,
                unsafe_allow_html=True,
            )
            state_col_b.markdown(
                f"""
                <div class="metric-card">
                  <strong>State components</strong><br/>
                  Risk tier: {current_state["risk_tier"]}<br/>
                  Chronic burden: {current_state["chronic_burden"]}<br/>
                  Utilization intensity: {current_state["utilization_intensity"]}<br/>
                  Baseline risk: {current_state["baseline_risk_probability"]:.1%}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Current MDP state is unavailable right now.")

        if recommendation is not None:
            st.markdown("### Recommended action")
            action_col_a, action_col_b = st.columns([0.95, 1.05])
            action_col_a.markdown(
                f"""
                <div class="metric-card">
                  <strong>Policy recommendation</strong><br/>
                  {recommendation["recommended_action_display"]}<br/><br/>
                  <strong>Estimated long-run value:</strong> {recommendation["expected_long_run_value"]:.2f}<br/>
                  <strong>Policy explanation:</strong> {recommendation["policy_explanation"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
            q_values_df = pd.DataFrame(recommendation["action_values"])
            q_values_df["is_recommended"] = q_values_df["action"] == recommendation["recommended_action"]
            q_values_df["bar_color"] = q_values_df["is_recommended"].map(
                {True: "#0f766e", False: "#94a3b8"}
            )
            q_values_df["display_value"] = q_values_df["q_value"].map(lambda value: f"{value:.2f}")
            q_fig = px.bar(
                q_values_df.sort_values("q_value"),
                x="q_value",
                y="action_label",
                orientation="h",
                color="bar_color",
                color_discrete_map="identity",
                text="display_value",
                title="Estimated long-run value by action",
            )
            q_fig.update_traces(
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Estimated long-run value: %{x:.2f}<extra></extra>",
                cliponaxis=False,
            )
            q_fig.update_layout(
                height=340,
                showlegend=False,
                margin=dict(l=10, r=35, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                xaxis_title="Estimated long-run value (higher is better)",
                yaxis_title="",
                xaxis=dict(
                    zeroline=True,
                    zerolinecolor="#cbd5e1",
                    gridcolor="#e2e8f0",
                    tickfont=dict(color="#10243e"),
                    title_font=dict(color="#10243e"),
                ),
                yaxis=dict(
                    tickfont=dict(color="#10243e"),
                ),
                title_font=dict(color="#10243e"),
                font=dict(color="#10243e"),
            )
            action_col_b.plotly_chart(q_fig, use_container_width=True)
        else:
            st.warning("Recommendation output is unavailable right now.")

        if simulation is not None:
            st.markdown("### Action-by-action comparison")
            comparison_df = pd.DataFrame(simulation["comparisons"])
            comparison_df["Expected next risk"] = comparison_df["expected_next_risk_probability"].map(lambda value: f"{value:.1%}")
            comparison_df["Risk delta"] = comparison_df["expected_risk_delta"].map(lambda value: f"{value:+.1%}")
            comparison_df["Immediate reward"] = comparison_df["expected_immediate_reward"].map(lambda value: f"{value:.2f}")
            comparison_df["Long-run value"] = comparison_df["q_value"].map(lambda value: f"{value:.2f}")
            comparison_table = comparison_df[
                ["action_label", "Expected next risk", "Risk delta", "Immediate reward", "Long-run value", "is_recommended"]
            ].rename(columns={"action_label": "Action", "is_recommended": "Recommended"})
            st.dataframe(comparison_table, use_container_width=True, hide_index=True)

            st.markdown(
                f"""
                <div class="metric-card">
                  <strong>Modeling note</strong><br/>
                  The risk engine is empirically trained on observed data, while the reinforcement-learning policy layer is a simulated decision prototype built on stylized transition and reward assumptions.<br/><br/>
                  Risk engine: empirically trained supervised model.<br/>
                  MDP state: discrete operational state derived from beneficiary features and current risk.<br/>
                  Policy recommendation: action with highest estimated long-run value within the simulated environment.<br/>
                  Recommendation: action with highest estimated long-run value within that simulated environment, not a causal claim about real intervention effects.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Action-by-action simulation is unavailable right now.")

        if result is not None:
            with st.expander("Show engineered feature snapshot"):
                engineered = pd.DataFrame(
                    [{"feature": key, "value": value} for key, value in result["engineered_features"].items()]
                )
                st.dataframe(engineered, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="subtle-note">
              Choose a profile and click <strong>Score Beneficiary</strong> to run the risk engine and simulated policy layer.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
foot_a, foot_b, foot_c = st.columns(3)
foot_a.caption("Model: gradient boosting")
foot_b.caption("Target: next-year high-cost beneficiary")
foot_c.caption("Pipeline: Databricks bronze → silver → gold → MLflow → FastAPI → Streamlit")
