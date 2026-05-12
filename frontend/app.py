from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE_URL = os.environ.get("PREDICTION_API_URL", "http://127.0.0.1:8000")

AGE_OPTIONS = ["under_65", "65_74", "75_84", "85_plus", "unknown"]
SEX_OPTIONS = ["female", "male", "unknown"]
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

def fetch_model_metrics() -> list[tuple[str, str, str]]:
    try:
        response = requests.get(f"{API_BASE_URL}/model_metrics", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data:
            model_display_name = data.get('model_name', 'model').replace('_', ' ').capitalize()
            return [
                ("Test PR-AUC", f"{data.get('pr_auc', 0.0):.3f}", f"{model_display_name} holdout result"),
                ("Top-5% Capture", f"{data.get('top_5_capture_rate', 0.0):.1%}", "High-cost cases found in highest-risk 5%"),
                ("Top-10% Capture", f"{data.get('top_10_capture_rate', 0.0):.1%}", "High-cost cases found in highest-risk 10%"),
                ("Brier Score", f"{data.get('brier_score', 0.0):.3f}", "Probability calibration error, lower is better"),
            ]
    except requests.RequestException:
        pass
    return [
        ("Test PR-AUC", "N/A", "Model metrics unavailable. Check backend artifact metadata."),
        ("Top-5% Capture", "N/A", "Model metrics unavailable. Check backend artifact metadata."),
        ("Top-10% Capture", "N/A", "Model metrics unavailable. Check backend artifact metadata."),
        ("Brier Score", "N/A", "Model metrics unavailable. Check backend artifact metadata."),
    ]


def post_json(endpoint: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def display_tier(tier: str) -> str:
    return str(tier).replace("_", " ").title()


def load_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f7f9fc;
            --ink: #102a43;
            --muted: #52616b;
            --navy: #1f4e79;
            --blue: #2457b8;
            --line: #dde6f0;
            --soft: #eef3f8;
            --white: #ffffff;
        }
        .stApp {
            background: var(--page-bg);
            color: var(--ink);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: #eef3f8;
            border-right: 1px solid #dbe4ee;
        }
        [data-testid="stSidebar"] * {
            color: #102a43;
        }
        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1rem 0 1.4rem 0;
            color: var(--ink);
        }
        .brand {
            font-size: 1.05rem;
            font-weight: 900;
            color: var(--ink);
            letter-spacing: 0;
        }
        .nav-links {
            display: flex;
            gap: 1.35rem;
            flex-wrap: wrap;
            font-size: 0.95rem;
        }
        .nav-links a {
            color: var(--navy) !important;
            text-decoration: none !important;
            font-weight: 700;
        }
        .hero {
            padding: 4rem 3rem 3.6rem 3rem;
            background: linear-gradient(135deg, #ffffff 0%, #eef3f8 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 18px 45px rgba(31, 78, 121, 0.08);
            margin-bottom: 1.4rem;
        }
        .eyebrow {
            color: var(--navy);
            font-size: 0.84rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.7rem;
        }
        .hero h1 {
            color: var(--ink);
            font-size: 3.35rem;
            line-height: 1.05;
            margin: 0 0 1rem 0;
            letter-spacing: 0;
        }
        .subtitle {
            color: #334e68;
            font-size: 1.22rem;
            line-height: 1.62;
            max-width: 720px;
            margin: 0;
        }
        .button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            margin-top: 1.7rem;
        }
        .primary-btn, .secondary-btn {
            border-radius: 999px;
            display: inline-block;
            font-weight: 800;
            padding: 0.78rem 1.15rem;
            text-decoration: none !important;
        }
        .primary-btn {
            background: var(--navy);
            color: white !important;
        }
        .secondary-btn {
            background: white;
            border: 1px solid #afc4d8;
            color: var(--navy) !important;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1.1rem 0 2rem 0;
        }
        .metric-card, .project-card, .note-card {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
        }
        .metric-card {
            padding: 1rem 1.05rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 700;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 850;
            margin: 0.2rem 0;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .section-heading {
            color: var(--ink);
            font-size: 1.85rem;
            font-weight: 850;
            margin: 2rem 0 0.7rem 0;
            letter-spacing: 0;
        }
        .section-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
            margin-bottom: 1rem;
            max-width: 760px;
        }
        .project-card {
            min-height: 178px;
            padding: 1.25rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .project-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.075);
        }
        .card-kicker {
            color: var(--navy);
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .project-card h3 {
            color: var(--ink);
            font-size: 1.08rem;
            margin: 0 0 0.55rem 0;
        }
        .project-card p {
            color: var(--muted);
            line-height: 1.56;
            margin: 0;
        }
        .stack-band {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .stack-list {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .stack-item {
            background: #f7f9fc;
            border: 1px solid #e4ecf4;
            border-radius: 14px;
            padding: 0.9rem;
            color: var(--ink);
            font-weight: 750;
            text-align: center;
        }
        .app-module {
            background: linear-gradient(180deg, #ffffff 0%, #f4f8fc 100%);
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: 0 18px 45px rgba(31, 78, 121, 0.07);
            margin-top: 2rem;
            padding: 2rem;
        }
        .scorer-shell {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.055);
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .scorer-heading {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 850;
            margin-bottom: 0.3rem;
        }
        .method-note {
            background: #f4f8fc;
            border-left: 5px solid var(--blue);
            border-radius: 14px;
            color: #334e68;
            line-height: 1.55;
            margin-bottom: 1rem;
            padding: 0.95rem 1rem;
        }
        .result-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 0.6rem;
            margin: 1rem 0 1.2rem 0;
        }
        .result-strip > div {
            background: white;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.8rem 0.6rem;
            text-align: center;
        }
        .result-label {
            display: block;
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .result-strip strong {
            color: var(--ink);
            display: block;
            font-size: 1.25rem;
            line-height: 1.2;
        }
        .recommendation-card {
            background: #e9f7ef;
            border: 1px solid #bfe7cd;
            border-left: 6px solid #16803c;
            border-radius: 12px;
            color: #0f5132;
            padding: 0.8rem 1rem;
        }
        .recommendation-label {
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
        }
        .recommendation-title {
            font-size: 1.15rem;
            font-weight: 850;
            line-height: 1.2;
            margin-bottom: 0.3rem;
        }
        .recommendation-value {
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .recommendation-card p {
            font-size: 0.85rem;
            line-height: 1.4;
            margin: 0;
            color: #146c43;
        }
        .state-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #64748b;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #334155;
            height: 100%;
            font-size: 0.9rem;
        }
        .state-card strong {
            color: #0f172a;
            font-size: 1rem;
        }
        .state-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 1.5rem;
        }
        .footer {
            border-top: 1px solid var(--line);
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 2.5rem;
            padding: 1.2rem 0 0.2rem 0;
            text-align: center;
        }
        .stButton > button {
            border-radius: 999px;
            font-weight: 800;
        }
        .stAlert {
            border-radius: 14px;
        }
        .stMetric {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.9rem;
        }
        @media (max-width: 900px) {
            .hero {
                padding: 2.2rem 1.35rem;
            }
            .hero h1 {
                font-size: 2.35rem;
            }
            .subtitle {
                font-size: 1.04rem;
            }
            .metric-grid, .stack-list, .result-strip, .state-grid {
                grid-template-columns: 1fr;
            }
            .app-module {
                padding: 1.1rem;
            }
            .top-nav {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards() -> None:
    metrics = fetch_model_metrics()
    cards = "".join(
        (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-note">{note}</div>'
            f"</div>"
        )
        for label, value, note in metrics
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def render_project_cards() -> None:
    col1, col2, col3 = st.columns(3)
    cards = [
        (
            col1,
            "Predictive Modeling",
            "High-Cost Claim Classifier",
            "Prospective healthcare risk model using medallion data engineering, calibrated scoring, lift curves, and top-k capture diagnostics.",
        ),
        (
            col2,
            "Policy Prototype",
            "RL Decision Support Layer",
            "A simulated intervention-policy prototype that maps scored beneficiaries into discrete MDP states and compares long-run action values.",
        ),
        (
            col3,
            "Cloud Pipeline",
            "Distributed ML Pipeline",
            "Databricks bronze, silver, and gold layers feeding MLflow artifacts, FastAPI serving, and public Streamlit deployment.",
        ),
    ]
    for column, kicker, title, body in cards:
        with column:
            st.markdown(
                f'<div class="project-card"><div class="card-kicker">{kicker}</div><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )


def render_stack() -> None:
    items = ["Databricks", "scikit-learn", "XGBoost", "FastAPI", "Streamlit"]
    html_items = "".join(f'<div class="stack-item">{item}</div>' for item in items)
    st.markdown(
        f'<div class="stack-band"><div class="stack-list">{html_items}</div></div>',
        unsafe_allow_html=True,
    )


def build_payload() -> tuple[dict, bool]:
    st.markdown('<div class="scorer-shell">', unsafe_allow_html=True)
    st.markdown('<div class="scorer-heading">Beneficiary risk scorer</div>', unsafe_allow_html=True)
    st.caption("Enter current-year utilization and cost signals. The app estimates next-year high-cost risk and returns a simulated policy recommendation.")

    preset_name = st.selectbox(
        "Scenario preset",
        list(PRESETS.keys()),
        help="Load a demo profile, then edit individual inputs before scoring.",
    )
    preset = PRESETS[preset_name]

    demographic_col, enrollment_col = st.columns(2)
    with demographic_col:
        age_band = st.selectbox("Age band", AGE_OPTIONS, index=AGE_OPTIONS.index(preset["age_band"]))
        sex = st.selectbox("Sex", SEX_OPTIONS, index=SEX_OPTIONS.index(preset["sex"]))
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

    st.markdown("#### Utilization and cost profile")
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
    return payload, score_clicked


def fetch_decision_support(payload: dict) -> tuple[dict | None, dict | None, dict | None, dict | None, list[str]]:
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
        section_errors.append(f"Unified decision-support endpoint unavailable: {exc}")
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
                section_errors.append(f"{section_name} unavailable from {endpoint}: {endpoint_exc}")

    return result, state_response, recommendation, simulation, section_errors


def render_results(result: dict | None, state_response: dict | None, recommendation: dict | None, simulation: dict | None) -> None:
    st.markdown('<div class="scorer-shell">', unsafe_allow_html=True)
    st.markdown('<div class="scorer-heading">Decision-support output</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="method-note">
            The risk engine is empirically trained on observed data. The intervention policy layer is a simulated decision prototype, so recommendations are planning aids rather than causal claims.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result is None:
        st.warning("Run a beneficiary profile to generate predictions.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    metrics = result["prediction"]
    metadata = result["metadata"]
    reasons = result["reason_codes"]
    
    probability = metrics["calibrated_probability"]
    raw_probability = metrics["raw_model_probability"]
    risk_score = metrics["risk_score_0_100"]
    annual_cost_proxy = result["annual_claim_cost_proxy"]
    intervention_flag = "Yes" if metrics["intervention_flag"] else "No"
    
    st.markdown(
        (
            '<div class="result-strip">'
            f'<div><span class="result-label">Live Risk Score</span><strong>{risk_score} / 100</strong></div>'
            f'<div><span class="result-label">Risk Tier</span><strong>{display_tier(metrics["risk_tier"])}</strong></div>'
            f'<div><span class="result-label">Calibrated Probability</span><strong>{probability:.1%}</strong></div>'
            f'<div><span class="result-label">Intervention Flag</span><strong>{intervention_flag}</strong></div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    
    if reasons:
        st.markdown("#### Top Drivers")
        for reason in reasons:
            st.markdown(f"- {reason}")
        st.write("")

    chart_col, cost_col = st.columns([0.95, 1.05])
    with chart_col:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Next-year high-cost risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f4e79"},
                    "steps": [
                        {"range": [0, metrics["decision_threshold"] * 100], "color": "#e8f1f9"},
                        {"range": [metrics["decision_threshold"] * 100, 100], "color": "#cfe0f2"},
                    ],
                    "threshold": {
                        "line": {"color": "#b45309", "width": 4},
                        "thickness": 0.75,
                        "value": metrics["decision_threshold"] * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=15, r=15, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
    with cost_col:
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
            color_discrete_sequence=["#1f4e79", "#2457b8", "#4f7cac", "#8aa6c1"],
            title=f"Current-year cost mix (${annual_cost_proxy:,.0f} total)",
        )
        cost_fig.update_layout(height=350, showlegend=False, margin=dict(l=140, r=20, t=50, b=35), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(cost_fig, use_container_width=True)

    current_state = recommendation["current_state"] if recommendation is not None else None
    if current_state is None and state_response is not None:
        current_state = state_response["current_state"]

    if current_state is not None:
        st.markdown("#### Current MDP state")
        st.markdown(
            f"""
            <div class="state-grid">
                <div class="state-card">
                    <strong>{current_state['label']}</strong><br><br>
                    State ID: {current_state['state_id']}<br>
                    Prior intervention: {INTERVENTION_OPTIONS[current_state['prior_intervention_status']]}
                </div>
                <div class="state-card">
                    Risk tier: {current_state['risk_tier']}<br>
                    Chronic burden: {current_state['chronic_burden']}<br>
                    Utilization intensity: {current_state['utilization_intensity']}<br>
                    Baseline risk: {current_state['baseline_risk_probability']:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if recommendation is not None:
        st.markdown("#### Recommended action")
        action_col_a, action_col_b = st.columns([0.95, 1.05])
        action_col_a.markdown(
            (
                '<div class="recommendation-card">'
                '<div class="recommendation-label">Recommended action</div>'
                f'<div class="recommendation-title">{recommendation["recommended_action_display"]}</div>'
                f'<div class="recommendation-value">Long-run value: {recommendation["expected_long_run_value"]:.2f}</div>'
                f'<p>{recommendation["policy_explanation"]}</p>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        q_values_df = pd.DataFrame(recommendation["action_values"])
        q_values_df["is_recommended"] = q_values_df["action"] == recommendation["recommended_action"]
        q_values_df["bar_color"] = q_values_df["is_recommended"].map({True: "#1f4e79", False: "#9fb3c8"})
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
        q_fig.update_traces(textposition="outside", cliponaxis=False)
        q_fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=180, r=50, t=50, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            xaxis_title="Estimated long-run value",
            yaxis_title="",
        )
        action_col_b.plotly_chart(q_fig, use_container_width=True)

    if simulation is not None:
        st.markdown("#### Action-by-action comparison")
        comparison_df = pd.DataFrame(simulation["comparisons"])
        comparison_df["Expected next risk"] = comparison_df["expected_next_risk_probability"].map(lambda value: f"{value:.1%}")
        comparison_df["Risk delta"] = comparison_df["expected_risk_delta"].map(lambda value: f"{value:+.1%}")
        comparison_df["Immediate reward"] = comparison_df["expected_immediate_reward"].map(lambda value: f"{value:.2f}")
        comparison_df["Long-run value"] = comparison_df["q_value"].map(lambda value: f"{value:.2f}")
        st.dataframe(
            comparison_df[
                ["action_label", "Expected next risk", "Risk delta", "Immediate reward", "Long-run value", "is_recommended"]
            ].rename(columns={"action_label": "Action", "is_recommended": "Recommended"}),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Diagnostic Model Internals"):
        st.write("#### Raw Model Information")
        st.write(f"**Model Name:** {metadata.get('model_name')}")
        st.write(f"**Contract Version:** {metadata.get('feature_contract_version')}")
        st.write(f"**Calibration Method:** {metadata.get('calibration_method')}")
        st.write(f"**Raw Model Probability:** {raw_probability:.1%}")
        st.write("#### Engineered Features Evaluated")
        engineered = pd.DataFrame([{"feature": key, "value": value} for key, value in result["engineered_features"].items()])
        st.dataframe(engineered, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Ray Odian | High-Cost Claim Classifier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_css()

st.markdown(
    """
    <div class="top-nav">
        <div class="brand">Ray Odian</div>
        <div class="nav-links">
            <a href="#projects">Projects</a>
            <a href="#model">Model</a>
            <a href="#scorer">Risk Scorer</a>
            <a href="#stack">Stack</a>
        </div>
    </div>
    <section class="hero">
        <div class="eyebrow">Data Science • Machine Learning • Actuarial Analytics</div>
        <h1>High-Cost Claim Classifier</h1>
        <p class="subtitle">
            A deployed healthcare risk application that uses distributed data engineering,
            statistically defensible model selection, and calibrated decision support to
            identify beneficiaries likely to become high-cost next year.
        </p>
        <div class="button-row">
            <a class="primary-btn" href="#scorer">Use the Risk Scorer</a>
            <a class="secondary-btn" href="https://rayodian-ncf.com/health" target="_blank">Check API Health</a>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

render_metric_cards()

st.markdown('<h2 id="projects" class="section-heading">Selected Projects</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="section-copy">
        This portfolio page centers the deployed Project 2 application while making the broader
        engineering story visible: model development, cloud pipeline design, and operational serving.
    </p>
    """,
    unsafe_allow_html=True,
)
render_project_cards()

st.markdown('<h2 id="model" class="section-heading">Model Approach</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="section-copy">
        The modeling frame is prospective: year-t beneficiary features predict year-t+1 high-cost status.
        Logistic regression is retained as an interpretable actuarial benchmark, while random forest,
        gradient boosting, and XGBoost are evaluated as flexible challengers using PR-AUC, top-k capture,
        lift, calibration, and Brier score rather than raw accuracy alone.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h2 id="stack" class="section-heading">Technical Stack</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="section-copy">
        The application combines a Databricks medallion pipeline, model artifacts served through FastAPI,
        and a public Streamlit interface deployed through AWS and Hugging Face Spaces.
    </p>
    """,
    unsafe_allow_html=True,
)
render_stack()

st.markdown('<div id="scorer" class="app-module">', unsafe_allow_html=True)
st.markdown('<h2 class="section-heading" style="margin-top:0;">Live Risk Scorer</h2>', unsafe_allow_html=True)
left, right = st.columns([1.04, 0.96], gap="large")

with left:
    payload, score_clicked = build_payload()

with right:
    if score_clicked:
        result, state_response, recommendation, simulation, section_errors = fetch_decision_support(payload)
        for message in section_errors:
            st.warning(message)
        render_results(result, state_response, recommendation, simulation)
    else:
        render_results(None, None, None, None)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
        2026 Ray Odian · Data Science Portfolio · High-cost claim prediction with Streamlit, FastAPI, Databricks, and AWS
    </div>
    """,
    unsafe_allow_html=True,
)
