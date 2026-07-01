const API_BASE_URL = window.PREDICTION_API_URL || "https://rayodian-ncf.com";

const AGE_OPTIONS = ["under_65", "65_74", "75_84", "85_plus", "unknown"];
const SEX_OPTIONS = ["female", "male", "unknown"];
const STATE_OPTIONS = ["CA", "FL", "NY", "TX", "PA", "OH", "IL", "NC", "GA", "MI", "unknown"];
const RACE_OPTIONS = ["1", "2", "3", "4", "5", "6", "unknown"];
const INTERVENTION_OPTIONS = {
    none: "No recent intervention",
    recent_low_touch: "Recent low-touch outreach",
    recent_intensive: "Recent intensive intervention",
};

const PRESETS = {
    "Low-risk routine beneficiary": {
        age_band: "65_74",
        sex: "female",
        race_code: "1",
        state_code: "FL",
        enrollment_months_count: 12,
        chronic_condition_count: 1,
        inpatient_claim_count: 0,
        outpatient_claim_count: 2,
        carrier_claim_count: 4,
        pde_claim_count: 8,
        total_claim_days: 3,
        unique_provider_count: 3,
        rx_total_cost: 650,
        inpatient_total_cost: 0,
        outpatient_total_cost: 420,
        carrier_total_cost: 310,
        prior_intervention_status: "none",
    },
    "Moderate chronic-care beneficiary": {
        age_band: "75_84",
        sex: "male",
        race_code: "1",
        state_code: "TX",
        enrollment_months_count: 12,
        chronic_condition_count: 4,
        inpatient_claim_count: 1,
        outpatient_claim_count: 8,
        carrier_claim_count: 12,
        pde_claim_count: 20,
        total_claim_days: 11,
        unique_provider_count: 7,
        rx_total_cost: 3200,
        inpatient_total_cost: 6400,
        outpatient_total_cost: 2800,
        carrier_total_cost: 2400,
        prior_intervention_status: "recent_low_touch",
    },
    "Very-high-risk complex beneficiary": {
        age_band: "85_plus",
        sex: "female",
        race_code: "2",
        state_code: "NY",
        enrollment_months_count: 12,
        chronic_condition_count: 7,
        inpatient_claim_count: 3,
        outpatient_claim_count: 14,
        carrier_claim_count: 25,
        pde_claim_count: 26,
        total_claim_days: 29,
        unique_provider_count: 13,
        rx_total_cost: 6400,
        inpatient_total_cost: 18800,
        outpatient_total_cost: 5400,
        carrier_total_cost: 4300,
        prior_intervention_status: "none",
    },
    "Recently managed very-high-risk beneficiary": {
        age_band: "85_plus",
        sex: "female",
        race_code: "2",
        state_code: "NY",
        enrollment_months_count: 12,
        chronic_condition_count: 8,
        inpatient_claim_count: 2,
        outpatient_claim_count: 12,
        carrier_claim_count: 22,
        pde_claim_count: 26,
        total_claim_days: 24,
        unique_provider_count: 11,
        rx_total_cost: 5900,
        inpatient_total_cost: 15100,
        outpatient_total_cost: 5100,
        carrier_total_cost: 4100,
        prior_intervention_status: "recent_intensive",
    },
};

const NUMBER_FIELDS = [
    "enrollment_months_count",
    "chronic_condition_count",
    "inpatient_claim_count",
    "outpatient_claim_count",
    "carrier_claim_count",
    "pde_claim_count",
    "total_claim_days",
    "unique_provider_count",
    "rx_total_cost",
    "inpatient_total_cost",
    "outpatient_total_cost",
    "carrier_total_cost",
];

const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const formatMoney = (value) => `$${Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const displayTier = (tier) => String(tier || "").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());

function fillSelect(id, values, formatter = (value) => value) {
    const select = document.getElementById(id);
    select.innerHTML = values.map((value) => `<option value="${value}">${formatter(value)}</option>`).join("");
}

function renderMetricCards(metrics) {
    const container = document.getElementById("model-metrics");
    const cards = metrics.map(([label, value, note]) => `
        <article class="metric-card">
            <div class="metric-label">${label}</div>
            <div class="metric-value">${value}</div>
            <div class="metric-note">${note}</div>
        </article>
    `);
    container.innerHTML = cards.join("");
}

async function fetchModelMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/model_metrics`, { headers: { Accept: "application/json" } });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const modelName = String(data.model_name || "model").replaceAll("_", " ");
        const modelDisplayName = modelName.charAt(0).toUpperCase() + modelName.slice(1);
        renderMetricCards([
            ["Test PR-AUC", Number(data.pr_auc || 0).toFixed(3), `${modelDisplayName} holdout result`],
            ["Top-5% Capture", formatPercent(data.top_5_capture_rate), "High-cost cases found in highest-risk 5%"],
            ["Top-10% Capture", formatPercent(data.top_10_capture_rate), "High-cost cases found in highest-risk 10%"],
            ["Brier Score", Number(data.brier_score || 0).toFixed(3), "Probability calibration error, lower is better"],
        ]);
    } catch {
        renderMetricCards([
            ["Test PR-AUC", "N/A", "Model metrics unavailable. Check backend artifact metadata."],
            ["Top-5% Capture", "N/A", "Model metrics unavailable. Check backend artifact metadata."],
            ["Top-10% Capture", "N/A", "Model metrics unavailable. Check backend artifact metadata."],
            ["Brier Score", "N/A", "Model metrics unavailable. Check backend artifact metadata."],
        ]);
    }
}

function applyPreset(name) {
    const preset = PRESETS[name];
    Object.entries(preset).forEach(([key, value]) => {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
        }
    });
}

function readPayload() {
    const payload = {};
    ["age_band", "sex", "race_code", "state_code", "prior_intervention_status"].forEach((field) => {
        payload[field] = document.getElementById(field).value;
    });
    NUMBER_FIELDS.forEach((field) => {
        payload[field] = Number(document.getElementById(field).value || 0);
    });
    return payload;
}

async function postJson(endpoint, payload) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });
    if (!response.ok) {
        throw new Error(`${endpoint} returned HTTP ${response.status}`);
    }
    return response.json();
}

async function fetchDecisionSupport(payload) {
    const errors = [];
    try {
        const decisionSupport = await postJson("/decision_support", payload);
        return {
            result: decisionSupport.prediction,
            stateResponse: decisionSupport.state,
            recommendation: decisionSupport.recommendation,
            simulation: decisionSupport.simulation,
            errors,
        };
    } catch (error) {
        errors.push(`Unified decision-support endpoint unavailable: ${error.message}`);
    }

    const output = {
        result: null,
        stateResponse: null,
        recommendation: null,
        simulation: null,
        errors,
    };
    const endpointMap = [
        ["Risk prediction", "/predict", "result"],
        ["Current state", "/state", "stateResponse"],
        ["Recommendation", "/recommend_action", "recommendation"],
        ["Simulation", "/simulate", "simulation"],
    ];

    for (const [sectionName, endpoint, key] of endpointMap) {
        try {
            output[key] = await postJson(endpoint, payload);
        } catch (error) {
            output.errors.push(`${sectionName} unavailable from ${endpoint}: ${error.message}`);
        }
    }

    return output;
}

function renderWarnings(errors) {
    if (!errors.length) {
        return "";
    }
    return errors.map((message) => `<div class="warning-message">${message}</div>`).join("");
}

function renderState(currentState) {
    if (!currentState) {
        return "";
    }
    return `
        <h3>Current MDP state</h3>
        <div class="state-grid">
            <div class="state-card">
                <strong>${currentState.label}</strong><br><br>
                State ID: ${currentState.state_id}<br>
                Prior intervention: ${INTERVENTION_OPTIONS[currentState.prior_intervention_status] || currentState.prior_intervention_status}
            </div>
            <div class="state-card">
                Risk tier: ${currentState.risk_tier}<br>
                Chronic burden: ${currentState.chronic_burden}<br>
                Utilization intensity: ${currentState.utilization_intensity}<br>
                Baseline risk: ${formatPercent(currentState.baseline_risk_probability)}
            </div>
        </div>
    `;
}

function renderComparisonTable(simulation) {
    if (!simulation || !Array.isArray(simulation.comparisons)) {
        return "";
    }
    const rows = simulation.comparisons.map((item) => `
        <tr>
            <td>${item.action_label}</td>
            <td>${formatPercent(item.expected_next_risk_probability)}</td>
            <td>${formatPercent(item.expected_risk_delta).replace("-", "-")}</td>
            <td>${Number(item.expected_immediate_reward || 0).toFixed(2)}</td>
            <td>${Number(item.q_value || 0).toFixed(2)}</td>
            <td>${item.is_recommended ? "Yes" : "No"}</td>
        </tr>
    `).join("");

    return `
        <h3>Action-by-action comparison</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Action</th>
                    <th>Expected next risk</th>
                    <th>Risk delta</th>
                    <th>Immediate reward</th>
                    <th>Long-run value</th>
                    <th>Recommended</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderDiagnostics(result) {
    const metadata = result.metadata || {};
    const engineered = Object.entries(result.engineered_features || {}).map(([key, value]) => `
        <tr><td>${key}</td><td>${value}</td></tr>
    `).join("");

    return `
        <details class="details">
            <summary>Diagnostic Model Internals</summary>
            <p><strong>Model Name:</strong> ${metadata.model_name || "N/A"}</p>
            <p><strong>Contract Version:</strong> ${metadata.feature_contract_version || "N/A"}</p>
            <p><strong>Calibration Method:</strong> ${metadata.calibration_method || "N/A"}</p>
            <table class="data-table">
                <thead><tr><th>Feature</th><th>Value</th></tr></thead>
                <tbody>${engineered}</tbody>
            </table>
        </details>
    `;
}

function drawCharts(result, recommendation) {
    const metrics = result.prediction;
    const probability = Number(metrics.calibrated_probability || 0);
    const threshold = Number(metrics.decision_threshold || 0);
    const gaugeElement = document.getElementById("risk-gauge");
    const costElement = document.getElementById("cost-chart");
    const qElement = document.getElementById("q-chart");

    Plotly.newPlot(gaugeElement, [{
        type: "indicator",
        mode: "gauge+number",
        value: probability * 100,
        number: { suffix: "%" },
        title: { text: "Next-year high-cost risk" },
        gauge: {
            axis: { range: [0, 100] },
            bar: { color: "#1f4e79" },
            steps: [
                { range: [0, threshold * 100], color: "#e8f1f9" },
                { range: [threshold * 100, 100], color: "#cfe0f2" },
            ],
            threshold: {
                line: { color: "#b45309", width: 4 },
                thickness: 0.75,
                value: threshold * 100,
            },
        },
    }], {
        height: 300,
        margin: { l: 15, r: 15, t: 50, b: 10 },
        paper_bgcolor: "rgba(0,0,0,0)",
    }, { displayModeBar: false, responsive: true });

    const costMix = result.cost_mix || {};
    Plotly.newPlot(costElement, [{
        type: "bar",
        orientation: "h",
        x: [costMix.inpatient || 0, costMix.outpatient || 0, costMix.carrier || 0, costMix.prescription || 0],
        y: ["Inpatient", "Outpatient", "Carrier", "Prescription"],
        marker: { color: ["#1f4e79", "#2457b8", "#4f7cac", "#8aa6c1"] },
    }], {
        title: `Current-year cost mix (${formatMoney(result.annual_claim_cost_proxy)} total)`,
        height: 320,
        margin: { l: 120, r: 20, t: 50, b: 35 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#ffffff",
        showlegend: false,
    }, { displayModeBar: false, responsive: true });

    if (recommendation && Array.isArray(recommendation.action_values)) {
        const actionValues = [...recommendation.action_values].sort((a, b) => a.q_value - b.q_value);
        Plotly.newPlot(qElement, [{
            type: "bar",
            orientation: "h",
            x: actionValues.map((item) => item.q_value),
            y: actionValues.map((item) => item.action_label),
            text: actionValues.map((item) => Number(item.q_value || 0).toFixed(2)),
            textposition: "outside",
            marker: {
                color: actionValues.map((item) => item.action === recommendation.recommended_action ? "#1f4e79" : "#9fb3c8"),
            },
            cliponaxis: false,
        }], {
            title: "Estimated long-run value by action",
            height: 360,
            margin: { l: 160, r: 55, t: 50, b: 45 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "#ffffff",
            showlegend: false,
            xaxis: { title: "Estimated long-run value" },
        }, { displayModeBar: false, responsive: true });
    }
}

function renderResults({ result, stateResponse, recommendation, simulation, errors }) {
    const status = document.getElementById("status-message");
    const container = document.getElementById("results");
    status.hidden = true;
    container.hidden = false;

    if (!result) {
        status.hidden = false;
        status.textContent = "The prediction API did not return a result. Check API availability and CORS settings.";
        container.hidden = true;
        return;
    }

    const metrics = result.prediction;
    const reasons = result.reason_codes || [];
    const probability = Number(metrics.calibrated_probability || 0);
    const riskScore = metrics.risk_score_0_100;
    const interventionFlag = metrics.intervention_flag ? "Yes" : "No";
    const currentState = recommendation?.current_state || stateResponse?.current_state;

    const reasonList = reasons.length
        ? `<h3>Top Drivers</h3><ul class="drivers">${reasons.map((reason) => `<li>${reason}</li>`).join("")}</ul>`
        : "";

    const recommendationMarkup = recommendation ? `
        <h3>Recommended action</h3>
        <div class="action-grid">
            <div class="recommendation-card">
                <div class="recommendation-label">Recommended action</div>
                <div class="recommendation-title">${recommendation.recommended_action_display}</div>
                <div><strong>Long-run value:</strong> ${Number(recommendation.expected_long_run_value || 0).toFixed(2)}</div>
                <p>${recommendation.policy_explanation}</p>
            </div>
            <div id="q-chart" class="chart-box"></div>
        </div>
    ` : "";

    container.innerHTML = `
        ${renderWarnings(errors)}
        <div class="result-strip">
            <div><span class="result-label">Live Risk Score</span><strong>${riskScore} / 100</strong></div>
            <div><span class="result-label">Risk Tier</span><strong>${displayTier(metrics.risk_tier)}</strong></div>
            <div><span class="result-label">Calibrated Probability</span><strong>${formatPercent(probability)}</strong></div>
            <div><span class="result-label">Intervention Flag</span><strong>${interventionFlag}</strong></div>
        </div>
        ${reasonList}
        <div class="chart-grid">
            <div id="risk-gauge" class="chart-box"></div>
            <div id="cost-chart" class="chart-box"></div>
        </div>
        ${renderState(currentState)}
        ${recommendationMarkup}
        ${renderComparisonTable(simulation)}
        ${renderDiagnostics(result)}
    `;

    drawCharts(result, recommendation);
}

function setupForm() {
    fillSelect("preset", Object.keys(PRESETS));
    fillSelect("age_band", AGE_OPTIONS, displayTier);
    fillSelect("sex", SEX_OPTIONS, displayTier);
    fillSelect("race_code", RACE_OPTIONS);
    fillSelect("state_code", STATE_OPTIONS);
    fillSelect("prior_intervention_status", Object.keys(INTERVENTION_OPTIONS), (key) => INTERVENTION_OPTIONS[key]);

    const presetSelect = document.getElementById("preset");
    presetSelect.addEventListener("change", () => applyPreset(presetSelect.value));
    applyPreset(presetSelect.value);

    const form = document.getElementById("risk-form");
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = form.querySelector("button[type='submit']");
        const status = document.getElementById("status-message");
        const results = document.getElementById("results");
        button.disabled = true;
        button.textContent = "Scoring...";
        status.hidden = false;
        results.hidden = true;
        status.textContent = "Scoring beneficiary profile...";

        try {
            const payload = readPayload();
            const output = await fetchDecisionSupport(payload);
            renderResults(output);
        } catch (error) {
            status.hidden = false;
            results.hidden = true;
            status.textContent = `Unable to score beneficiary: ${error.message}`;
        } finally {
            button.disabled = false;
            button.textContent = "Score Beneficiary";
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupForm();
    fetchModelMetrics();
});
