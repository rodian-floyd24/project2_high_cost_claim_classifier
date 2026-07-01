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
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

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

    if (!output.result) {
        return estimateDecisionSupport(payload, errors);
    }

    return output;
}

function estimateDecisionSupport(payload, apiErrors) {
    const annualCost = payload.inpatient_total_cost
        + payload.outpatient_total_cost
        + payload.carrier_total_cost
        + payload.rx_total_cost;
    const totalClaims = payload.inpatient_claim_count
        + payload.outpatient_claim_count
        + payload.carrier_claim_count
        + payload.pde_claim_count;
    const ageLift = {
        under_65: -0.1,
        "65_74": 0.15,
        "75_84": 0.35,
        "85_plus": 0.55,
        unknown: 0,
    }[payload.age_band] || 0;
    const interventionOffset = {
        none: 0,
        recent_low_touch: -0.12,
        recent_intensive: -0.32,
    }[payload.prior_intervention_status] || 0;

    const linearScore = -4.2
        + payload.chronic_condition_count * 0.23
        + payload.inpatient_claim_count * 0.42
        + payload.outpatient_claim_count * 0.035
        + payload.carrier_claim_count * 0.018
        + payload.pde_claim_count * 0.012
        + payload.total_claim_days * 0.016
        + payload.unique_provider_count * 0.025
        + Math.log1p(annualCost) * 0.22
        + ageLift
        + interventionOffset;
    const probability = clamp(1 / (1 + Math.exp(-linearScore)), 0.015, 0.94);
    const riskTier = probability >= 0.75
        ? "very_high"
        : probability >= 0.45
            ? "high"
            : probability >= 0.22
                ? "moderate"
                : "low";
    const chronicBurden = payload.chronic_condition_count >= 6
        ? "high"
        : payload.chronic_condition_count >= 3
            ? "moderate"
            : "low";
    const utilizationIntensity = totalClaims >= 45 || payload.total_claim_days >= 25
        ? "high"
        : totalClaims >= 18 || payload.total_claim_days >= 10
            ? "moderate"
            : "low";
    const stateId = `${riskTier}_${chronicBurden}_${utilizationIntensity}_${payload.prior_intervention_status}`;

    const reasons = [
        payload.chronic_condition_count >= 5 ? `Chronic condition count is elevated at ${payload.chronic_condition_count}.` : null,
        payload.inpatient_claim_count > 0 ? `${payload.inpatient_claim_count} inpatient claim(s) materially increase expected risk.` : null,
        annualCost >= 10000 ? `Current-year claim cost proxy is high at ${formatMoney(annualCost)}.` : null,
        payload.total_claim_days >= 14 ? `Total claim days show sustained utilization at ${payload.total_claim_days} days.` : null,
        payload.unique_provider_count >= 10 ? `Provider fragmentation is elevated across ${payload.unique_provider_count} providers.` : null,
    ].filter(Boolean);
    if (!reasons.length) {
        reasons.push("Low utilization and modest current-year costs keep estimated risk comparatively low.");
    }

    const recommendedAction = riskTier === "very_high" || riskTier === "high"
        ? "intensive"
        : riskTier === "moderate"
            ? "low_touch"
            : "none";
    const actionDefinitions = [
        { action: "none", action_label: "No new intervention", riskDelta: 0, costPenalty: 0 },
        { action: "low_touch", action_label: "Low-touch outreach", riskDelta: -0.055, costPenalty: 0.6 },
        { action: "intensive", action_label: "Intensive care management", riskDelta: -0.14, costPenalty: 1.3 },
    ];
    const comparisons = actionDefinitions.map((action) => {
        const expectedNextRisk = clamp(probability + action.riskDelta, 0.01, 0.95);
        const policyFitBonus = action.action === recommendedAction ? 2 : 0;
        const qValue = (1 - expectedNextRisk) * 5 - action.costPenalty + policyFitBonus;
        return {
            action: action.action,
            action_label: action.action_label,
            expected_next_risk_probability: expectedNextRisk,
            expected_risk_delta: expectedNextRisk - probability,
            expected_immediate_reward: (probability - expectedNextRisk) * 10 - action.costPenalty,
            q_value: qValue,
            is_recommended: false,
        };
    });
    comparisons.sort((a, b) => b.q_value - a.q_value);
    comparisons[0].is_recommended = true;
    const recommended = comparisons[0];

    return {
        result: {
            prediction: {
                calibrated_probability: probability,
                raw_model_probability: probability,
                risk_score_0_100: Math.round(probability * 100),
                risk_tier: riskTier,
                decision_threshold: 0.42,
                intervention_flag: probability >= 0.42,
            },
            metadata: {
                model_name: "browser_demo_estimator",
                feature_contract_version: "static-fallback-v1",
                calibration_method: "heuristic client-side fallback",
            },
            reason_codes: reasons,
            annual_claim_cost_proxy: annualCost,
            cost_mix: {
                inpatient: payload.inpatient_total_cost,
                outpatient: payload.outpatient_total_cost,
                carrier: payload.carrier_total_cost,
                prescription: payload.rx_total_cost,
            },
            engineered_features: {
                age_band: payload.age_band,
                chronic_condition_count: payload.chronic_condition_count,
                total_claim_count: totalClaims,
                total_claim_days: payload.total_claim_days,
                unique_provider_count: payload.unique_provider_count,
                annual_claim_cost_proxy: annualCost,
                prior_intervention_status: payload.prior_intervention_status,
                api_fallback_reason: apiErrors[0] || "Live API unavailable",
            },
        },
        stateResponse: {
            current_state: {
                state_id: stateId,
                label: `${displayTier(riskTier)} risk | ${displayTier(chronicBurden)} chronic burden | ${displayTier(utilizationIntensity)} utilization`,
                prior_intervention_status: payload.prior_intervention_status,
                risk_tier: riskTier,
                chronic_burden: chronicBurden,
                utilization_intensity: utilizationIntensity,
                baseline_risk_probability: probability,
            },
        },
        recommendation: {
            current_state: {
                state_id: stateId,
                label: `${displayTier(riskTier)} risk | ${displayTier(chronicBurden)} chronic burden | ${displayTier(utilizationIntensity)} utilization`,
                prior_intervention_status: payload.prior_intervention_status,
                risk_tier: riskTier,
                chronic_burden: chronicBurden,
                utilization_intensity: utilizationIntensity,
                baseline_risk_probability: probability,
            },
            recommended_action: recommended.action,
            recommended_action_display: recommended.action_label,
            expected_long_run_value: recommended.q_value,
            policy_explanation: "Live API scoring is unavailable, so this browser-side estimate uses the same inputs to keep the demo interactive. Treat this as a portfolio demonstration, not a deployed clinical or actuarial decision.",
            action_values: comparisons.map((item) => ({
                action: item.action,
                action_label: item.action_label,
                q_value: item.q_value,
            })),
        },
        simulation: {
            comparisons,
        },
        errors: ["Live prediction API unavailable; showing a browser-only demo estimate."],
    };
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
    if (!window.Plotly) {
        document.getElementById("risk-gauge").innerHTML = "<p>Chart library unavailable.</p>";
        document.getElementById("cost-chart").innerHTML = "<p>Chart library unavailable.</p>";
        const qElement = document.getElementById("q-chart");
        if (qElement) {
            qElement.innerHTML = "<p>Chart library unavailable.</p>";
        }
        return;
    }

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
