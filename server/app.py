"""Auxiliary server entrypoint required by OpenEnv local validation checks."""

import os

from flask import Flask, Response, jsonify, request

from environment import EmailTriageEnv
from tasks import get_task_scenario_count, list_task_ids

FRONTEND_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Inbox Helper Practice</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #f5f1e9;
            --paper: #fffaf2;
            --ink: #102433;
            --accent: #ea6a2a;
            --accent-soft: #ffd6bf;
            --line: #d7cabb;
            --ok: #0f7b6c;
            --warn: #9a3a12;
            --radius: 14px;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(1100px 460px at -10% -20%, #f2bc9f 0%, transparent 60%),
                radial-gradient(1100px 520px at 120% 115%, #b8d7cf 0%, transparent 62%),
                var(--bg);
            min-height: 100vh;
        }

        .wrap {
            max-width: 1100px;
            margin: 28px auto;
            padding: 0 16px;
            animation: reveal .45s ease-out;
        }

        @keyframes reveal {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .title {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 14px;
            margin-bottom: 14px;
        }

        h1 {
            margin: 0;
            font-size: clamp(1.5rem, 2vw, 2.2rem);
            letter-spacing: .4px;
        }

        .subtitle {
            margin: 6px 0 0;
            font-size: .95rem;
            opacity: .8;
        }

        .badge {
            background: var(--accent-soft);
            border: 1px solid #f2b693;
            color: #7f2e0b;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: .85rem;
            font-weight: 600;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
        }

        @media (min-width: 900px) {
            .grid { grid-template-columns: 1fr 1fr; }
            .wide { grid-column: span 2; }
        }

        .card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 14px;
            box-shadow: 0 8px 28px rgba(16, 36, 51, 0.08);
        }

        .card h2 {
            margin: 0 0 10px;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            opacity: .86;
        }

        .row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin-bottom: 10px;
        }

        select, input, textarea, button {
            font-family: inherit;
            font-size: .95rem;
        }

        select, input, textarea {
            width: 100%;
            border: 1px solid #cdbba6;
            border-radius: 10px;
            padding: 9px 10px;
            background: #fff;
            color: var(--ink);
        }

        textarea {
            min-height: 92px;
            resize: vertical;
        }

        button {
            border: 0;
            border-radius: 10px;
            padding: 9px 12px;
            font-weight: 700;
            background: var(--ink);
            color: #fff;
            cursor: pointer;
            transition: transform .12s ease, opacity .12s ease;
        }

        button.secondary {
            background: #285066;
        }

        button.accent {
            background: var(--accent);
        }

        button:hover { transform: translateY(-1px); }
        button:active { transform: translateY(0); opacity: .92; }

        .status {
            padding: 8px 10px;
            border-radius: 10px;
            background: #eef7f5;
            border: 1px solid #c7e4de;
            color: var(--ok);
            font-weight: 600;
            min-height: 40px;
            display: flex;
            align-items: center;
        }

        .status.error {
            background: #fff1ea;
            border-color: #ffc8ae;
            color: var(--warn);
        }

        pre {
            margin: 0;
            white-space: pre-wrap;
            background: #0f1b24;
            color: #d9efe9;
            border-radius: 10px;
            padding: 12px;
            max-height: 340px;
            overflow: auto;
            font-family: 'IBM Plex Mono', monospace;
            font-size: .85rem;
            border: 1px solid #21313f;
        }

        .email-block {
            background: #fff;
            border: 1px solid #d9ccbc;
            border-radius: 10px;
            padding: 12px;
        }

        .email-row {
            margin-bottom: 8px;
            font-size: .95rem;
            line-height: 1.35;
        }

        .email-row strong {
            display: inline-block;
            min-width: 66px;
        }

        .help {
            margin: 0 0 10px;
            font-size: .9rem;
            opacity: .8;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px dashed #dbcfbe;
            font-size: .95rem;
        }

        .metric strong {
            font-weight: 700;
        }

        .coach {
            background: #fff7ed;
            border: 1px solid #f2caa9;
            border-radius: 10px;
            padding: 10px;
            min-height: 74px;
            line-height: 1.4;
            font-size: .92rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .chip {
            background: #eaf3ff;
            border: 1px solid #b9d1ef;
            color: #184469;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: .84rem;
            cursor: pointer;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="title">
            <div>
                <h1>Inbox Helper Practice</h1>
                <p class="subtitle">Practice deciding priority, category, and who should handle each email.</p>
            </div>
            <span class="badge" id="badge">connecting...</span>
        </div>

        <div class="grid">
            <section class="card">
                <h2>Start a Scenario</h2>
                <p class="help">Pick a difficulty, then click Start.</p>
                <div class="row">
                    <select id="taskId">
                        <option value="task_easy">Easy: one clear email</option>
                        <option value="task_medium">Medium: mixed inbox</option>
                        <option value="task_hard">Hard: high-risk complaint</option>
                        <option value="task_production">Production: full inbox simulator</option>
                    </select>
                </div>
                <div id="productionControls" style="display:none;">
                    <div class="row">
                        <select id="productionProfile">
                            <option value="light">Workload: Light</option>
                            <option value="standard" selected>Workload: Standard</option>
                            <option value="heavy">Workload: Heavy</option>
                        </select>
                    </div>
                    <div class="row">
                        <select id="businessHoursMode">
                            <option value="false" selected>Time Profile: 24x7 inbox</option>
                            <option value="true">Time Profile: business hours focus</option>
                        </select>
                    </div>
                    <div class="row">
                        <select id="escalationMode">
                            <option value="low">Escalation: Low</option>
                            <option value="normal" selected>Escalation: Normal</option>
                            <option value="high">Escalation: High</option>
                        </select>
                    </div>
                </div>
                <div class="row">
                    <button class="accent" id="btnReset">Start</button>
                    <button class="secondary" id="btnState">Check Progress</button>
                </div>
                <div class="status" id="status">Ready. Start a scenario.</div>
            </section>

            <section class="card">
                <h2>Your Decision</h2>
                <p class="help">Choose priority, who should handle it, and a short reason.</p>
                <div class="row">
                    <select id="label">
                        <option value="urgent">Urgent</option>
                        <option value="normal" selected>Normal</option>
                        <option value="spam">Spam</option>
                        <option value="archive">Archive</option>
                    </select>
                </div>
                <div class="row">
                    <input id="routeTo" placeholder="Who should handle this? (billing, safety, engineering, support)" value="general" />
                </div>
                <div class="row">
                    <textarea id="summary" placeholder="Write one clear sentence with key clues from the email.">Needs review.</textarea>
                </div>
                <div class="row">
                    <button id="btnStep">Send Decision</button>
                </div>
            </section>

            <section class="card wide">
                <h2>Current Email</h2>
                <div class="email-block">
                    <div class="email-row"><strong>Subject:</strong> <span id="mailSubject">No email loaded yet.</span></div>
                    <div class="email-row"><strong>From:</strong> <span id="mailSender">-</span></div>
                    <div class="email-row"><strong>Message:</strong> <span id="mailBody">Start a scenario to load an email.</span></div>
                </div>
            </section>

            <section class="card">
                <h2>Live Progress</h2>
                <div class="metric"><span>Task</span><strong id="insightTask">-</strong></div>
                <div class="metric"><span>Scenario</span><strong id="insightScenario">-</strong></div>
                <div class="metric"><span>Progress</span><strong id="insightProgress">0/0</strong></div>
                <div class="metric"><span>Last Reward</span><strong id="insightReward">-</strong></div>
                <div class="metric"><span>Base Score</span><strong id="insightBase">-</strong></div>
            </section>

            <section class="card">
                <h2>Coach Notes</h2>
                <p class="help">Use this to improve your next triage action.</p>
                <div class="coach" id="coachNotes">Start a scenario and submit one decision to get feedback.</div>
                <div class="chip-row">
                    <button class="chip" id="chipSafety">Quick Fill: Urgent + Safety</button>
                    <button class="chip" id="chipBilling">Quick Fill: Normal + Billing</button>
                    <button class="chip" id="chipSpam">Quick Fill: Spam + General</button>
                </div>
            </section>

            <section class="card wide">
                <h2>Details (Advanced)</h2>
                <pre id="output">Waiting for your first action...</pre>
            </section>
        </div>
    </div>

    <script>
        const statusEl = document.getElementById('status');
        const badgeEl = document.getElementById('badge');
        const outEl = document.getElementById('output');
        const mailSubjectEl = document.getElementById('mailSubject');
        const mailSenderEl = document.getElementById('mailSender');
        const mailBodyEl = document.getElementById('mailBody');
        const taskIdEl = document.getElementById('taskId');
        const productionControlsEl = document.getElementById('productionControls');
        const insightTaskEl = document.getElementById('insightTask');
        const insightScenarioEl = document.getElementById('insightScenario');
        const insightProgressEl = document.getElementById('insightProgress');
        const insightRewardEl = document.getElementById('insightReward');
        const insightBaseEl = document.getElementById('insightBase');
        const coachNotesEl = document.getElementById('coachNotes');

        function setStatus(msg, isError = false) {
            statusEl.textContent = msg;
            statusEl.classList.toggle('error', isError);
        }

        function writeOutput(value) {
            outEl.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
        }

        function updateEmailPanel(data) {
            if (!data || !data.observation) {
                return;
            }
            const obs = data.observation;
            mailSubjectEl.textContent = obs.subject || 'No subject';
            mailSenderEl.textContent = obs.sender || '-';
            mailBodyEl.textContent = obs.body || '';
        }

        function updateProductionControlsVisibility() {
            const isProduction = taskIdEl.value === 'task_production';
            productionControlsEl.style.display = isProduction ? 'block' : 'none';
        }

        function safeNumber(value) {
            return typeof value === 'number' && !Number.isNaN(value) ? value : null;
        }

        function updateInsights(data) {
            const info = (data && data.info) ? data.info : {};
            const taskValue = info.task_id || data.task_id || (data.observation && data.observation.task_id) || '-';
            const scenarioValue = info.scenario_id || '-';

            insightTaskEl.textContent = taskValue;
            insightScenarioEl.textContent = scenarioValue;

            const emailsProcessed = safeNumber(info.emails_processed);
            const emailsTotal = safeNumber(info.emails_total);
            if (emailsProcessed !== null && emailsTotal !== null) {
                insightProgressEl.textContent = `${emailsProcessed}/${emailsTotal}`;
            } else if (safeNumber(data.current_step) !== null && safeNumber(data.total_steps) !== null) {
                insightProgressEl.textContent = `${data.current_step}/${data.total_steps}`;
            }

            const rewardValue = safeNumber(data.reward);
            insightRewardEl.textContent = rewardValue !== null ? rewardValue.toFixed(2) : '-';

            const baseScoreValue = safeNumber(info.base_score);
            insightBaseEl.textContent = baseScoreValue !== null ? baseScoreValue.toFixed(2) : '-';

            const tips = [];
            if (info.validation_error) {
                tips.push('Action format is invalid. Keep label/summary/route_to filled correctly.');
            }

            const routeNoise = safeNumber(info.grade_route_noise_penalty);
            if (routeNoise !== null && routeNoise > 0.01) {
                tips.push('Route to one best owner team. Avoid sending to many teams at once.');
            }

            const summaryMatch = safeNumber(info.grade_summary_match);
            if (summaryMatch !== null && summaryMatch < 0.6) {
                tips.push('Summary is weak. Include concrete clues from subject/body/thread.');
            }

            const labelMatch = safeNumber(info.grade_label_match);
            if (labelMatch !== null && labelMatch < 1.0) {
                tips.push('Priority label may be off. Re-check urgency and risk signals.');
            }

            const routeMatch = safeNumber(info.grade_route_match);
            if (routeMatch !== null && routeMatch < 1.0) {
                tips.push('Routing looks off. Pick the team that directly owns this issue.');
            }

            const urgencyComponent = safeNumber(info.grade_urgency_component);
            if (urgencyComponent !== null && urgencyComponent < 0.2) {
                tips.push('For high-risk complaints, mark urgent and route to safety first.');
            }

            if (!tips.length && typeof info.grading_feedback === 'string' && info.grading_feedback) {
                tips.push(info.grading_feedback);
            }

            coachNotesEl.textContent = tips.length
                ? tips.join(' ')
                : 'Looks good. Keep your next route precise and your summary evidence-based.';
        }

        function prefillAction(label, routeTo, summary) {
            document.getElementById('label').value = label;
            document.getElementById('routeTo').value = routeTo;
            document.getElementById('summary').value = summary;
        }

        async function postJson(path, payload) {
            const response = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload || {}),
            });
            const text = await response.text();
            let data = text;
            try { data = JSON.parse(text); } catch (e) {}
            if (!response.ok) {
                throw new Error('HTTP ' + response.status + ' - ' + text);
            }
            return data;
        }

        async function warmup() {
            try {
                const res = await fetch('/meta');
                const data = await res.json();
                badgeEl.textContent = data.status === 'ok' ? 'ready' : 'check service';
            } catch (e) {
                badgeEl.textContent = 'offline';
            }
        }

        document.getElementById('btnReset').addEventListener('click', async () => {
            const taskId = taskIdEl.value;
            setStatus('Starting a new scenario...');
            try {
                const payload = { task_id: taskId };
                if (taskId === 'task_production') {
                    payload.production_profile = document.getElementById('productionProfile').value;
                    payload.business_hours_mode = document.getElementById('businessHoursMode').value === 'true';
                    payload.escalation_mode = document.getElementById('escalationMode').value;
                }
                const data = await postJson('/reset', payload);
                setStatus('Scenario started. Read the email below.');
                updateEmailPanel(data);
                updateInsights(data);
                writeOutput(data);
            } catch (e) {
                setStatus('Could not start scenario. See details below.', true);
                writeOutput(String(e));
            }
        });

        document.getElementById('btnState').addEventListener('click', async () => {
            setStatus('Checking progress...');
            try {
                const data = await postJson('/state', {});
                setStatus('Progress updated.');
                updateInsights(data);
                writeOutput(data);
            } catch (e) {
                setStatus('Could not fetch progress. See details below.', true);
                writeOutput(String(e));
            }
        });

        document.getElementById('btnStep').addEventListener('click', async () => {
            const payload = {
                label: document.getElementById('label').value,
                summary: document.getElementById('summary').value,
                route_to: document.getElementById('routeTo').value,
            };
            setStatus('Sending your decision...');
            try {
                const data = await postJson('/step', payload);
                setStatus('Decision saved.');
                updateEmailPanel(data);
                updateInsights(data);
                writeOutput(data);
            } catch (e) {
                setStatus('Could not submit decision. See details below.', true);
                writeOutput(String(e));
            }
        });

        document.getElementById('chipSafety').addEventListener('click', () => {
            prefillAction('urgent', 'safety', 'Potential safety risk with immediate escalation needed.');
        });

        document.getElementById('chipBilling').addEventListener('click', () => {
            prefillAction('normal', 'billing', 'Customer billing issue needs finance team review and response.');
        });

        document.getElementById('chipSpam').addEventListener('click', () => {
            prefillAction('spam', 'general', 'Likely phishing or irrelevant message with suspicious external request.');
        });

        taskIdEl.addEventListener('change', updateProductionControlsVisibility);

        updateProductionControlsVisibility();
        warmup();
    </script>
</body>
</html>
"""

app = Flask(__name__)
current_env = EmailTriageEnv(task_id="task_easy")
SCENARIO_COUNTERS = {task_id: 0 for task_id in list_task_ids()}
DEFAULT_EVAL_SPLIT = os.getenv("OPENENV_EVAL_SPLIT", "public")
ALLOW_CLIENT_EVAL_OVERRIDE = (
    os.getenv("OPENENV_ALLOW_CLIENT_EVAL_OVERRIDE", "false").strip().lower() == "true"
)


@app.get("/")
def root_page():
    """Render a lightweight frontend for interacting with the environment."""
    return Response(FRONTEND_HTML, mimetype="text/html")


@app.get("/meta")
def root_endpoint():
    """Return service metadata for health checks and machine clients."""
    return jsonify(
        {
            "name": "email-triage-env",
            "status": "ok",
            "endpoints": {
                "reset": {"method": "POST", "path": "/reset"},
                "step": {"method": "POST", "path": "/step"},
                "state": {"method": "POST", "path": "/state"},
            },
            "scenario_pools": {
                "public": {
                    task_id: get_task_scenario_count(task_id, "public")
                    for task_id in list_task_ids()
                },
            },
            "eval_split": DEFAULT_EVAL_SPLIT,
            "production_runtime_controls": {
                "production_profile": ["light", "standard", "heavy"],
                "business_hours_mode": [True, False],
                "escalation_mode": ["low", "normal", "high"],
            },
        }
    )


@app.post("/reset")
def reset_endpoint():
    """Reset the environment with a selected task and return ResetResult JSON.

    Returns:
        Flask response containing reset payload.
    """
    global current_env
    global SCENARIO_COUNTERS

    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    elif not isinstance(payload, dict):
        return jsonify({"error": "Malformed JSON payload."}), 400

    task_id = payload.get("task_id", "task_easy")
    if not isinstance(task_id, str):
        return jsonify({"error": "Field 'task_id' must be a string."}), 400

    runtime_options: dict[str, object] = {}
    if task_id == "task_production":
        production_profile = payload.get("production_profile", "standard")
        if not isinstance(production_profile, str) or production_profile not in {
            "light",
            "standard",
            "heavy",
        }:
            return (
                jsonify(
                    {
                        "error": (
                            "Field 'production_profile' must be one of "
                            "light/standard/heavy."
                        )
                    }
                ),
                400,
            )

        escalation_mode = payload.get("escalation_mode", "normal")
        if not isinstance(escalation_mode, str) or escalation_mode not in {
            "low",
            "normal",
            "high",
        }:
            return (
                jsonify(
                    {
                        "error": (
                            "Field 'escalation_mode' must be one of "
                            "low/normal/high."
                        )
                    }
                ),
                400,
            )

        business_hours_mode = payload.get("business_hours_mode", False)
        if isinstance(business_hours_mode, str):
            business_hours_mode = business_hours_mode.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        elif not isinstance(business_hours_mode, bool):
            return jsonify({"error": "Field 'business_hours_mode' must be boolean."}), 400

        runtime_options = {
            "production_profile": production_profile,
            "business_hours_mode": business_hours_mode,
            "escalation_mode": escalation_mode,
        }

    if not ALLOW_CLIENT_EVAL_OVERRIDE and (
        "eval_split" in payload or "scenario_index" in payload
    ):
        return jsonify(
            {
                "error": (
                    "Client overrides for eval_split/scenario_index are disabled "
                    "by server policy."
                )
            }
        ), 400

    eval_split = DEFAULT_EVAL_SPLIT
    if ALLOW_CLIENT_EVAL_OVERRIDE:
        requested_split = payload.get("eval_split", DEFAULT_EVAL_SPLIT)
        if not isinstance(requested_split, str):
            return jsonify({"error": "Field 'eval_split' must be a string."}), 400
        eval_split = requested_split

    requested_index = payload.get("scenario_index") if ALLOW_CLIENT_EVAL_OVERRIDE else None
    if requested_index is not None and (not isinstance(requested_index, int) or requested_index < 0):
        return jsonify({"error": "Field 'scenario_index' must be a non-negative integer."}), 400

    try:
        scenario_count = get_task_scenario_count(task_id, eval_split)
        if requested_index is None:
            scenario_index = SCENARIO_COUNTERS.get(task_id, 0)
            if scenario_count > 0:
                SCENARIO_COUNTERS[task_id] = (scenario_index + 1) % scenario_count
        else:
            scenario_index = requested_index

        current_env = EmailTriageEnv(
            task_id=task_id,
            scenario_index=scenario_index,
            split=eval_split,
            runtime_options=runtime_options,
        )
        reset_result = current_env.reset()
    except KeyError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(reset_result.model_dump())


@app.post("/step")
def step_endpoint():
    """Advance environment by one action and return StepResult JSON.

    Returns:
        Flask response containing step payload.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Malformed JSON payload."}), 400

    step_result = current_env.step(payload)
    return jsonify(step_result.model_dump())


@app.post("/state")
def state_endpoint():
    """Return read-only EnvironmentState JSON snapshot.

    Returns:
        Flask response containing state payload.
    """
    state_result = current_env.state()
    return jsonify(state_result.model_dump())


def main() -> None:
    """Run the Flask app for local and script-based launches."""
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
