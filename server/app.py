"""Auxiliary server entrypoint required by OpenEnv local validation checks."""

from flask import Flask, Response, jsonify, request

from environment import EmailTriageEnv

FRONTEND_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OpenEnv Email Triage Console</title>
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
    </style>
</head>
<body>
    <div class="wrap">
        <div class="title">
            <h1>OpenEnv Email Triage Console</h1>
            <span class="badge" id="badge">connecting...</span>
        </div>

        <div class="grid">
            <section class="card">
                <h2>Reset Episode</h2>
                <div class="row">
                    <select id="taskId">
                        <option value="task_easy">task_easy</option>
                        <option value="task_medium">task_medium</option>
                        <option value="task_hard">task_hard</option>
                    </select>
                </div>
                <div class="row">
                    <button class="accent" id="btnReset">POST /reset</button>
                    <button class="secondary" id="btnState">POST /state</button>
                </div>
                <div class="status" id="status">Ready</div>
            </section>

            <section class="card">
                <h2>Step Action</h2>
                <div class="row">
                    <select id="label">
                        <option value="urgent">urgent</option>
                        <option value="normal" selected>normal</option>
                        <option value="spam">spam</option>
                        <option value="archive">archive</option>
                    </select>
                </div>
                <div class="row">
                    <input id="routeTo" placeholder="route_to (e.g. billing, safety, engineering)" value="general" />
                </div>
                <div class="row">
                    <textarea id="summary" placeholder="Write a contextual summary with supporting clues.">No summary yet.</textarea>
                </div>
                <div class="row">
                    <button id="btnStep">POST /step</button>
                </div>
            </section>

            <section class="card wide">
                <h2>Response</h2>
                <pre id="output">Waiting for request...</pre>
            </section>
        </div>
    </div>

    <script>
        const statusEl = document.getElementById('status');
        const badgeEl = document.getElementById('badge');
        const outEl = document.getElementById('output');

        function setStatus(msg, isError = false) {
            statusEl.textContent = msg;
            statusEl.classList.toggle('error', isError);
        }

        function writeOutput(value) {
            outEl.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
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
                badgeEl.textContent = data.status === 'ok' ? 'service online' : 'service degraded';
            } catch (e) {
                badgeEl.textContent = 'offline';
            }
        }

        document.getElementById('btnReset').addEventListener('click', async () => {
            const taskId = document.getElementById('taskId').value;
            setStatus('Resetting ' + taskId + '...');
            try {
                const data = await postJson('/reset', { task_id: taskId });
                setStatus('Reset complete');
                writeOutput(data);
            } catch (e) {
                setStatus(e.message, true);
                writeOutput(String(e));
            }
        });

        document.getElementById('btnState').addEventListener('click', async () => {
            setStatus('Fetching state...');
            try {
                const data = await postJson('/state', {});
                setStatus('State loaded');
                writeOutput(data);
            } catch (e) {
                setStatus(e.message, true);
                writeOutput(String(e));
            }
        });

        document.getElementById('btnStep').addEventListener('click', async () => {
            const payload = {
                label: document.getElementById('label').value,
                summary: document.getElementById('summary').value,
                route_to: document.getElementById('routeTo').value,
            };
            setStatus('Submitting action...');
            try {
                const data = await postJson('/step', payload);
                setStatus('Step completed');
                writeOutput(data);
            } catch (e) {
                setStatus(e.message, true);
                writeOutput(String(e));
            }
        });

        warmup();
    </script>
</body>
</html>
"""

app = Flask(__name__)
current_env = EmailTriageEnv(task_id="task_easy")


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
        }
    )


@app.post("/reset")
def reset_endpoint():
    """Reset the environment with a selected task and return ResetResult JSON."""
    global current_env

    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    elif not isinstance(payload, dict):
        return jsonify({"error": "Malformed JSON payload."}), 400

    task_id = payload.get("task_id", "task_easy")
    if not isinstance(task_id, str):
        return jsonify({"error": "Field 'task_id' must be a string."}), 400

    try:
        current_env = EmailTriageEnv(task_id=task_id)
        reset_result = current_env.reset()
    except KeyError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(reset_result.model_dump())


@app.post("/step")
def step_endpoint():
    """Advance environment by one action and return StepResult JSON."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Malformed JSON payload."}), 400

    step_result = current_env.step(payload)
    return jsonify(step_result.model_dump())


@app.post("/state")
def state_endpoint():
    """Return read-only EnvironmentState JSON snapshot."""
    state_result = current_env.state()
    return jsonify(state_result.model_dump())


def main() -> None:
    """Run the Flask app for local and script-based launches."""
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
