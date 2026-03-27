"""Flask server wrapper for the OpenEnv email triage environment."""

from flask import Flask, jsonify, request

from environment import EmailTriageEnv

app = Flask(__name__)
current_env = EmailTriageEnv(task_id="task_easy")


@app.post("/reset")
def reset_endpoint():
    """Reset the environment with a selected task and return ResetResult JSON.

    Returns:
        Flask response containing reset payload.
    """
    global current_env

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Malformed JSON payload."}), 400

    task_id = payload.get("task_id")
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
