# OpenEnv Email Triage Environment

A real-world AI agent training environment that simulates professional email triage.
Built to the OpenEnv specification for standardized agent evaluation and benchmarking.

- **Status:** In Development
- **Domain:** Email Triage
- **Deployment:** Hugging Face Spaces (Docker)

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Who Is This For?](#who-is-this-for)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Tasks](#tasks)
- [Reward Function](#reward-function)
- [Quick Start](#quick-start)
- [Running Inference](#running-inference)
- [Inference Architecture](#inference-architecture)
- [Score Table](#score-table)
- [Docker Deployment](#docker-deployment)
- [Hugging Face Space](#hugging-face-space)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## What Is This?

This environment simulates a professional email inbox where an AI agent must:

1. Read incoming emails with realistic metadata (sender, subject, body, thread history).
2. Classify each email with the correct priority label.
3. Route each email to the appropriate department or person.
4. Summarize the email's key information.

Think of it as OpenAI Gym for office work. Instead of balancing a pole, the agent triages an
inbox. The environment provides structured observations, accepts structured actions, and
returns graded rewards with partial credit.

Every decision is scored by a deterministic programmatic grader: no LLM-as-judge,
no randomness, fully reproducible.

---

## Who Is This For?

| Audience | Use Case |
|---|---|
| AI Safety Researchers | Measure agent behavior on realistic tasks with known ground truth |
| LLM Agent Developers | Benchmark models and prompting strategies on real-world work |
| RL Researchers | Train agents with shaped rewards in a professional task environment |
| Companies | Evaluate LLM agents before deploying them to handle real email |

---

## Observation Space

What the agent sees at each step:

| Field | Type | Description |
|---|---|---|
| `email_id` | `str` | Unique identifier for this email |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text |
| `sender` | `str` | Sender's email address |
| `timestamp` | `str` | ISO 8601 timestamp of when the email was received |
| `thread_history` | `list[str]` | Previous messages in the email thread (may be empty) |
| `task_id` | `str` | Which task is currently active |
| `step_number` | `int` | Current step in the episode (0-indexed) |
| `total_emails` | `int` | Total number of emails to process in this task |

The observation never contains the correct answer. The agent must reason from email content.

---

## Action Space

What the agent must output at each step:

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `label` | `Literal` | `"urgent"`, `"normal"`, `"spam"`, `"archive"` | Priority classification |
| `summary` | `str` | Free text | Brief summary of the email's content and intent |
| `route_to` | `str` | Free text (`"billing"`, `"safety"`, `"engineering"`) | Department or person |

### Example action JSON

```json
{
  "label": "urgent",
  "summary": "Customer reports a safety issue with product overheating",
  "route_to": "safety"
}
```

---

## Tasks

### Task 1 — Easy (`task_easy`)

Objective: Correctly classify a single unambiguous email.

Scoring:

- Correct label: 1.0
- Wrong label but correct routing: 0.3
- Everything wrong: 0.0

### Task 2 — Medium (`task_medium`)

Objective: Triage a queue of 5 emails with mixed priority signals.

Scoring:

- Each email scored individually
- Score = (correct labels / total emails) * priority weight factor
- Higher-priority misclassifications are penalized more heavily
- Final score = weighted mean of all individual scores

### Task 3 — Hard (`task_hard`)

Objective: Handle a complex complaint that crosses multiple categories.

Scoring:

- Escalated to safety: 0.4 weight
- Correct routing: 0.3 weight
- Marked as urgent: 0.3 weight
- Penalty: -0.2 if marked as spam
- Final score = weighted sum of sub-scores (clipped to 0.0 minimum)

---

## Reward Function

The reward function provides dense training signal at every step, not just binary pass/fail.

### Formula

```text
final_reward = base_score - (step_count * 0.01) + trajectory_bonus - penalties
```

### Components

| Component | Value | Condition |
|---|---|---|
| Base score | 0.0-1.0 | Raw grader score for the current step |
| Step penalty | -0.01 per step | Encourages efficiency |
| Trajectory bonus | +0.2 | If all tasks completed with mean score > 0.8 |
| Destructive action penalty | -0.5 | Agent archives or deletes without reading |
| Loop detection penalty | -0.3 | Same action repeated 3+ times consecutively |

The final reward is clipped to [-1.0, 1.0] before being returned.

---

## Quick Start

### Prerequisites

- Python 3.11+
- API endpoint, model name, and token for inference

### Installation

```bash
pip install -r requirements.txt
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-token-here"
```

### Run the environment locally

```bash
python server.py

curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"label": "urgent", "summary": "Test", "route_to": "billing"}'

curl -X POST http://localhost:7860/state
```

---

## Running Inference

```bash
python inference.py --task all
python inference.py --task 1
```

The script reads API settings from environment variables and uses fallback actions when
model output is unparseable, so episodes still complete.

---

## Inference Architecture

The inference script (inference.py) follows this loop:

```text
1. Initialize OpenAI client + environment
2. reset() to get first observation
3. Loop until done or MAX_STEPS:
  - Build prompt from observation + history
  - Call LLM with OpenAI client (catch request errors)
  - Parse response into action (fallback on parse failure)
  - env.step(action)
  - Record reward and history
4. Print score table
```

### Environment Variables Required

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-token-here"
```

Fallback behavior when parsing fails:

```json
{"label": "normal", "summary": "Unable to parse response", "route_to": "general"}
```

---

## Score Table

Placeholder until inference is run.

| Model | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Mean |
|---|---|---|---|---|
| MODEL_NAME | TBD | TBD | TBD | TBD |

Expected rough ranges:

- GPT-4o: 0.8-1.0 on easy, 0.5-0.8 on medium, 0.4-0.7 on hard

---

## Docker Deployment

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

For Apple Silicon:

```bash
docker build --platform linux/amd64 -t email-triage-env .
```

---

## Hugging Face Space

Live URL placeholder:

`https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env`

Example interaction:

```bash
export SPACE_URL="https://YOUR_USERNAME-email-triage-env.hf.space"

curl -X POST "$SPACE_URL/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

---

## API Reference

### POST /reset

Request:

```json
{"task_id": "task_easy"}
```

Response: `EmailObservation` JSON object.

### POST /step

Request:

```json
{
  "label": "urgent",
  "summary": "Customer needs immediate help",
  "route_to": "support"
}
```

Response:

```json
{
  "observation": {},
  "reward": {"score": 0.85, "breakdown": {}, "feedback": "..."},
  "done": false,
  "info": {"step": 1, "task_id": "task_easy"}
}
```

### POST /state

No request body required.

Response: `EnvironmentState` JSON object.

---

## Project Structure

```text
.
├── models.py
├── tasks.py
├── graders.py
├── environment.py
├── server.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── RULES.md
```

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Static email data | No dynamic email generation |
| Single-agent server instance | Concurrent agents can conflict |
| No live thread simulation | Thread history is static |
| English-only content | No multilingual coverage |
| No attachments | Text-only triage |
| Simplified routing | No org chart or availability modeling |
| No temporal dynamics | Queue is fixed at reset |
| String-matching grader edges | Equivalent routes may not always get credit |

What an agent cannot exploit:

- The correct answer is never present in observations
- The grader is a pure function and cannot be manipulated
- Step penalty cannot be bypassed except by efficient actions

---

## Summary of Revision 2 Changes

| What Changed | Before | After | Why |
|---|---|---|---|
| Return type of step() | tuple | StepResult object | Match sample result.observation pattern |
| Return type of reset() | EmailObservation | ResetResult object | Match sample result.observation pattern |
| New models | 4 models | 6 models (+StepResult, +ResetResult) | Match sample interface |
| API key reading | OPENAI_API_KEY style | HF_TOKEN or API_KEY via os.getenv | Match sample fallback pattern |
| Temperature guidance | 0 | 0.2 | Match sample behavior |
| Response parsing | JSON-only assumption | Text parsing with fallback action | Robustness to non-JSON model output |
| History tracking | Optional | Mandatory | Match sample architecture |
| Step cap | Not explicit | MAX_STEPS constant | Runtime safety and reproducibility |

---

## Contributing

Read `RULES.md` before contributing.

Key constraints:

- Type hints and Pydantic models required
- No extra dependencies without explicit approval
- No features beyond project brief
- Graders must remain deterministic pure functions

---

## License

MIT License.
