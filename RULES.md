# RULES.md - Project Constitution & AI Guardrails
# OpenEnv Email Triage Environment

EVERY AI agent, copilot, or assistant working on this project MUST read and obey this file before generating ANY code.

REVISION 2: Updated based on sample inference.py analysis.
Where submission rules conflict with the original brief, SUBMISSION RULES WIN.
Where the sample script reveals patterns, MATCH THE PATTERNS.

## 0. GOLDEN RULE

> Do NOT generate code that you cannot explain line by line.
> Do NOT add features not listed in this document.
> Do NOT deviate from the file map, naming conventions, or interfaces defined here.
> When in doubt, do LESS, not more.

---

## 1. SCOPE - What This Project Is

- An OpenEnv-compliant AI agent training environment
- Domain: Email Triage (classify, prioritise, route emails)
- Deployed as a Docker-based Hugging Face Space
- Evaluated by inference.py using OpenAI Client with configurable endpoint

### What this project is NOT

- A chatbot
- A web app with a UI
- A game or toy problem
- A fine-tuning pipeline
- A multi-agent system
- An LLM wrapper with extra features
- A BrowserGym environment (the sample uses BrowserGym - we do NOT)

---

## 2. SUBMISSION CHECKLIST - DISQUALIFICATION CRITERIA

These are automated checks. Failing ANY ONE means disqualification.

| # | Check | What the validator does |
|---|---|---|
| 1 | HF Space deploys | Pings Space URL - must return HTTP 200 and respond to reset() |
| 2 | OpenEnv spec compliance | Validates openenv.yaml, typed models, /step, /reset, /state |
| 3 | Dockerfile builds | Runs docker build on the submitted repo - must succeed |
| 4 | Inference reproduces | Runs inference.py - must complete without error and produce scores |
| 5 | 3+ tasks with graders | Enumerates tasks, runs each grader, verifies scores in [0.0, 1.0] |

### Infrastructure constraints

| Constraint | Limit |
|---|---|
| vCPU | 2 |
| Memory | 8 GB |
| Inference runtime | < 20 minutes |

---

## 3. ENVIRONMENT VARIABLES - Mandatory

```python
import os

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
```

How to use in code (EXACT PATTERN - matches sample):

```python
from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[...],
    temperature=0.2,
    max_tokens=200,
    stream=False,
)

response_text = completion.choices[0].message.content or ""
```

Rules:

- NEVER hard-code any of these values
- NEVER use os.environ["VAR"] (use os.getenv() - matches sample)
- NEVER use any LLM client other than openai.OpenAI
- Support both HF_TOKEN and API_KEY with or fallback (matches sample)

---

## 4. FILE MAP - Strict Build Order

| Order | File | Purpose | May import from |
|---|---|---|---|
| 1st | models.py | Pydantic models + StepResult wrapper | stdlib, pydantic only |
| 2nd | tasks.py | Task definitions + hard-coded email data | models.py only |
| 3rd | graders.py | Deterministic grader functions | models.py, tasks.py only |
| 4th | environment.py | Core env class: step, reset, state | models, tasks, graders |
| 5th | server.py | Flask HTTP wrapper: /reset, /step, /state | environment.py, models.py |
| 6th | inference.py | OpenAI Client inference script | models.py, environment.py |
| 7th | openenv.yaml | Spec metadata | N/A (data file) |
| 8th | Dockerfile | Container build | N/A (config file) |
| 8th | requirements.txt | Pinned dependencies | N/A (config file) |
| 9th | README.md | Full documentation | N/A (documentation) |

### Rules about files

- Do NOT create files not listed above. No utils.py, helpers.py, or config.py.
- Do NOT merge files. Each file has one responsibility.
- Do NOT create subdirectories. All files live in the project root.
- Do NOT add init.py. This is not a package.

---

## 5. DEPENDENCY RULES

### Allowed dependencies

```txt
pydantic>=2.0,<3.0
flask>=3.0,<4.0
openai>=1.0,<2.0
gunicorn>=21.0,<23.0
```

### Conditionally allowed (only if needed)

```txt
numpy
Pillow
```

### Forbidden

- No LangChain, LlamaIndex, or any agent framework
- No pandas or scipy
- No database libraries
- No async frameworks (FastAPI, aiohttp) - use Flask
- No frontend frameworks (Streamlit, Gradio)
- No ML libraries (torch, transformers, sklearn)

---

## 6. PYDANTIC MODEL RULES

### models.py constraints

- ALL models MUST inherit from pydantic.BaseModel
- ALL fields MUST have explicit type annotations
- ALL Literal types MUST use typing.Literal with exhaustive values
- NO methods on models (except StepResult and ResetResult wrappers)
- NO validators that call external services
- NO default_factory that uses randomness
- Field names MUST be snake_case
- NO nested models deeper than 2 levels

### Required models (exact names)

```python
class EmailObservation(BaseModel): ...
class TriageAction(BaseModel): ...
class RewardResult(BaseModel): ...
class EnvironmentState(BaseModel): ...
class StepResult(BaseModel): ...
class ResetResult(BaseModel): ...
```

### StepResult and ResetResult interface (mandatory)

```python
class StepResult(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: dict[str, str | int | float | bool]

class ResetResult(BaseModel):
    observation: EmailObservation
    info: dict[str, str | int | float | bool]
```

### EmailObservation required fields

| Field | Type | Required |
|---|---|---|
| email_id | str | Yes |
| subject | str | Yes |
| body | str | Yes |
| sender | str | Yes |
| timestamp | str | Yes |
| thread_history | list[str] | Yes |
| task_id | str | Yes |
| step_number | int | Yes |
| total_emails | int | Yes |

### TriageAction required fields

| Field | Type | Required |
|---|---|---|
| label | Literal["urgent", "normal", "spam", "archive"] | Yes |
| summary | str | Yes |
| route_to | str | Yes |

### RewardResult required fields

| Field | Type | Required |
|---|---|---|
| score | float | Yes |
| breakdown | dict[str, float] | Yes |
| feedback | str | Yes |

### EnvironmentState required fields

| Field | Type | Required |
|---|---|---|
| task_id | str | Yes |
| current_step | int | Yes |
| total_steps | int | Yes |
| done | bool | Yes |
| action_history | list | Yes |
| reward_history | list | Yes |

---

## 7. ENVIRONMENT CLASS RULES

- Class name: EmailTriageEnv
- Constructor: __init__(self, task_id: str)
- MUST accept a task_id string
- MUST NOT call any external API
- MUST NOT use randomness

### reset() -> ResetResult

- MUST return a ResetResult object (not a bare observation)
- result.observation must contain the first email
- MUST reset all internal state
- MUST be callable multiple times without side effects
- HF Space validator will call /reset and expect HTTP 200 + valid JSON

### step(action: TriageAction) -> StepResult

- MUST return a StepResult object (not a tuple)
- result.observation: next email or terminal observation
- result.reward: float score for this step
- result.done: bool indicating episode end
- result.info: metadata dict
- MUST never raise an exception from bad agent input
- If action validation fails: return StepResult with reward=0.0 and continue
- MUST increment step counter
- MUST set done=True when all emails processed or max_steps hit

### state() -> EnvironmentState

- MUST return the full current internal state
- MUST be read-only

### Hard rules for environment.py

- NO randomness
- NO API calls
- NO file I/O during step/reset/state
- NO global mutable state
- NO threading or async
- NO print statements

---

## 8. TASK DATA RULES

Unchanged from previous version.

- All email data MUST be hard-coded
- NO loading from external files, URLs, or databases
- Task IDs: task_easy, task_medium, task_hard
- Each task defines: task_id, description, emails, ground_truth
- Ground truth MUST NOT be in observations (no answer leakage)
- Realistic professional email content
- NO offensive or NSFW content

---

## 9. GRADER RULES

Unchanged from previous version.

- Pure functions
- Deterministic
- Partial credit
- Scores in [0.0, 1.0]

---

## 10. REWARD FUNCTION RULES

Unchanged from previous version.

```text
final_reward = base_score - (step_count * 0.01) + trajectory_bonus - penalties
```

Final reward is clipped to [-1.0, 1.0].

---

## 11. SERVER RULES

### server.py constraints

- MUST use Flask
- Exactly THREE routes:
  - POST /reset: accepts {"task_id": str}, returns ResetResult JSON
  - POST /step: accepts TriageAction JSON, returns StepResult JSON
  - POST /state: returns EnvironmentState JSON
- MUST listen on port 7860
- MUST handle malformed JSON gracefully (return 400)
- All responses must include Content-Type: application/json
- Validator will ping and call /reset, which must return HTTP 200

### /step response format

```json
{
  "observation": {},
  "reward": 0.85,
  "done": false,
  "info": {"step": 1, "task_id": "task_easy"}
}
```

### /reset response format

```json
{
  "observation": {},
  "info": {"task_id": "task_easy"}
}
```

---

## 12. INFERENCE SCRIPT RULES

CRITICAL PATTERNS FROM SAMPLE - MUST FOLLOW

### Architecture (matches sample)

```text
1. Initialize OpenAI client with env vars
2. Create environment instance
3. Call reset(), get initial observation
4. Loop up to MAX_STEPS:
   a. Build prompt from observation + history
   b. Call LLM
   c. Parse response into action (with fallback)
   d. Call step(action)
   e. Record history
   f. Check done flag
5. Print results
```

### Mandatory constants

```python
MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = ...
```

### Response parsing rules

- Do NOT rely only on response_format={"type": "json_object"}
- Parse free-text responses with regex or string matching
- If parsing fails, use a fallback action
- Strip prefixes like action: or next action: before parsing
- Regex parsing with fallback is preferred

### History tracking

```python
history: list[str] = []
history_line = f"Step {step}: {action} -> reward {reward:+.2f}"
history.append(history_line)
```

### Error handling

```python
try:
    completion = client.chat.completions.create(...)
    response_text = completion.choices[0].message.content or ""
except Exception as exc:
    print(f"Model request failed ({exc}). Using fallback action.")
    response_text = ""
```

### Output format

```text
Episode: task_easy
Step 1: label=urgent, route=safety -> reward +0.85
Final score: 0.85

=== SCORE TABLE ===
Task         Score    Steps
task_easy    0.85     1
task_medium  0.62     5
task_hard    0.45     2
Mean         0.64
```

### File naming and location

- File MUST be named inference.py
- MUST be in the project root directory
- MUST be runnable with python inference.py
- MUST complete in under 20 minutes

---

## 13. DOCKERFILE RULES

- Base image: python:3.11-slim
- WORKDIR: /app
- Copy requirements.txt first, pip install, then copy source
- EXPOSE 7860
- Create non-root user
- CMD starts the server
- Must build with --platform linux/amd64
- Must run within 2 vCPU / 8 GB memory
- No unnecessary system packages
- No CUDA/GPU dependencies

---

## 14. CODE STYLE RULES

- Python 3.11+
- Type hints on ALL function signatures
- Docstrings on ALL public functions (Google style)
- No single-letter variable names except i in loops
- Comments explain WHY, not WHAT
- Max line length: 100 characters
- f-strings only
- No wildcard imports
- Import order: stdlib -> third-party -> local

---

## 15. WHAT AI MUST NEVER DO

- Never add features not in this spec
- Never use an LLM inside a grader
- Never generate fake scores
- Never create a UI
- Never use randomness in the environment
- Never store API keys in code
- Never skip error handling in step()
- Never use bare dicts where Pydantic models are specified
- Never name the inference script baseline.py
- Never use OPENAI_API_KEY; use HF_TOKEN/API_KEY
- Never use response_format={"type": "json_object"} without text-parsing fallback
- Never return tuples from step/reset; use StepResult/ResetResult objects
- Never skip the fallback action pattern
- Never skip history tracking in inference

---

## 16. DEFINITION OF DONE - Per Phase Checklist

### Phase 1 complete when

- models.py exists with all 6 models (including StepResult, ResetResult)
- All fields match this document
- Models instantiate with sample data without errors
- StepResult has observation, reward, done, info attributes

### Phase 2 complete when

- tasks.py exists with 3 tasks
- All email data is realistic and hard-coded
- Ground truth exists for every email
- No answer leakage

### Phase 3 complete when

- graders.py has 3 pure grader functions
- Partial credit works
- All scores in [0.0, 1.0]

### Phase 4 complete when

- environment.py has EmailTriageEnv class
- reset() returns ResetResult
- step() returns StepResult
- step() handles invalid input without crashing
- Full episode runs to completion

### Phase 5 complete when

- server.py has /reset, /step, /state routes
- /reset returns {"observation": ..., "info": ...}
- /step returns {"observation": ..., "reward": ..., "done": ..., "info": ...}
- Malformed requests return 400
- Port 7860

### Phase 6 complete when

- inference.py follows sample architecture
- Uses os.getenv() for API_BASE_URL, HF_TOKEN/API_KEY, MODEL_NAME
- Has MAX_STEPS, TEMPERATURE, MAX_TOKENS, FALLBACK constants
- Has history tracking
- Has response parsing with fallback
- Has try/except around API calls
- Prints score table
- Completes in under 20 minutes

### Phase 7-9

Unchanged from previous version.

---

## 17. WHEN IN DOUBT

- Re-read this file
- Re-read the project briefing
- Re-read the sample inference.py
- Match the sample patterns
- Choose the simpler option
- Ask the human, do not guess

This file is the law. Code that violates it gets deleted.
