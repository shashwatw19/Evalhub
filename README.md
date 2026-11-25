# EvalHub

EvalHub is a local evaluation platform for testing AI agents. It supports:

1. **Evaluate saved agent results** - use recorded inputs and outputs already produced by an agent.
2. **Evaluate a live agent** - have EvalHub call an HTTP endpoint or trusted Python function for every test case.
3. **Evaluate registered agent results** - register a read-only GET API, retrieve existing
   agent runs by ID, and evaluate their recorded outputs. This mode never triggers an investigation.

Evaluations are scored with DeepEval metrics, stored in SQLite, shown in a Streamlit dashboard, and can be compared against an earlier baseline.

## What you need

- Python 3.10 or newer
- Windows PowerShell commands are shown below
- Optional: a Gemini API key for LLM-based metrics

## Install

Run the setup from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Verify the installation:

```powershell
.\.venv\Scripts\evalhub.exe --help
.\.venv\Scripts\python.exe -m pytest
```

For Gemini judge metrics, also install the optional dependency and set a key in your shell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gemini]"
$env:GOOGLE_API_KEY = "<your API key>"
```

The key is read from the environment and should not be stored in a dataset or config.

## Choose an evaluation mode

### 1. Saved-output evaluation

Use this when your agent has already run and you have a JSONL file containing the recorded output.

No agent is called during the evaluation. The platform loads each `actual_output` and applies the configured metrics.

Example dataset:

```json
{"case_id":"case-001","input":"What is 2 + 2?","actual_output":"4","expected_output":"4","tags":["smoke-test"]}
{"case_id":"case-002","input":{"task":"classify","text":"The service is excellent."},"actual_output":{"category":"positive"},"expected_output":{"category":"positive"}}
```

Example config:

```json
{
  "adapter": "generic",
  "metrics": [
    {"name": "exact_match", "threshold": 1.0, "required": true}
  ]
}
```

Run it:

```powershell
.\.venv\Scripts\evalhub.exe validate --dataset examples\generic\cases.jsonl
.\.venv\Scripts\evalhub.exe run --dataset examples\generic\cases.jsonl --config examples\generic\config.json --label baseline
.\.venv\Scripts\evalhub.exe list-runs
```

The first command confirms the dataset shape. The run command creates a new run in `evalhub.db` and prints its run ID. The list command prints every stored run and its counts.

### 2. Live HTTP agent evaluation

Use this when EvalHub should send each case to an agent endpoint.

The live dataset only needs a stable `case_id`, a non-null `input`, and any fields needed by the selected metrics:

```json
{"case_id":"greeting","input":"hello","expected_output":"HELLO"}
```

Example HTTP config:

```json
{
  "agent": {
    "type": "http",
    "url": "https://agent.example.com/invoke",
    "input_field": "prompt",
    "response_path": "result.answer",
    "tool_calls_path": "result.tools",
    "retrieval_context_path": "result.context",
    "auth_env": "AGENT_API_TOKEN",
    "timeout_seconds": 30,
    "retries": 0
  },
  "metrics": [
    {"name": "exact_match", "threshold": 1.0, "required": true}
  ]
}
```

Before running:

```powershell
$env:AGENT_API_TOKEN = "<token>"
.\.venv\Scripts\evalhub.exe validate --dataset agent_cases.jsonl --input-only
.\.venv\Scripts\evalhub.exe run --dataset agent_cases.jsonl --config http_agent_config.json --label candidate
```

The connector sends a JSON POST:

```json
{"prompt":"hello"}
```

It expects a JSON response. Dot-separated paths such as `result.answer`, `result.tools`, and `result.context` extract fields from the response. Omit `response_path` to use the whole response as `actual_output`.

Security and behavior details:

- HTTP redirects are rejected.
- Calls run sequentially.
- The default is zero retries because retrying a POST may execute the agent more than once.
- Timeout is applied per attempt.
- The default authorization header is `Authorization: Bearer <token>`. Use `auth_header` and `auth_prefix` to customize it.
- Never put credentials in a config or URL. Configurations are stored with every run.
- Use HTTPS for authenticated endpoints.
- A case-level HTTP or timeout error is stored as an execution error and does not become a successful output.

### 3. Live Python agent evaluation

Use this for a local, trusted Python callable.

The callable receives the case `input` and must return a JSON-serializable value:

```python
# my_agent.py
def respond(value):
    return value.upper()
```

Configuration:

```json
{
  "agent": {
    "type": "python",
    "callable": "my_agent:respond",
    "response_path": null,
    "tool_calls_path": null,
    "retrieval_context_path": null
  },
  "metrics": [
    {"name": "exact_match", "threshold": 1.0, "required": true}
  ]
}
```

The module must be importable from the process. Run the included example:

```powershell
.\.venv\Scripts\evalhub.exe validate --dataset examples\generic\agent_cases.jsonl --input-only
.\.venv\Scripts\evalhub.exe run --dataset examples\generic\agent_cases.jsonl --config examples\generic\python_agent_config.json --label baseline
```

Python connectors execute arbitrary local code. The callable should implement its own timeout and only trusted modules should be loaded. If the callable returns an envelope, set `response_path`, `tool_calls_path`, or `retrieval_context_path` to extract the required values.

### 4. Registered agent results (GET)

Register a results source once, then select it for subsequent evaluations. Agent
registrations and evaluation runs use the same SQLite database. Registering, updating,
listing, or disabling an agent does not call its API.

The included `card_risk_agent.json` registers this request:

```text
GET http://127.0.0.1:8000/api/evalhub/investigate?run_id=<source-run-id>
```

Its required response is one JSON object containing `recommendation`. All other
response mappings are optional and default to `null`:

| Connector setting | Value to extract |
|---|---|
| `response_path` | Output to score (`recommendation` in the example) |
| `input_path` | Original task/input given to the agent |
| `run_id_path` | Source run ID; checked against a `run_id` lookup when both exist |
| `status_path` | Source execution status |
| `explanation_path` | Explanation retained for inspection |
| `evidence_path` | Claimed evidence retained for inspection, not authoritative ground truth |
| `tool_calls_path` | Canonical tool call records |
| `retrieval_context_path` | Reference passages |

Set paths only when the API returns those fields. Configured paths must exist.
Dot-separated nested paths are supported. With `status_path`, only statuses in
`completed_statuses` (default: `completed`, `success`, case-insensitive) are evaluated;
other statuses are retained as source-not-completed errors without invoking the judge.
Incomplete runs need only return their configured status field, not output or detail fields.
Without a status mapping, EvalHub assumes a returned record is ready for evaluation.
Do not register an endpoint that returns incomplete records without configuring status.
List responses are rejected rather than silently selecting the first/latest run.

A retrieval dataset separates lookup values from the original agent task:

```json
{"case_id":"case-001","lookup":{"run_id":"source-run-123"},"expected_output":"ESCALATE"}
```

Replace `REPLACE_WITH_SOURCE_RUN_ID` in `card_risk_cases.jsonl` with a real source
run ID. Do not substitute an alert ID unless the endpoint actually accepts it.
If an original task is available, provide `input` in the dataset or configure
`input_path` to extract it. A dataset-supplied input takes precedence.
Lookup IDs are never used as substitute task text.

The card-risk evaluation profile now defaults to offline `exact_match` for the
recommendation. Enable `judge_enabled`, supply a supported `judge_model`, and add
judge metrics only when desired. Judge metrics require the original task;
if absent, a required metric reports an error and an optional metric reports
`not_applicable`. Scoring explanation quality requires selecting output that actually
includes the explanation; capturing `explanation_path` alone does not change scored output.

```powershell
.\venv\Scripts\evalhub.exe agents register --file card_risk_agent.json
.\venv\Scripts\evalhub.exe agents list
.\venv\Scripts\evalhub.exe agents show --agent-id card-risk
.\venv\Scripts\evalhub.exe validate --dataset card_risk_cases.jsonl --lookup-only
.\venv\Scripts\evalhub.exe run --dataset card_risk_cases.jsonl --config card_risk_config.json --label card-risk
```

Use your environment's executable path (`.venv` if installed using the setup above).
To preview a record without evaluating it, explicitly run:

```powershell
.\venv\Scripts\evalhub.exe agents test --agent-id card-risk --lookup '{"run_id":"source-run-123"}'
```

This command fetches and displays the record, including potentially sensitive data.
It does not persist an evaluation. The dashboard also provides **Test retrieval** and
**Preview first case** actions. A subsequent evaluation retrieves fresh data again.

Edit the registration JSON and use `agents update --file card_risk_agent.json` to
append a revision. `agents show --agent-id card-risk --revision 1` shows an older
revision. `agents disable --agent-id card-risk` blocks new retrievals, including
references to older revisions. To re-enable, update with `"enabled": true`.
All agent commands accept `--db`; use the same database for `run`.

Evaluation config:

```json
{
  "agent_ref": {"agent_id": "card-risk"},
  "metrics": [{"name": "exact_match", "threshold": 1.0, "required": true}]
}
```

Optionally set `agent_ref.revision` to pin a revision. Otherwise the latest active
revision is resolved once at evaluation start. Supplying both `agent` and `agent_ref`
is rejected. Existing inline POST/Python and saved-output configs still work.

For query mapping, `"query_params": {"run_id": "id"}` sends dataset
`lookup.id` as the `run_id` query parameter. For path mapping, use a URL such as
`https://agent.example.com/runs/{id}` and `"path_params": {"id": "run_id"}`.
Declare required lookup types in `lookup_fields`, for example `{"run_id":"string"}`.
Supported types are `string`, `integer`, `number`, and `boolean`; all declared fields
are required and must match the mapping. Values are URL-encoded and GET has no body.
Inline HTTP configs can also explicitly set `"method": "GET"`; POST remains the default.

Each persisted evaluation captures the selected registry revision, resolved
configuration (without authentication tokens), lookup values, retrieval time,
source run ID when available, raw JSON response, and normalized evaluation data.
Registry edits never modify past runs. Expected answers and authoritative evidence
remain dataset-owned. Comparison fingerprints use the original task and evaluation
requirements, not source lookup IDs or returned output.

Keep tokens in the environment referenced by `auth_env`. Do not put credentials in
URLs, registration JSON, datasets, or returned payloads. Treat response snapshots as
sensitive. The registry is local/trusted, not a multi-user endpoint allowlist or
access-control system. Retrieval is sequential with bounded timeouts and optional
retries; scheduling, automatic polling, and investigation triggering are not included.
Opening an existing database upgrades it non-destructively to schema version 2.

## JSONL dataset format

Each non-empty line must be one JSON object. A valid generic record uses these fields:

| Field | Required | Meaning |
|---|---:|---|
| `case_id` | Yes | Stable, unique identifier for the case |
| `input` | Saved/live runs; metric-dependent for retrieval | Original task; never a replacement for lookup IDs |
| `lookup` | Registered results only | Parameters identifying the existing source run |
| `actual_output` | Saved runs only | Previously observed agent output |
| `expected_output` | For exact match/correctness | Ground-truth output |
| `expected_labels` | Metric-specific | Optional structured labels |
| `tool_calls` | Tool metrics | Observed tool trace |
| `expected_required_tools` | Tool coverage | Tool names that should be called |
| `evidence` | Evidence metric | Authoritative evidence records |
| `retrieval_context` | Faithfulness | Reference passages used by the judge |
| `tags` | No | Labels for organizing cases |
| `source_run_id` | No | Source agent run identifier |
| `metadata` | No | Extra structured metadata |

Tool trace objects use this shape:

```json
{
  "name": "search",
  "arguments": {"query": "example"},
  "result": "returned result",
  "status": "success",
  "call_id": "call-123"
}
```

Case IDs must be unique within a dataset. Any fields not recognized by the generic adapter are preserved in the case's `metadata`.

For a source-specific native schema, register an adapter in Python:

```python
from evalhub.adapters import register_adapter

def adapt(record):
    # Return an EvaluationCase
    ...

register_adapter("my_format", adapt)
```

Then set `"adapter": "my_format"` in the evaluation config.

## Metrics

| Metric | Prerequisites | Behavior |
|---|---|---|
| `exact_match` | `expected_output` | Compares actual and expected values exactly |
| `tool_coverage` | `expected_required_tools`; observed `tool_calls` | Checks whether every required tool was called |
| `evidence_reference_validity` | An `evidence` list and output containing an `evidence` list | Verifies claimed record IDs and sources against authoritative evidence |
| `explanation_quality` | `judge_enabled: true` | Offline word-count heuristic with 30 words as a full score |
| `answer_relevancy` | Gemini config | LLM judge for relevance of the answer to the input |
| `answer_correctness` | Gemini config and `expected_output` | LLM judge for factual consistency with the expected answer |
| `faithfulness` | Gemini config and `retrieval_context` or successful tool results | LLM judge for grounding in supplied reference passages or successful tool outputs |

For `faithfulness`, explicitly supplied `retrieval_context` takes precedence. If it is absent, EvalHub builds judge context from the `result` values of successful (`status: "success"`) tool calls. Failed calls and calls without results are ignored. Live-agent responses must expose tool calls with `tool_calls_path`; saved-output cases must include them as `tool_calls`.

For every metric, `threshold` is the minimum score (from `0.0` to `1.0`) and `required` controls missing prerequisites:

- `required: true` makes missing prerequisites an error or the metric is a strict failure.
- `required: false` normally marks missing optional prerequisites as `not_applicable`.

Example deterministic config:

```json
{
  "metrics": [
    {"name": "exact_match", "threshold": 1.0, "required": true},
    {"name": "tool_coverage", "threshold": 1.0, "required": false},
    {"name": "evidence_reference_validity", "threshold": 1.0, "required": false}
  ]
}
```

Example Gemini config:

```json
{
  "judge_enabled": true,
  "judge_model": "gemini-2.0-flash",
  "metrics": [
    {"name": "answer_relevancy", "threshold": 0.7, "required": true},
    {"name": "answer_correctness", "threshold": 0.7, "required": true},
    {"name": "faithfulness", "threshold": 0.7, "required": false}
  ]
}
```

To use Vertex AI instead of the Gemini API key, install the Gemini extra and configure
Application Default Credentials, a project with the Vertex AI API enabled, and a
supported model and location. In Cloud Shell, for example:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="$(gcloud config get-value project)"
export GOOGLE_CLOUD_LOCATION="global"
gcloud auth application-default login
```

Set `judge_model` to an available Vertex AI model, such as `gemini-4-argon`.
Unset `GOOGLE_GENAI_USE_VERTEXAI` or set it to `false` to use `GOOGLE_API_KEY`
instead. The card-risk example extracts only `recommendation`, so the judge cannot
assess the agent's investigation or reasoning unless the selected output includes it.

Gemini metrics send inputs and outputs to Google and may incur charges. Only enable them for data approved for that transfer. Deterministic metrics run offline and do not contact Gemini.

## Read evaluation results

Run a summary contains counts for:

- **passed**
- **failed**
- **error**
- **not_applicable**
- **safety_failures**

The CLI also prints diagnostics for failed cases and connector errors.

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | Evaluation passed; no failures or safety failures |
| `1` | One or more cases failed, or a safety failure was detected |
| `2` | Invalid input, dataset/configuration error, execution error, or comparison incompatibility |

Save a machine-readable run ID by copying it from the run summary. The run ID is required for CLI comparisons.

## Compare baseline and candidate runs

Both runs must use the same:

- Stable case IDs
- Inputs
- Expected outputs
- Expected labels
- Required tools
- Evidence and retrieval context
- Metric names, thresholds, options, and required settings
- DeepEval and metric implementations

The platform calculates a strict comparison fingerprint from these requirements. Agent outputs and tool traces are intentionally excluded, so a regression compares candidate behavior against the baseline requirements rather than against prior agent output.

List runs:

```powershell
.\.venv\Scripts\evalhub.exe list-runs
```

Compare:

```powershell
.\.venv\Scripts\evalhub.exe compare --baseline <baseline-run-id> --candidate <candidate-run-id>
.\.venv\Scripts\evalhub.exe compare --baseline <baseline-run-id> --candidate <candidate-run-id> --json-out comparison.json
```

The comparison reports new failures, fixed cases, new errors, missing and extra cases, and safety failures. It exits with `1` for regressions and `2` when runs are incompatible or invalid.

## Use the dashboard

Start the dashboard locally:

```powershell
.\.venv\Scripts\streamlit.exe run src\evalhub\dashboard.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit.

The dashboard contains four tabs:

- **New evaluation** - upload JSONL, choose a saved-output, HTTP, or Python agent, and run with guided or advanced JSON settings.
- **Agents** - register, inspect, update, disable, and explicitly test read-only result sources. New evaluations can select registered agents and preview the first case.
- **Runs** - list runs and inspect inputs, outputs, execution errors, tool calls, retrieval context, and metric diagnostics.
- **Compare** - select a baseline and candidate run and view a comparison.

Uploaded datasets and generated configs are stored under `.evalhub`. Results are stored in `evalhub.db` by default. Set a custom database before launch:

```powershell
$env:EVALHUB_DB = "C:\data\evalhub.db"
.\.venv\Scripts\streamlit.exe run src\evalhub\dashboard.py --server.address 127.0.0.1
```

The dashboard has no authentication and can execute Python connectors or make HTTP/Gemini requests. Keep it bound to `127.0.0.1` and do not expose it to untrusted users.

## Storage and privacy

- SQLite database: `evalhub.db` by default, or `EVALHUB_DB` / CLI `--db`
- Dashboard uploads and configs: `.evalhub`
- DeepEval cache: `.deepeval`
- Virtual environment: `.venv`

Treat datasets, run results, and dashboard uploads as sensitive. Their contents can include prompts, agent answers, tool arguments, retrieval passages, and evidence records.

## Project structure

```text
src/evalhub/
  adapters.py    JSONL normalization and custom adapter registration
  cli.py         validate, run, compare, and list-runs commands
  connectors.py  HTTP and trusted Python live-agent connectors
  dashboard.py   Streamlit interface
  metrics.py     deterministic and Gemini metric implementations
  models.py      Pydantic dataset, config, result, and comparison schemas
  runner.py      evaluation orchestration and comparison fingerprints
  storage.py     SQLite persistence and run retrieval
examples/generic/
  cases.jsonl     saved-output example
  config.json     saved-output configuration
  agent_cases.jsonl
  python_agent_config.json
  sample_agent.py
tests/
  platform test suite
```

## Common workflows

### Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Evaluate a saved candidate

```powershell
.\.venv\Scripts\evalhub.exe run --dataset candidate.jsonl --config config.json --label candidate
```

### Use a different database

```powershell
.\.venv\Scripts\evalhub.exe run --dataset candidate.jsonl --config config.json --label candidate --db C:\data\evalhub.db
.\.venv\Scripts\evalhub.exe list-runs --db C:\data\evalhub.db
.\.venv\Scripts\evalhub.exe compare --baseline <baseline-id> --candidate <candidate-id> --db C:\data\evalhub.db
```

### Troubleshoot

- **Unknown adapter**: use `generic` or register the adapter before calling `validate_dataset`.
- **Duplicate case ID**: make every `case_id` unique within the dataset.
- **Missing input**: live and saved records must have a non-null `input`.
- **Missing expected output**: add `expected_output` for `exact_match` or `answer_correctness`.
- **Missing tool trace**: add `tool_calls` or set `tool_coverage.required` to `false`.
- **Missing retrieval context**: add `retrieval_context` or set `faithfulness.required` to `false`.
- **Gemini credential error**: install the Gemini extra and set `GOOGLE_API_KEY`, or configure Vertex AI environment variables and Application Default Credentials.
- **Incompatible comparison**: baseline and candidate requirements differ, or the dataset/metric configuration changed.
- **Dashboard cannot find a run**: use the same `EVALHUB_DB` value as the CLI database.
