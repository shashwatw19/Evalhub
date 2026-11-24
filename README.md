# EvalHub

Standalone evaluation harness for live agents and saved runs using DeepEval.

## Features
- **DeepEval Evaluation Engine**: Runs deterministic metrics offline and optional Gemini-judged metrics through DeepEval's `evaluate()` API.
- **Generic Agent Adapter**: Accepts a documented JSONL case format; source-specific fields are preserved as metadata.
- **Live Agent Connectors**: Call a JSON HTTP endpoint or a trusted Python callable for each case.
- **SQLite Persistence**: Stores runs, normalized test cases, and metric results with schema versioning.
- **Baseline vs Candidate Comparison**: Compares runs based on stable case IDs and strict comparison fingerprints.
- **Standalone Dashboard**: Streamlit interface for run overviews and case detail inspections.

## Quickstart
```bash
# Validate dataset
evalhub validate --dataset examples/generic/cases.jsonl

# Run baseline evaluation
evalhub run --dataset examples/generic/cases.jsonl --config examples/generic/config.json --label baseline

# Compare baseline vs candidate
evalhub compare --baseline <baseline_run_id> --candidate <candidate_run_id>

# Launch standalone dashboard
streamlit run src/evalhub/dashboard.py
```

## Evaluate a live agent

Live datasets need a unique `case_id` and non-null `input`; include `expected_output`
when using exact match or answer correctness. Outputs and observed tool calls in a
live dataset are ignored and replaced with the agent's response.

```json
{"case_id":"greeting","input":"hello","expected_output":"HELLO"}
```

From the project directory, try the included local Python agent:

```powershell
.\.venv\Scripts\evalhub.exe validate --dataset examples\generic\agent_cases.jsonl --input-only
.\.venv\Scripts\evalhub.exe run --dataset examples\generic\agent_cases.jsonl --config examples\generic\python_agent_config.json --label baseline
```

### Python connector

Set `"agent"` in the evaluation config to `{"type":"python",
"callable":"my_package.agent:respond"}`. The module must be importable in the
EvalHub process. The synchronous function receives the case input as one
argument and returns a JSON-serializable output. It executes arbitrary local
code, so use only trusted modules; the function must implement its own timeout.
If it returns an envelope, configure `response_path` to extract the output.
Optional `tool_calls_path` and `retrieval_context_path` extract traces and
context from the same response. Paths are dot-separated object keys.

### HTTP connector

For an agent that accepts `{"prompt": ...}` and returns
`{"result":{"answer": ...,"tools": [...]}}`, use:

```json
{
  "agent": {
    "type": "http",
    "url": "https://agent.example.com/invoke",
    "input_field": "prompt",
    "response_path": "result.answer",
    "tool_calls_path": "result.tools",
    "auth_env": "AGENT_API_TOKEN",
    "timeout_seconds": 30,
    "retries": 0
  },
  "metrics": [{"name": "exact_match", "threshold": 1.0, "required": true}]
}
```

The connector sends a JSON POST and expects JSON. Omitting `response_path` uses
the whole response as the output. `auth_env` reads the token from an environment
variable; the default header is `Authorization: Bearer <token>`. `auth_header`
and `auth_prefix` can customize this. Do not put credentials in the config or
URL: the config is saved with each run. Use HTTPS for credentials. Redirects
are rejected. Calls run sequentially; retries default to zero because retrying
a POST may execute the agent more than once. HTTP timeout applies per attempt.
For other protocols or request shapes, use a Python wrapper callable.

Tool traces must be a list of objects with `name` and optional `arguments`,
`result`, `status`, and `call_id`. `retrieval_context` is a list of strings,
either supplied in the dataset or extracted from the response via
`retrieval_context_path`. Case-level connector errors are saved as errors rather
than successful outputs.

## Metrics and Gemini judge

Deterministic metrics are `exact_match`, `tool_coverage`,
`evidence_reference_validity`, and `explanation_quality` (a length heuristic,
enabled by `judge_enabled`). They remain offline unless an HTTP agent is used.
`exact_match` requires `expected_output`; tool coverage requires
`expected_required_tools` and an observed trace; evidence validation requires
the corresponding references and records.

For optional LLM judging, install the Gemini extra, set `GOOGLE_API_KEY` in the
environment, and explicitly configure a Gemini model supported by your installed
DeepEval version:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[gemini]'
$env:GOOGLE_API_KEY = '<your API key>'
```

```json
{
  "judge_enabled": true,
  "judge_model": "gemini-2.0-flash",
  "metrics": [
    {"name": "answer_relevancy", "threshold": 0.7},
    {"name": "answer_correctness", "threshold": 0.7},
    {"name": "faithfulness", "threshold": 0.7, "required": false}
  ]
}
```

`answer_correctness` needs `expected_output`; `faithfulness` needs
`retrieval_context`. Missing optional prerequisites are marked not applicable.
These metrics send case content to Google, may incur charges, and require
network access. Use only data approved for that transfer. Gemini is not
contacted for deterministic metrics or unless an LLM metric is configured and
`judge_enabled` is true.

## Dashboard and comparison

Run the dashboard locally with:

```powershell
.\.venv\Scripts\streamlit.exe run src\evalhub\dashboard.py --server.address 127.0.0.1
```

The **New evaluation** tab uploads JSONL and provides guided connector and metric
settings or an advanced JSON configuration. Uploads and configuration are
stored in `.evalhub`, and results in
`evalhub.db`. Treat both as sensitive. The **Runs** and **Compare** tabs inspect
results and compare runs. The dashboard has no authentication and can execute
Python code or make network requests: do not expose it to untrusted users.
Set `EVALHUB_DB` before starting it to use another database.

Use the same case IDs, inputs, expectations, reference context and metric
settings for baseline and candidate comparisons. Agent outputs, tool traces,
and agent-returned retrieval context may vary between runs. Configured connector
details and judge model are recorded with each run.

## Generic agent dataset format

Export one JSON object per line. `case_id`, `input`, and `actual_output` are the core fields; expected output, labels, tool traces, evidence, and metadata are optional and used by metrics that need them.

```json
{
  "case_id": "case-001",
  "input": "Question or structured input sent to the agent",
  "actual_output": "The agent's saved response",
  "expected_output": "The expected response",
  "tool_calls": [
    {"name": "search", "arguments": {"query": "example"}, "result": "result"}
  ],
  "expected_required_tools": ["search"],
  "tags": ["smoke-test"],
  "source_run_id": "agent-run-123"
}
```

For CLI use, transform agent-specific records to the canonical format. For a Python integration that reads a different native schema, register an adapter callable with `register_adapter(name, callable)` before calling `EvaluationRunner.validate_dataset(..., adapter_name=name)`. Unrecognized fields in canonical records are retained in the case's `metadata`.
