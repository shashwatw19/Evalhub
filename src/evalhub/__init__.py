"""
EvalHub: Standalone evaluation harness for live agents and saved runs using DeepEval.

Purpose:
    Assess agent outputs, tool calls, and citations; compare runs; inspect results.

Key Design Decisions:
    - Offline deterministic evaluation by default (telemetry disabled).
    - Optional agent connectors and Gemini judging are explicitly configured.
"""

__version__ = "0.1.0"
