"""
conftest.py

Purpose:
    Pytest configuration and shared fixtures for EvalHub testing.
"""

import os
import pytest

# Ensure telemetry is opted out before any test imports DeepEval
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
