"""
test_runner.py

Purpose:
    Unit tests for EvaluationRunner orchestration, fingerprint calculation, and dataset validation.
"""

from pathlib import Path
import pytest
from evalhub.runner import EvaluationRunner, compute_comparison_fingerprint
from evalhub.storage import StorageEngine
from evalhub.models import EvaluationCase, EvaluationConfig, MetricConfig, MetricStatus


def test_load_and_validate_dataset(tmp_path):
    dataset_file = tmp_path / "test.jsonl"
    dataset_file.write_text(
        '{"case_id":"ALT-1","input":"prompt","actual_output":"response"}\n',
        encoding="utf-8"
    )
    runner = EvaluationRunner(StorageEngine(db_path=str(tmp_path / "test.db")))
    cases = runner.validate_dataset(str(dataset_file))
    assert len(cases) == 1
    assert cases[0].case_id == "ALT-1"


def test_comparison_fingerprint_deterministic():
    cases = [
        EvaluationCase(case_id="C-1", input="in1", actual_output="out1", expected_output="expected1"),
        EvaluationCase(case_id="C-2", input="in2", actual_output="out2", expected_output="expected2"),
    ]
    config = EvaluationConfig(
        adapter="generic",
        metrics=[MetricConfig(name="exact_match", threshold=1.0)]
    )

    fp1 = compute_comparison_fingerprint(cases, config, "4.3.4", {"exact_match": "3.0.0"})
    
    # Candidate cases with DIFFERENT actual outputs must produce the EXACT SAME comparison fingerprint!
    candidate_cases = [
        EvaluationCase(case_id="C-1", input="in1", actual_output="DIFFERENT_OUTPUT_1", expected_output="expected1"),
        EvaluationCase(case_id="C-2", input="in2", actual_output="DIFFERENT_OUTPUT_2", expected_output="expected2"),
    ]
    fp2 = compute_comparison_fingerprint(candidate_cases, config, "4.3.4", {"exact_match": "3.0.0"})

    assert fp1 == fp2


def test_run_uses_deepeval_and_preserves_metric_diagnostics(tmp_path):
    dataset_file = tmp_path / "cases.jsonl"
    dataset_file.write_text(
        '{"case_id":"C-1","input":"question","actual_output":'
        '{"answer":"same","evidence":[{"source":"docs","record_id":"MISSING"}]},'
        '"expected_output":{"answer":"same","evidence":['
        '{"source":"docs","record_id":"MISSING"}]},'
        '"expected_required_tools":["search"],"tool_calls":[],"evidence":['
        '{"source":"docs","record_id":"DOC-1"}]}\n'
        '{"case_id":"C-2","input":"question","actual_output":'
        '{"answer":"same","evidence":[{"source":"docs","record_id":"DOC-2"}]},'
        '"expected_output":{"answer":"same","evidence":['
        '{"source":"docs","record_id":"DOC-2"}]},'
        '"expected_required_tools":["search"],"tool_calls":[{"name":"search"}],'
        '"evidence":[{"source":"docs","record_id":"DOC-2"}]}\n',
        encoding="utf-8",
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"adapter":"generic","metrics":['
        '{"name":"exact_match","threshold":1.0,"required":true},'
        '{"name":"evidence_reference_validity","threshold":1.0,"required":true},'
        '{"name":"tool_coverage","threshold":1.0,"required":true}]}',
        encoding="utf-8",
    )
    runner = EvaluationRunner(StorageEngine(db_path=str(tmp_path / "test.db")))

    summary, case_results = runner.run_evaluation(
        str(dataset_file), str(config_file), label="deepeval-engine"
    )

    assert summary.total_cases == 2
    assert summary.failed_cases == 1
    assert summary.passed_cases == 1
    assert summary.safety_failures == 0
    assert [result.name for result in case_results[0].metric_results] == [
        "exact_match",
        "evidence_reference_validity",
        "tool_coverage",
    ]
    assert case_results[0].metric_results[1].status == MetricStatus.FAILED
    assert case_results[0].metric_results[2].status == MetricStatus.FAILED
    assert case_results[1].overall_status.value == "passed"


def test_deepeval_execution_preserves_prerequisite_errors(tmp_path):
    dataset_file = tmp_path / "cases.jsonl"
    dataset_file.write_text(
        '{"case_id":"C-1","input":"question","actual_output":"response"}\n',
        encoding="utf-8",
    )
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"adapter":"generic","metrics":['
        '{"name":"exact_match","threshold":1.0,"required":true}]}',
        encoding="utf-8",
    )
    runner = EvaluationRunner(StorageEngine(db_path=str(tmp_path / "test.db")))

    summary, case_results = runner.run_evaluation(
        str(dataset_file), str(config_file), label="missing-ground-truth"
    )

    assert summary.error_cases == 1
    assert case_results[0].metric_results[0].status == MetricStatus.ERROR


def test_gemini_judge_is_evaluated_through_deepeval(tmp_path, monkeypatch):
    class FakeJudge:
        reason = "The answer is relevant."

        def measure(self, case):
            assert case.input == "question"
            assert case.actual_output == "answer"
            return 0.8

    calls = []

    def make_judge(name, model, threshold):
        calls.append((name, model, threshold))
        return FakeJudge()

    monkeypatch.setattr("evalhub.metrics._make_gemini_metric", make_judge)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"case_id":"c","input":"question","actual_output":"answer"}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        '{"judge_enabled":true,"judge_model":"gemini-test",'
        '"metrics":[{"name":"answer_relevancy","threshold":0.7}]}',
        encoding="utf-8",
    )
    runner = EvaluationRunner(StorageEngine(str(tmp_path / "test.db")))
    summary, results = runner.run_evaluation(str(dataset), str(config))
    assert calls == [("answer_relevancy", "gemini-test", 0.7)]
    assert summary.passed_cases == 1
    assert results[0].metric_results[0].score == 0.8
    assert results[0].metric_results[0].explanation == "The answer is relevant."


def test_native_deepeval_metrics_with_stubbed_gemini(tmp_path, monkeypatch):
    from deepeval.models import DeepEvalBaseLLM

    class StubGemini(DeepEvalBaseLLM):
        def __init__(self, model, api_key, use_vertexai):
            super().__init__(model)

        def load_model(self):
            return self

        def get_model_name(self):
            return "stub-gemini"

        def supports_log_probs(self):
            return False

        def generate(self, prompt, schema=None):
            if schema.__name__ == "Statements":
                return schema(statements=["answer"])
            if schema.__name__ == "Verdicts":
                return schema(verdicts=[{"verdict": "yes"}])
            if schema.__name__ == "AnswerRelevancyScoreReason":
                return schema(reason="Relevant answer")
            if schema.__name__ == "Truths":
                return schema(truths=["answer"])
            if schema.__name__ == "Claims":
                return schema(claims=["answer"])
            if schema.__name__ == "FaithfulnessScoreReason":
                return schema(reason="Grounded answer")
            if schema.__name__ == "Steps":
                return schema(steps=["Compare the answers"])
            if schema.__name__ == "ReasonScore":
                return schema(score=8, reason="Correct answer")
            raise AssertionError(f"Unexpected schema {schema}")

        async def a_generate(self, prompt, schema=None):
            return self.generate(prompt, schema)

    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("deepeval.models.GeminiModel", StubGemini)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"case_id":"c","input":"question","actual_output":"answer",'
        '"expected_output":"answer","retrieval_context":["answer"]}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        '{"judge_enabled":true,"judge_model":"gemini-test",'
        '"metrics":[{"name":"answer_relevancy","threshold":0.7},'
        '{"name":"answer_correctness","threshold":0.7},'
        '{"name":"faithfulness","threshold":0.7}]}',
        encoding="utf-8",
    )
    summary, results = EvaluationRunner(StorageEngine(str(tmp_path / "test.db"))).run_evaluation(
        str(dataset), str(config)
    )
    assert summary.passed_cases == 1
    assert results[0].metric_results[0].score == 1.0
    assert results[0].metric_results[0].explanation == "Relevant answer"
    assert results[0].metric_results[1].score == 0.8
    assert results[0].metric_results[2].score == 1.0
