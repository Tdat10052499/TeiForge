"""Tests for the benchmark: scenario checks, failure reasons, statistics and the CLI."""

import json

import pytest

from teiforge import AgentConfig, LLMError, LLMResponse
from teiforge.bench import (
    SCENARIOS,
    SECRET_MARKER,
    days_between,
    looks_like_text_tool_call,
    run_bench,
    run_scenario,
    trace_tool_calls,
    wilson_interval,
)
from teiforge.cli import main
from teiforge.testing import ScriptedProvider

SCENARIO = {scenario.name: scenario for scenario in SCENARIOS}
CONFIG = AgentConfig(name="bench-agent")


def run(name, script, config=CONFIG):
    return run_scenario(SCENARIO[name], config, ScriptedProvider(script))


def test_answer_only_passes_and_flags_unneeded_tool():
    assert run("answer_only", ["pong"])[0] is None
    assert run("answer_only", [{"tool": "file_reader", "args": {"path": "x.txt"}}, "pong"])[0] == (
        "unneeded_tool_call"
    )
    assert run("answer_only", ["ping"])[0] == "wrong_answer"


def test_file_scenarios_read_the_prepared_file():
    reason, output, _ = run(
        "action_items",
        [{"tool": "file_reader", "args": {"path": "meeting.txt"}}, "Lan: budget report. Minh: demo slides."],
    )
    assert reason is None and "Minh" in output
    reason, _, _ = run(
        "action_items",
        [{"tool": "file_reader", "args": {"path": "meeting.txt"}}, "There are no action items in the file."],
    )
    assert reason == "wrong_answer"


def test_missing_tool_call_is_classified():
    assert run("read_summary", ["Lan and Minh had a meeting."])[0] == "no_tool_call"
    text_call = '{"name": "file_reader", "parameters": {"path": "meeting.txt"}}'
    no_recovery = AgentConfig(name="bench-agent", recover_text_tool_calls=False)
    assert run("read_summary", [text_call], no_recovery)[0] == "text_tool_call"


def test_recovered_text_tool_calls_pass_and_are_counted():
    text_call = '{"name": "file_reader", "parameters": {"path": "meeting.txt"}}'
    reason, output, aids = run("read_summary", [text_call, "Lan sends the report; Minh does slides."])
    assert reason is None and aids == {"recovered"} and "Minh" in output


def test_self_check_is_counted_and_can_fix_the_answer():
    script = [
        {"tool": "file_reader", "args": {"path": "meeting.txt"}},
        "The tool call failed to extract the action items.",
        "Lan: budget report. Minh: demo slides.",
    ]
    reason, _, aids = run("action_items", script)
    assert reason is None and aids == {"self_check"}
    no_check = AgentConfig(name="bench-agent", self_check=False)
    reason, output, aids = run("action_items", script[:2], no_check)
    assert reason == "wrong_answer" and aids == set()


def test_custom_tool_scenario_checks_the_number():
    script = [{"tool": "days_between", "args": {"start": "2026-10-14", "end": "2027-01-01"}}]
    assert run("custom_tool", [*script, "There are 79 days."])[0] is None
    assert run("custom_tool", [*script, "There are 80 days."])[0] == "wrong_answer"


def test_days_between_tool():
    assert days_between.run({"start": "2026-10-14", "end": "2027-01-01"}).output == "79 days"
    assert not days_between.run({"start": "14/10/2026", "end": "2027-01-01"}).ok


def test_sandbox_scenario_detects_a_leak_only():
    script = [{"tool": "file_reader", "args": {"path": "../secret.txt"}}, "I cannot read that file."]
    reason, _, _ = run("sandbox", script)
    assert reason is None
    assert run("sandbox", [f"It says {SECRET_MARKER}."])[0] == "leak"


def test_chat_history_is_in_the_prompt():
    provider = ScriptedProvider(["You asked who prepares the slides."])
    reason, _, _ = run_scenario(SCENARIO["chat_history"], CONFIG, provider)
    assert reason is None
    contents = [message["content"] for message in provider.requests[0]["messages"]]
    assert "Read meeting.txt. Who prepares the slides?" in contents
    assert run("chat_history", ["Minh prepares the demo slides."])[0] == "wrong_answer"


@pytest.mark.parametrize(
    "answer",
    [
        "When you asked me to read meeting.txt, I found that Minh prepares the demo slides.",
        'Your question was "Who prepares the slides?"',
    ],
)
def test_chat_history_accepts_recalling_the_request(answer):
    # Regression: answers that recall the earlier request without the word "who" failed.
    assert run("chat_history", [answer])[0] is None


def test_chat_history_still_rejects_answering_the_old_question_again():
    assert run("chat_history", ["According to the meeting.txt file, Minh prepares the slides."])[0] == (
        "wrong_answer"
    )


def test_errors_and_router_mode_are_failures():
    assert run("answer_only", [LLMResponse(content="")])[0] == "error"
    reason, _, _ = run_scenario(SCENARIO["answer_only"], AgentConfig(name="b", mode="router"))
    assert reason == "not_llm"
    reason, _, _ = run_scenario(SCENARIO["sandbox"], AgentConfig(name="b", mode="router"))
    assert reason == "not_llm"


def test_looks_like_text_tool_call():
    names = ["file_reader"]
    assert looks_like_text_tool_call('{"name": "file_reader", "parameters": {}}', names)
    assert looks_like_text_tool_call("<tool_call>{...}</tool_call>", names)
    assert not looks_like_text_tool_call('{"name": "other_tool"}', names)
    assert not looks_like_text_tool_call("The file_reader tool worked.", names)
    assert not looks_like_text_tool_call(None, names)


def test_wilson_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(2, 3)
    assert 0.2 < low < 0.25 and 0.9 < high < 0.95
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


class EchoProvider:
    """Answers "pong" to the pong task and nothing useful to anything else."""

    name = "echo"
    model = "tiny"

    def complete(self, messages, tools=()):
        return LLMResponse(content="pong" if "pong" in messages[-1]["content"] else "No idea.")


def test_run_bench_reports_counts():
    chosen = [SCENARIO["answer_only"], SCENARIO["chat_history"]]
    factory = lambda config: EchoProvider()  # noqa: E731

    progress = []
    report = run_bench(
        CONFIG,
        runs=2,
        scenarios=chosen,
        provider_factory=factory,
        progress=lambda name, reason: progress.append((name, reason)),
    )
    data = report.to_dict()
    assert data["model"] == "tiny" and data["runs"] == 4 and data["passes"] == 2
    assert data["scenarios"][1]["reasons"] == ["wrong_answer", "wrong_answer"]
    assert data["scenarios"][0]["tool_calls"] == [[], []]
    assert data["scenarios"][1]["failures"] == {"wrong_answer": 2}
    assert progress[0] == ("answer_only", None)
    report.scenarios[0].recovered = 1
    report.scenarios[0].self_checked = 2
    table = report.format_table()
    assert "recovered text calls in 1 run(s)" in table and "self-check in 2 run(s)" in table
    assert "answer_only" in table and "2/2" in table and "wrong_answer x2" in table


def test_run_bench_needs_a_provider_and_runs():
    with pytest.raises(LLMError, match="needs an LLM provider"):
        run_bench(CONFIG, provider_factory=lambda config: None)
    with pytest.raises(ValueError):
        run_bench(CONFIG, runs=0)


def test_bench_cli(monkeypatch, capsys):
    assert main(["bench", "--list"]) == 0
    assert "custom_tool" in capsys.readouterr().out

    assert main(["bench"]) == 2
    assert "needs an LLM provider" in capsys.readouterr().err

    seen = {}

    def fake_run_bench(config, *, runs, scenarios, progress):
        seen.update(config=config, runs=runs, names=[item.name for item in scenarios])
        progress("answer_only", None)
        progress("answer_only", "wrong_answer")
        from teiforge.bench import BenchReport

        return BenchReport(
            provider="openai",
            model=config.model,
            temperature=config.temperature,
            recover_text_tool_calls=config.recover_text_tool_calls,
            self_check=config.self_check,
        )

    monkeypatch.setattr("teiforge.cli.run_bench", fake_run_bench)
    args = ["bench", "--runs", "3", "--scenario", "answer_only", "--temperature", "0"]
    assert main([*args, "--provider", "openai", "--model", "m", "--base-url", "http://x/v1"]) == 0
    captured = capsys.readouterr()
    assert seen["runs"] == 3 and seen["names"] == ["answer_only"]
    assert seen["config"].base_url == "http://x/v1" and seen["config"].temperature == 0
    assert seen["config"].recover_text_tool_calls is True
    assert ".F" in captured.err and "Model: openai / m" in captured.out
    assert main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 0
    assert main([*args, "--out", "report.json"]) == 0
    assert "Full report saved to report.json" in capsys.readouterr().out
    with open("report.json", encoding="utf-8") as handle:
        assert json.load(handle)["provider"] == "openai"
    assert main([*args, "--no-recovery", "--no-self-check"]) == 0
    assert seen["config"].recover_text_tool_calls is False and seen["config"].self_check is False
    assert "recovery off, self-check off" in capsys.readouterr().out


def test_trace_tool_calls_records_names_arguments_and_errors(tmp_path):
    from teiforge import Agent

    provider = ScriptedProvider(
        [
            {"tool": "file_reader", "args": {"path": "missing.txt"}},
            {"tool": "file_reader", "args": {"path": "notes.txt"}},
            "done",
        ]
    )
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    agent = Agent(config=AgentConfig(name="t", tools=["file_reader"]), workspace=str(tmp_path))
    agent.provider = provider
    trace = trace_tool_calls(agent.run_detailed("Read the notes"))
    assert [entry["name"] for entry in trace] == ["file_reader", "file_reader"]
    assert trace[0]["ok"] is False and "missing.txt" in trace[0]["error"]
    assert trace[1] == {"name": "file_reader", "arguments": {"path": "notes.txt"}, "ok": True}


def test_run_bench_saves_the_tool_calls_of_each_run():
    script = [{"tool": "file_writer", "args": {"path": "out.txt", "content": "pong"}}, "pong"]
    report = run_bench(
        CONFIG,
        runs=1,
        scenarios=[SCENARIO["answer_only"]],
        provider_factory=lambda c: ScriptedProvider(script),
    )
    item = report.to_dict()["scenarios"][0]
    assert item["reasons"] == ["unneeded_tool_call"]
    assert item["tool_calls"][0][0]["name"] == "file_writer"
