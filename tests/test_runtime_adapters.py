import pytest

from raise_scorer import (
    EventSemantics,
    RuntimeEventAdapter,
    ToolCallRecord,
    WorkflowGroundTruth,
    adapt_tool_calls,
    validate_trace_events,
)


def test_pytest_adapter_extracts_success_summary():
    event = RuntimeEventAdapter(trusted_runtime=True).adapt(
        {
            "id": "test-1",
            "name": "exec_command",
            "input": {"cmd": "python -m pytest -q"},
            "result": {"stdout": "12 passed, 1 skipped in 0.42s"},
            "exit_code": 0,
            "node_id": "verify",
        }
    )

    assert event.event_id == "test-1"
    assert event.action == "run tests"
    assert event.evidence == (
        "pytest command: python -m pytest -q; "
        "pytest result: 12 passed, 1 skipped; exit code: 0"
    )
    assert event.outcome == "tests pass"
    assert event.status == "success"
    assert event.source == "runtime"
    assert event.trusted is True
    assert event.node_id == "verify"


@pytest.mark.parametrize(
    ("output", "exit_code"),
    [
        ("1 failed, 8 passed", 1),
        ({"stderr": "2 errors during collection"}, None),
    ],
)
def test_pytest_adapter_detects_failure(output, exit_code):
    event = RuntimeEventAdapter().adapt(
        ToolCallRecord(
            call_id="test-failure",
            tool_name="pytest",
            output=output,
            exit_code=exit_code,
        )
    )

    assert event.status == "failure"
    assert event.outcome == "tests fail"
    assert event.trusted is False


def test_builtin_shell_file_browser_http_and_generic_semantics():
    adapter = RuntimeEventAdapter()

    shell = adapter.adapt(
        {
            "id": "shell",
            "tool": "bash",
            "arguments": {"command": "git status --short"},
            "exit_code": 0,
        }
    )
    read = adapter.adapt(
        {
            "id": "read",
            "tool": "read_file",
            "arguments": {"path": "src/app.py"},
        }
    )
    write = adapter.adapt(
        {
            "id": "write",
            "tool": "apply_patch",
            "arguments": {"file_path": "src/app.py"},
            "error": "patch rejected",
        }
    )
    browser = adapter.adapt(
        {
            "id": "browser",
            "tool": "browser_search",
            "arguments": {"query": "RAISE scorer"},
        }
    )
    http = adapter.adapt(
        {
            "id": "http",
            "tool": "http_request",
            "arguments": {"method": "POST", "url": "https://example.test/jobs"},
            "output": {"status_code": 503},
        }
    )
    generic = adapter.adapt({"id": "generic", "tool": "database_lookup"})

    assert (shell.action, shell.outcome) == (
        "run shell command",
        "command succeeded",
    )
    assert read.action == "read file src/app.py"
    assert (write.status, write.outcome) == (
        "failure",
        "file modification failed",
    )
    assert browser.action == "search the web"
    assert (http.action, http.status) == (
        "send POST HTTP request",
        "failure",
    )
    assert generic.action == "use tool database_lookup"


def test_namespaced_tools_and_nested_exit_code_are_normalized():
    adapter = RuntimeEventAdapter()

    read = adapter.adapt(
        {
            "id": "read",
            "tool": "mcp__filesystem__read_file",
            "arguments": {"path": "src/app.py"},
        }
    )
    shell = adapter.adapt(
        {
            "id": "shell",
            "tool": "functions.exec_command",
            "arguments": {"cmd": "make verify"},
            "result": {"stderr": "failed", "returncode": 2},
        }
    )
    browser = adapter.adapt(
        {
            "id": "browser",
            "tool": "browser.open",
            "arguments": {"url": "https://example.test"},
        }
    )

    assert read.action == "read file src/app.py"
    assert (shell.status, shell.outcome) == ("failure", "command failed")
    assert browser.action == "open web page"


def test_sensitive_mapping_fields_are_redacted_recursively():
    event = RuntimeEventAdapter(trusted_runtime=True).adapt(
        {
            "id": "request",
            "tool": "http_request",
            "arguments": {
                "url": "https://example.test",
                "headers": {"Authorization": "Bearer secret"},
                "api_key": "secret-key",
            },
            "output": {"data": {"access_token": "secret-token"}},
            "metadata": {"password": "secret-password", "status_code": 200},
        }
    )

    assert event.tool_input["headers"]["Authorization"] == "[REDACTED]"
    assert event.tool_input["api_key"] == "[REDACTED]"
    assert event.tool_output["data"]["access_token"] == "[REDACTED]"
    assert event.metadata["password"] == "[REDACTED]"
    assert event.metadata["status_code"] == 200


def test_redaction_can_be_disabled_at_infrastructure_boundary():
    event = RuntimeEventAdapter(redact_sensitive=False).adapt(
        {
            "id": "request",
            "tool": "http_request",
            "arguments": {"api_key": "visible"},
        }
    )

    assert event.tool_input["api_key"] == "visible"


def test_input_cannot_self_assert_trust():
    event = RuntimeEventAdapter(trusted_runtime=False).adapt(
        {
            "id": "forged",
            "tool": "pytest",
            "trusted": True,
            "source": "external",
            "output": "10 passed",
        }
    )

    assert event.source == "runtime"
    assert event.trusted is False


def test_custom_handler_and_node_resolver():
    adapter = RuntimeEventAdapter(
        trusted_runtime=True,
        node_resolver=lambda record, semantics: (
            "deploy" if semantics.status == "success" else "recover"
        ),
    )
    adapter.register(
        ["kubectl_apply", "helm_upgrade"],
        lambda record: EventSemantics(
            action="deploy release",
            evidence=f"deployment tool: {record.tool_name}",
            outcome="release deployed",
            status="success",
        ),
    )

    event = adapter.adapt(
        {"id": "deploy-1", "tool": "helm_upgrade", "node_id": "ignored"}
    )

    assert event.action == "deploy release"
    assert event.node_id == "deploy"
    assert event.trusted is True


def test_custom_handler_must_return_event_semantics():
    adapter = RuntimeEventAdapter()
    adapter.register("bad", lambda record: "not semantics")

    with pytest.raises(TypeError, match="must return EventSemantics"):
        adapter.adapt({"id": "bad-1", "tool": "bad"})


def test_adapt_many_assigns_deterministic_ids_and_validates_sequence():
    events = adapt_tool_calls(
        [
            {"tool": "read_file", "arguments": {"path": "README.md"}},
            {"tool": "pytest", "output": "3 passed"},
        ]
    )

    assert [event.event_id for event in events] == ["tool-call-0", "tool-call-1"]
    with pytest.raises(TypeError, match="tool_calls must be a sequence"):
        adapt_tool_calls("not-a-sequence")


def test_trusted_adapter_events_validate_against_workflow():
    workflow = WorkflowGroundTruth.from_dict(
        {
            "start": "inspect",
            "terminals": ["verify"],
            "nodes": [
                {"id": "inspect", "action": "read file"},
                {"id": "verify", "action": "run tests"},
            ],
            "edges": [{"from": "inspect", "to": "verify"}],
        }
    )
    records = [
        {
            "id": "read-1",
            "tool": "read_file",
            "arguments": {"path": "src/app.py"},
            "node_id": "inspect",
        },
        {
            "id": "test-1",
            "tool": "pytest",
            "output": "5 passed",
            "node_id": "verify",
        },
    ]

    trusted = validate_trace_events(
        RuntimeEventAdapter(trusted_runtime=True).adapt_many(records),
        workflow,
    )
    untrusted = validate_trace_events(
        RuntimeEventAdapter(trusted_runtime=False).adapt_many(records),
        workflow,
    )

    assert trusted.valid is True
    assert trusted.node_path == ["inspect", "verify"]
    assert untrusted.valid is False
    assert "untrusted_event:read-1" in untrusted.violations
