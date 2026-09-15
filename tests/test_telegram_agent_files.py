from pathlib import Path
from typing import Literal, cast

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner import RunnerTurnControl
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.mock import Emit, Return, ScriptRunner
from takopi.settings import TelegramFilesSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.commands.agent_files import _MAX_AGENT_FILES, _AgentFileRunner
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import DEFAULT_ENGINE_ID, FakeBot, FakeTransport


async def _events(runner: _AgentFileRunner, prompt: str) -> list:
    return [event async for event in runner.run(prompt, None)]


@pytest.mark.anyio
async def test_agent_file_runner_requests_files_and_hides_directives() -> None:
    inner = ScriptRunner(
        [
            Return(
                answer=(
                    "Prepared both files.\n\n"
                    "<takopi-file>reports/one.md</takopi-file>\n"
                    "<takopi-file>reports/two.csv</takopi-file>"
                )
            )
        ],
        engine=DEFAULT_ENGINE_ID,
    )
    runner = _AgentFileRunner(inner)

    events = await _events(runner, "send the reports")

    completed = next(event for event in events if isinstance(event, CompletedEvent))
    assert completed.answer == "Prepared both files."
    assert runner.paths == ["reports/one.md", "reports/two.csv"]
    assert "<takopi-file>relative/path</takopi-file>" in inner.calls[0][0]


@pytest.mark.anyio
async def test_agent_file_runner_limits_files_per_response() -> None:
    paths = [f"reports/{index}.txt" for index in range(_MAX_AGENT_FILES + 1)]
    answer = "Done.\n" + "\n".join(
        f"<takopi-file>{path}</takopi-file>" for path in paths
    )
    inner = ScriptRunner([Return(answer=answer)], engine=DEFAULT_ENGINE_ID)
    runner = _AgentFileRunner(inner)

    events = await _events(runner, "send the reports")

    completed = next(event for event in events if isinstance(event, CompletedEvent))
    assert completed.answer == "Done."
    assert runner.paths == paths[:_MAX_AGENT_FILES]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        "Use <takopi-file>report.md</takopi-file> in a final line.",
        "Example:\n<takopi-file>report.md</takopi-file>\nThat is the format.",
        "```xml\n<takopi-file>report.md</takopi-file>\n```",
    ],
)
async def test_agent_file_runner_ignores_non_trailing_directives(answer: str) -> None:
    inner = ScriptRunner([Return(answer=answer)], engine=DEFAULT_ENGINE_ID)
    runner = _AgentFileRunner(inner)

    events = await _events(runner, "explain the protocol")

    completed = next(event for event in events if isinstance(event, CompletedEvent))
    assert completed.answer == answer
    assert runner.paths == []


@pytest.mark.anyio
async def test_agent_file_runner_disables_files_after_steer() -> None:
    class _Control:
        def __init__(self) -> None:
            self.steered: list[str] = []

        async def steer(self, text: str) -> None:
            self.steered.append(text)

        async def interrupt(self) -> bool:
            return True

    control = _Control()
    token = ResumeToken(engine=DEFAULT_ENGINE_ID, value="sid")
    inner = ScriptRunner(
        [
            Emit(
                StartedEvent(
                    engine=DEFAULT_ENGINE_ID,
                    resume=token,
                    meta={"control": control},
                )
            ),
            Return(answer="Done.\n<takopi-file>report.md</takopi-file>"),
        ],
        engine=DEFAULT_ENGINE_ID,
        emit_session_start=False,
    )
    runner = _AgentFileRunner(inner)
    events = []

    async for event in runner.run("send the report", None):
        events.append(event)
        if isinstance(event, StartedEvent):
            assert event.meta is not None
            wrapped = cast(RunnerTurnControl, event.meta["control"])
            await wrapped.steer("send another file")

    completed = next(event for event in events if isinstance(event, CompletedEvent))
    assert completed.answer == "Done."
    assert control.steered == ["send another file"]
    assert runner.steered is True
    assert runner.paths == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mode", "agent_path", "expected_document"),
    [
        ("command", None, False),
        ("prompt", "report.md", True),
        ("prompt", "/other secret.txt", False),
    ],
)
async def test_run_main_loop_respects_auto_get_mode(
    tmp_path: Path,
    mode: Literal["command", "prompt"],
    agent_path: str | None,
    expected_document: bool,
) -> None:
    current = tmp_path / "current"
    current.mkdir()
    (current / "report.md").write_bytes(b"report")
    other = tmp_path / "other"
    other.mkdir()
    (other / "secret.txt").write_bytes(b"secret")
    answer = "Done."
    if agent_path is not None:
        answer += f"\n<takopi-file>{agent_path}</takopi-file>"
    runner = ScriptRunner(
        [Return(answer=answer)],
        engine=DEFAULT_ENGINE_ID,
    )
    runtime = TransportRuntime(
        router=AutoRouter(
            entries=[RunnerEntry(engine=runner.engine, runner=runner)],
            default_engine=runner.engine,
        ),
        projects=ProjectsConfig(
            projects={
                "proj": ProjectConfig(
                    alias="proj",
                    path=current,
                    worktrees_dir=Path(".worktrees"),
                ),
                "other": ProjectConfig(
                    alias="other",
                    path=other,
                    worktrees_dir=Path(".worktrees"),
                ),
            },
            default_project="proj",
        ),
    )
    transport = FakeTransport()
    bot = FakeBot()
    cfg = TelegramBridgeConfig(
        bot=bot,
        runtime=runtime,
        chat_id=123,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=transport,
            presenter=MarkdownPresenter(),
            final_notify=True,
        ),
        forward_coalesce_s=0,
        media_group_debounce_s=0,
        files=TelegramFilesSettings(enabled=True, auto_get_mode=mode),
    )

    async def poller(_cfg: TelegramBridgeConfig):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=123,
            message_id=12,
            text="send the report",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=1,
            chat_type="private",
        )

    await run_main_loop(cfg, poller)

    assert bool(bot.document_calls) is expected_document
    prompt = runner.calls[0][0]
    assert ("<takopi-file>relative/path</takopi-file>" in prompt) is (mode == "prompt")
    response_texts = [call["message"].text for call in transport.send_calls]
    assert any("Done." in text for text in response_texts)
    assert all("takopi-file" not in text for text in response_texts)
    if agent_path == "/other secret.txt":
        assert any("invalid download path" in text for text in response_texts)
