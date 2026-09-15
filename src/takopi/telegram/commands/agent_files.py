from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import cast

from ...model import CompletedEvent, ResumeToken, StartedEvent, TakopiEvent
from ...runner import Runner, RunnerTurnControl

_FILE_DIRECTIVE_RE = re.compile(
    r"^[ \t]*<takopi-file>(?P<path>[^<>\r\n]+)</takopi-file>[ \t]*$"
)
_MAX_AGENT_FILES = 5
_FILE_DELIVERY_INSTRUCTIONS = f"""[takopi file delivery]
If the user explicitly asks you to send an existing file back to Telegram, add
one line per requested file at the end of your final answer using this exact
format:
<takopi-file>relative/path</takopi-file>
Use paths relative to the current working directory. Do not use absolute paths.
Attach no more than {_MAX_AGENT_FILES} files in one response.
Do not add these lines unless the user asked to receive the file."""


def _extract_file_directives(answer: str) -> tuple[str, list[str]]:
    lines = answer.splitlines()
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    start = end
    reversed_paths: list[str] = []
    while start > 0:
        match = _FILE_DIRECTIVE_RE.fullmatch(lines[start - 1])
        if match is None:
            break
        path = match.group("path").strip()
        if path:
            reversed_paths.append(path)
        start -= 1
    if not reversed_paths:
        return answer, []
    cleaned = "\n".join(lines[:start]).strip()
    paths = list(dict.fromkeys(reversed(reversed_paths)))[:_MAX_AGENT_FILES]
    return cleaned or "Attached the requested file.", paths


@dataclass(slots=True)
class _AgentFileTurnControl:
    control: RunnerTurnControl
    on_steer: Callable[[], None]

    async def steer(self, text: str) -> None:
        self.on_steer()
        await self.control.steer(text)

    async def interrupt(self) -> bool:
        return await self.control.interrupt()


@dataclass(slots=True)
class _AgentFileRunner:
    runner: Runner
    paths: list[str] = field(default_factory=list)
    steered: bool = False

    @property
    def engine(self) -> str:
        return self.runner.engine

    def is_resume_line(self, line: str) -> bool:
        return self.runner.is_resume_line(line)

    def format_resume(self, token: ResumeToken) -> str:
        return self.runner.format_resume(token)

    def extract_resume(self, text: str | None) -> ResumeToken | None:
        return self.runner.extract_resume(text)

    async def run(
        self, prompt: str, resume: ResumeToken | None
    ) -> AsyncIterator[TakopiEvent]:
        self.paths.clear()
        self.steered = False
        prompt_with_instructions = f"{prompt}\n\n{_FILE_DELIVERY_INSTRUCTIONS}"
        async for event in self.runner.run(prompt_with_instructions, resume):
            if isinstance(event, StartedEvent) and event.meta is not None:
                control = event.meta.get("control")
                if control is not None:
                    meta = dict(event.meta)
                    meta["control"] = _AgentFileTurnControl(
                        cast(RunnerTurnControl, control), self._mark_steered
                    )
                    event = replace(event, meta=meta)
            if isinstance(event, CompletedEvent) and event.ok:
                answer, paths = _extract_file_directives(event.answer)
                if not self.steered:
                    self.paths.extend(path for path in paths if path not in self.paths)
                event = replace(event, answer=answer)
            yield event

    def _mark_steered(self) -> None:
        self.steered = True
        self.paths.clear()
