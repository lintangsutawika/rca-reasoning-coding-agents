import os
import shlex
from collections.abc import Sequence

from pydantic import Field

from openhands.tools.execute_bash import (
    BashExecutor,
    ExecuteBashAction,
)
from openhands.sdk.tool import (
    Tool,
    ToolExecutor,
    register_tool,
)

from openhands.sdk import (
    Action,
    ImageContent,
    Observation,
    TextContent,
    ToolDefinition,
    get_logger,
)

logger = get_logger(__name__)


# --- Action / Observation ---


class SubmitAction(Action):
    pattern: str = Field(description="Regex to search for")
    path: str = Field(
        default=".", description="Directory to search (absolute or relative)"
    )
    include: str | None = Field(
        default=None, description="Optional glob to filter files (e.g. '*.py')"
    )


class SubmitObservation(Observation):
    matches: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    count: int = 0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if not self.count:
            return [TextContent(text="No matches found.")]
        files_list = "\n".join(f"- {f}" for f in self.files[:20])
        sample = "\n".join(self.matches[:10])
        more = "\n..." if self.count > 10 else ""
        ret = (
            f"Found {self.count} matching lines.\n"
            f"Files:\n{files_list}\n"
            f"Sample:\n{sample}{more}"
        )
        return [TextContent(text=ret)]


# --- Executor ---


class SubmitExecutor(ToolExecutor[SubmitAction, SubmitObservation]):
    def __init__(self, bash: BashExecutor):
        self.bash = bash

    def __call__(self, action: SubmitAction) -> SubmitObservation:
        root = os.path.abspath(action.path)
        pat = shlex.quote(action.pattern)
        root_q = shlex.quote(root)

        cmd = "git add -A && git diff --cached"

        result = self.bash(ExecuteBashAction(command=cmd))

        matches: list[str] = []
        files: set[str] = set()

        # Submit returns exit code 1 when no matches; treat as empty
        if result.output.strip():
            for line in result.output.strip().splitlines():
                matches.append(line)
                # Expect "path:line:content" — take the file part before first ":"
                file_path = line.split(":", 1)[0]
                if file_path:
                    files.add(os.path.abspath(file_path))

        return SubmitObservation(
            matches=matches, files=sorted(files), count=len(matches)
        )


# Tool description
_Submit_DESCRIPTION = """Fast content search tool.
* Searches file contents using regular expressions
* Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
* Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
* Returns matching file paths sorted by modification time.
* Only the first 100 results are returned. Consider narrowing your search with stricter regex patterns or provide path parameter if you need more results.
* Use this tool when you need to find files containing specific patterns
* When you are doing an open ended search that may require multiple rounds of globbing and Submitping, use the Agent tool instead
"""  # noqa: E501

# Tools - demonstrating both simplified and advanced patterns
cwd = os.getcwd()


def _make_bash_and_Submit_tools(conv_state) -> list[ToolDefinition]:
    """Create execute_bash and custom Submit tools sharing one executor."""

    bash_executor = BashExecutor(working_dir=conv_state.workspace.working_dir)
    Submit_executor = SubmitExecutor(bash_executor)
    Submit_tool = ToolDefinition(
        name="Submit",
        description=_Submit_DESCRIPTION,
        action_type=SubmitAction,
        observation_type=SubmitObservation,
        executor=Submit_executor,
    )

    return [Submit_tool]


register_tool("SubmitToolSet", _make_bash_and_Submit_tools)

tools = [
    Tool(name="SubmitToolSet"),
]
