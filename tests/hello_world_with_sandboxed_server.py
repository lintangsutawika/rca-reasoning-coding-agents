import os
import time

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Conversation,
    RemoteConversation,
    get_logger,
)
# from openhands.sdk.workspace import DockerWorkspace, LocalWorkspace
from openhands.tools.preset.default import get_default_agent

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.spec import Tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

from rca.environments.singularity_workspace import SingularityWorkspace
from rca.environments.local_workspace import TestLocalWorkspace

logger = get_logger(__name__)


def main() -> None:
    # 1) Ensure we have LLM API key
    # api_key = os.getenv("LITELLM_API_KEY")
    # assert api_key is not None, "LITELLM_API_KEY environment variable is not set."

    api_key = os.getenv("CMU_KEY")
    api_url = os.getenv("CMU_URL")
    assert api_key is not None, "CMU_KEY environment variable is not set."
    assert api_url is not None, "CMU_URL environment variable is not set."
    llm = LLM(
        service_id="agent",
        # model="litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
        # base_url="https://llm-proxy.eval.all-hands.dev",
        model="litellm_proxy/neulab/claude-sonnet-4-20250514",
        base_url=api_url,
        api_key=SecretStr(api_key),
    )

    register_tool("BashTool", BashTool)
    register_tool("FileEditorTool", FileEditorTool)
    register_tool("TaskTrackerTool", TaskTrackerTool)



    # 2) Create a Docker-based remote workspace that will set up and manage
    #    the Docker container automatically
    # with DockerWorkspace(
    #     base_image="nikolaik/python-nodejs:python3.12-nodejs22",
    #     host_port=8010,
    #     # TODO: Change this to your platform if not linux/arm64
    #     platform="linux/arm64",
    #     forward_env=["LITELLM_API_KEY"],  # Forward API key to container
    # ) as workspace:
    # SingularityWorkspace
    with TestLocalWorkspace(
        working_dir="./",
    ) as workspace:
        # 3) Create agent
        agent = get_default_agent(
            llm=llm,
            # tools=[
            #     Tool(name="BashTool"),
            #     Tool(name="FileEditorTool"),
            #     Tool(name="TaskTrackerTool"),
            # ],
            cli_mode=True,
        )

        # 4) Set up callback collection
        received_events: list = []
        last_event_time = {"ts": time.time()}

        def event_callback(event) -> None:
            event_type = type(event).__name__
            logger.info(f"🔔 Callback received event: {event_type}\n{event}")
            received_events.append(event)
            last_event_time["ts"] = time.time()

        # 5) Test the workspace with a simple command
        result = workspace.execute_command(
            "echo 'Hello from sandboxed environment!' && pwd"
        )
        logger.info(
            f"Command '{result.command}' completed with exit code {result.exit_code}"
        )
        logger.info(f"Output: {result.stdout}")
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[event_callback],
            visualize=True,
        )
        # assert isinstance(conversation, Conversation)

        try:
            logger.info(f"\n📋 Conversation ID: {conversation.state.id}")

            logger.info("📝 Sending first message...")
            conversation.send_message(
                "Read the current repo and write 3 facts about the project into "
                "FACTS.txt."
            )
            logger.info("🚀 Running conversation...")
            conversation.run()
            logger.info("✅ First task completed!")
            logger.info(f"Agent status: {conversation.state.agent_status}")

            # Wait for events to settle (no events for 2 seconds)
            logger.info("⏳ Waiting for events to stop...")
            while time.time() - last_event_time["ts"] < 2.0:
                time.sleep(0.1)
            logger.info("✅ Events have stopped")

            logger.info("🚀 Running conversation again...")
            conversation.send_message("Great! Now delete that file.")
            conversation.run()
            logger.info("✅ Second task completed!")
        finally:
            print("\n🧹 Cleaning up conversation...")
            conversation.close()


if __name__ == "__main__":
    main()
