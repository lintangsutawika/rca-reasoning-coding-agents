import os
import time

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Conversation,
    RemoteConversation,
    Event,
    LLMConvertibleEvent,
    Message,
    TextContent,
    get_logger,
)
# from openhands.sdk.workspace import DockerWorkspace
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_agent

from openhands.sdk import Agent
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.spec import Tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

logger = get_logger(__name__)


def main() -> None:
    # 1) Ensure we have LLM API key
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL")
    # assert api_key is not None, "LLM_API_KEY environment variable is not set."

    # llm = LLM(
    #     service_id="agent",
    #     # model="litellm_proxy/anthropic/claude-sonnet-4-5-20250929",
    #     # model="litellm_proxy/neulab/claude-sonnet-4-20250514",
    #     model="neulab/claude-sonnet-4-20250514",
    #     base_url=api_url,
    #     # api_key=SecretStr(api_key),
    #     api_key="sk-A3F2OynOu5ryarjMIBbQmw",
    # )

    llm = LLM(
        service_id="agent",
        model="hosted_vllm/Qwen/Qwen3-8B",
        # base_url="http://host.docker.internal:8080/v1/",
        base_url="http://172.212.177.81:8080/v1/",
        # base_url="http://0.0.0.0:8080/v1/",
        api_key="",
    )


    from openhands.sdk import Agent
    from openhands.sdk.tool.registry import register_tool
    from openhands.sdk.tool.spec import Tool
    from openhands.tools.execute_bash import BashTool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    register_tool("BashTool", BashTool)
    register_tool("FileEditorTool", FileEditorTool)
    register_tool("TaskTrackerTool", TaskTrackerTool)

    # # Provide Tool so Agent can lazily materialize tools at runtime.
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name="BashTool"),
            Tool(name="FileEditorTool"),
            Tool(name="TaskTrackerTool"),
        ],
    )


    # Test completion
    # messages = [Message(role="user", content=[TextContent(text="Hello")])]
    # response = llm.completion(messages=messages)
    # print("LLM response:", response)
    # import sys; sys.exit()

    # 2) Create a Docker-based remote workspace that will set up and manage
    #    the Docker container automatically
    with DockerWorkspace(
        base_image="nikolaik/python-nodejs:python3.12-nodejs22",
        host_port=None,
        # TODO: Change this to your platform if not linux/arm64
        # platform="linux/arm64",
        platform="linux/amd64",
        forward_env=["LLM_API_KEY", "API_KEY", "OPENAI_API_KEY"],  # Forward API key to container
    ) as workspace:
        # 3) Create agent
        # agent = get_default_agent(
        #     llm=llm,
        #     # cli_mode=True,
        # )

        # 4) Set up callback collection
        received_events: list = []
        last_event_time = {"ts": time.time()}

        def event_callback(event) -> None:
            event_type = type(event).__name__
            logger.info(f"🔔 Callback received event: {event_type}\n{event}")
            received_events.append(event)
            last_event_time["ts"] = time.time()

        llm_messages = []  # collect raw LLM messages

        def conversation_callback(event: Event):
            print("HERE Event", event, type(event))
            if isinstance(event, LLMConvertibleEvent):
                llm_messages.append(event.to_llm_message())

        # # 5) Test the workspace with a simple command
        # result = workspace.execute_command(
        #     "echo 'Hello from sandboxed environment!' && pwd"
        # )
        # logger.info(
        #     f"Command '{result.command}' completed with exit code {result.exit_code}"
        # )
        # logger.info(f"Output: {result.stdout}")
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            # callbacks=[event_callback, conversation_callback],
            callbacks=[conversation_callback],
            visualize=True,
            # visualize=False,
        )
        assert isinstance(conversation, RemoteConversation)

        try:
            logger.info(f"\n📋 Conversation ID: {conversation.state.id}")

            logger.info("📝 Sending first message...")
            conversation.send_message(
                "Write a fizzbuzz program in Python and save it to a file named fizzbuzz.py."
            )
            logger.info(f"Agent status: {conversation.state.agent_status}")
            logger.info("🚀 Running conversation...")
            conversation.run()
            logger.info("✅ First task completed!")
            logger.info(f"Agent status: {conversation.state.agent_status}")

            # # Wait for events to settle (no events for 2 seconds)
            # logger.info("⏳ Waiting for events to stop...")
            # while time.time() - last_event_time["ts"] < 2.0:
            #     time.sleep(0.1)
            # logger.info("✅ Events have stopped")

            # logger.info("🚀 Running conversation again...")
            # conversation.send_message("Great! Now delete that file.")
            # conversation.run()
            # logger.info("✅ Second task completed!")
        except Exception as e:
            logger.error(f"Error during conversation: {e}", exc_info=True)
        finally:
            print("\n🧹 Cleaning up conversation...")
            conversation.close()

    print("=" * 100)
    print("Conversation finished. Got the following LLM messages:")
    for i, message in enumerate(llm_messages):
        print(f"Message {i}: {str(message)[:200]}")

    return llm_messages

if __name__ == "__main__":
    llm_messages = main()
