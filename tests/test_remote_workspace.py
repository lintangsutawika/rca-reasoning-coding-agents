import os
import time
import json
from pydantic import SecretStr

from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.sdk import (
    Agent,
    LLM,
    Conversation,
    RemoteConversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
)
import ast

from rca.utils.mini_swe import get_docker_image_name, evaluate_trajectory


from datasets import load_dataset
# import pytest

dataset = load_dataset("SWE-bench/SWE-smith", split="train").to_pandas()
instance = dataset.iloc[0].to_dict()
instance["data_source"] = "swe-smith"
instance_id = instance["instance_id"]
# patch = instance["model_patch"]
patch = instance["patch"]

logger = get_logger(__name__)


def main() -> None:

    api_key = os.getenv("CMU_KEY")
    api_url = os.getenv("CMU_URL")
    assert api_key is not None, "CMU_KEY environment variable is not set."
    assert api_url is not None, "CMU_URL environment variable is not set."
    llm = LLM(
        service_id="agent",
        model="litellm_proxy/neulab/claude-sonnet-4-20250514",
        base_url=api_url,
        api_key=SecretStr(api_key),
    )

    # import requests
    # public_ip = requests.get('https://api.ipify.org').text
    # print(f"Public IP: {public_ip}")
    # llm = LLM(
    #     service_id="agent",
    #     model="hosted_vllm/Qwen/Qwen3-4B",
    #     base_url=f"http://{public_ip}:8080/v1/",
    #     api_key="",
    # )

    messages = []
    result = None
    working_dir = "/testbed"

    def conversation_callback(event: Event):
            # print("HERE Event", event, type(event))
            if isinstance(event, LLMConvertibleEvent):
                messages.append(event.to_llm_message())

    data_source = instance.get("data_source", "swe-smith")
    print("data_source", data_source)
    image_name = get_docker_image_name(instance, data_source=data_source)
    print("image_name", image_name)
    with DockerWorkspace(
        base_image=image_name,
        host_port=None,
        working_dir=working_dir,
        platform="linux/amd64", # "linux/arm64"
        forward_env=["AGENT_SDK_PATH", "API_KEY", "CMU_KEY", "OPENAI_API_KEY"],  # Forward API key to container
    ) as workspace:

        cli_mode = True
        agent = Agent(
            llm=llm,
            tools=get_default_tools(
                # Disable browser tools in CLI mode
                enable_browser=not cli_mode,
            ),
            cli_mode=cli_mode,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[conversation_callback],
            visualize=True,
        )
        assert isinstance(conversation, RemoteConversation)
        try:
            logger.info("Conversation Starting")
            conversation.send_message(instance["problem_statement"])
            conversation.run()
        except Exception as e:
            logger.error(f"Error is sending conversation: {e}", exc_info=True)
        finally:
            conversation.close()
            logger.info("Conversation Finished")

    print("=" * 100)
    print("Conversation finished. Got the following LLM messages:")
    for i, message in enumerate(messages):
        print(f"Message {i}: {str(message)[:200]}")

    constructed_messages = []
    for idx, message in enumerate(messages):
        if message.role == "assistant":
            tool_name = message.tool_calls[0].name
            tool_args = ast.literal_eval(message.tool_calls[0].arguments)
            if tool_name != "finish":
                if len(message.content) == 0:
                    message_text = ""
                else:
                    message_text = message.content[0].text
            else:
                message_text = tool_args["message"]

            full_text = message_text + "\n\n" + f"<function={tool_name}>"
            for k, v in tool_args.items():
                full_text += f"\n<parameter={k}>{v}</parameter>\n"
            full_text += f"</function>\n"
        else:
            full_text = message.content[0].text
            if full_text.startswith("diff --git"):
                result = full_text
        constructed_messages.append({"role": message.role, "text": full_text})

    print("full_text", full_text)
    print("result", result)
    result = evaluate_trajectory(instance, result, {"cwd": working_dir}, data_source)
    print(result)

    # Save trajectory for debugging
    with open(f"traj_{instance_id}.jsonl", "w") as f:
        f.writelines(json.dumps(msg) + "\n" for msg in constructed_messages)


if __name__ == "__main__":
    main()
