import os
<<<<<<< HEAD
import time
=======
>>>>>>> ae7c2f7 (add all)
import json
from pydantic import SecretStr

from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.sdk import (
    Agent,
    LLM,
    Conversation,
    RemoteConversation,
<<<<<<< HEAD
    Event,
    LLMConvertibleEvent,
    get_logger,
    Message,
    TextContent
)
import ast
=======
    get_logger,
)
>>>>>>> ae7c2f7 (add all)

from rca.utils.mini_swe import get_docker_image_name, evaluate_trajectory


from datasets import load_dataset
# import pytest

logger = get_logger(__name__)

import requests
<<<<<<< HEAD
public_ip = requests.get('https://api.ipify.org').text
print(f"Public IP: {public_ip}")

def test_workspace(instance) -> None:

    # api_key = os.getenv("CMU_KEY")
    # api_url = os.getenv("CMU_URL")
    # assert api_key is not None, "CMU_KEY environment variable is not set."
    # assert api_url is not None, "CMU_URL environment variable is not set."
    # llm = LLM(
    #     service_id="agent",
    #     model="litellm_proxy/neulab/claude-sonnet-4-20250514",
    #     base_url=api_url,
    #     api_key=SecretStr(api_key),
    # )

    instance_id = instance["instance_id"]
    llm = LLM(
        service_id="agent",
        model="hosted_vllm/Qwen/Qwen3-8B",
        base_url=f"http://{public_ip}:8080/v1/",
        api_key="",
    )

    messages = []
    result = None
    working_dir = "/testbed"

    def conversation_callback(event: Event):
            # print("HERE Event", event, type(event))
            if isinstance(event, LLMConvertibleEvent):
                messages.append(event.to_llm_message())

    data_source = instance.get("data_source", "swe-smith")
=======

public_ip = requests.get("https://api.ipify.org").text
print(f"Public IP: {public_ip}")


def test_workspace(instance) -> None:
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

    # llm = LLM(
    #     service_id="agent",
    #     model="hosted_vllm/Qwen/Qwen3-8B",
    #     base_url=f"http://{public_ip}:8080/v1/",
    #     api_key="",
    # )

    messages = []
    result = None
    working_dir = "/workspace"

    data_source = instance.get("data_source", "swe-smith")
    instance_id = instance["instance_id"]
>>>>>>> ae7c2f7 (add all)
    print("data_source", data_source)
    image_name = get_docker_image_name(instance, data_source=data_source)
    print("image_name", image_name)
    with DockerWorkspace(
        base_image=image_name,
<<<<<<< HEAD
        host_port=None,
        detach_logs=False,
        working_dir=working_dir,
        platform="linux/amd64", # "linux/arm64"
        # forward_env=["AGENT_SDK_PATH", "API_KEY", "CMU_KEY", "OPENAI_API_KEY"],  # Forward API key to container
        forward_env=["AGENT_SDK_PATH"],  # Forward API key to container
    ) as workspace:

=======
        image="ghcr.io/all-hands-ai/agent-server",
        host_port=None,
        detach_logs=False,
        working_dir=working_dir,
        platform="linux/amd64",  # "linux/arm64"
        # forward_env=["AGENT_SDK_PATH", "API_KEY", "CMU_KEY", "OPENAI_API_KEY"],  # Forward API key to container
        # forward_env=["AGENT_SDK_PATH"],  # Forward API key to container
    ) as workspace:
>>>>>>> ae7c2f7 (add all)
        agent = Agent(
            llm=llm,
            tools=get_default_tools(
                enable_browser=False,
            ),
            cli_mode=False,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
<<<<<<< HEAD
            callbacks=[conversation_callback],
=======
>>>>>>> ae7c2f7 (add all)
            # visualize=True,
            visualize=False,
        )
        assert isinstance(conversation, RemoteConversation)
        print("Starting conversation...")
        print(instance["problem_statement"])
        try:
            logger.info("Conversation Starting")
            conversation.send_message(instance["problem_statement"])
            conversation.run()
        except Exception as e:
            logger.error(f"Error is sending conversation: {e}", exc_info=True)
        finally:
<<<<<<< HEAD
            workspace_result = workspace.execute_command("git add -A && git diff --cached", cwd=working_dir)
=======
            workspace_result = workspace.execute_command(
                "git add -A && git diff --cached", cwd=working_dir
            )
>>>>>>> ae7c2f7 (add all)
            result = workspace_result.stdout
            conversation.close()
            logger.info("Conversation Finished")

<<<<<<< HEAD
=======
    messages = list(map(lambda event: event.model_dump(), conversation.state.events))
>>>>>>> ae7c2f7 (add all)
    print("=" * 100)
    print("Conversation finished. Got the following LLM messages:")
    for i, message in enumerate(messages):
        print(f"Message {i}: {str(message)[:200]}")

<<<<<<< HEAD
    constructed_messages = []
    for idx, message in enumerate(messages):
        full_text = ""
        if message.role == "assistant":
            if message.content is not None and len(message.content) > 0:
                full_text += message.content[0].text

            if message.tool_calls is not None and len(message.tool_calls) > 0:
                tool_name = message.tool_calls[0].name
                tool_args = ast.literal_eval(message.tool_calls[0].arguments)
                if tool_name == "finish":
                    full_text += tool_args["message"]
                else:
                    full_text += "\n\n" + f"<function={tool_name}>"
                    for k, v in tool_args.items():
                        full_text += f"\n<parameter={k}>{v}</parameter>\n"
                    full_text += f"</function>\n"
        else:
            full_text += message.content[0].text
        constructed_messages.append({"role": message.role, "text": full_text})

    constructed_messages.append({"role": "system", "text": result})
    print("result", result)
    result = evaluate_trajectory(instance, result, {"cwd": working_dir}, data_source)
    print("Evaluation:", result)
    # Save trajectory for debugging
    with open(f"traj_{instance_id}.jsonl", "w") as f:
        f.writelines(json.dumps(msg) + "\n" for msg in constructed_messages)
=======
    # constructed_messages = []
    # for idx, message in enumerate(messages):
    #     full_text = ""
    #     if message.role == "assistant":
    #         if message.content is not None and len(message.content) > 0:
    #             full_text += message.content[0].text

    #         if message.tool_calls is not None and len(message.tool_calls) > 0:
    #             tool_name = message.tool_calls[0].name
    #             tool_args = ast.literal_eval(message.tool_calls[0].arguments)
    #             if tool_name == "finish":
    #                 full_text += tool_args["message"]
    #             else:
    #                 full_text += "\n\n" + f"<function={tool_name}>"
    #                 for k, v in tool_args.items():
    #                     full_text += f"\n<parameter={k}>{v}</parameter>\n"
    #                 full_text += f"</function>\n"
    #     else:
    #         full_text += message.content[0].text
    #     constructed_messages.append({"role": message.role, "text": full_text})

    # constructed_messages.append({"role": "system", "text": result})
    print("result", result)
    result = evaluate_trajectory(instance, result, {"cwd": working_dir}, data_source)
    print("Evaluation:", result)
    with open(f"traj_{instance_id}.jsonl", "w") as f:
        f.writelines(json.dumps(msg) + "\n" for msg in messages)
>>>>>>> ae7c2f7 (add all)

    return result


if __name__ == "__main__":

<<<<<<< HEAD
    import random
    dataset = load_dataset("SWE-bench/SWE-smith", split="train").to_pandas()
    dataset = dataset[dataset['problem_statement'] != ""]
    while True:
        try:
            idx = random.randint(0, len(dataset) - 1)
=======
    dataset = load_dataset("SWE-bench/SWE-smith", split="train").to_pandas()
    dataset = dataset[dataset["problem_statement"] != ""]
    while True:
        try:
            # idx = random.randint(0, len(dataset) - 1)
            idx = 0
>>>>>>> ae7c2f7 (add all)
            print(f"Testing instance {idx}")
            instance = dataset.iloc[idx].to_dict()
            result = test_workspace(instance)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error occurred: {e}")
            break
<<<<<<< HEAD
=======
        break
>>>>>>> ae7c2f7 (add all)
