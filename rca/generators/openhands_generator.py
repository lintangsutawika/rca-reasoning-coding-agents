import asyncio
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import yaml
import traceback
import ray
import requests
from pathlib import Path
import random
import os
import ast
# from minisweagent.run.utils.save import save_traj
from rca.utils.mini_swe import evaluate_trajectory, get_sb_environment, get_docker_image_name

from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    # encode_messages_subset,
)

from pydantic import SecretStr

from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.sdk import (
    Agent,
    LLM,
    Conversation,
    RemoteConversation,
    Event,
    Message,
    TextContent,
    LLMConvertibleEvent,
    get_logger,
)

from rca.utils.mini_swe import get_docker_image_name

logger = get_logger(__name__)

public_ip = requests.get('https://api.ipify.org').text
print(f"Public IP: {public_ip}")

additional_command = """<SUBMISSION>
* When you've completed your work (reading, editing, testing), and cannot make further progress:
  - Use the bash tool to execute `git add -A && git diff --cached`
</SUBMISSION>"""

def encode_messages_subset(messages: ConversationType, tokenizer):
    """Encodes a subset of messages from a multi-turn conversation using the fixed base approach.

    This function tokenizes messages as if they are part of a larger conversation, ensuring
    no additional default system messages are prepended by the tokenizer's chat template

    The "fixed base approach" works by:
    - Creating a dummy base conversation to establish context
    - Appending the target messages to this base
    - Tokenizing the full conversation and extracting only the tokens for the target messages

    For simple chat templates without complex token splitting behavior, this produces the same
    result as directly tokenizing the messages. For templates like Qwen's ChatML format where
    a default system prompt can be appended, this ensures correct tokenization

    Reference: https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach

    Args:
        messages: List of message dicts with 'role' and 'content' keys. Must contain at least
                 one message. These are assumed to be a subset from a larger conversation.
        tokenizer: HuggingFace tokenizer with chat_template support and eos_token_id defined.

    Returns:
        List[int]: Token IDs for the given messages, with proper multi-turn context handling.
    """
    assert len(messages), "messages list cannot be empty"
    # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
    base_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
    ]
    if messages[0]["role"] != "assistant":
        # add an assistant message as well if the first role is user/tool
        base_conversation.append({"role": "assistant", "content": "I am an assistant."})
    base_conversation_token_ids = tokenizer.apply_chat_template(
        base_conversation,
        add_generation_prompt=False,
        tokenize=True,
    )

    full_conversation = base_conversation + messages
    full_conversation_token_ids = tokenizer.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
        tokenize=True,
    )
    conversation_token_ids = full_conversation_token_ids[len(base_conversation_token_ids) :]
    return conversation_token_ids


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    litellm_base_url: dict,
    generator_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    messages = []
    full_messages = []

    working_dir = "/testbed"
    def conversation_callback(event: Event):
        # print("HERE Event", event, type(event))
        if isinstance(event, LLMConvertibleEvent):
            messages.append(event.to_llm_message())

    try:
        print("data_source", data_source)
        image_name = get_docker_image_name(instance, data_source=data_source)
        print("image_name", image_name)
        with DockerWorkspace(
            # base_image="nikolaik/python-nodejs:python3.12-nodejs22",
            # server_image="ghcr.io/all-hands-ai/agent-server:latest-python",
            base_image=image_name,
            host_port=None,
            working_dir=working_dir,
            platform="linux/amd64", # "linux/arm64"
            # forward_env=["AGENT_SDK_PATH", "API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"],  # Forward API key to container
            forward_env=["AGENT_SDK_PATH", "API_KEY", "CMU_KEY", "OPENAI_API_KEY"],  # Forward API key to container
        ) as workspace:

            cli_mode = True
            agent = Agent(
                llm=LLM(
                    service_id="agent",
                    model=litellm_model_name,
                    # base_url="http://host.docker.internal:8080/v1/",
                    base_url=f"http://{public_ip}:8080/v1/",
                    api_key=os.getenv("API_KEY"),
                ),
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
                visualize=False,
                # visualize=True,
            )
            assert isinstance(conversation, RemoteConversation)
            try:
                logger.info("Conversation Starting")
                conversation.send_message(instance["problem_statement"]+"/no_think")
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

        for idx, message in enumerate(messages):
            if message.role == "assistant":
                tool_name = message.tool_calls[0].name
                tool_args = ast.literal_eval(message.tool_calls[0].arguments)
                message_text = message.content[0].text

                full_text = message_text + "\n\n" + f"<function={tool_name}>"
                for k, v in tool_args.items():
                    full_text += f"\n<parameter={k}>{v}</parameter>\n"
                full_text += f"</function>\n"
            else:
                full_text = message.content[0].text
            full_messages.append({"role": message.role, "content": full_text})

    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, {"cwd": working_dir}, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            # save_traj(agent, path, exit_status=exit_status, result=result, extra_info=extra_info, reward=reward, eval_error=eval_error)  # type: ignore[arg-type]

    return (full_messages, reward, error)


class OpenhandsGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        self.litellm_model_name = "hosted_vllm/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("OpenhandsGenerator doesn't support custom chat template")

    async def openhands_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:

        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        instance = env_extras["instance"]
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            # sweagent_config,
            self.base_url,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        if not len(messages):
            return None, None, None, None, None, None

        # response_messages = [{"role": msg.role, "content": msg.content[0].text} for msg in messages]

        # TODO Properly handle the right system prompt.
        input_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instance["problem_statement"]},
        ]

        initial_input_ids = self.tokenizer.apply_chat_template(input_prompt, add_generation_prompt=False, tokenize=True)
        initial_prompt_length = len(initial_input_ids)

        response_ids: List[int] = []
        loss_mask: List[int] = []

        for message in messages:
            # Apply chat template and tokenize each message
            msg_encoding = encode_messages_subset([message], self.tokenizer)

            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)

            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] in ["user", "tool"]:
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))
        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.openhands_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[0] is not None]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
