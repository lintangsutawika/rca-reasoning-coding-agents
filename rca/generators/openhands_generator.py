import subprocess
import re
import json
import asyncio
from socket import timeout
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
from omegaconf import DictConfig
import traceback
import ray
import requests
from pathlib import Path
import os
import ast

# from minisweagent.run.utils.save import save_traj
from rca.utils.prompt import get_instruction
from rca.utils.mini_swe import (
    evaluate_trajectory,
    get_docker_image_name,
)

from skyrl_train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    GeneratorOutput,
    GeneratorInput,
)
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    encode_messages_subset,
)

from openhands.workspace import DockerWorkspace, APIRemoteWorkspace, LocalWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.sdk import (
    Agent,
    LLM,
    Event,
    Conversation,
    RemoteConversation,
    # LLMConvertibleEvent,
    # TokenEvent,
    LLMSummarizingCondenser,
    get_logger,
)

from rca.agent.agent import CustomAgent
from rca.utils.tunnel import create_localtunnel
from rca.utils.containers import get_agent_server_docker_image
from rca.utils.timeout import timeout

import logging

logger = get_logger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)

# public_ip = requests.get("https://api.ipify.org").text
# print(f"Public IP: {public_ip}")

import ngrok


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    litellm_base_url: dict,
    generator_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: Union[TrajectoryID, Any],
    global_step: int,
    training_phase: Union[TrainingPhase, Any],
):

    agent = None
    patch = ""
    error = None
    eval_error = None
    messages = []
    working_dir = "/testbed"

    print("data_source", data_source)
    repo_path = f"/workspace/{instance['repo'].split('/')[-1]}_{global_step}_{trajectory_id.repetition_id}/"
    image_name = get_docker_image_name(instance, data_source=data_source)
    image_source = "lintangsutawika/agent-swe-smith"
    server_image = get_agent_server_docker_image(image_source, image_name)
    print("server_image", server_image)
    print("repo_path", repo_path)

    with APIRemoteWorkspace(
        runtime_api_url="https://runtime.eval.all-hands.dev",
        runtime_api_key=os.getenv("OPENHANDS_RUNTIME_API_KEY"),
        working_dir=repo_path,
        server_image=server_image,
        target_type="source",
        api_timeout=600,
    ) as workspace:

        instance["repo_path"] = repo_path
        logger.info(f"repo_path: {repo_path}")
        cp_testebed_repo = workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
        assert cp_testebed_repo.exit_code == 0, (
            f"cp_testebed_repo failed: {cp_testebed_repo.stderr}"
        )

        delimiter = f"PATCH_{uuid.uuid4().hex}"
        bug_patch = instance["patch"]
        apply_bug_patch = f"git apply --verbose <<'{delimiter}'\n{bug_patch}\n{delimiter}"
        _ = workspace.execute_command(apply_bug_patch, cwd=repo_path)
        # print("Applied bug patch")
        # print(workspace.execute_command("git status", cwd=repo_path).stdout)
        # print(_.stdout)
        # print(_.stderr)
        workspace.execute_command('git config --global user.email "sweft@anon.com"', cwd=repo_path)
        workspace.execute_command('git config --global user.name "sweft"', cwd=repo_path)
        commit_log = workspace.execute_command("git commit -am 'Initial commit'", cwd=repo_path)
        # print("Commit log:", commit_log)
        # print(workspace.execute_command("git log", cwd=repo_path).stdout)

        workspace.execute_command("git checkout --orphan new-main", cwd=repo_path)
        workspace.execute_command("git add .", cwd=repo_path)
        workspace.execute_command("git commit -m 'Initial commit'", cwd=repo_path)
        workspace.execute_command("git branch -D main", cwd=repo_path)
        workspace.execute_command("git branch -m main", cwd=repo_path)
        # print(workspace.execute_command("git log", cwd=repo_path).stdout)

        api_key = os.getenv("CMU_KEY")
        api_url = os.getenv("CMU_URL")
        assert api_key is not None, "CMU_KEY environment variable is not set."
        assert api_url is not None, "CMU_URL environment variable is not set."

        # model_as_condenser = True
        model_as_condenser = False
        if model_as_condenser:
            llm = LLM(
                service_id="agent",
                model="litellm_proxy/neulab/claude-sonnet-4-20250514",
                # model="litellm_proxy/neulab/gpt-oss-120b",
                base_url=api_url,
                api_key=api_key,
            )

            condenser = LLMSummarizingCondenser(
                llm=LLM(
                    service_id="condenser",
                    model=litellm_model_name,
                    # base_url="http://host.docker.internal:8080/v1/",
                    # base_url=f"http://{public_ip}:8080/v1/",
                    base_url=litellm_base_url,
                    api_key="sk-x",
                    litellm_extra_body={
                        "return_token_ids": True,
                        "include_stop_str_in_output": True,
                        "session_id": f"{instance['instance_id']}_{global_step}_{trajectory_id.repetition_id}",
                    }
                ),
                max_size=8,
                keep_first=2,
            )
        else:
            llm=LLM(
                service_id="agent",
                model=litellm_model_name,
                # base_url="http://host.docker.internal:8080/v1/",
                # base_url=f"http://{public_ip}:8080/v1/",
                base_url=litellm_base_url,
                api_key="sk-x",
                litellm_extra_body={
                    "return_token_ids": True,
                    "include_stop_str_in_output": True,
                    "session_id": f"{instance['instance_id']}_{global_step}_{trajectory_id.repetition_id}",
                }
            )
            condenser=None

        agent = CustomAgent(
            llm=llm,
            tools=get_default_tools(
                enable_browser=False,
            ),
            condenser=condenser,
            cli_mode=False,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=50,
            stuck_detection=True,
            visualizer=None,
        )

        input_message = get_instruction(instance, None, repo_path)
        assert isinstance(conversation, RemoteConversation)

        logger.info("Conversation Starting")
        conversation.send_message(input_message)
        conversation.run()
        
        messages = list(map(lambda event: event.model_dump(), conversation.state.events))
        workspace_result = workspace.execute_command(
            "git diff HEAD", cwd=repo_path
        )
        patch = workspace_result.stdout
        conversation.close()
        logger.info("Conversation Finished")

    return (messages, patch)


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
        super().__init__(
            generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name
        )

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_endpoint_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_endpoint_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        # self.litellm_model_name = "hosted_vllm/" + self.model_name
        self.litellm_model_name = "litellm_proxy/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError(
                "OpenhandsGenerator doesn't support custom chat template"
            )

        # base_url = "http://0.0.0.0:8080"
        # listener = ngrok.forward(
        #         addr=base_url,
        #         authtoken=os.getenv("NGROK_KEY")
        #     )
        # self.base_url = f"{listener.url()}/v1/"
        # self.base_url = f"https://loud-terms-accept.loca.lt/v1/"
        # self.base_url = create_localtunnel(port=8080)+"/v1/"
        # self.base_url = base_url + "/v1/"
        # print("Localtunnel URL:", self.base_url)

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


        port = self.base_url.split(":")[-1].split("/")[0]
        url_tunnel, process = create_localtunnel(port=int(port))
        print("Localtunnel URL:", url_tunnel)

        instance = env_extras["instance"]
        generator_cfg = self.generator_cfg
        global_step = batch_metadata.global_step
        training_phase = batch_metadata.training_phase
        data_source = env_extras["data_source"]
        try:
            messages, patch = await init_and_run.remote(
                instance,
                self.litellm_model_name,
                self.base_url,
                generator_cfg,
                data_source,
                sampling_params,
                trajectory_id,
                global_step,
                training_phase,
            )

        except Exception as e:
            logger.error(
                f"Error processing instance {instance['instance_id']}: {e}", exc_info=True
            )
            # # exit_status, result = type(e).__name__, str(e)
            # error = f"Error with image: {server_image}\n"+str(e)
            # # extra_info = {"traceback": traceback.format_exc()}

            messages = []
            patch = ""
            error = f"Error in init_and_run: {e}"

        reward = 0
        result = None
        if patch != "":
           # Reward if a patch is generated
            if patch.startswith("diff --git"):
                reward = 1
            else:
                reward = 0

            # with 5 minutes timeout
            try:
                with timeout(300):
                    result = evaluate_trajectory(
                        instance, patch, {"environment": {"environment_class": "singularity"}, "cwd": "/testbed"}, data_source
                    )
            except TimeoutError as e:
                result = None
                error = f"TimeoutError during evaluation: {e}"

            if isinstance(result, dict) and "partial_score" in result:
                reward += float(result["partial_score"])

        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        rollout_list = []
        if len(token_messages) > 0:
            for idx, message in enumerate(token_messages):
                current_prompt_ids = message["prompt_token_ids"]
                current_response_ids = message["response_token_ids"]
                step_reward = reward

                rollout_list.append(
                    (
                        current_response_ids,
                        step_reward,
                        "complete",
                        [1]*len(current_response_ids),
                        current_prompt_ids,
                        None,
                    )
                )
        else:
            response_ids = [151643]
            stop_reason = "error"
            loss_mask = [1]
            initial_input_ids = [151643]
            rollout_list.append(
                (response_ids, reward, stop_reason, loss_mask, initial_input_ids, None)
            )

        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        print("generator_cfg.traj_dir", generator_cfg.traj_dir)
        path = Path(generator_cfg.traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.jsonl"
        # path = path / filename

        result_dict = {
            "reward": reward,
            "detailed_result": result,
            "error": error,
            "patch": patch,
            "messages": messages,
        }

        with open(os.path.join(path, f"train_traj_{instance_id}_{trajectory_id.repetition_id}.json"), "w") as f:
            json.dump(result_dict, f, indent=2)

        if patch.startswith("diff --git"):
            with open(os.path.join(path, f"train_traj_{instance_id}_{trajectory_id.repetition_id}.diff"), "w") as f:
                f.write(patch)

        return rollout_list

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

        task_rollouts = []
        for i in range(len(prompts)):
            rollout = self.openhands_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            
            task_rollouts.append(rollout)

        collected_task_rollouts = await asyncio.gather(*task_rollouts)
        all_outputs = [rollout[0] for rollout in collected_task_rollouts]

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = sum([[output[0] for output in step_outputs] for step_outputs in all_outputs], [])
        rewards = sum([[output[1] for output in step_outputs] for step_outputs in all_outputs], [])
        stop_reasons = sum([[output[2] for output in step_outputs] for step_outputs in all_outputs], [])
        loss_masks = sum([[output[3] for output in step_outputs] for step_outputs in all_outputs], [])
        prompt_token_ids = sum([[output[4] for output in step_outputs] for step_outputs in all_outputs], [])

        out_trajectory_ids = []
        is_last_step = []
        for i in range(len(all_outputs)):
            step_outputs = all_outputs[i]
            for step_id in range(len(step_outputs)):
                out_trajectory_id = copy.deepcopy(trajectory_ids[i])
                out_trajectory_id.step = step_id
                out_trajectory_ids.append(out_trajectory_id.instance_id)
                is_last_step.append(step_id == len(step_outputs) - 1)

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
            "is_last_step": is_last_step,
        }

        return generator_output
