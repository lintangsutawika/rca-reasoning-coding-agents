from typing import TypedDict, Optional
import traceback
import uuid

from typing import Dict, Any
from loguru import logger

from jinja2 import StrictUndefined, Template

from swebench.harness.constants import DOCKER_WORKDIR
from swesmith.profiles import registry
from swesmith.constants import (
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
)

from minisweagent.environments import Environment, get_environment
from rca.environments import ApptainerEnvironment

class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]

def get_docker_image_name(instance: dict, data_source: str="swe-bench") -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")  # to comply with docker image naming convention
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-smith" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/jyangballin/swesmith.x86_64.{id_docker_compatible}:latest".lower()
        else:
            raise NotImplementedError(f"Data source: {data_source} is not supported")
    return image_name

def get_sb_environment(config: dict, instance: dict, data_source: str) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "apptainer")
    image_name = get_docker_image_name(instance, data_source=data_source)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
        env = get_environment(env_config)
    else:
        env_config["image"] = f"docker://{image_name}"
        if env_config["environment_class"] == "singularity":
            env = get_environment(env_config)
        elif env_config["environment_class"] == "apptainer":
            env_config.pop("environment_class")
            env = ApptainerEnvironment(**env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env

def evaluate_trajectory(
    instance: Dict[str, Any],
    model_patch: str,
    sweagent_config: dict,
    data_source: str
) -> MiniSWEEvaluationResult:

    ret = MiniSWEEvaluationResult(instance_id=instance["instance_id"], resolved=False, eval_error=None)

    env = None
    try:
        # env = get_environment(
        env = get_sb_environment(
            sweagent_config,
            instance,
            data_source
            )
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}")
        return ret

    profile = registry[".".join(instance["instance_id"].split(".")[:-1])]()
    f2p_files, p2p_files = profile.get_test_files(instance)
    test_files = " ".join(f2p_files + p2p_files)
    if test_files:
        env.execute(f"git checkout -- {test_files}", cwd=sweagent_config["cwd"])

    # apply git patch
    # NOTE (sumanthrh): This applies patch in-line, and the maximum patch size is limited by the OS limits for `ARG_MAX`.
    # In modern systems, this is typically ~ 1 MB, which is pretty generous.
    # For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail
    delimiter = f"PATCH_{uuid.uuid4().hex}"  # unlikely to collide with symbols in the patch
    command = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"
    obs = env.execute(command, cwd=sweagent_config["cwd"])

    if obs["returncode"] != 0:
        ret["eval_error"] = obs["output"]
    else:
        # run eval script in-line
        # eval_script = instance["eval_script"]
        test_command, _ = profile.get_test_cmd(instance) #, f2p_only=f2p_only)
        eval_script = "\n".join(
                [
                    "#!/bin/bash",
                    "set -uxo pipefail",
                    f"cd {DOCKER_WORKDIR}",
                    f": '{TEST_OUTPUT_START}'",
                    test_command,
                    f": '{TEST_OUTPUT_END}'",
                ]
            ) + "\n"

        eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
        # add longer timeout for evaluation
        obs = env.execute(eval_cmd, cwd=sweagent_config["cwd"], timeout=3600)
        # use the return value
        ret["resolved"] = obs["returncode"] == 0
        # truncate to last 1000 characters for brevity
        ret["eval_error"] = (
            f"(truncated to last 1000 characters)\n{obs["output"][-1000:]}" if not ret["resolved"] else None
        )
    return ret
