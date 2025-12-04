from typing import TypedDict, Optional
import traceback
import uuid

from typing import Dict, Any
from loguru import logger

from jinja2 import Template

from swebench.harness.constants import DOCKER_WORKDIR
from swesmith.profiles import registry
from swesmith.constants import (
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
)

from swebench.harness.constants import (
    ResolvedStatus,
)
from swesmith.harness.grading import get_eval_tests_report, get_resolution_status
from minisweagent.environments import Environment, get_environment


class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]


def get_docker_image_name(instance: dict, data_source: str = "swe-bench") -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")
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
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_docker_image_name(instance, data_source)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = f"docker://{image_name}"
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def evaluate_trajectory(
    instance: Dict[str, Any], model_patch: str, sweagent_config: dict, data_source: str
) -> MiniSWEEvaluationResult:
    ret = MiniSWEEvaluationResult(
        instance_id=instance["instance_id"], resolved=False, eval_error=None
    )

    env = None
    try:
        # env = get_environment(
        env = get_sb_environment(sweagent_config, instance, data_source)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(
            f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}"
        )
        return ret

    # apply git patch
    # NOTE (sumanthrh): This applies patch in-line, and the maximum patch size is limited by the OS limits for `ARG_MAX`.
    # In modern systems, this is typically ~ 1 MB, which is pretty generous.
    # For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail
    profile = registry.get_from_inst(instance)

    # # run eval script in-line
    # if data_source.lower() == "swe-gym":
    #     # Checkout Base Commit
    #     # "git checkout {base_commit}"

    if data_source.lower() == "swe-smith":
        # In Swe-Smith, the patch applies the buggy patch
        # to fix by our model.
        bug_patch = instance["patch"]
        delimiter = (f"PATCH_{uuid.uuid4().hex}")
        apply_bug_patch = f"git apply --verbose <<'{delimiter}'\n{bug_patch}\n{delimiter}"
        _ = env.execute(apply_bug_patch, cwd=sweagent_config["cwd"])
        _ = env.execute('git config --global user.email "sweft@anon.com"', cwd=sweagent_config["cwd"])
        _ = env.execute('git config --global user.name "sweft"', cwd=sweagent_config["cwd"])
        _ = env.execute("git commit -am 'Initial commit'", cwd=sweagent_config["cwd"])

        # Remove commit history.
        _ = env.execute("git checkout --orphan new-main", cwd=sweagent_config["cwd"])
        _ = env.execute("git add .", cwd=sweagent_config["cwd"])
        _ = env.execute("git commit -m 'Initial commit'", cwd=sweagent_config["cwd"])
        _ = env.execute("git branch -D main", cwd=sweagent_config["cwd"])
        _ = env.execute("git branch -m main", cwd=sweagent_config["cwd"])

    delimiter = (f"PATCH_{uuid.uuid4().hex}")
    apply_patch = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"
    _ = env.execute(apply_patch, cwd=sweagent_config["cwd"])

    # "git checkout {base_commit}"
    f2p_files, p2p_files = profile.get_test_files(instance)
    test_files = " ".join(f2p_files + p2p_files)
    if test_files:
        env.execute(f"git checkout -- {test_files}", cwd=sweagent_config["cwd"])
    test_command, _ = profile.get_test_cmd(instance)  # , f2p_only=f2p_only)
    eval_script = (
        "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                f"cd {DOCKER_WORKDIR}",
                f": '{TEST_OUTPUT_START}'",
                test_command,
                f": '{TEST_OUTPUT_END}'",
            ]
        )
        + "\n"
    )

    eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
    # add longer timeout for evaluation
    obs = env.execute(eval_cmd, cwd=sweagent_config["cwd"], timeout=3600)
    content = obs["output"]
    start_sep = f"+ : '{TEST_OUTPUT_START}'"
    end_sep = f"+ : '{TEST_OUTPUT_END}'"
    start_idx = content.find(start_sep)
    end_idx = content.find(end_sep)

    # Get evaluation test report
    test_logs = content[start_idx:end_idx][len(start_sep) :]
    test_status_map = profile.log_parser(test_logs)
    report = get_eval_tests_report(test_status_map, instance)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        ret["resolved"] = True
        # ret["eval_error"] = None
    else:
        ret["resolved"] = False
    
    ret["eval_error"] = report
    try:
        f2p_success = len(ret["eval_error"]["FAIL_TO_PASS"]["success"])
        f2p_failure = len(ret["eval_error"]["FAIL_TO_PASS"]["failure"])

        p2p_success = len(ret["eval_error"]["PASS_TO_PASS"]["success"])
        p2p_failure = len(ret["eval_error"]["PASS_TO_PASS"]["failure"])

        f2p_score = f2p_success / (f2p_success + f2p_failure)
        p2p_score = p2p_success / (p2p_success + p2p_failure)

        resolution_score = (f2p_score + p2p_score) / 2.0
        ret["partial_score"] = resolution_score
    except Exception as e:
        logger.error(f"Error computing partial score: {e}")
        ret["partial_score"] = 0.0

    # truncate to last 1000 characters for brevity
    # ret["eval_error"] = (
    #     f"(truncated to last 1000 characters)\n{obs['output'][-1000:]}"
    #     if not ret["resolved"]
    #     else None
    # )
    return ret
