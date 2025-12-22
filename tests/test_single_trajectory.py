from datasets import load_dataset

# import pytest
import traceback

from rca.utils.mini_swe import evaluate_trajectory
from rca.utils.tunnel import create_localtunnel
import fire
import pandas as pd
import os

from tqdm import tqdm

from omegaconf import DictConfig
from rca.generators.openhands_generator import init_and_run
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata

from rca.utils.mini_swe import (
    get_docker_image_name,
)
import ngrok
from rca.utils.containers import get_agent_server_docker_image

from openhands.sdk import get_logger
import logging

logger = get_logger(__name__)
logger.setLevel(logging.ERROR)

import asyncio

async def test_workspace(instance, base_url):
    instance_id = instance["instance_id"]
    data_source = "swe-smith"
    trajectory_id = TrajectoryID(instance_id=instance_id, repetition_id="test")
    training_phase = "train"

    # path of this file
    current_file_path = __file__
    test_path = os.path.join(os.path.dirname(current_file_path), "test_outputs")
    generator_cfg = DictConfig({"traj_dir": test_path})
        
    messages, reward, error = await init_and_run.remote(
        instance=instance,
        litellm_model_name="litellm_proxy/Qwen/Qwen3-8B",
        litellm_base_url=base_url,
        generator_cfg=generator_cfg,
        data_source=data_source,
        sampling_params={},
        trajectory_id=trajectory_id,
        global_step=0,
        training_phase=training_phase,
        )
    
    return messages, reward, error

if __name__ == "__main__":
    import pandas as pd
    import argparse
    import subprocess
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    dataset = pd.read_parquet(args.data_path)
    image_names = dataset["image_name"].unique().tolist()

    # base_url, process = create_localtunnel(port=8000)
    # base_url = base_url+"/v1/"
    base_url = "http://0.0.0.0:8080"

    for image_name in image_names:
        print(image_name)
        instance = dataset[dataset["image_name"] == image_name].iloc[0].to_dict()
        messages, patch = asyncio.run(test_workspace(instance, base_url))

        for msg in messages:
            print(msg)
        # print("Reward:", reward)
        # print(messages)
        break