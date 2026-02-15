from datasets import load_dataset

# import pytest
import traceback

from rca.utils.mini_swe import evaluate_trajectory

import fire
import pandas as pd
import os

from tqdm import tqdm
# print("Using ground truth patch:")
# ret = evaluate_trajectory(instance, patch, config, "swe-smith")
# print(ret)
# # {'instance_id': 'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc', 'resolved': True, 'eval_error': None}


# print("Using empty patch:")
# ret = evaluate_trajectory(instance, "", config, "swe-smith")
# print(ret)
# # {'instance_id': 'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc', 'resolved': False, 'eval_error': 'error: unrecognized input\n'}

from openhands.sdk import get_logger
import logging

logger = get_logger(__name__)
logger.setLevel(logging.ERROR)


def test_training_data(data_path: str, data_source: str = "swe-smith", split: str = "test"):
    """
    Evaluate trajectory with different patches.

    Args:
        data_path: Path to the directory containing the training data.
    """
    # config = {"cwd": "/testbed"}
    config = {"environment": {"environment_class": "singularity"}, "cwd": "/testbed"}

    if data_path.endswith(".parquet"):
        training_dataset = pd.read_parquet(data_path)
    else:
        # check if data_path is a directory
        if not os.path.isdir(data_path):
            training_dataset = load_dataset(data_path, split=split).to_pandas()
        else:
            training_dataset = pd.read_parquet(os.path.join(data_path, "train.parquet"))

    for i, instance in tqdm(training_dataset.iterrows(), total=len(training_dataset)):
        try:
            instance = instance.to_dict()
            # patch = instance["patch"]
            patch = ""
            print(f"Evaluating instance {i} with patch")
            print(patch)
            ret = evaluate_trajectory(instance, patch, config, data_source)
            print(ret)
        except Exception as e:
            print(f"Error evaluating instance {i}: {e}\n, {traceback.format_exc()}")
            continue
        break


if __name__ == "__main__":
    fire.Fire(test_training_data)
