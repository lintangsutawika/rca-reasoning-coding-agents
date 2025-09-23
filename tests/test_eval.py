from datasets import load_dataset
import pytest

from rca.utils.mini_swe import get_environment, evaluate_trajectory

dataset = load_dataset("SWE-bench/SWE-smith", split="train").to_pandas()
instance = dataset.iloc[0].to_dict()
instance["data_source"] = "swe-smith"
patch = instance["model_patch"]
config = {"cwd": "/testbed"}
env = get_environment(config, instance, "swe-smith")

# evaluate_trajectory
result = evaluate_trajectory(instance, patch, config, "swe-smith")