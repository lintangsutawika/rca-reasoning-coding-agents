import uuid
from datasets import load_dataset
# import pytest

from rca.utils.mini_swe import get_sb_environment, evaluate_trajectory

from swesmith.profiles import registry

dataset = load_dataset("SWE-bench/SWE-smith", split="train").to_pandas()
instance = dataset.iloc[0].to_dict()
instance["data_source"] = "swe-smith"
instance_id = instance["instance_id"]
# patch = instance["model_patch"]
patch = instance["patch"]
config = {"cwd": "/testbed"}



print("Using ground truth patch:")
ret = evaluate_trajectory(instance, patch, config, "swe-smith")
print(ret)
# {'instance_id': 'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc', 'resolved': True, 'eval_error': None}


print("Using empty patch:")
ret = evaluate_trajectory(instance, "", config, "swe-smith")
print(ret)
# {'instance_id': 'oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc', 'resolved': False, 'eval_error': 'error: unrecognized input\n'}