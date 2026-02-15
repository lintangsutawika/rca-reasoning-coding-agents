from datasets import load_dataset

# import pytest
import traceback

from rca.utils.mini_swe import evaluate_trajectory
from rca.utils.timeout import timeout

import fire
import pandas as pd
import os

from tqdm import tqdm


def test_training_data():

    config = {"environment": {"environment_class": "singularity"}, "cwd": "/testbed"}
    training_dataset = load_dataset("SWE-bench/SWE-smith", split="train").to_pandas()
    # Get one of each unique image_name
    training_dataset = training_dataset.drop_duplicates(subset=["image_name"])
    # Get first 1000 samples for testing
    instance_list = []
    for i, instance in tqdm(training_dataset.iterrows(), total=len(training_dataset)):
        instance = instance.to_dict()
        patch = ""
        try:
            # with 2 minutes timeout
            with timeout(120):
                result = evaluate_trajectory(
                    instance, patch, config, "data_source"
                )
            instance_list.append(instance["instance_id"])
        except Exception as e:
            print(f"Error evaluating instance {i}: {e}\n, {traceback.format_exc()}")
            result = None
            error = f"TimeoutError during evaluation: {e}"

    # Save the instance_list to a csv file
    df = pd.DataFrame(instance_list, columns=["instance_id"])
    output_path = "test_training_data_instances.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved instance IDs to {output_path}")

if __name__ == "__main__":
    # fire.Fire(test_training_data)
    test_training_data()
