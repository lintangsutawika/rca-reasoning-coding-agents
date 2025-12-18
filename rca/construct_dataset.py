"""
Preprocess the SWEBench dataset to SkyRL format
"""

import argparse
import os

import datasets

valid_env = [
    "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc",
    "pytest-dev__iniconfig.16793ead.combine_file__06k2m7dd",
    "jd__tenacity.0d40e76f.combine_file__1n8ql04e",
    "Cog-Creators__Red-DiscordBot.33e0eac7.combine_file__041av9lh",
    "agronholm__typeguard.b6a7e438.combine_file__3qg8gxw1",
    "pdfminer__pdfminer.six.1a8bd2f7.combine_file__06wx35ad",
    "cknd__stackprinter.219fcc52.combine_file__1i9gep13",
    "pudo__dataset.5c2dc8d3.combine_file__09k00ucq",
    "seatgeek__thefuzz.8a05a3ee.combine_file__18e0miwg",
 ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/swe_smith")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "SWE-bench/SWE-smith"
    # eval_data_source = "SumanthRH/SWE-bench_Verified"

    def filter_empty_examples(example):
        return example["problem_statement"] != ""
    
    def filter_valid_envs(example):
        iid = example["instance_id"].split(".")[:-1]
        iid = ".".join(iid)
        
        # Find if any of valid_env starts with iid
        for env in valid_env:
            if env.startswith(iid):
                return True
        return False
        

    # Use filter to remove empty problem statements
    raw_data = datasets.load_dataset(data_source, "default", split="train")
    # Shuffle
    raw_data = raw_data.shuffle(seed=42)
    # # Select 10000
    # raw_data = raw_data.select([i for i in list(range(5000))])
    # Filter out empty examples
    raw_data = raw_data.filter(filter_empty_examples)
    raw_data = raw_data.filter(filter_valid_envs)

    print(len(raw_data), "examples after filtering")

    # train_dataset = raw_data.select(range(1000))
    # val_dataset = raw_data.select(range(1000, 1500))
    train_dataset = raw_data #.select(range(35))
    val_dataset = raw_data #.select(range(35))
    # val_dataset = raw_data.select(range(40, 50))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                # "data_source": data_source if split == "train" else eval_data_source,
                "data_source": "swe-smith",
                "prompt": [
                    {
                        "role": "user",
                        "content": example["problem_statement"],
                    }
                ],
                "env_extras": {
                    "data_source": "swe-smith",
                },
                "env_class": "null",
                "instance": example,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
