"""
Preprocess the SWEBench dataset to SkyRL format
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/swe_smith")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "SWE-bench/SWE-smith"
    # eval_data_source = "SumanthRH/SWE-bench_Verified"

<<<<<<< HEAD
    train_dataset = datasets.load_dataset(data_source, "default", split="train[:1000]")
    val_dataset = datasets.load_dataset(data_source, "default", split="train[1000:1200]")
=======
    def filter_empty_examples(example):
        return example["problem_statement"] != ""

    # Use filter to remove empty problem statements
    raw_data = datasets.load_dataset(data_source, "default", split="train")
    # Shuffle
    raw_data = raw_data.shuffle(seed=42)
    # Select 10000
    raw_data = raw_data.select([i for i in list(range(5000))])
    # Filter out empty examples
    raw_data = raw_data.filter(filter_empty_examples)

    train_dataset = raw_data.select(range(1000))
    val_dataset = raw_data.select(range(1000, 1500))
>>>>>>> ae7c2f7 (add all)

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
<<<<<<< HEAD
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
=======
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
>>>>>>> ae7c2f7 (add all)
