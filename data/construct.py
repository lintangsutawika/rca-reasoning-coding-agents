import os

import argparse
import pandas as pd
from datasets import load_dataset

from emac.utils import parse_action

def main(args):

    # Load dataset
    if args.data_name:
        dataset = load_dataset(args.data_path, args.data_name, split='train')
        task_name = args.task_name or args.data_name
    else:
        dataset = load_dataset(args.data_path, split='train')
        task_name = args.task_name or args.data_path

    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    def process_row(row):
        system_message = row["system"]
        conversations = row["conversations"]
        messages = []
        ongoing_message = [{"role": "system", "content": system_message}]
        for step in conversations:
            role = step["from"]
            content = step["value"]

            content = parse_action(content, string_only=True) if role == "gpt" else content
            new_role = "assistant" if role == "gpt" else "user"
            ongoing_message.append({"role": new_role, "content": content})

            if role == "gpt":
                messages.append(
                    ongoing_message.copy()
                )
    
        return pd.Series(messages)
    
    _df = df.apply(lambda x: process_row(x), axis=1)  # Example operation, replace with actual logic
    data_df = pd.DataFrame()
    _df = _df.stack().reset_index()
    data_df["full_trajectory"] = _df[0]
    data_df["id"] = _df["level_0"]
    data_df["turn"] = _df["level_1"]
    data_df["input"] = data_df["full_trajectory"].apply(lambda x: x[:-1])
    data_df["output"] = data_df["full_trajectory"].apply(lambda x: x[-1]["content"])
    data_df["N"] = data_df["input"].apply(lambda x: len(x))

    data_df['data_source'] = f"{args.data_path}/{args.data_name}" if args.data_name else args.data_path

    data_df['reward_model'] = data_df.apply(
        lambda row: {
            "ground_truth": row["output"],
        },
        axis=1
    )

    data_df["extra_info"] = data_df.apply(
        lambda row: {
            "task": row["data_source"],
            "ground_truth": row["output"],
            "reward_partial": True,
        }, 
        axis=1
    )

    # Sort by input length to ensure even distribution
    data_df = data_df.sort_values('N').reset_index(drop=True)
    data_df = data_df.drop(columns=['full_trajectory']) 
    
    task_size = len(data_df)
    indices = list(range(task_size))
    # Create stratified splits by distributing every 10th sample to maintain length distribution
    train_indices = [i for i in indices if i % 10 < 8]
    valid_indices = [i for i in indices if i % 10 == 8]
    test_indices = [i for i in indices if i % 10 == 9]

    train_df = data_df.loc[train_indices].reset_index(drop=True)
    valid_df = data_df.loc[valid_indices].reset_index(drop=True)
    test_df = data_df.loc[test_indices].reset_index(drop=True)

    save_path = os.path.join(args.output_path, task_name)
    os.makedirs(save_path, exist_ok=True)
    train_df.to_parquet(os.path.join(save_path, "train.parquet"))
    valid_df.to_parquet(os.path.join(save_path, "valid.parquet"))
    test_df.to_parquet(os.path.join(save_path, "test.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct data from datasets')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--data_name', type=str, help='Name of the dataset configuration')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed data')
    parser.add_argument('--task_name', type=str, default=None, help='Name of the task')
    args = parser.parse_args()
    main(args)