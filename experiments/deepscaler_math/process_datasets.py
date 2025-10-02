from datasets import load_dataset

# train data: "agentica-org/DeepScaleR-Preview-Dataset"
# test data: "opencompass/AIME2025", "HuggingFaceH4/aime_2024", "rawsh/2024_AMC12", "HuggingFaceH4/MATH-500", "https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl", "math-ai/minervamath"


import argparse
import os
import datasets
import random

instruction = "Let's think step by step and output the final answer within \\boxed{}."

def make_map_fn(split: str, data_source: str):

    def preprocess_fn(example, idx):
        question = example.pop("problem") if "problem" in example else example.pop("question")
        answer = example.pop("answer")
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user", 
                    "content": f"{question} {instruction}"
                }
            ],
            "env_class": "math_hard",
            "reward_spec": {
                "method": "rule", 
                "ground_truth": str(answer),
                "wrong_answer": str(int(random.random() * 100))
            },
            "noise_spec": {
                "method": "randomly_replace",
                "param": str(random.random())
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
                "answer": str(answer),
                "question": question,
            }
        }
        return data

    return preprocess_fn

def prepare_deepscaler_data():
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    
    dataset = dataset.map(function=make_map_fn("train", "deepscelar_train"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from DeepScaler training dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_aime2024_data():
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "aime2024"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AIME2024 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_aime2025_data():
    dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    dataset = datasets.concatenate_datasets([dataset1, dataset2])

    dataset = dataset.map(function=make_map_fn("test", "aime2025"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AIME2025 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_amc2024_data():
    dataset = load_dataset("rawsh/2024_AMC12", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "amc2024"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AMC2024 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")
    return dataset

def prepare_math500_data():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "math500"), with_indices=True)
    # show a few examples
    print("="*100)
    print("Sample examples from MATH500 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_amc2023_data(file_path):
    if not os.path.exists(file_path):
        print("Please download the AMC2023 test dataset from https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl and save it to", file_path)
        return {}
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.map(function=make_map_fn("test", "amc2023"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from AMC2023 test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

def prepare_minervamath_data():
    dataset = load_dataset("math-ai/minervamath", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "minervamath"), with_indices=True)

    # show a few examples
    print("="*100)
    print("Sample examples from MinervaMath test dataset:")
    for i in range(3):
        for key, value in dataset[i].items():
            print(f"{key}: {value}")
        print("\n")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/deepscaler_math")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = prepare_deepscaler_data()
    test_datasets = {
        "aime2024": prepare_aime2024_data(),
        "aime2025": prepare_aime2025_data(),
        "amc2024": prepare_amc2024_data(),
        "math500": prepare_math500_data(),
        "amc2023": prepare_amc2023_data(args.output_dir + "/amc2023.jsonl"),
        "minervamath": prepare_minervamath_data()
    }

    train_output_path = os.path.join(args.output_dir, "deepscaler_train.parquet")
    train_dataset.to_parquet(train_output_path)
    print(f"Train dataset saved to {train_output_path}")

    all_test_datasets = []
    for name, dataset in test_datasets.items():
        all_test_datasets.append(dataset)
        test_output_path = os.path.join(args.output_dir, f"{name}_test.parquet")
        dataset.to_parquet(test_output_path)
        print(f"Test dataset {name} saved to {test_output_path}")
    test_dataset = datasets.concatenate_datasets(all_test_datasets)
    test_dataset.to_parquet(os.path.join(args.output_dir, f"deepscaler_test.parquet"))





