import ast

import torch
from datasets import Dataset, load_dataset


def to_list(val):
    try:
        return ast.literal_eval(val)
    except:
        return val


def load_train_val_datasets(
    lang: str,  # e.g., 'en' | 'ja' | 'id'
    task: str,  # 'wikipedia' | 'squad'
    train_size: int,
    val_size: int,
) -> tuple[Dataset, Dataset]:
    # Validate that if the task is 'squad', the language must be 'en'
    if task == "squad":
        assert lang == "en", "SQuAD is English-only."

    # Define dataset configurations for each task
    data_configs = {
        "wikipedia": {
            "data_id": "wikimedia/wikipedia",
            "data_dir": f"20231101.{lang}",
            "train_split": "train",
            "val_split": "train",
        },
        "squad": {
            "data_id": "rajpurkar/squad",
            "data_dir": None,
            "train_split": "train",
            "val_split": "validation",
        },
    }

    # Validate that the specified task is supported
    assert task in data_configs, (
        f"Unsupported task: {task}. Supported tasks: {list(data_configs.keys())}"
    )

    # Set up Hugging Face dataset configuration
    data_id = data_configs[task]["data_id"]
    data_dir = data_configs[task]["data_dir"]
    train_split = data_configs[task]["train_split"]
    val_split = data_configs[task]["val_split"]

    if train_split == val_split:
        # If the train and validation splits are the same, we need to sample from the same dataset stream
        dataset_stream = load_dataset(
            data_id,
            data_dir=data_dir,
            split=train_split,
            streaming=True,
        )

        train_data = []
        val_data = []

        for i, example in enumerate(dataset_stream):
            if i < train_size:
                train_data.append(example)
            elif i < train_size + val_size:
                val_data.append(example)
            else:
                break

    else:
        # If the train and validation splits are different, we can sample from each split separately
        def sample_split(split, size):
            dataset_stream = load_dataset(
                data_id,
                data_dir=data_dir,
                split=split,
                streaming=True,
            )

            data = []
            for i, example in enumerate(dataset_stream):
                if i >= size:
                    break
                data.append(example)
            return data

        train_data = sample_split(train_split, train_size)
        val_data = sample_split(val_split, val_size)

    return (
        Dataset.from_list(train_data),
        Dataset.from_list(val_data),
    )


def get_device(device: str = "auto") -> str:
    """Get the best available device, or return the specified device as-is."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_amp_config(device: str) -> tuple[bool, bool]:
    """Determine the best mixed-precision settings for the target device.

    Returns (bf16, fp16) flags compatible with TrainingArguments.
    Priority: bf16 > fp16 > neither (CPU / unsupported MPS).
    """
    if device == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, False
        return False, True
    return False, False


def get_optim(optim: str, device: str) -> str:
    """Return a compatible optimizer name for the target device.

    bitsandbytes 8-bit optimizers only support CUDA. Fall back to the
    standard PyTorch equivalent on other devices.
    """
    if device == "cuda":
        return optim
    # Map bitsandbytes 8-bit variants to their standard PyTorch equivalents
    fallback = {
        "adamw_8bit": "adamw_torch",
    }
    return fallback.get(optim, optim)
