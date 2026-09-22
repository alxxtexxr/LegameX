import ast
import logging
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import ListConfig


@contextmanager
def capture_to_log():
    """Redirect stdout so print() output also goes to the Hydra log file."""
    class _Tee:
        def __init__(self, original, file_handler):
            self._original = original
            self._file = file_handler.stream
            self._formatter = file_handler.formatter or logging.Formatter()
            # Create a record-like object so we can reuse the file handler's formatter
            self._record = logging.LogRecord(
                name="__main__", level=logging.INFO,
                pathname="", lineno=0, msg="", args=(), exc_info=None,
            )

        def write(self, s):
            self._original.write(s)
            if s.strip():  # skip blank newlines to avoid double-spacing
                self._record.msg = s.rstrip("\n")
                self._record.args = ()
                formatted = self._formatter.format(self._record)
                self._file.write(formatted + "\n")
                self._file.flush()

        def flush(self):
            self._original.flush()
            self._file.flush()

        def __getattr__(self, name):
            return getattr(self._original, name)

    # Find the Hydra file handler on the root logger
    file_handler = None
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.FileHandler):
            file_handler = h
            break

    if file_handler is None:
        yield  # no file handler, nothing to capture
        return

    original = sys.stdout
    assert file_handler is not None  # guaranteed by early return above
    sys.stdout = _Tee(original, file_handler)
    try:
        yield
    finally:
        sys.stdout = original


def to_list(val):
    # Handle Hydra's ListConfig by converting to a plain Python list
    if isinstance(val, ListConfig):
        return list(val)
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
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


def preprocess_train_val_datasets(
    task: str,  # 'wikipedia' | 'squad'
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer,
    seed: int,

    # Wikipedia dataset configuration
    wiki_max_length: int = 510,  # 510 + 2 (BOS + EOS) = 512, matching XLM-R's max_position_embeddings
    num_chunks_per_wiki_article: int = 3,
) -> tuple[Dataset, Dataset]:
    if task == 'squad':

        def preprocess_squad(examples):
            tokenized = tokenizer(
                examples['question'],
                examples['context'],
                truncation='only_second', # Only truncate the context
                max_length=384,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )

            # Each generated window points back to its original SQuAD example.
            sample_mapping = tokenized.pop('overflow_to_sample_mapping') # Global index

            # Save offsets separately because the model does not need them.
            offset_mapping = tokenized.pop('offset_mapping') # Local index

            start_positions = []
            end_positions = []

            # Indices of windows that contain the complete answer.
            keep_indices = []

            for feature_idx, offsets in enumerate(offset_mapping):

                # Which original example produced this window?
                sample_idx = sample_mapping[feature_idx]

                answer = examples['answers'][sample_idx]

                answer_start_char = answer['answer_start'][0]
                answer_text = answer['text'][0]
                answer_end_char = (
                    answer_start_char + len(answer_text)
                )

                # sequence_ids tells us which tokens belong to:
                # 0 = question
                # 1 = context
                # None = special tokens
                sequence_ids = tokenized.sequence_ids(feature_idx)

                # Find the first and last context tokens in this window.
                context_token_indices = [
                    idx
                    for idx, seq_id in enumerate(sequence_ids)
                    if seq_id == 1
                ]

                if not context_token_indices:
                    continue

                context_start = context_token_indices[0]
                context_end = context_token_indices[-1]

                # Check whether the COMPLETE answer is inside this window.
                #
                # If the answer starts before the first context token,
                # or ends after the last context token, this window
                # does not contain the complete answer.
                if offsets[context_start][0] > answer_start_char:
                    continue

                if offsets[context_end][1] < answer_end_char:
                    continue

                # Find the token containing the answer start.
                token_start = context_start

                while (
                    token_start <= context_end
                    and offsets[token_start][1] <= answer_start_char
                ):
                    token_start += 1

                # Find the token containing the answer end.
                token_end = context_end

                while (
                    token_end >= context_start
                    and offsets[token_end][0] >= answer_end_char
                ):
                    token_end -= 1

                # Safety check.
                if token_start > context_end or token_end < context_start:
                    continue

                # Keep this window.
                keep_indices.append(feature_idx)
                start_positions.append(token_start)
                end_positions.append(token_end)

            # Keep only windows containing the complete answer.
            tokenized = {
                key: [
                    values[i]
                    for i in keep_indices
                ]
                for key, values in tokenized.items()
            }

            tokenized['start_positions'] = start_positions
            tokenized['end_positions'] = end_positions

            return tokenized


        train_dataset = train_dataset.map(
            preprocess_squad,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        val_dataset = val_dataset.map(
            preprocess_squad,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

    else:

        def tokenize_text(examples, indices):
            all_input_ids = []
            all_attention_masks = []
            all_labels = []

            for text, article_idx in zip(examples['text'], indices, strict=True):

                # Tokenize the entire article without truncation.
                # Special tokens are added after chunking.
                article_tokens = tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                )['input_ids']

                # Create non-overlapping 512-token chunks.
                chunks = [
                    article_tokens[i:i + wiki_max_length]
                    for i in range(
                        0,
                        len(article_tokens),
                        wiki_max_length,
                    )
                ]

                # Keep only complete 512-token chunks.
                chunks = [
                    chunk
                    for chunk in chunks
                    if len(chunk) == wiki_max_length
                ]

                # Select up to N chunks randomly.
                num_chunks = min(
                    num_chunks_per_wiki_article,
                    len(chunks),
                )

                if num_chunks == 0:
                    continue

                # Use a deterministic per-article RNG.
                article_rng = random.Random(
                    seed + article_idx
                )

                selected_chunks = article_rng.sample(
                    chunks,
                    num_chunks,
                )

                for chunk in selected_chunks:

                    # Add special tokens in the same way
                    # the XLM-R tokenizer normally does.
                    input_ids = (
                        [tokenizer.bos_token_id]
                        + chunk
                        + [tokenizer.eos_token_id]
                    )
                    attention_mask = [1] * len(input_ids)

                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_mask)
                    all_labels.append(input_ids.copy())

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
                'labels': all_labels,
            }

        train_dataset = train_dataset.map(
            tokenize_text,
            batched=True,
            with_indices=True,
            remove_columns=train_dataset.column_names,
        )

        val_dataset = val_dataset.map(
            tokenize_text,
            batched=True,
            with_indices=True,
            remove_columns=val_dataset.column_names,
        )

    return train_dataset, val_dataset


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


# Compute metrics for SQuAD (token-level span F1 and Exact Match)
def compute_metrics_squad(eval_pred):
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids

    # Take argmax to get predicted positions
    pred_starts = np.argmax(start_logits, axis=-1)
    pred_ends = np.argmax(end_logits, axis=-1)

    f1_scores = []
    em_scores = []
    for pred_start, pred_end, true_start, true_end in zip(
        pred_starts, pred_ends, start_positions, end_positions, strict=True
    ):
        # Ensure valid spans
        if pred_start > pred_end:
            pred_start, pred_end = pred_end, pred_start

        # Exact Match: both start and end must match exactly
        try:
            em_scores.append(float(pred_start == true_start and pred_end == true_end))
        except (ValueError, TypeError):
            em_scores.append(0.0)

        predicted_tokens = set(range(pred_start, pred_end + 1))
        true_tokens = set(range(true_start, true_end + 1))

        if len(predicted_tokens) == 0 or len(true_tokens) == 0:
            f1_scores.append(0.0)
            continue

        intersection = predicted_tokens & true_tokens
        precision = len(intersection) / len(predicted_tokens)
        recall = len(intersection) / len(true_tokens)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return {'f1': np.mean(f1_scores), 'exact_match': np.mean(em_scores)}
