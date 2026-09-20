import math
import os
from datetime import datetime
from typing import cast

import hydra
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from utils import to_list, load_train_val_datasets, get_device, get_amp_config, get_optim


@hydra.main(
    version_base="1.3",
    config_path="../conf/lora_training_xlmr",
    config_name="wikipedia-en-10K",
)
def main(cfg: DictConfig):
    print(
        "================================================================================================================================"
    )
    print("Configuration")
    print(
        "================================================================================================================================"
    )
    print("Seed:", cfg.seed)
    print("Device:", cfg.device)

    # Resume training configuration
    resume_from_checkpoint = None
    if resume_from_checkpoint:
        model_name = cfg.resume.model_id
        run_name = model_name.split("/")[-1]
        hub_model_id = cfg.resume.model_id

        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=hub_model_id, local_dir=model_name)

        if cfg.resume.ckpt_step:
            resume_from_checkpoint = f"{hub_model_id}/checkpoint-{cfg.resume.ckpt_step}"
            # Ensure the checkpoint exists
            assert os.path.exists(resume_from_checkpoint), (
                f"Checkpoint {resume_from_checkpoint} does not exist."
            )

    else:
        model_name = cfg.model_name
        run_name = f"{model_name}-{cfg.task}-{cfg.lang}-{cfg.data.train_size / 1000:g}K-LoRA-v{datetime.now().strftime('%y%m%d%H%M%S')}"

        # Automatically retrieve the username of the currently logged-in user
        from huggingface_hub import HfApi

        user_info = HfApi().whoami()
        username = user_info["name"]

        hub_model_id = f"{username}/{run_name}"
    base_hub_model_id, version = hub_model_id.rsplit("-v", 1)
    hub_merged_model_id = f"{base_hub_model_id}-Merged-v{version}"

    print("Resume from checkpoint:", resume_from_checkpoint)
    print("Model name:", model_name)
    print("Run name:", run_name)
    print("Hub model ID:", hub_model_id)
    print("Hub merged model ID:", hub_merged_model_id)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # Determine the appropriate model classes based on the task
    if cfg.task == "squad":
        model_cls = AutoModelForQuestionAnswering
        task_type = "QUESTION_ANS"
        data_collator = DataCollatorWithPadding(tokenizer)
        label_names = ["start_positions", "end_positions"]
    else:
        model_cls = AutoModelForMaskedLM
        task_type = "FEATURE_EXTRACTION"
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=cfg.data.mlm_prob,
        )
        label_names = ["labels"]

    print()
    print(
        "================================================================================================================================"
    )
    print("Model")
    print(
        "================================================================================================================================"
    )
    # Load the model
    model = model_cls.from_pretrained(cfg.model_id)

    if resume_from_checkpoint:
        # Load the LoRA adapter from the checkpoint and ensure it's in training mode
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
        model.train()                   # Ensure the resumed model is in training mode
        model.enable_adapter_layers()   # Explicitly unfreeze LoRA weights
    else:
        # Set up a LoRA configuration and apply it to the model
        lora_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            task_type=task_type,
            target_modules=to_list(cfg.lora.target_modules),
        )
        model = get_peft_model(model, lora_config)
        model = cast(PeftModel, model)  # Explicitly cast to PeftModel
    device = get_device(cfg.device)
    model = model.to(device)
    model.print_trainable_parameters()
    print("device:", model.device)

    print()
    print(
        "================================================================================================================================"
    )
    print("Data")
    print(
        "================================================================================================================================"
    )
    # Load the dataset
    train_dataset, val_dataset = load_train_val_datasets(
        lang=cfg.lang,
        task=cfg.task,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
    )

    print("Train dataset:")
    print(train_dataset)
    print()
    print("Validation dataset:")
    print(val_dataset)

    # Preprocess the dataset
    if cfg.task == "squad":

        def preprocess_squad(examples):
            # Tokenize question + context with offset mapping
            tokenized = tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=384,
                # stride=128,
                return_offsets_mapping=True,
                # padding='max_length',
            )

            # Prepare label lists
            start_positions = []
            end_positions = []

            for i, offsets in enumerate(tokenized["offset_mapping"]):
                # Find which tokens belong to the context (not the question, not special tokens)
                sequence_ids = tokenized.sequence_ids(i)

                # SQuAD v1.1 always has exactly one answer; take the first
                answer = examples["answers"][i]
                answer_start_char = answer["answer_start"][0]
                answer_text = answer["text"][0]
                answer_end_char = answer_start_char + len(answer_text)

                # Locate the token span that corresponds to the answer
                token_start = None
                token_end = None
                for idx, (offset_start, offset_end) in enumerate(offsets):
                    # Ignore question tokens and special tokens
                    if sequence_ids[idx] != 1:
                        continue
                    # Token fully inside answer
                    if (
                        offset_start >= answer_start_char
                        and offset_end <= answer_end_char
                        or offset_start < answer_end_char
                        and offset_end > answer_start_char
                    ):
                        if token_start is None:
                            token_start = idx
                        token_end = idx

                # If answer is out of bounds (truncated), set to CLS token index
                if token_start is None or token_end is None:
                    token_start = 0
                    token_end = 0

                start_positions.append(token_start)
                end_positions.append(token_end)

            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions

            return tokenized

        train_dataset = train_dataset.map(
            preprocess_squad, batched=True, remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            preprocess_squad, batched=True, remove_columns=val_dataset.column_names
        )
    else:

        def tokenize_text(examples):
            tokenized = tokenizer(
                examples["text"],
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        train_dataset = train_dataset.map(
            tokenize_text, batched=True, remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            tokenize_text, batched=True, remove_columns=val_dataset.column_names
        )

    print()
    print(
        "================================================================================================================================"
    )
    print("Training")
    print(
        "================================================================================================================================"
    )
    # Calculate the maximum number of training steps
    max_steps = (
        math.ceil(
            len(train_dataset)
            / (cfg.train.mini_batch_size * cfg.train.grad_accum_steps)
        )
        * cfg.train.num_epochs
    )
    print("Calculated max steps:", max_steps)

    # Set up the output directory
    output_dir = os.path.join(os.getcwd(), "outputs_training", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set up the training arguments
    bf16, fp16 = get_amp_config(device)
    optim = get_optim(cfg.train.optim, device)
    report_to = to_list(cfg.train.report_to)
    
    print(f"AMP: bf16={bf16}, fp16={fp16}")
    print(f"Optimizer: {optim} (config was: {cfg.train.optim})")
    print("Report to:", report_to)
    
    # Initialize wandb
    if "wandb" in report_to:
        import wandb
        wandb.init(
            project="legamex",
            name=run_name,
        )
    
    # Set up a trainer
    train_args = TrainingArguments(
        # Training arguments
        seed=cfg.seed,
        bf16=bf16,
        fp16=fp16,
        per_device_train_batch_size=cfg.train.mini_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        max_steps=max_steps,
        warmup_steps=cfg.train.warmup_steps,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        optim=optim,
        max_grad_norm=cfg.train.max_grad_norm,
        weight_decay=cfg.train.weight_decay,
        # Validation arguments
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        # Logging arguments
        logging_strategy="steps",
        logging_steps=cfg.train.logging_steps,
        # logging_first_step=True,
        report_to=report_to,
        # Saving arguments
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        # save_total_limit=5, # 1 best + 4 recent checkpoints. WARN: It doesn't work
        # With load_best_model_at_end=True, your save_strategy will be ignored and default to eval_strategy.
        # So you will find one checkpoint at the end of each epoch.
        # https://discuss.huggingface.co/t/trainer-not-saving-after-save-steps/5464
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        run_name=run_name,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        push_to_hub=cfg.train.push_to_hub,
        hub_strategy="all_checkpoints",
        hub_always_push=True,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=train_args,
        # label_names=['labels'],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.train.early_stopping_patience,
                # early_stopping_threshold = 0.001,
            )
        ],
    )
    trainer.label_names = label_names

    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print()
    print(
        "================================================================================================================================"
    )
    print("Merging and Uploading")
    print(
        "================================================================================================================================"
    )
    # If the task is SQuAD, upload the merged model and tokenizer
    if cfg.task == "squad":
        # After the training finishes, merge the LoRA into the base model and save everything
        model = model.eval()  # Good practice
        merged_model = model.merge_and_unload()

        # Upload the merged model to Hugging Face
        merged_model.push_to_hub(hub_merged_model_id)
        tokenizer.push_to_hub(hub_merged_model_id)

        print(f"Merged model uploaded to: https://huggingface.co/{hub_merged_model_id}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]  # Hydra @hydra.main() injects cfg automatically
