import os
import math
import logging
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
from utils import (
    to_list,
    load_train_val_datasets,
    preprocess_train_val_datasets,
    get_device,
    get_amp_config,
    get_optim,
    compute_metrics_squad,
    capture_to_log,
)

WIKI_MAX_LENGTH = 510  # 510 + 2 (BOS + EOS) = 512, matching XLM-R's max_position_embeddings
NUM_CHUNKS_PER_WIKI_ARTICLE = 3


@hydra.main(
    version_base="1.3",
    config_path="../conf/lora_training_xlmr",
    config_name="wikipedia-en-10K",
)
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Replace Hydra's default console handler with a cleaner format
    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    start_time = datetime.now()
    log.info("================================================================")
    log.info(f"Starting: {start_time}")
    log.info(f"Configuration name: {cfg.config_name}")
    log.info("================================================================")

    log.info("")
    log.info("================================================================")
    log.info("Configuration")
    log.info("================================================================")
    log.info(f"Seed: {cfg.seed}")
    log.info(f"Device: {cfg.device}")

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
        run_name = f"{model_name}-{cfg.task}-{cfg.lang}-{cfg.data.train_size / 1000:g}K-s{cfg.seed}-LoRA-v{datetime.now().strftime('%y%m%d%H%M%S')}"

        # Automatically retrieve the username of the currently logged-in user
        from huggingface_hub import HfApi

        user_info = HfApi().whoami()
        username = user_info["name"]

        hub_model_id = f"{username}/{run_name}"
    base_hub_model_id, version = hub_model_id.rsplit("-v", 1)
    hub_merged_model_id = f"{base_hub_model_id}-Merged-v{version}"

    log.info(f"Resume from checkpoint: {resume_from_checkpoint}")
    log.info(f"Model name: {model_name}")
    log.info(f"Run name: {run_name}")
    log.info(f"Hub model ID: {hub_model_id}")
    log.info(f"Hub merged model ID: {hub_merged_model_id}")

    log.info("")
    log.info("================================================================")
    log.info("Model")
    log.info("================================================================")
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
    with capture_to_log():
        model.print_trainable_parameters()
    log.info(f"device: {model.device}")

    log.info("")
    log.info("================================================================")
    log.info("Data")
    log.info("================================================================")
    # Load the dataset
    train_dataset, val_dataset = load_train_val_datasets(
        lang=cfg.lang,
        task=cfg.task,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
    )

    log.info("Train dataset:")
    log.info(train_dataset)
    log.info("")
    log.info("Validation dataset:")
    log.info(val_dataset)

    # Preprocess the dataset
    train_dataset, val_dataset = preprocess_train_val_datasets(
        task=cfg.task,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        seed=cfg.seed,
        wiki_max_length=WIKI_MAX_LENGTH,
        num_chunks_per_wiki_article=NUM_CHUNKS_PER_WIKI_ARTICLE,
    )

    if cfg.task == "wikipedia":
        log.info(f"Total Wikipedia chunks: {len(train_dataset)}")

    log.info("")
    log.info("================================================================")
    log.info("Training")
    log.info("================================================================")
    if isinstance(cfg.train.max_steps, int):
        # Manually set the maximum number of steps
        max_steps = cfg.train.max_steps
        log.info(f"Max steps: {max_steps} (manually set from configuration)")
    else:
        # Calculate the maximum number of steps
        steps_per_epoch = math.ceil(len(train_dataset) / (cfg.train.mini_batch_size * cfg.train.grad_accum_steps))
        max_steps = steps_per_epoch * cfg.train.num_epochs
        log.info(f"Steps per epoch: {steps_per_epoch}")
        log.info(f"Max steps: {max_steps}")

    # Set up the output directory
    output_dir = os.path.join(os.getcwd(), "outputs_training", run_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        log.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    # Set up the training arguments
    bf16, fp16 = get_amp_config(device)
    optim = get_optim(cfg.train.optim, device)
    report_to = to_list(cfg.train.report_to)

    log.info(f"AMP: bf16={bf16}, fp16={fp16}")
    log.info(f"Optimizer: {optim} (configuration was: {cfg.train.optim})")
    log.info(f"Report to: {report_to}")

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
        eval_strategy=cfg.train.eval_strategy,
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
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model='f1' if cfg.task == 'squad' else 'eval_loss',
        greater_is_better=(cfg.task == 'squad'),
        run_name=run_name,
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        push_to_hub=cfg.train.push_to_hub,
        hub_strategy="all_checkpoints",
        hub_always_push=True,
    )
    # Only add EarlyStoppingCallback when evaluation is enabled
    trainer_callbacks = []
    if cfg.train.eval_strategy != "no" and cfg.train.load_best_model_at_end:
        trainer_callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.train.early_stopping_patience,
                # early_stopping_threshold = 0.001,
            )
        )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=train_args,
        compute_metrics=compute_metrics_squad if cfg.task == 'squad' else None,
        # label_names=['labels'],
        callbacks=trainer_callbacks,
    )
    trainer.label_names = label_names

    # Start training
    with capture_to_log():
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # If the task is SQuAD, upload the merged model and tokenizer
    if cfg.task == "squad" and cfg.train.push_to_hub:
        log.info("")
        log.info("================================================================")
        log.info("Merging and Uploading")
        log.info("================================================================")
        # After the training finishes, merge the LoRA into the base model and save everything
        model = model.eval()  # Good practice
        merged_model = model.merge_and_unload()

        # Upload the merged model to Hugging Face
        merged_model.push_to_hub(hub_merged_model_id)
        tokenizer.push_to_hub(hub_merged_model_id)

        log.info(f"Merged model uploaded to: https://huggingface.co/{hub_merged_model_id}")

    end_time = datetime.now()
    elapsed = end_time - start_time
    log.info("")
    log.info("================================================================")
    log.info(f"Finished: {end_time}")
    log.info(f"Elapsed: {elapsed}")
    log.info("================================================================")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]  # Hydra @hydra.main() injects cfg automatically
