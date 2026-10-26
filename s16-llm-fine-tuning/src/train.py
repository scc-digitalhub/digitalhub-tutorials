import os

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import huggingface_hub
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# ============================================================
# Logging Callback
# ============================================================


class LoggingCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return

        metrics = {}

        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v

            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics[k] = v.item()

        for k, v in metrics.items():
            try:
                self.run.log_metric(k, v)

            except Exception:
                pass


# ============================================================
# Formatting Function
# ============================================================


def formatting_func(example):

    output_texts = []

    for q, a in zip(example["question"], example["answer"]):
        text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{q}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{a}"
            "<|eot_id|>"
        )

        output_texts.append(text)

    return output_texts


# ============================================================
# Train Function
# ============================================================


def train(
    model_id: str,
    from_base: int,
    hf_dataset_name: str,
    train_data_path: str,
    dev_data_path: str,
    output_dir: str,
    final_dir: str,
    project_name: str,
    run_name: str,
    quantization: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    max_sequence_length: int,
    early_stopping_patience: int,
    learning_rate: float,
    scheduler_type: str,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    hf_token: str = None,
    wandb_key: str = None,
    logger: TrainerCallback = None,
) -> None:

    # ========================================================
    # HF Login
    # ========================================================

    if hf_token is not None:
        try:
            huggingface_hub.login(token=hf_token)

        except Exception:
            raise RuntimeError("Error logging into Hugging Face")

    # ========================================================
    # WandB Login
    # ========================================================

    if wandb_key is not None:
        try:
            wandb.login(key=wandb_key)

            wandb.init(
                project=project_name,
                name=run_name,
            )

        except Exception:
            raise RuntimeError("Error logging into WandB")

    else:
        wandb.init(mode="disabled")

    # ========================================================
    # CUDA INFO
    # ========================================================

    print("=" * 80)
    print("CUDA DEBUG")
    print("=" * 80)

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )

    # ========================================================
    # Tokenizer
    # ========================================================

    print("=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # ========================================================
    # Quantization Config
    # ========================================================

    quantization_config = None

    if quantization == 4:
        print("4bit quantization enabled")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    else:
        print("No quantization")

    # ========================================================
    # Model Loading
    # ========================================================

    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )

    except Exception as e:
        print("=" * 80)
        print("MODEL LOAD ERROR")
        print("=" * 80)

        print(str(e))
        traceback.print_exc()

        raise

    print("=" * 80)
    print("MODEL LOADED")
    print("=" * 80)

    print("HF device map:", model.hf_device_map)

    # ========================================================
    # Gradient Checkpointing
    # ========================================================

    model.config.use_cache = False
    gradient_checkpointing_kwargs = {"use_reentrant": False}
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    )

    # ========================================================
    # Prepare for QLoRA
    # ========================================================

    if quantization == 4:
        print("=" * 80)
        print("PREPARING MODEL FOR QLORA")
        print("=" * 80)

        model = prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

    # ========================================================
    # LoRA
    # ========================================================

    print("=" * 80)
    print("APPLYING LORA")
    print("=" * 80)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # ========================================================
    # Dataset
    # ========================================================

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    try:
        dataset = load_dataset(
            hf_dataset_name,
            data_files={
                "train": train_data_path,
                "validation": dev_data_path,
            },
        )

        train_dataset = dataset["train"].shuffle(seed=42)
        dev_dataset = dataset["validation"].shuffle(seed=42)

    except Exception as e:
        print("DATASET ERROR")
        traceback.print_exc()

        raise RuntimeError(f"Error loading dataset: {e!s}")

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(dev_dataset))

    # ========================================================
    # Callbacks
    # ========================================================

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    if logger is not None:
        callbacks.append(logger)

    # ========================================================
    # Trainer
    # ========================================================

    print("=" * 80)
    print("BUILDING TRAINER")
    print("=" * 80)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        formatting_func=formatting_func,
        callbacks=callbacks,
        args=SFTConfig(
            output_dir=output_dir,
            max_seq_length=max_sequence_length,
            learning_rate=learning_rate,
            lr_scheduler_type=scheduler_type,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb" if wandb_key is not None else "none",
            fp16=True,
            bf16=False,
            seed=42,
        ),
    )

    print("=" * 80)
    print("TRAINER READY")
    print("=" * 80)

    # ========================================================
    # Train
    # ========================================================

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.train()

    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    # ========================================================
    # Save
    # ========================================================

    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"Model saved to: {final_dir}")


# ============================================================
# Train + DigitalHub Logging
# ============================================================


def train_and_log_model(
    project,
    model_name: str,
    model_id: str,
    hf_dataset_name: str,
    train_data_path: str,
    dev_data_path: str,
    from_base: int = 0,
    quantization: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_sequence_length: int = 2048,
    early_stopping_patience: int = 5,
    learning_rate: float = 2e-4,
    scheduler_type: str = "cosine",
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    grad_accum_steps: int = 4,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 50,
    wandb_project: str = None,
    wandb_run: str = None,
):

    output_dir = "/app/local_data/checkpoints/ground"
    final_dir = "/app/local_data/weights/ground"

    hf_token = None
    wandb_key = None

    # ========================================================
    # Secrets
    # ========================================================

    try:
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()

    except Exception:
        pass

    try:
        wandb_key = project.get_secret("WANDB_KEY").read_secret_value()

    except Exception:
        pass

    # ========================================================
    # Train
    # ========================================================

    train(
        model_id=model_id,
        from_base=from_base,
        hf_dataset_name=hf_dataset_name,
        train_data_path=train_data_path,
        dev_data_path=dev_data_path,
        output_dir=output_dir,
        final_dir=final_dir,
        project_name=wandb_project,
        run_name=wandb_run,
        quantization=quantization,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_sequence_length=max_sequence_length,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        scheduler_type=scheduler_type,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        grad_accum_steps=grad_accum_steps,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        hf_token=hf_token,
        wandb_key=wandb_key,
        logger=LoggingCallback(project.get_run(os.environ["RUN_ID"])),
    )

    # ========================================================
    # Model Parameters
    # ========================================================

    model_params = {
        "quantization": quantization,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_sequence_length": max_sequence_length,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate": learning_rate,
        "scheduler_type": scheduler_type,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
    }

    # ========================================================
    # DigitalHub Model Logging
    # ========================================================

    model = project.log_model(
        name=model_name,
        source=final_dir + "/",
        model_id=model_id,
        model_revision="main",
    )

    run = project.get_run(os.environ["RUN_ID"])

    metrics = run.status.metrics

    for k in metrics:
        try:
            model.log_metric(k, metrics[k])

        except Exception:
            pass
