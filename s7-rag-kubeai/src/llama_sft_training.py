import json
import torch
import huggingface_hub
import os
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset, load_dataset
import sys

class LoggingCallback(TrainerCallback):

    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics[k] = v.item()
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                    "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                )
            if k in metrics:
                self.run.log_metric(k, metrics[k])

def formatting_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id>\nYou are a helpful assistant.<|eot_id><|start_header_id|>user<|end_header_id>{example['question'][i]}<|eot_id><|start_header_id|>assistant<|end_header_id>{example['answer'][i]}<|eot_id>"
        output_texts.append(text)
    return output_texts
    
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
    logger: TrainerCallback = None
) -> None:
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        from_base (int): From Base? (0 or 1)
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub.
        train_data_path (str): Training dataset path
        dev_data_path (str): Dev dataset path
        output_dir (str): Output directory for model and checkpoints
        final_dir (str): Output directory for final model
        project_name (str): Project name for logging
        run_name (str): Run name for logging
        quantization (int): Quantization: 0 if none, 4 for 4bit quantization
        lora_rank (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout rate
        max_sequence_length (int): Maximum sequence length
        early_stopping_patience (int): Early stopping patience
        learning_rate (float): Learning rate for training
        scheduler_type (str): Learning rate scheduler type
        train_batch_size (int): Training batch size
        eval_batch_size (int): Evaluation batch size
        grad_accum_steps (int): Gradient accumulation steps
        num_epochs (int): Number of training epochs
        weight_decay (float): Weight decay for optimizer
        warmup_ratio (float): Warmup ratio for learning rate schedule
        logging_steps (int): Number of steps between logging
        eval_steps (int): Number of steps between evaluations
        save_steps (int): Number of steps between model checkpoints
        hf_token (str): Hugging Face API token. Required only for private models; not needed when using public models.
        wandb_key (str): Weights & Biases API key
    """
    # Logging in Hugging Face and WandB
    if hf_token is not None:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            raise RuntimeError("Error logging into Hugging Face. Check your token.")
        
    if wandb_key is not None:
        try:
            "Wandb configuration initialization..."
            wandb.login(key=wandb_key)
            wandb.init(project=project_name, name=run_name)
        except Exception as e:
            raise RuntimeError("Error logging into WandB Face. Check your key.")
    else:
        wandb.init(mode="disabled")
        
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

    # Loading the model, applying quantization if requested
    if quantization == 4:
        print("4bit quantization applied")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("No quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Checking if GPU is available
    print("Model is using:", model.device)
    if not torch.cuda.is_available():
        raise Exception("GPU not available")

    # Apply LoRA adapters
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

    try:
        dataset = load_dataset(hf_dataset_name, data_files={
        "train": train_data_path,
        "validation": dev_data_path,
        })
        train_dataset = dataset["train"].shuffle(seed=42)
        dev_dataset = dataset["validation"].shuffle(seed=42)
    except Exception as e:
        raise RuntimeError("Error loading dataset. Check your data paths.")
        

    # Setting training arguments
    max_seq_length = max_sequence_length
    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
    callbacks = [early_stopping]
    if logger is not None:
        callbacks.append(logger)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=callbacks,
        formatting_func=formatting_func,
        args=SFTConfig(
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            report_to="wandb" if wandb_key is not None else "none",
            learning_rate=learning_rate,
            lr_scheduler_type=scheduler_type,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=num_epochs,
            logging_steps=logging_steps,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            output_dir=output_dir,
            seed=0,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    # Launch training and save best model
    trainer.train()
    model.save_pretrained(final_dir)

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
    lora_dropout: float = 0,
    max_sequence_length: int =  6500,
    early_stopping_patience: int = 5,
    learning_rate: float = 5e-5,
    scheduler_type: str = "cosine",
    train_batch_size: int = 3,
    eval_batch_size: int = 2,
    grad_accum_steps: int = 3,
    num_epochs: int = 10,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    logging_steps: int = 20,
    eval_steps: int = 20,
    save_steps: int = 20,
    wandb_project: str = None,
    wandb_run: str = None
    ):
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        model_name (str): name of the model to log
        from_base (int): From Base? (0 or 1)
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub.
        train_data_path (str): Training dataset path
        dev_data_path (str): Dev dataset path
        quantization (int): Quantization: 0 if none, 4 for 4bit quantization
        lora_rank (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout rate
        max_sequence_length (int): Maximum sequence length
        early_stopping_patience (int): Early stopping patience
        learning_rate (float): Learning rate for training
        scheduler_type (str): Learning rate scheduler type
        train_batch_size (int): Training batch size
        eval_batch_size (int): Evaluation batch size
        grad_accum_steps (int): Gradient accumulation steps
        num_epochs (int): Number of training epochs
        weight_decay (float): Weight decay for optimizer
        warmup_ratio (float): Warmup ratio for learning rate schedule
        logging_steps (int): Number of steps between logging
        eval_steps (int): Number of steps between evaluations
        save_steps (int): Number of steps between model checkpoints
    """

    output_dir = '/app/local_data/checkpoints/ground'
    final_dir = '/app/local_data/weights/ground'    

    hf_token = None
    wandb_key = None
    try:    
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()
    except Exception:
        pass

    try:    
        wandb_key = project.get_secret("WANDB_KEY").read_secret_value()
    except Exception:
        pass

    train(
        model_id, 
        from_base, 
        hf_dataset_name,
        train_data_path,
        dev_data_path,
        output_dir,
        final_dir,
        wandb_project,
        wandb_run,
        quantization,
        lora_rank,
        lora_alpha,
        lora_dropout,
        max_sequence_length,
        early_stopping_patience,
        learning_rate,
        scheduler_type,
        train_batch_size,
        eval_batch_size,
        grad_accum_steps,
        num_epochs,
        weight_decay,
        warmup_ratio,
        logging_steps,
        eval_steps,
        save_steps,
        hf_token=hf_token,
        wandb_key=wandb_key,
        logger=LoggingCallback(project.get_run(os.environ['RUN_ID']))
    )

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
        "save_steps": save_steps
        
    }
    
    model = project.log_model(
        name=model_name,
        kind="huggingface",
        base_model=model_id,
        parameters=model_params,
        source=final_dir +"/",
    )      
    run = project.get_run(os.environ['RUN_ID'])
    metrics = run.status.metrics
    for k in metrics:
        model.log_metric(k, metrics[k])
    