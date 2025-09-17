from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
import os
import huggingface_hub

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate

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
                
class DataCollatorSpeechSeq2SeqWithPadding:

    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
        
def train(
    model_id: str,
    hf_dataset_name: str,
    language_code: str,
    language: str,
    data_dir: str,
    output_dir: str,
    final_dir: str,
    max_sequence_length: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum_steps: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    warmup_steps: int,
    max_steps: int,
    hf_token: str = None,
    logger: TrainerCallback = None
) -> None:
    """
    Train the Whisper model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        language_code (str): Language (2-letter ISO code)
        language (str): Language to use 
        output_dir (str): Output directory for model and checkpoints
        final_dir (str): Output directory for final model
        data_dir (str): directory with data
        max_sequence_length (int): Maximum sequence length
        early_stopping_patience (int): Early stopping patience
        learning_rate (float): Learning rate for training
        train_batch_size (int): Training batch size
        eval_batch_size (int): Evaluation batch size
        grad_accum_steps (int): Gradient accumulation steps
        logging_steps (int): Number of steps between logging
        eval_steps (int): Number of steps between evaluations
        save_steps (int): Number of steps between model checkpoints
        warmup_steps (int): Number of warmpup steps
        max_steps (int): Max number of steps
        hf_token (str): Hugging Face API token. Required only for private models; not needed when using public models.
    """
    # Logging in Hugging Face and WandB
    if hf_token is not None:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            raise RuntimeError("Error logging into Hugging Face. Check your token.")

    tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language, task="transcribe")            
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task="transcribe")

    common_voice = DatasetDict.load_from_disk(data_dir)

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
    
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
    
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        generation_max_length=max_sequence_length,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False
    )

    callbacks = [logger] if logger is not None else None

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    # Launch training and save best model
    trainer.train()
    model.save_pretrained(final_dir)

def train_and_log_model(
    project,
    model_name: str,
    model_id: str, 
    hf_dataset_name: str = None,
    artifact_name: str = None,
    language_code: str = "it",
    language: str = "Italian",
    max_sequence_length: int = 225,
    learning_rate: float = 1e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 8,
    grad_accum_steps: int = 1,
    logging_steps: int = 25,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    warmup_steps: int = 500,
    max_steps: int = 5000
    ):
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        model_name (str): name of the model to log
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub. If not specified, data is loaded from artifact
        artifact_name (str): Name of the artifact to use for the data. If not specified, data is loaded from HuggingFace dataset
        language_code (str): Language (2-letter ISO code)
        language (str): Language to use 
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

    output_dir = '/shared/data/checkpoints/ground'
    final_dir = '/shared/data/weights/ground'   
    data_dir = '/shared/data/dataset'

    hf_token = None
    wandb_key = None
    try:    
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()
    except Exception:
        pass

    if artifact_name is not None:
        project.get_artifact(artifact_name).download(data_dir)
        
    elif hf_dataset_name is not None:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language, task="transcribe")  
        
        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]
            # compute log-Mel input features from input audio array 
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            # encode target text to label ids 
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            return batch
    
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(hf_dataset_name, language_code, split=f"train+validation", trust_remote_code=True)
        common_voice["test"] = load_dataset(hf_dataset_name, language_code, split=f"test", trust_remote_code=True)
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
        common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
        common_voice.save_to_disk(data_dir)
        
    train(
        model_id, 
        hf_dataset_name,
        language_code,
        language,
        data_dir,
        output_dir,
        final_dir,
        max_sequence_length,
        learning_rate,
        train_batch_size,
        eval_batch_size,
        grad_accum_steps,
        logging_steps,
        eval_steps,
        save_steps,
        warmup_steps,
        max_steps,
        hf_token=hf_token,
        logger=LoggingCallback(project.get_run(os.environ['RUN_ID']))
    )

    model_params = {
        "max_sequence_length": max_sequence_length,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
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