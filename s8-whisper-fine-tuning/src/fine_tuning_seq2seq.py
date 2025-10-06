"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


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


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to read preprocessed dataset"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default="text",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
        },
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(
        self, features: list[dict[str, Union[list[int], torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor(
                [feature["attention_mask"] for feature in features]
            )

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main(callback=None, args=None):
    if args is None:
        args = sys.argv
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )

    if len(args) == 2 and args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(args[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = DatasetDict()
    preprocessed = False
    if data_args.data_dir is not None:
        raw_datasets = DatasetDict.load_from_disk(data_args.data_dir)
        preprocessed = True
    else:
        if training_args.do_train:
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.train_split_name,
                cache_dir=model_args.cache_dir,
                trust_remote_code=True,
            )

        if training_args.do_eval:
            raw_datasets["eval"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.eval_split_name,
                cache_dir=model_args.cache_dir,
                trust_remote_code=True,
            )

    if (
        data_args.audio_column_name
        not in next(iter(raw_datasets.values())).column_names
    ):
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": None})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if (
        hasattr(model.generation_config, "is_multilingual")
        and model.generation_config.is_multilingual
    ):
        # We only need to set the language and task ids in a multilingual setting
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.generation_config.language = data_args.language
        model.generation_config.task = data_args.task
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 6. Resample speech dataset if necessary
    dataset_sampling_rate = (
        next(iter(raw_datasets.values()))
        .features[data_args.audio_column_name]
        .sampling_rate
    )
    if not preprocessed:
        if dataset_sampling_rate != feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                data_args.audio_column_name,
                datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
            )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = (
        data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    )
    min_input_length = (
        data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(data_args.max_train_samples)
        )

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(
            range(data_args.max_eval_samples)
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask,
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = (
            batch[text_column_name].lower()
            if do_lower_case
            else batch[text_column_name]
        )
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    if preprocessed:
        vectorized_datasets = raw_datasets
    else:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=next(iter(raw_datasets.values())).column_names,
                num_proc=data_args.preprocessing_num_workers,
                desc="preprocess train dataset",
            )

        # filter data that is shorter than min_input_length or longer than
        # max_input_length
        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        vectorized_datasets.save_to_disk(training_args.output_dir)
        return

    # 8. Load Metric
    metric = evaluate.load("wer", cache_dir=model_args.cache_dir)

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )
    callbacks = [callback] if callback is not None else None
    # 11. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        callbacks=callbacks,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets["train"])
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(
            max_eval_samples, len(vectorized_datasets["eval"])
        )

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        results["metrics"] = metrics

    # 14. Write Training Stats
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    results.update(kwargs)
    return results


def train_and_log_model(
    project,
    model_name: str,
    model_id: str,
    dataset_name: str,
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
    max_steps: int = 5000,
    max_train_samples: int = None,
    max_eval_samples: int = None,
):
    """
    Train the Whisper model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        model_name (str): name of the model to log
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        artifact_name (str): Name of the artifact to use for the data being saved to cache dir. If not specified, data is loaded from HuggingFace dataset
        language_code (str): Language (2-letter ISO code)
        language (str): Language to use
        eval_steps (int): Number of steps between evaluations
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
        max_train_samples (int): Number of entries to consider for training (for debugging)
        max_eval_samples (int): Number of entries to consider for evaluation (for debugging)

    """

    output_dir = "/shared/data/model"
    cache_dir = "/shared/data/cache"
    # final_dir = '/shared/data/final'
    # data_dir = '/shared/data/dataset'

    if artifact_name is not None:
        project.get_artifact(artifact_name).download(cache_dir)

    hf_token = None
    try:
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()
    except Exception:
        pass

    args = [
        f"--model_name_or_path={model_id}",
        f"--dataset_name={dataset_name}",
        f"--dataset_config_name={language_code}",
        f"--language={language}",
        f"--task=transcribe",
        f"--train_split_name=train+validation",
        f"--eval_split_name=test",
        f"--max_steps={max_steps}",
        f"--output_dir={output_dir}",
        f"--per_device_train_batch_size={train_batch_size}",
        f"--logging_steps={logging_steps}",
        f"--learning_rate={learning_rate}",
        f"--warmup_steps={warmup_steps}",
        f"--eval_strategy=steps",
        f"--eval_steps={eval_steps}",
        f"--save_strategy=steps",
        f"--save_steps={save_steps}",
        f"--save_total_limit=1",
        f"--generation_max_length={max_sequence_length}",
        f"--preprocessing_num_workers=16",
        f"--max_duration_in_seconds=30",
        f"--text_column_name=sentence",
        f"--cache_dir={cache_dir}",
        f"--gradient_checkpointing",
        f"--fp16",
        f"--overwrite_output_dir",
        f"--do_train",
        f"--do_eval",
        f"--predict_with_generate",
    ]
    if max_train_samples is not None:
        args.append(f"--max_train_samples={max_train_samples}")
    if max_eval_samples is not None:
        args.append(f"--max_eval_samples={max_eval_samples}")

    result = main(
        callback=LoggingCallback(project.get_run(os.environ["RUN_ID"])), args=args
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
        "save_steps": save_steps,
    }

    model = project.log_model(
        name=model_name,
        kind="huggingface",
        base_model=model_id,
        parameters=model_params,
        source=output_dir + "/",
    )
    run = project.get_run(os.environ["RUN_ID"])
    metrics = run.status.metrics
    for k in metrics:
        model.log_metric(k, metrics[k])


def preprocess_dataset(
    project,
    artifact_name: str,
    model_id: str,
    dataset_name: str,
    language_code: str = "it",
    language: str = "Italian",
    max_train_samples: int = None,
    max_eval_samples: int = None,
):
    """
    Preprocess dataset and model and store as artifact.

    Args:
        model_id (str): Model ID
        artifact_name (str): name of the artifact to log
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        language_code (str): Language (2-letter ISO code)
        language (str): Language to use
        max_train_samples (int): Number of entries to consider for training (for debugging)
        max_eval_samples (int): Number of entries to consider for evaluation (for debugging)
    """

    output_dir = "/shared/data/model"
    cache_dir = "/shared/data/cache"
    # final_dir = '/shared/data/weights/ground'
    # data_dir = '/shared/data/dataset'

    hf_token = None
    try:
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()
    except Exception:
        pass

    args = [
        f"--model_name_or_path={model_id}",
        f"--dataset_name={dataset_name}",
        f"--dataset_config_name={language_code}",
        f"--language={language}",
        f"--task=transcribe",
        f"--train_split_name=train+validation",
        f"--eval_split_name=test",
        f"--output_dir={output_dir}",
        f"--preprocessing_num_workers=16",
        f"--max_duration_in_seconds=30",
        f"--text_column_name=sentence",
        f"--cache_dir={cache_dir}",
        f"--overwrite_output_dir",
        f"--preprocessing_only",
        f"--do_train",
        f"--do_eval",
    ]
    if max_train_samples is not None:
        args.append(f"--max_train_samples={max_train_samples}")
    if max_eval_samples is not None:
        args.append(f"--max_eval_samples={max_eval_samples}")

    main(args=args)

    artifact = project.log_artifact(
        name=artifact_name,
        kind="artifact",
        framework="whisper",
        source=output_dir + "/",
    )


if __name__ == "__main__":
    main()
