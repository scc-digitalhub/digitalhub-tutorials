from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import os
import huggingface_hub

from dataclasses import dataclass
from typing import Any, Dict, List, Union
            
def create_dataset(
    model_id: str,
    hf_dataset_name: str,
    language_code: str,
    language: str,
    output_dir: str,
    ratio: int,
    hf_token: str = None,
) -> None:
    """
    Prepare the Whisper train dataset.

    Args:
        model_id (str): Model ID
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub.
        language_code (str): Language (2-letter ISO code)
        language (str): Language to use 
        output_dir (str): Output directory for dataset
        hf_token (str): Hugging Face API token. Required only for private models; not needed when using public models.
    """
    # Logging in Hugging Face and WandB
    if hf_token is not None:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            raise RuntimeError("Error logging into Hugging Face. Check your token.")

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
    common_voice["train"] = load_dataset(hf_dataset_name, language_code, split=f"train[:{ratio}%]+validation[:{ratio}%]", trust_remote_code=True)
    common_voice["test"] = load_dataset(hf_dataset_name, language_code, split=f"test[:{ratio}%]", trust_remote_code=True)
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
    common_voice.save_to_disk(output_dir)


def create_and_log_dataset(
    project,
    dataset_name: str,
    model_id: str, 
    hf_dataset_name: str,
    language_code: str = "it",
    language: str = "Italian",
    ratio: int = 10,
    ):
    """
    Train the LLM model with the given dataset and configuration.

    Args:
        model_id (str): Model ID
        dataset_name (str): name of the artifact to log
        hf_dataset_name (str): Name of the dataset on Hugging Face Hub.
        language_code (str): Language (2-letter ISO code)
        language (str): Language to use 
        ratio (int): percentage of the dataset to consider (default 10%)
    """

    output_dir = '/shared/data/dataset'

    hf_token = None
    wandb_key = None
    try:    
        hf_token = project.get_secret("HF_TOKEN").read_secret_value()
    except Exception:
        pass

    create_dataset(
        model_id, 
        hf_dataset_name,
        language_code,
        language,
        output_dir,
        hf_token=hf_token
    )

    
    project.log_artifact(
        name=dataset_name,
        kind="artifact",
        source=output_dir
    )      
