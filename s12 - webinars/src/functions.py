import functools
import operator
import os

from datasets import load_dataset
from digitalhub_runtime_python import handler
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@handler()
def train(project):
    model_id = "distilbert/distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("imdb")

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    )

    chunk_size = 128

    def group_texts(examples):
        concatenated_examples = {k: functools.reduce(operator.iadd, examples[k], []) for k in examples}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        total_length = (total_length // chunk_size) * chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    train_size = 10_000
    test_size = int(0.1 * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    batch_size = 64
    logging_steps = len(downsampled_dataset["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir=f"{model_id}-finetuned-imdb",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps=logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    save_dir = "model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    project.log_huggingface(
        name="test_llm_model",
        base_model=model_id,
        source=save_dir,
    )
