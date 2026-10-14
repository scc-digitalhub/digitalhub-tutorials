# LLM Fine-Tuning Pipeline with DigitalHub & QLoRA

This tutorial shows how to fine-tune large language models (such as Llama 3.1) using DigitalHub, Hugging Face TRL (`SFTTrainer`), and PEFT (QLoRA). It demonstrates how to package the training code into a DigitalHub Python function, build the container image, attach persistent storage, and execute the fine-tuning run on a GPU.

## Jupyter Notebook

Open the notebook [train-llm.ipynb](train-llm.ipynb) in your environment and run it step by step:

1. **Initialize the Project**: Connect to or create the DigitalHub project (`llm-finetunning`).
2. **Configure Prerequisites**: Set up your `HF_TOKEN` (and optionally `WANDB_KEY`) as DigitalHub project secrets.
3. **Deploy Training Function**: Package [train.py](train.py) with [requirements.txt](requirements.txt) as a Python job and trigger the build.
4. **Perform Training**: Launch the training job specifying GPU profile (`1xrtx5000`), model parameters, and persistent storage volumes.

---

## Repository Structure

- [train-llm.ipynb](train-llm.ipynb): Interactive orchestration notebook for project setup, building the function image, submitting fine-tuning jobs, and serving.
- [train.py](train.py): Core training logic supporting 4-bit QLoRA, Llama-3 chat formatting, dataset loading, WandB logging, and model saving.
- [serve.py](serve.py): DigitalHub serving script implementing `init_context` (loads base model + LoRA adapters) and `serve_predictions` (REST inference handler).
- [requirements.txt](requirements.txt): Runtime dependencies for PyTorch, Transformers, TRL, PEFT, and BitsAndBytes.

---

## Prerequisites

1. **DigitalHub Platform**: Access to a DigitalHub environment with GPU runtime profiles (e.g., `1xrtx5000`).
2. **Hugging Face Token**:
   - Generate an access token on [Hugging Face](https://huggingface.co/settings/tokens) with read access to the target model (e.g. `meta-llama/Llama-3.1-8B`).
   - Add it to your DigitalHub project secrets as `HF_TOKEN`.
3. *(Optional)* **Weights & Biases**:
   - Add your WandB key as a project secret named `WANDB_KEY` if metric tracking is desired.

---

## Training Script Configuration

The entrypoint function `train_and_log_model` in [train.py](train.py) accepts the following parameters:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `model_id` | *Required* | Base model repo identifier on Hugging Face (e.g. `meta-llama/Llama-3.1-8B`) |
| `hf_dataset_name` | *Required* | Hugging Face dataset identifier (e.g. `team-bay/data-science-qa`) |
| `train_data_path` | *Required* | Training split data file / path |
| `dev_data_path` | *Required* | Validation split data file / path |
| `quantization` | `4` | `4` for 4-bit NF4 QLoRA, `0` for fp16 without quantization |
| `lora_rank` | `16` | LoRA rank dimension ($r$) |
| `lora_alpha` | `16` | LoRA scaling factor ($\alpha$) |
| `lora_dropout` | `0.05` | LoRA dropout probability |
| `learning_rate` | `2e-4` | Peak learning rate for cosine scheduler |
| `num_epochs` | `3` | Number of training epochs |
| `train_batch_size` | `1` | Per-device training batch size |
| `grad_accum_steps` | `4` | Gradient accumulation steps |
| `max_sequence_length` | `2048` | Maximum token sequence length |

---

## Outputs & Artifacts

- **Checkpoints**: Stored during training under `/app/local_data/checkpoints/ground`.
- **Final Weights**: Saved adapter weights and tokenizer files written to `/app/local_data/weights/ground`.

---

## 4. Serving the Model (REST API)

Following the DigitalHub model serving pattern (with `init_context` and `serve_predictions` handlers), deploy [serve.py](serve.py) as a serving service:

```python
# 1. Define the serving function
serving_func = proj.new_function(
    name="serve-llm",
    kind="python",
    python_version="PYTHON3_11",
    code_src="serve.py",
    init_function="init_context",
    handler="serve_predictions",
    requirements="requirements.txt"
)

# 2. Build the serving container image
serving_func.run(action="build")

# 3. Deploy the service on GPU with persistent volume and secrets
serve_run = serving_func.run(
    action="serve",
    profile="1xrtx5000",
    secrets=["HF_TOKEN"],
    envs=[
        {"name": "HF_HOME", "value": "/app/local_data/huggingface"},
        {"name": "TRANSFORMERS_CACHE", "value": "/app/local_data/huggingface"},
        {"name": "MODEL_NAME", "value": "llmpa-tracked"},
        {"name": "BASE_MODEL_ID", "value": "meta-llama/Llama-3.1-8B"},
        {"name": "QUANTIZATION", "value": "4"}
    ],
    volumes=[{
        "volume_type": "persistent_volume_claim",
        "name": "volume-llmpa",
        "mount_path": "/app/local_data",
        "spec": {"size": "20Gi"}
    }]
)
```

### Inference Request Example

Once deployed, send POST requests to the serving endpoint:

```json
{
  "prompt": "Explain gradient descent in machine learning.",
  "system_prompt": "You are an expert AI assistant.",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response format:
```json
{
  "generated_text": "Gradient descent is an optimization algorithm used to minimize...",
  "model": "meta-llama/Llama-3.1-8B"
}
```

---

## Dependency Notes

The packages in [requirements.txt](requirements.txt) are pinned to ensure compatibility between PyTorch, Pandas, and PyArrow on Python 3.11:
- `pandas==2.1.4` and `numpy==1.26.4`
- `pyarrow<18.0.0` (prevents binary incompatibility with NumPy 1.x)

