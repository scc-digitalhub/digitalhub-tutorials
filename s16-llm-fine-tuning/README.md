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

Deploy the trained model with the DigitalHub `vllmserve-text` runtime. This example serves a QLoRA adapter alongside its base model: `base_model` and `model` must refer to DigitalHub storage objects containing the base model and the adapter saved under `/app/local_data/weights/ground`, respectively. Replace these references and the base model name with the values for your run. The adapter is registered as `adapter`; use that name in inference requests.

```python
# 1. Define the vLLM text-serving runtime function.
#    `base_model` and `model` are DigitalHub storage objects for this run.
vllm_func = proj.new_function(
    name="vllm-text-serve",
    kind="vllmserve-text",
    model_name="base-model",
    url=base_model.key,
    adapters=[{"name": "adapter", "url": model.key}],
)

# 2. Deploy on a GPU. Include HF_TOKEN if the base model requires access.
serve_run = vllm_func.run(
    action="serve",
    profile="1xrtxa5000-shared",
    secrets=["HF_TOKEN"],
    args=["--gpu_memory_utilization", "0.95", "--max_model_len", "32768"],
    resources={"mem": "24Gi", "disk": "20Gi"},
    wait=True,
)
```

### Send an Inference Request

After deployment, use the service endpoint shown in DigitalHub as `SERVING_ENDPOINT` and send a chat completion request to the OpenAI-compatible vLLM API. For more details, see the [notebook](train-llm.ipynb):

```python
import requests

response = requests.post(
    f"{SERVING_ENDPOINT.rstrip('/')}/v1/chat/completions",
    json={
        "model": "adapter",
        "messages": [{"role": "user", "content": "What is fine-tuning?"}],
        "max_tokens": 128,
    },
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```


---

## Dependency Notes

The packages in [requirements.txt](requirements.txt) are pinned to ensure compatibility between PyTorch, Pandas, and PyArrow on Python 3.11:
- `pandas==2.1.4` and `numpy==1.26.4`
- `pyarrow<18.0.0` (prevents binary incompatibility with NumPy 1.x)

