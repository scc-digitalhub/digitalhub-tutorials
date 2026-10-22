import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def init_context(context):
    """
    Initialize serving context by loading base model and fine-tuned LoRA adapters.
    DigitalHub Nuclio serving injects `context` which includes project and logger access.
    """
    context.logger.info("Initializing serving context for HuggingFace model...")

    # Read configuration from context/environment
    # The model name or key can be passed via parameters or env var MODEL_NAME
    model_name = os.environ.get("MODEL_NAME", "llmpa-tracked")
    base_model_id = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.1-8B")
    quantization = int(os.environ.get("QUANTIZATION", "4"))
    local_weights_dir = os.environ.get(
        "LOCAL_WEIGHTS_DIR", "/app/local_data/weights/ground"
    )

    adapter_path = local_weights_dir

    # Check if weights exist locally in mounted volume; otherwise download via DigitalHub
    if not os.path.exists(adapter_path) or not os.listdir(adapter_path):
        context.logger.info(
            f"Local weights not found at {adapter_path}. Downloading model artifact '{model_name}' from DigitalHub..."
        )
        model_artifact = context.project.get_model(model_name)
        adapter_path = model_artifact.download()

    context.logger.info(f"Loading adapter and tokenizer from: {adapter_path}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Configure Quantization (NF4 4-bit by default for efficient inference)
    quantization_config = None
    if quantization == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # 3. Load Base Model
    context.logger.info(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )

    # 4. Attach Fine-Tuned LoRA Adapters
    context.logger.info("Applying LoRA adapter to base model...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Store model and tokenizer in the serving context
    context.model = model
    context.tokenizer = tokenizer
    context.logger.info("Serving context initialization complete.")


def serve_predictions(context, event):
    """
    Serve text generation / inference requests via REST API.
    Expected request body JSON structure:
    {
        "prompt": "What is overfitting?",
        "system_prompt": "You are a helpful assistant.",  # optional
        "max_new_tokens": 256,                             # optional
        "temperature": 0.7,                               # optional
        "top_p": 0.9                                      # optional
    }
    """
    if isinstance(event.body, (bytes, bytearray)):
        body = json.loads(event.body.decode("utf-8"))
    elif isinstance(event.body, str):
        body = json.loads(event.body)
    else:
        body = event.body

    context.logger.info(f"Received inference request: {body}")

    prompt = body.get("prompt")
    if not prompt:
        return {"error": "Missing 'prompt' in request body"}

    system_prompt = body.get("system_prompt", "You are a helpful assistant.")
    max_new_tokens = int(body.get("max_new_tokens", 256))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.9))
    do_sample = temperature > 0.0

    # Format prompt using the Llama-3 instruction format used during training
    formatted_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    tokenizer = context.tokenizer
    model = context.model

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False

        output_tokens = model.generate(**inputs, **generate_kwargs)

    # Slice output tokens to remove prompt tokens
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = output_tokens[0][input_len:]
    generated_text = tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    ).strip()

    return {
        "generated_text": generated_text,
        "model": getattr(model, "name_or_path", "hf-lora-model"),
    }
