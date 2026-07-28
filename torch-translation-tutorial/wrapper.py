import sys
import traceback
sys.path.append("./torch-translation-tutorial/")

# WORKAROUND for SpaCy language model download. To use SpaCy models,
# the model must be downloaded and installed before the training script. 
# TO make it dynamic, we call the spacy.cli.download() function to download the model 
# to a specific location, and then import it from that location.
def ensure_lang_model(model_name: str):
    from importlib import import_module
    from spacy.cli import download
    import spacy

    OLD_MODEL_SHORTCUTS = (
                        spacy.errors.OLD_MODEL_SHORTCUTS if hasattr(spacy.errors, "OLD_MODEL_SHORTCUTS") else {}
                    )
    
    model_name = OLD_MODEL_SHORTCUTS[model_name] if model_name in OLD_MODEL_SHORTCUTS else model_name
    download(model_name, False, False, None, "-t", "/shared/language_models/")    
    sys.path.append("/shared/language_models/")
    model_module = import_module(model_name)
    model_module.load()
    
from main import main
def train(
   project,
   run,
   training_data,
   validation_data,
   src_lang="de",
   tgt_lang="en",
   epochs=30,     
   lr=1e-4,
   batch_size=128,
   backend="cpu",
   attn_heads=8,
   enc_layers=5,
   dec_layers=5,
   embed_size=512,
   dim_feedforward=512,
   dropout=0.1,
   model_name="translator-model",
):
    print("Running training script...")
    opts = type("Namespace", (), {})()
    setattr(opts, "src", src_lang)
    setattr(opts, "tgt", tgt_lang)
    setattr(opts, "epochs", epochs)
    setattr(opts, "lr", lr)
    setattr(opts, "batch", batch_size)
    setattr(opts, "backend", backend)
    setattr(opts, "attn_heads", attn_heads)
    setattr(opts, "enc_layers", enc_layers)
    setattr(opts, "dec_layers", dec_layers)    
    setattr(opts, "embed_size", embed_size)
    setattr(opts, "dim_feedforward", dim_feedforward)
    setattr(opts, "dropout", dropout)
    
    model_dir = "./data/output/"
    # fixed logging dir
    setattr(opts, "logging_dir", model_dir)
    
    try:
        ensure_lang_model(src_lang)
        ensure_lang_model(tgt_lang)
    except Exception as e:
        print(f"Error downloading language models: {e}")
        print(traceback.format_exc())
        raise e

    training_data.download("./data/input/training.tar.gz", overwrite=True)
    validation_data.download("./data/input/validation.tar.gz", overwrite=True)
    setattr(opts, "train_file", "./data/input/training.tar.gz")
    setattr(opts, "valid_file", "./data/input/validation.tar.gz")

    setattr(opts, "dry_run", False)

    setattr(opts, "run", run)

    try:
        metrics = main(opts)
        print(f"Model metrics:{metrics}")
    except Exception as e:
        print(f"Error running training script: {e}")
        print(traceback.format_exc())
        raise e

    parameters = {
        "src": src_lang,
        "tgt": tgt_lang,
        "epochs": epochs,
        "lr": lr,
        "batch": batch_size,
        "backend": backend,
        "attn_heads": attn_heads,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "embed_size": embed_size,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout
    }

    # log model
    model_artifact = project.log_model(
        name=model_name,
        kind="model",
        framework="pytorch",
        source=model_dir + "best.pt",
        parameters=parameters,
    )
    model_artifact.log_metrics(metrics)
