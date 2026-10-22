import sys
import time

import hydra
from omegaconf import DictConfig

sys.path.append("./s14-lightning-hydra-template/")
sys.path.append("./s14-lightning-hydra-template/src/")

from src import train


def init(context) -> None:
    print("init() called")
    time.sleep(1)
    print("init() finished")


def complete(context) -> None:
    print("complete() called")
    import os

    import optuna

    try:
        study = optuna.load_study(study_name="hpo", storage="sqlite:///hpo.db")
        best_run = study.best_trial
        print(f"Best Trial: #{best_run.number} with Value: {best_run}")
        p = f"/shared/hpo_results/{best_run.number}/checkpoints/last.ckpt"
        print(f"Checkpoint: {p}, exists: {os.path.isfile(p)}")

    except Exception:
        print("loading best model failed")
    print("complete() finished")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:

    print("Bypass http proxy")
    import os

    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["no_proxy"] = "*"

    return train.main(cfg)
