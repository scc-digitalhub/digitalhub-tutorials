import sys
import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple

sys.path.append("./s14-lightning-hydra-template/")
sys.path.append("./s14-lightning-hydra-template/src/")

from src import simple

def init(context) -> None:
    print("init() called")
    time.sleep(1)
    print("init() finished")

def complete(context) -> None:
    print("complete() called")
    time.sleep(1)
    print("complete() finished")
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    simple.main(cfg)
