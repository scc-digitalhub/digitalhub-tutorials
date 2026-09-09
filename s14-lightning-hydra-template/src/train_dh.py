import sys
import hydra

sys.path.append("./s14-lightning-hydra-template/")
sys.path.append("./s14-lightning-hydra-template/src/")

from src import train

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
    train.main(cfg)
