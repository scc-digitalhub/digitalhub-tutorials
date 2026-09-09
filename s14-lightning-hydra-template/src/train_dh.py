import sys


sys.path.append("./s14-lightning-hydra-template/src/")

import train

def init(context) -> None:
    print("init() called")
    time.sleep(1)
    print("init() finished")

def complete(context) -> None:
    print("complete() called")
    time.sleep(1)
    print("complete() finished")
    
def main():
    train.main()
