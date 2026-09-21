# Lighting Hydra Template

This tutorial is based on the Pytorch Lightning training and validation scenario template as defined in the following [GitHub repository](https://github.com/ashleve/lightning-hydra-template). 

## Overview

In the tutorial we adopt the original project in order to execute it in the platform using the corresponding **Hydra runtime**. The original project
implements two main functionalities: train and evaluate. They are defined as the Hydra application (annotated with ``@hydra.main``) and allow for wide range of the configurations and customizations. More specifically, 

- the source code files the main functions refer to are defined in the ``src`` module of the project.
- the configurations are defined in the ``configs`` folder of the project and define the different variations of the application execution, such as hyper parameter search with Optuna, wide range of loggers, experiment configurations, MNIST model parameters, etc.
- The core logic adapts [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable) library.

While it is possible to run the different configurations directly with the Hydra runtime in the platform, for the sake of this tutorial we introduce an extra step, which allows us to perform Hyper Parameter Optimization and identify the best model configuration and checkpoints to be stored in the platform catalog.

## Adapting the Code

To perform the extra steps, without modifying the original code, we introduce a new Python script, namely ``train_dh.py``. This script is used to wrap the original train function and adds the initialization and finalizer to our project: 
the latter is used to upload the best checkpoint.

```python
import sys
import hydra
import time
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple

sys.path.append("./s14-lightning-hydra-template/")
sys.path.append("./s14-lightning-hydra-template/src/")

from src import train

def init(context) -> None:
    print("init() called")

def complete(context) -> None:
    print("complete() called")
    import optuna
    import os
    try:
        study = optuna.load_study(
            study_name="hpo", 
            storage="sqlite:///hpo.db"
        )
        best_run = study.best_trial
        print(f"Best Trial: #{best_run.number} with Value: {best_run}")
        p = f"/shared/hpo_results/{best_run.number}/checkpoints/last.ckpt"
        print(f"Checkpoint: {p}, exists: {os.path.isfile(p)}")

    except  Exception as ex: 
        print("loading best model failed")
    print("complete() finished")
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    return train.main(cfg)
```

As you can see from the code, 

- the ``main()`` function simply wraps the original ``train()`` function.
- the ``init()`` function does not do anything particular. The experiment setup and storage will be configured by Hydra overrides. 
- the ``complete()`` function is used to identify the best checkpoint so it can be uploaded to the platform.  It loads the stody from the optuna storage, identifies the best run of the study, prints the parameters and the checkpoint path. 

Please note some relevant considerations to take into account:

- when the GitHub project is cloned, the execution starts from the project root, so the  modules in subpaths should be explicitly added to the system path.
- when executed in the platform, Hydra runtime creates and mounts the ReadWriteMAny persistent volume available to all the subtasks under ``/shared`` path. So as in this case, if the results of the multirun are stored in a subfolder of this shared volume, the ``complete`` operation can access them. 
 
## Executing the Hydra job within the platform 

Once the code is ready, we can execute it in the platform. As all the operations of the platform, we start from defining the context of our experiments and executions, or **project**. Project is a logical container for data, artifacts, models, executable operations, and their executions.

### Define the context

It is possible to create the project via platform UI or programmatically using the platform SDK:

```python

import digitalhub as dh

project = dh.get_or_create_project("hydra-example")
```

### Register function

Next, we need to describe and register our Hydra training function. Our code is pure Hydra Job application, so we can use `hydra` runtime for this purpose. Again, it is possible to do it via UI or programmatically:

```python
func = project.new_function(
    "lightning-hpo",
    kind="hydra",
    code_src="git+https://github.com/scc-digitalhub/digitalhub-tutorials#0.16.x",
    handler="s14-lightning-hydra-template.src.train_dh:main",
    init_function="init",
    complete_function="complete",
    config_path="s14-lightning-hydra-template/configs",
    python_version="PYTHON3_13",
    requirements=[
        "torch>=2.0.0", "torchvision>=0.15.0", "lightning>=2.0.0", "torchmetrics>=0.11.4", "hydra-colorlog==1.2.0", "hydra-optuna-sweeper==1.2.0", "SQLAlchemy==1.4.46", "rootutils"
    ]
)
```

Note the reference to the git repository where the training function is located: by default it will point out to the **current** version of the code in the ``0.16.x`` branch. So each time the function is executed, the latest version of the code at this branch will be used.

If the code is at the private repository, it is possile to [configure the credentials](https://scc-digitalhub.github.io/docs/tasks/code-source/#remote-git-repository) to access it, using environment variables or [secrets](https://scc-digitalhub.github.io/docs/tasks/secrets/), such as, e.g, `GITHUB_TOKEN`. 

The entry point is defined as `handler` attribute composed of python path to the containing module and the function name. The ``config_path`` define the relative path to the folder containg the Hydra configuration files.

The dependencies are listed explicitly with their versions. A specific version of the SQLAlchemy is required as it is used by the Hydra Optuna Sweeper storage support.

The Hydra runtime allows for executing the code either `locally` or `remotely` (controlled by local_execution=True/False when `function.run` method is called).  Local execution means the code will be downloaded and executed in the same space where SDK is being called. This may be your PC or an interactive workspace like Jupyter notebook. Provided the dependencies are already installed (and GPU is available if necessary), the SDK will download the repo and call the function directly. This may work for testing purposes, but may be not practical for heavy and long-running tasks. In this case we should use remote execution.

In case of remote execution, the execution takes place in the computational cluster behind the platform. The SDK will call the platform API to trigger this execution, and the plaform will instantiate a **Kubernetes Job**, that will be scheduled on the cluster and will download the code and execute it with the parameters and configuration specified. For this to happen the underlying function container image should be created.

### Building a container image

The build operation may be triggered by the UI or programmatically:

```python

func.run(action="build")
```

This will result in the container image being created within the platform cluster, the image will be published in the internal image registry, and the function will be associated with the image. If the list of the dependencies changes, the container image should be rebuild. The images with the same dependencies and base image may be reused across different functions and executions.

If the build operation is not performed, the each new execution will require the dependencies to be installed from scratch. Note that in some enivroments the platform may be configured to skip the custom dependency installation for performance purposes, in which case the execution will fail.

### Executing the function

The function can be executed via UI or programmatically:

```python
run = func.run(
    action="job",
    parameters={
        "trainer.max_epochs": 1,
        "hparams_search": "mnist_optuna",
        "hydra.sweeper.storage": "sqlite:///hpo.db",
        "hydra.sweeper.study_name": "hpo",
        "hydra.sweep.dir": "./hpo_results"
    },
    workers=5,
    local_execution=False
)
```

As you can see, here we run the job with 5 workers available for parallel execution, and a set of custom parameters to override the default configuration and use HPO with Optuna, custom sweeper storage and study name, and a custom directory to store the results (relative to the project root at `/shared` folder).

Once the execution terminates successfully, the resulting information is logged:

``` log
complete() called 
Best Trial: #9 with Value: FrozenTrial(number=9, values=[0.9696000218391418], datetime_start=datetime.datetime(2026, 9, 20, 18, 52, 15, 767308), datetime_complete=datetime.datetime(2026, 9, 20, 19, 5, 31, 18455), params={'data.batch_size': 32, 'model.net.lin1_size': 256, 'model.net.lin2_size': 64, 'model.net.lin3_size': 32, 'model.optimizer.lr': 0.013488718479284397}, distributions={'data.batch_size': CategoricalDistribution(choices=(32, 64, 128, 256)), 'model.net.lin1_size': CategoricalDistribution(choices=(64, 128, 256)), 'model.net.lin2_size': CategoricalDistribution(choices=(64, 128, 256)), 'model.net.lin3_size': CategoricalDistribution(choices=(32, 64, 128, 256)), 'model.optimizer.lr': UniformDistribution(high=0.1, low=0.0001)}, user_attrs={'hparams_search': 'mnist_optuna', 'trainer.max_epochs': '1'}, system_attrs={}, intermediate_values={}, trial_id=10, state=TrialState.COMPLETE, value=None) 
Checkpoint: /shared/hpo_results/9/checkpoints/last.ckpt, exists: True 
complete() finished 
```

It is possible to further extend this scenario to, e.g., upload the best model to the model registry, to perform some post-processing of the results, or to add custom logging callback support to log execution metrics for single runs.
