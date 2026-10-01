# Ray MNIST Example

This tutorial is based on the Ray tutorial that creates a resnet model based on MNIST dataset using Ray Train library for distributed training. 

While it is possible to run the original code as is, in the following we show how to adapt the original code in order to leverage the the platform capabilities for better management of data, for experiment tracing, and for artifact management.  

## Overview

In the tutorial we adopt the original project in order to execute it in the platform using the corresponding **Ray runtime**. The original project implements the training procedure in a single python script. It downloads the standard MNIST dataset, configures the scalability config for parallel/distributed training, and logs checkpoints to some preconfigured storage (e.g., S3 storage).

We adopt the code in order to provide the experiment tracking and model artifact management capabilities. When the code is executed with the platform using the **Ray runtime** a corresponding cluster is created to distribute the work.

## Adapting the Code

The changes in the code presented in src/ray_train.py`` script are the following:

- introduce an explicit entry point (handler function) that wraps the original ``main()`` function and allows for passing the hyper parameters explicitly and to provide the references to the project and run of the experiment as registered in the platform. 
- add the code to report metrics on each epoch to the platform
- store checkpoints to the platform datalake under the current project subpath
- store the resulting model as a model artifact in the platform with the produced metrics and hyper parameters configuration for further comparison. 
 
## Executing the Ray job within the platform 

Once the code is ready, we can execute it in the platform. As all the operations of the platform, we start from defining the context of our experiments and executions, or **project**. Project is a logical container for data, artifacts, models, executable operations, and their executions.

### Define the context

It is possible to create the project via platform UI or programmatically using the platform SDK:

```python

import digitalhub as dh

project = dh.get_or_create_project("ray-example")
```

### Register function

Next, we need to describe and register our Hydra training function. Our code is pure Hydra Job application, so we can use `hydra` runtime for this purpose. Again, it is possible to do it via UI or programmatically:

```python
func = project.new_function(
    name="ray-training",
    kind="ray",
    requirements=["torch==2.2.2", "torchvision==0.17.2", "numpy==1.24.1"],
    code_src="src/ray_train.py",
    handler="ray_handler"
)
```

The entry point is defined as `handler` attribute pointing to the function name. 
Code source is read fron the specified python script.

The dependencies are listed explicitly with their versions. 


### Building a container image

A good practive is to build a container image for the function so that the dependencies are not downloaded each time the function is executed.

The build operation may be triggered by the UI or programmatically:

```python

func.run(action="build")
```

This will result in the container image being created within the platform cluster, the image will be published in the internal image registry, and the function will be associated with the image. If the list of the dependencies changes, the container image should be rebuild. The images with the same dependencies and base image may be reused across different functions and executions.

If the build operation is not performed, the each new execution will require the dependencies to be installed from scratch. Note that in some enivroments the platform may be configured to skip the custom dependency installation for performance purposes, in which case the execution will fail.

### Executing the function

The function can be executed via UI or programmatically:

```python
train_run = func.run(
    action="job",
    replicas=2,
    min_replicas=1,
    max_replicas=4,
    parameters={"epochs": 10},
    volumes=[
        {
            "name": "data",
            "mount_path": "/data",
            "volume_type": "persistent_volume_claim",
            "spec": {"size": "1Gi"} 
        }
    ]
)
```

As you can see, here we run the job with a cluster that has a minimum of 1 and a maximum of 4 worker nodes, the initial number of worker nodes are 4. We also define a persistent volume claim to store and share the training data across the workers without downloading it each time.

During the execution the ``loss`` is logged as a metric with the experiment run.

Once the execution terminates successfully, the resulting model is logged in the platform with the provided hyper parameter values and the final model metrics as obtained from the ray execution result object.