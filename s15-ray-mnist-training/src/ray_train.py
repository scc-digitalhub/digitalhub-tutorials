import os
import tempfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

import ray.train.torch

def ray_handler(project, run, epochs=1, num_classes=10, lr=0.001):

    def train_func():
        # Model, Loss, Optimizer
        model = resnet18(num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # [1] Prepare model.
        model = ray.train.torch.prepare_model(model)
        # model.to("cuda")  # This is done by `prepare_model`
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Data
        transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
        train_data = FashionMNIST(root="/data", train=True, download=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        # [2] Prepare dataloader.
        train_loader = ray.train.torch.prepare_data_loader(train_loader)

        # Training
        for epoch in range(epochs):
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)

            for images, labels in train_loader:
                # This is done by `prepare_data_loader`!
                # images, labels = images.to("cuda"), labels.to("cuda")
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # [3] Report metrics and checkpoint.
            metrics = {"loss": loss.item(), "epoch": epoch}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pt")
                )
                ray.train.report(
                    metrics,
                    checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                )
            if ray.train.get_context().get_world_rank() == 0:
                print(metrics)
                # [PLATFORM] Specific platform change: logging metrics for the current experiment
                run.log_metrics({"loss": metrics["loss"]})

    # [0] Initialized dataset to the shared volume folder. To initialize dataset before the actual training starts. 
    # May use, for example, the dataset stored in the platform as artifact instead.
    train_data = FashionMNIST(root="/data", train=True, download=True)

    # [4] Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=False)

    # [5] Launch distributed training job.
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        # [5a] If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # [PLATFORM] log the checkpoints to the preconfigured project S3 storage path
        run_config=ray.train.RunConfig(storage_path=f"s3://{project.name}/ray", name=run.id),
    )
    result = trainer.fit()

    # [6] Load the trained model.
    with result.checkpoint.as_directory() as checkpoint_dir:
        mdl_path = os.path.join(checkpoint_dir, "model.pt")
        print(mdl_path)
        print(f"logging model from {mdl_path}")
        print(f"results: {result}")
        
        parameters = {
            "epochs": epochs,
            "num_classes": num_classes,
            "lr": lr
        }
        # [PLATFORM] Log model to the platform
        mdl = project.log_model(
                          name="resnet18_fashion_mnist",
                          framework="pytorch",
                          algorithm="resnet18",
                          parameters=parameters,
                          source=mdl_path
        )
        mdl.log_metrics({"loss" : result.metrics["loss"]})