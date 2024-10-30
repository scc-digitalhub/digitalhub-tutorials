

from digitalhub_runtime_python import handler

import pandas as pd
import numpy as np

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import NBEATSModel
from darts.metrics import mape, smape, mae

from zipfile import ZipFile

@handler()
def train_model(project):
    series = AirPassengersDataset().load()
    train, val = series[:-36], series[-36:]

    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=200,
        random_state=0
    )
    model.fit(train)
    pred = model.predict(n=36)

    model.save("predictor_model.pt")
    with ZipFile("predictor_model.pt.zip", "w") as z:
        z.write("predictor_model.pt")
        z.write("predictor_model.pt.ckpt")
    metrics = {
        "mape": mape(series, pred),
        "smape": smape(series, pred),
        "mae": mae(series, pred)
    }

    project.log_model(
        name="darts_model",
        kind="model",
        source="predictor_model.pt.zip",
        algorithm="darts.models.NBEATSModel",
        framework="darts",
        metrics=metrics
    )
