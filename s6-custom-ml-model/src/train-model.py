from zipfile import ZipFile

from darts.datasets import AirPassengersDataset
from darts.metrics import mae, mape, smape
from darts.models import NBEATSModel
from digitalhub_runtime_python import handler


@handler()
def train_model(project):
    series = AirPassengersDataset().load()
    train, _ = series[:-36], series[-36:]

    model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=200, random_state=0)
    model.fit(train)
    pred = model.predict(n=36)

    model.save("predictor_model.pt")
    with ZipFile("predictor_model.pt.zip", "w") as z:
        z.write("predictor_model.pt")
        z.write("predictor_model.pt.ckpt")
    metrics = {"mape": mape(series, pred), "smape": smape(series, pred), "mae": mae(series, pred)}

    model = project.log_model(
        name="darts_model",
        kind="model",
        source="predictor_model.pt.zip",
        algorithm="darts.models.NBEATSModel",
        framework="darts",
    )
    model.log_metrics(metrics)
    return model
