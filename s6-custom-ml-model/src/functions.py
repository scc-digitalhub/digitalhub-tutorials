import json
from zipfile import ZipFile

import pandas as pd
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mae, mape, smape
from darts.models import NBEATSModel
from digitalhub_runtime_python import handler


@handler(outputs=["model"])
def train_model(project):
    """
    Train a NBEATS model on the Air Passengers dataset
    """
    # Load Air Passengers dataset
    series = AirPassengersDataset().load()
    train, test = series[:-36], series[-36:]

    # Configure and train NBEATS model
    model = NBEATSModel(
        input_chunk_length=24, output_chunk_length=12, n_epochs=200, random_state=0
    )
    model.fit(train)

    # Make predictions for evaluation
    pred = model.predict(n=36)

    # Save model artifacts
    model.save("predictor_model.pt")
    with ZipFile("predictor_model.pt.zip", "w") as z:
        z.write("predictor_model.pt")
        z.write("predictor_model.pt.ckpt")

    # Calculate metrics
    metrics = {
        "mape": mape(test, pred),
        "smape": smape(test, pred),
        "mae": mae(test, pred),
    }

    # Register model in DigitalHub
    model_artifact = project.log_model(
        name="air-passengers-forecaster",
        kind="model",
        source="predictor_model.pt.zip",
        algorithm="darts.models.NBEATSModel",
        framework="darts",
    )
    model_artifact.log_metrics(metrics)
    return model_artifact


def init_context(context, model_key):
    """
    Initialize serving context by loading the trained model
    """
    model = context.project.get_model(model_key)
    path = model.download()
    local_path_model = "extracted_model/"

    # Extract model from zip file
    with ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(local_path_model)

    # Load the NBEATS model
    input_chunk_length = 24
    output_chunk_length = 12
    name_model_local = local_path_model + "predictor_model.pt"
    mm = NBEATSModel(input_chunk_length, output_chunk_length).load(name_model_local)

    setattr(context, "model", mm)


def serve_predictions(context, event):
    """
    Serve time series predictions via REST API
    """
    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    else:
        body = event.body

    context.logger.info(f"Received event: {body}")
    inference_input = body["inference_input"]

    # Convert input to Darts TimeSeries format
    pdf = pd.DataFrame(inference_input)
    pdf["date"] = pd.to_datetime(pdf["date"], unit="ms")

    ts = TimeSeries.from_dataframe(pdf, time_col="date", value_cols="value")

    # Make predictions
    output_chunk_length = 12
    result = context.model.predict(n=output_chunk_length * 2, series=ts)

    # Convert result to JSON format
    jsonstr = result.pd_dataframe().reset_index().to_json(orient="records")
    return json.loads(jsonstr)
