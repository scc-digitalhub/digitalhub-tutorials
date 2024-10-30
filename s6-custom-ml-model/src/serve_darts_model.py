
from darts.models import NBEATSModel
from zipfile import ZipFile
from darts import TimeSeries
import json
import pandas as pd

def init(context):
    model_name = "darts_model"

    model = context.project.get_model(model_name)
    path = model.download()
    local_path_model = "extracted_model/"

    with ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(local_path_model)

    input_chunk_length = 24
    output_chunk_length = 12
    name_model_local = local_path_model +"predictor_model.pt"
    mm = NBEATSModel(
            input_chunk_length,
            output_chunk_length
    ).load(name_model_local)

    setattr(context, "model", mm)

def serve(context, event):

    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    else:
        body = event.body
    context.logger.info(f"Received event: {body}")
    inference_input = body["inference_input"]

    pdf = pd.DataFrame(inference_input)
    pdf['date'] = pd.to_datetime(pdf['date'], unit='ms')

    ts = TimeSeries.from_dataframe(
        pdf,
        time_col="date",
        value_cols="value"
    )

    output_chunk_length = 12
    result = context.model.predict(n=output_chunk_length*2, series=ts)
    # Convert the result to a pandas DataFrame, reset the index, and convert to a list
    jsonstr = result.pd_dataframe().reset_index().to_json(orient='records')
    return json.loads(jsonstr)
