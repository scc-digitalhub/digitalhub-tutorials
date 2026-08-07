import pandas as pd
from digitalhub_runtime_python import handler

COLS = [
    "codice spira",
    "longitudine",
    "latitudine",
    "Livello",
    "tipologia",
    "codice",
    "codice arco",
    "codice via",
    "Nome via",
    "stato",
    "direzione",
    "angolo",
    "geopoint",
]

KEYS = [
    "00:00-01:00",
    "01:00-02:00",
    "02:00-03:00",
    "03:00-04:00",
    "04:00-05:00",
    "05:00-06:00",
    "06:00-07:00",
    "07:00-08:00",
    "08:00-09:00",
    "09:00-10:00",
    "10:00-11:00",
    "11:00-12:00",
    "12:00-13:00",
    "13:00-14:00",
    "14:00-15:00",
    "15:00-16:00",
    "16:00-17:00",
    "17:00-18:00",
    "18:00-19:00",
    "19:00-20:00",
    "20:00-21:00",
    "21:00-22:00",
    "22:00-23:00",
    "23:00-24:00",
]
COLUMNS = ["data", "codice spira"]


@handler(outputs=["dataset"])
def downloader(url):
    return url.as_df(file_format="csv", sep=";")


@handler(outputs=["dataset-spire"])
def process_spire(di):
    df = di.as_df()
    return df.groupby(["codice spira"]).first().reset_index()[COLS]


@handler(outputs=["dataset-measures"])
def process_measures(di):
    df = di.as_df()
    rdf = df[COLUMNS + KEYS]
    ls = []
    for key in KEYS:
        k = key.split("-")[0]
        xdf = rdf[COLUMNS + [key]]
        xdf["time"] = xdf.data.apply(lambda x: x + " " + k)
        xdf["value"] = xdf[key]
        ls.append(xdf[["time", "codice spira", "value"]])
    return pd.concat(ls)


def init_context(context, dataitem):
    di = context.project.get_dataitem(dataitem)
    df = di.as_df()
    context.df = df


def serve(context, event):
    df = context.df

    if df is None:
        return ""

    # Simple REST API
    fields = event.fields

    page = max(int(fields.get("page", 0)), 0)
    pageSize = min(max(int(fields.get("size", 50)), 1), 100)

    start = page * pageSize
    end = start + pageSize
    total = len(df)

    end = min(end, total)

    ds = df.iloc[start:end]
    json = ds.to_json(orient="records")

    return {"data": json, "page": page, "size": pageSize, "total": total}
