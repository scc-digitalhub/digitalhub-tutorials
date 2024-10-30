
def init_context(context):
    di = context.project.get_dataitem('dataset-measures')
    df = di.as_df()
    setattr(context, "df", df)

def handler(context, event):
    df = context.df

    if df is None:
        return ""

    # mock REST api
    method = event.method
    path = event.path
    fields = event.fields

    id = False

    # pagination
    page = 0
    pageSize = 50

    if "page" in fields:
        page = int(fields['page'])

    if "size" in fields:
        pageSize = int(fields['size'])

    if page < 0:
        page = 0

    if pageSize < 1:
        pageSize = 1

    if pageSize > 100:
        pageSize = 100

    start = page * pageSize
    end = start + pageSize
    total = len(df)

    if end > total:
        end = total

    ds = df.iloc[start:end]
    json = ds.to_json(orient="records")

    res = {"data": json, "page": page, "size": pageSize, "total": total}

    return res
