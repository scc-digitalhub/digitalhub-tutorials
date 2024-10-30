
from digitalhub_runtime_python import handler

@handler(outputs=["dataset"])
def downloader(url):
    # read and rewrite to normalize and export as data
    df = url.as_df(file_format='csv',sep=";")
    return df
