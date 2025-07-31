from digitalhub_runtime_python import handler


@handler(outputs=["dataset"])
def downloader(url):
    return url.as_df(file_format="csv", sep=";")
