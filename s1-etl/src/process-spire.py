
from digitalhub_runtime_python import handler

KEYS=['codice spira','longitudine','latitudine','Livello','tipologia','codice','codice arco','codice via','Nome via', 'stato','direzione','angolo','geopoint']

@handler(outputs=["dataset-spire"])
def process(project, di):
    df = di.as_df()
    sdf= df.groupby(['codice spira']).first().reset_index()[KEYS]
    return sdf
