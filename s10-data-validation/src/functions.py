import os

from digitalhub_runtime_python import handler
from frictionless import validate


@handler(outputs=["report"])
def main(project, di):
    # download as local file
    path = di.download(destination=di.name, overwrite=True)
    # validate
    report = validate(path)
    # update artifact with label
    label = "VALID" if report.valid else "INVALID"
    di.metadata.labels = (
        di.metadata.labels.append(label) if di.metadata.labels else [label]
    )
    di.save(update=True)
    # cleanup
    os.remove(path)

    with open("report.json", "w") as f:
        f.write(report.to_json())

    project.log_artifact(name=f"{di.name}_validation-report.json", source="report.json")

    # persist report
    return report.to_json()
