import requests
import os

base_dir="/app/local_data"
def extract_text(tika_url, artifact, project):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print(f"Downloading artifact {artifact.name}...")
    fp = artifact.download(os.path.join(base_dir, 'test.pdf'))
    if tika_url[:4] != "http":
        tika_url = "http://" + tika_url
    print(f"Sending {fp} to {tika_url}...")
    response = requests.put(
        tika_url.rstrip("/") + "/tika/html",
        headers={"Accept": "text/html"},
        data=open(fp, "rb").read(),
    )
    if response.status_code == 200:
        print("Extracted text with success")
        res = os.path.join(base_dir, "output.html")
        with open(res, "w") as tf:
            tf.write(response.text)
        project.log_artifact(
            kind="artifact", name=artifact.name + "_output.html", source=res
        )
        return res
    else:
        print(f"Received error: {response.status_code}")
        print(f"Request Accept: {response.request.headers.get('Accept')}")
        print(f"Response Content-Type: {response.headers.get('Content-Type')}")
        raise RuntimeError(
            f"Tika extraction failed at {response.url} "
            f"(HTTP {response.status_code}): {response.text[:2000]}"
        )
