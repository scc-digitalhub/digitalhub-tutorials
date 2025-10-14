import requests

def extract_text(tika_url, artifact, project):
    print(f"Downloading artifact {artifact.name}...")
    fp = artifact.as_file()
    if not (tika_url)[:4] == "http": 
        tika_url = "http://"+tika_url
    print(f"Sending {fp} to {tika_url}...")    
    response = requests.put(tika_url+"/tika",headers={"Accept":"text/html"}, data=open(fp,'rb').read())
    if response.status_code == 200:
        print("Extracted text with success")
        res = "/tmp/output.html"
        with open(res, "w") as tf:
            tf.write(response.text)
        project.log_artifact(kind="artifact", name=artifact.name+"_output.html", source=res)
        return res
    else:
        print(f"Received error: {response.status_code}")
        raise Exception("Error")
