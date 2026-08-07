import json
import os
from pathlib import Path

import digitalhub as dh
import streamlit as st
from dotenv import load_dotenv

st.title("Chat Demo")

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

if "model_config" not in st.session_state:
    st.session_state["model_config"] = {
        "model_name": os.environ["model_name"],
        "model_run": dh.get_run(os.environ["model_run_key"]),
    }

# initialize a list to store chat history in the session state between reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# display messages (box with avatar and some content) from chat history on reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_completion(prompt):
    input = {
        "inputs": [
            {"name": "input-0", "shape": [1], "datatype": "BYTES", "data": [prompt]}
        ]
    }

    model_name = st.session_state.model_config["model_name"]
    run = st.session_state.model_config["model_run"]

    data_string = ""
    for r in run.invoke(model_name=model_name, json=input):
        data_string += r.decode("utf-8")

    try:
        return json.loads(data_string)["outputs"][0]["data"][0]
    except:
        return "Sorry, something went wrong."


# display a chat input and store its input in prompt
if prompt := st.chat_input("Give me a sentence to fill (use [MASK] as placeholder)"):
    # store user input in history and display it as a chat message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # display a chat message with the model response
    with st.chat_message("assistant"):
        response = get_completion(prompt)
        st.markdown(response)
    # store model response in history
    st.session_state.messages.append({"role": "assistant", "content": response})
