# Digitalhub-tutorials

This project repository aims to provide some documented scenarios to showcase how to use the platform. Inside each scenario folder there are:

- Executable Jupyter Notebook
- Project descriptor files and code sources

In-depth descriptions of these scenarios, as well as more details on the platform, can be found in [the documentation](https://scc-digitalhub.github.io/docs).

Here follows a short description of each scenario:

## ETL (Extract, Transform, Load)

This scenario demonstrates how to collect data regarding traffic, analyze and transform it, then expose the resulting dataset.

## DBT (Database Transformation) Scenario

This scenario demonstrates how to collect data regarding organizations, analyze and transform it, then expose the resulting dataset.

## Scikit Learn Scenario

This scenario provides a quick overview of developing and deploying a scikit-learn machine learning application using the functionalities of the platform. We will prepare data, train a generic model, and expose it as a service.

## ML Flow Model Training and Serving

This scenario provides a quick overview of developing and deploying a machine learning application based on model tracked with MLFlow framework using the functionalities of the platform.

## LLM Flow Model Training and Serving

This scenario demonstrates how to create and serve LLM HuggingFace-compatible models. Specifically, it is possible to serve directly the LLM models from the HuggingFace catalog provided the id of the model or to serve the fine-tuned model from the specified path, such as S3. The scenario uses a GPU and a profile defined by the cluster owner.

## Custom ML Flow Model Training and Serving

This scenario provides a quick overview of developing and deploying generic machine learning applications using the functionalities of the platform. For this purpose, we use ML algorithms for time series management provided by the Darts framework.

## Retrieval-Augmented Generation (RAG)

This scenario builds a RAG application, using a chat model and an embedding model, to provide a chatbot.

## Whisper fine-tuning

This scenario demonstrates how to fine-tune [Whisper](https://huggingface.co/openai/whisper-small), a model for speech-to-text recognition.

## Flower federated learning

This scenario introduces the [Flower Federated Learning](https://flower.ai/) framework, which allows for running federated learning tasks where different client nodes perform local training and cooperatively create a more robust solution without exchanging the data but only the weight necessary to progress the training process.

## Data validation

This scenario implements a simple data validation function, which evaluates the correctness of a CSV table by leveraging an open source library, Frictionless.
