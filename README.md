# Digitalhub-tutorials

This project repository aims to provide some documented scenarios for the Digitalhub platform. Currently, there are six different scenarios, briefly explained below, that aim to showcase the functionality of the platform. Inside each scenario folder there are:

- Executable Jupyter Notebook
- Project descriptor files and code sources

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
