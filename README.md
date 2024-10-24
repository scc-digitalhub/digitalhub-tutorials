# digitalhub-tutorials

The project aims at executing the documented scenarios of digitalhub platform. Currently there are seven different scenarios briefly explained below that allows to demonstrate the functionality of the platform. Inside each scenario folder prex with s<number>-<scenario_name> (for e.g. s1-etl), one can find 

- Jypter notebook 
	- Import the Jupyter notebook for each scenariolocated inside project folder in the 'Coder' instance and execute it step by step.

- Project
	- Import the project and execute...
	

## 1. ETL (Extract transform load)

The scenario depict how to collect some data regarding traffic, analyze and transform it, then expose the resulting dataset. The 'etl' folder contains
a jypter notebook and a digitalhub project.


## 2. DBT (Database transformation) scenario 

This scenario depict how to collect some data regarding organizations, analyze and transform it, then expose the resulting dataset

## 3. Scikit Learn Scenario

This scenario provides a quick overview of developing and deploying a scikit-learn machine learning application using the functionalities of the platform. We will prepare data, train a generic model and expose it as a service

## 4. ML Flow Model Training and Serving

This scenario provides a quick overview of developing and deploying a machine learning application based on model tracked with MLFlow framework using the functionalities of the platform.

## 5. LLM Flow Model Training and Serving

This scenario depict how to create and serve LLM HuggingFace-compatible-models. Specifically, it is possible to serve directly the LLM models from the HuggingFace catalogue provided the id of the model or to serve the fine-tuned model from the specified path, such as S3.

## 6. Custom ML FLow Model Training and Serving

This scenario provides a quick overview of developing and deploying generic machine learning applications using the functionalities of the platform. For this purpose, we use ML algorithms for the time series management provided by the Darts framework.

## 7. Postgres REST API

In this scenario, we download some data into a Postgres database, then use PostgREST - a tool to make Postgres tables accessible via REST API - to expose this data and run a simple request to view the results.