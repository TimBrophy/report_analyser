# REPORT ANALYSER

This repo contains the code for a demo done at ElasticON London in 2024.
The code in this repo is intended as a way to learn how to implement Retrieval Augmented Generation architecural patterns, 
but should not be used as a production application in any sense, as many of the patterns implemented are orientated toward
educational purposes rather than robustness. In addition, the main library used to render this app is Streamlit (streamlit.io)
which is a rapid prototyping tool and not a production grade deployment platform (some may argue against this view).

## Prerequisites
An Elasticsearch cluster. The recommended option is an Elastic Cloud deployment which can be created easily and cost
effectively here: https://cloud.elastic.co

Node sizes can align to the following guidelines (but your own mileage may vary):
1. Vector search hardware profile selection
2. Content tier 2 zones, 2GB RAM in each
3. 8GB RAM in 1 zones for machine learning nodes
4. 1GB RAM in 1 zone for Kibana

Enable the .elser_model_2_linux-x86_64 model in the Trained Models section of Kibana. Most likely this will be a
download and then deploy operation.

I have also used the https://huggingface.co/ProsusAI/finbert model for sentiment analysis. You can change this if you want
to but would need to update the pipeline configuration to leverage any new model.

You need to download and deploy this model following the following Elastic documentation steps:
https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html

## Setup
Download the contents of the repo to your local computer.
In the root directory of the project (most likely 'report_analyser') create a python virtual environment (instructions here: https://docs.python.org/3/library/venv.html)
Activate the environment and install dependencies using the following command: $ pip install -r requirements.txt

Copy the example_secrets file and create a file called secrets.toml.
Complete all the details required, with at least one set of LLM credentials. Bear in mind that whichever LLM provider
you choose not to use, you need to comment that option out in the dropdown or remove the LLM dropdown completely from the app.py file.

## Run
Issue the command: streamlit run app.py and the application will open in a new browser window.
Setup your data using the 'elastic tasks' menu item
Check Indices looks for the required indices and if they dont exist, builds them
Delete Indices does what it says on the button
Check Pipelines does the same for pipelines as the Check Indices does for indices

# Dashboarding
I havent had time to rebuild the dashboards for the logging data - this should be a short exercise if you apply your mind.