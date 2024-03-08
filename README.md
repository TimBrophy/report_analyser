# REPORT ANALYSER

This repo contains the code for a demo done at ElasticON London in 2024.
The code in this repo is intended as a way to learn how to implement Retrieval Augmented Generation architecural patterns, 
but should not be used as a production application in any sense, as many of the patterns implemented are orientated toward
educational purposes rather than robustness. In addition, the main library used to render this app is Streamlit (streamlit.io)
which is a rapid prototyping tool and not a production grade deployment platform (some may argue against this view).

## Notable attributes of this repo
1. This is effectively everything in-a-box with the least amount of manual setup needed (save for the dashboards which I will get to in a future version).
2. It is a quick and easy way to learn RAG with simple to understand and verbose code that tries best not to obviscate the Elastic-goodness by using a library to do the vectorsearch portions of the app.
3. I've implemented a simple cache to speed up QA responses and save unnecessary calls to the Cloud provider for questions the model has already answered. (this could also be built so much more comprehensively if I had more time)
4. The cache includes a sentiment analysis feature to understand whether any LLM responses lean in either direction away from Neutral.
5. I've also logged all answers along with costs based on provider so we can learn what the true cost of generative AI within a use case actually is.
6. The way the PDF file is chunked upon ingest is also notable as it is done outside of Elastic within the app rather than inside an ingest pipeline. I've built it in Python purely based on convenience for me - it could be built in a pipeline using Painless scripting and essentially 'handed off' from the app (probably more robust and better for prod)

## Prerequisites
1. Python 3.x and up
2. An Elasticsearch cluster. The recommended option is an Elastic Cloud deployment which can be created easily and cost
effectively here: https://cloud.elastic.co

Node sizes can align to the following guidelines (but your own mileage may vary):
1. Vector search hardware profile selection
2. Content tier 2 zones, 2GB RAM in each
3. 8GB RAM in 1 zones for machine learning nodes
4. 1GB RAM in 1 zone for Kibana

Enable the .elser_model_2_linux-x86_64 model in the Trained Models section of Kibana. Most likely this will be a download, and then deploy operation.

I have also used the https://huggingface.co/ProsusAI/finbert model for sentiment analysis. You can change this if you want to, but would need to update the pipeline configuration to leverage any new model.

You need to download and deploy this model following the following Elastic documentation steps:
https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html

3. Access to an LLM hosted with either AWS, Azure or both. (and of course the associated credentials) 

## Setup
Download the contents of the repo to your local computer.
In the root directory of the project (most likely 'report_analyser') create a python virtual environment (instructions here: https://docs.python.org/3/library/venv.html)
Activate the environment and install dependencies using the following command: $ pip install -r requirements.txt

Copy the example_secrets file and create a file called secrets.toml.
Complete all the details required, with at least one set of LLM credentials. Bear in mind that whichever LLM provider you choose **not** to use, you need to comment that option out in the dropdown or remove the LLM dropdown completely from the app.py file. I would recommend using BOTH integrations so that you can have fun comparing answers. 

## Run
Issue the command: streamlit run app.py and the application will open in a new browser window.
Setup your data using the 'elastic tasks' menu item:
- Check Indices looks for the required indices and if they dont exist, builds them
- Delete Indices does what it says on the button
- Check Pipelines does the same for pipelines as the Check Indices does for indices

## Dashboarding
I havent had time to rebuild the dashboards for the logging data - this should be a short exercise if you’d like to try it. 
Here’s a few use cases I recommend:
- sentiment logger - are prompts and responses neutral, helpful, negative, positive?
- count users, interactions, queries
- count documents ingested, chunks created
- cost calculation for prompts, responses (token usage)
