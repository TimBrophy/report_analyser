import streamlit as st
from elasticsearch import Elasticsearch
import os
import json

es = Elasticsearch(os.environ['elastic_url'],
                   api_key=os.environ['elastic_api_key'])
report_index = 'search-reports'
report_pipeline = 'ml-inference-search-reports'
transformer_model = '.elser_model_2_linux-x86_64'
logging_index = 'llm_interactions'
logging_pipeline = 'ml-inference-llm_logging'


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


report_index_mapping = read_json_file(f'config/{report_index}-mapping.json')
report_index_settings = read_json_file(f'config/{report_index}-settings.json')
logging_index_mapping = read_json_file(f'config/{logging_index}-mapping.json')
report_pipeline_config = read_json_file(f'config/{report_pipeline}.json')
logging_pipeline_config = read_json_file(f'config/{logging_pipeline}.json')


def check_indices():
    task_report = []
    report_exists = es.indices.exists(index=report_index)
    if not report_exists:
        report_result = es.indices.create(index=report_index, mappings=report_index_mapping)
        task_report.append(report_result)
    elif report_exists:
        task_report.append("Report index exists already")
    logging_exists = es.indices.exists(index=logging_index)
    if not logging_exists:
        logging_result = es.indices.create(index=logging_index, mappings=logging_index_mapping,
                                           settings=report_index_settings)
        task_report.append(logging_result)
    elif logging_exists:
        task_report.append("Logging index exists already")
    return task_report


def delete_indices():
    task_report = []
    report_result = es.indices.delete(index=report_index)
    task_report.append(report_result)
    logging_result = es.indices.delete(index=logging_index)
    task_report.append(logging_result)
    return task_report

def check_pipelines():
    task_report = []
    report_pipeline_exists = es.ingest.get_pipeline(id=report_pipeline, ignore=[404])

    if len(report_pipeline_exists):
        task_report.append(report_pipeline_exists)
    else:
        pipeline_result = es.ingest.put_pipeline(id=report_pipeline, processors=report_pipeline_config)
        task_report.append(pipeline_result)

    logging_pipeline_exists = es.ingest.get_pipeline(id=logging_pipeline, ignore=[404])
    if len(logging_pipeline_exists):
        task_report.append(logging_pipeline_exists)
    else:
        pipeline_result = es.ingest.put_pipeline(id=logging_pipeline, processors=logging_pipeline_config)
        task_report.append(pipeline_result)
    return task_report


st.title("Elastic tasks")
check_index = st.button("Check indices")
clear_index = st.button("Delete indices")
check_pipeline = st.button("Check pipelines")
if check_index:
    outcome = check_indices()
    st.write(outcome)
elif clear_index:
    outcome = delete_indices()
    st.write(outcome)
elif check_pipeline:
    outcome = check_pipelines()
    st.write(outcome)