

import re
import random
    
    
def run_to_doc(df):
    all_data = []
    for i in range(len(df)):
        d = df.iloc[i]
        all_data.append({
            "id": i,
            "content": d['Title'] + "\\n\\n" + d['Excerpt']
        })
    
    return all_data
  
    
def parse_summary(xml_string):
    summary_pattern = r"<summary>(.*?)</summary>"
    explanation_pattern = r"<explanation>(.*?)</explanation>"
    summary_match = re.search(summary_pattern, xml_string, re.DOTALL)
    explanation_match = re.search(explanation_pattern, xml_string, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    return {"summary": summary, "explanation": explanation}
    


def get_content(state):
    docs = state["documents"]
    return [{"content": doc["content"]} for doc in docs]

    
def reduce_summaries(combined):
    summaries = combined["summaries"]
    documents = combined["documents"]
    return {
        "documents": [
            {
                "id": doc["id"],
                "content": doc["content"],
                "summary": summ_info["summary"],
                "explanation": summ_info["explanation"],
            }
            for doc, summ_info in zip(documents, summaries)
        ]
    }
# This is the summary node




def get_minibatches(state, config):
    batch_size = config["configurable"].get("batch_size", 200)
    original = state["documents"]
    indices = list(range(len(original)))
    random.shuffle(indices)
    if len(indices) < batch_size:
        # Don't pad needlessly if we can't fill a single batch
        return {"minibatches": [indices]}
    num_full_batches = len(indices) // batch_size
    batches = [
        indices[i * batch_size : (i + 1) * batch_size] for i in range(num_full_batches)
    ]
    leftovers = len(indices) % batch_size
    if leftovers:
        last_batch = indices[num_full_batches * batch_size :]
        elements_to_add = batch_size - leftovers
        last_batch += random.sample(indices, elements_to_add)
        batches.append(last_batch)
    return {
        "minibatches": batches,
    }
    


from typing import Dict
from langchain_core.runnables import Runnable


def parse_taxa(output_text):
    """Extract the taxonomy from the generated output."""
    
    cluster_pattern = r"<cluster>\s*<id>(.*?)</id>\s*<name>(.*?)</name>\s*<description>(.*?)</description>\s*</cluster>"

    cluster_matches = re.findall(cluster_pattern, output_text, re.DOTALL)
    clusters = [
        {"id": id.strip(), "name": name.strip(), "description": description.strip()}
        for id, name, description in cluster_matches
    ]
    return {"clusters": clusters}



def format_docs(docs):
    xml_table = "\\n"
    for doc in docs:
        xml_table += f'{doc["summary"]}\\n'
    xml_table += ""
    return xml_table


def format_taxonomy(clusters):
    xml = "\\n"
    for label in clusters:
        xml += "  \\n"
        xml += f'    {label["id"]}\\n'
        xml += f'    {label["name"]}\\n'
        xml += f'    {label["description"]}\\n'
        xml += "  \\n"
    xml += ""
    return xml
    
    
    
def invoke_taxonomy_chain(chain, state, config, mb_indices):
    configurable = config["configurable"]
    docs = state["documents"]
    minibatch = [docs[idx] for idx in mb_indices]
    data_table_xml = format_docs(minibatch)
    previous_taxonomy = state["clusters"][-1] if state["clusters"] else []
    cluster_table_xml = format_taxonomy(previous_taxonomy)
    updated_taxonomy = chain.invoke(
        {
            "data_xml": data_table_xml,
            "use_case": configurable["use_case"],
            "cluster_table_xml": cluster_table_xml,
            "suggestion_length": configurable.get("suggestion_length", 30),
            "cluster_name_length": configurable.get("cluster_name_length", 10),
            "cluster_description_length": configurable.get(
                "cluster_description_length", 30
            ),
            "explanation_length": configurable.get("explanation_length", 20),
            "max_num_clusters": configurable.get("max_num_clusters", 25),
        }
    )
    return {
        "clusters": [updated_taxonomy["clusters"]],
    }
    
