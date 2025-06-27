import pandas as pd
from datasets import load_dataset
from functions import *

import re
import random
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
    model="mistralai/mistral-small-3.2-24b-instruct:free"  # or any available model on OpenRouter
)


import operator
from typing import Annotated, List, TypedDict, Optional
class Doc(TypedDict):
    id: str
    content: str
    summary: Optional[str]
    explanation: Optional[str]
    category: Optional[str]
class TaxonomyGenerationState(TypedDict):
    # The raw docs; we inject summaries within them in the first step
    documents: List[Doc]
    # Indices to be concise
    minibatches: List[List[int]]
    # Candidate Taxonomies (full trajectory)
    clusters: Annotated[List[List[dict]], operator.add]

# Load the dataset
dataset = load_dataset("okite97/news-data")
# Access the different splits if available (e.g., train, test, validation)
train_data = dataset['train']
df = pd.DataFrame(train_data)
df = df.dropna()
df.reset_index(drop=True, inplace=True)

docs = run_to_doc(df[:20])
print('Data sample = ')
print(docs[0])
print('='*60)





summary_prompt = hub.pull("wfh/tnt-llm-summary-generation").partial(
    summary_length=20, explanation_length=30
)
# summary_prompt is an already saved prompt that can be pulled from langchain hub and reused.
# we use .partial to fix 2 parameters in the input parameters of the prompt.
# Now the prompt is a template that has only one input which is the 'content' (the text we want to summarize and explain ...)
print('type of the variable summary_prompt = ' , type(summary_prompt)) # we should see ChatPromptTemplate
print('input_variables = ', summary_prompt.input_variables) # we should see 'content'

# Fill with example input
formatted_messages = summary_prompt.format_messages(
    content="Show Template"
)

# Print the messages clearly
for msg in formatted_messages:
    print(f"{msg.type.upper()}:\n{msg.content}\n{'-' * 40}")
    
# So the template asks the llm to provide summary and an explanation of how he obtained the summary
# The template asks the llm to place the summary between <summary> and to place the explanation between <explanation>
# The following function, parses the output of the llm and returns a dictionary.


summary_llm_chain = (
    summary_prompt | model | StrOutputParser()
).with_config(run_name="GenerateSummary")

summary_chain = summary_llm_chain | parse_summary

input_text = """
What is Quantum computing?
Quantum computing is a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than on classical computers. The field of quantum computing includes hardware research and application development. Quantum computers are able to solve certain types of problems faster than classical computers by taking advantage of quantum mechanical effects, such as superposition and quantum interference. Some applications where quantum computers can provide such a speed boost include machine learning (ML), optimization, and simulation of physical systems. Eventual use cases could be portfolio optimization in finance or the simulation of chemical systems, solving problems that are currently impossible for even the most powerful supercomputers on the market.

What is the quantum computing advantage?
Currently, no quantum computer can perform a useful task faster, cheaper, or more efficiently than a classical computer. Quantum advantage is the threshold where we have built a quantum system that can perform operations that the best possible classical computer cannot simulate in any kind of reasonable time.

What is quantum mechanics?
Quantum mechanics is the area of physics that studies the behavior of particles at a microscopic level. At subatomic levels, the equations that describe how particles behave is different from those that describe the macroscopic world around us. Quantum computers take advantage of these behaviors to perform computations in a completely new way.

What is a qubit?
Quantum bits, or qubits, are represented by quantum particles. The manipulation of qubits by control devices is at the core of a quantum computer's processing power. Qubits in quantum computers are analogous to bits in classical computers. At its core, a classical machine's processor does all its work by manipulating bits. Similarly, the quantum processor does all its work by processing qubits.

How are qubits different from classical bits?
In classical computing, a bit is an electronic signal that is either on or off. The value of the classical bit can thus be one (on) or zero (off). However, because the qubit is based on the laws of quantum mechanics it can be placed in a superposition of states.

What are the principles of quantum computing?
A quantum computer works using quantum principles. Quantum principles require a new dictionary of terms to be fully understood, terms that include superposition, entanglement, and decoherence. Let's understand these principles below.
"""

#result = summary_chain.invoke(input_text)
#print(result)
#print('='*60)


map_step = RunnablePassthrough.assign(
    summaries=get_content
    | RunnableLambda(func=summary_chain.batch, afunc=summary_chain.abatch)
)

# map_step, is taking states as input, and adds a summaries as new key
# the value of the new key is computed by the chain get_content | RunnableLambda ...
# Example
print(type(map_step))


input_text1 = """
Overview¶
LangGraph is built for developers who want to build powerful, adaptable AI agents. Developers choose LangGraph for:

Reliability and controllability. Steer agent actions with moderation checks and human-in-the-loop approvals. LangGraph persists context for long-running workflows, keeping your agents on course.
Low-level and extensible. Build custom agents with fully descriptive, low-level primitives free from rigid abstractions that limit customization. Design scalable multi-agent systems, with each agent serving a specific role tailored to your use case.
First-class streaming support. With token-by-token streaming and streaming of intermediate steps, LangGraph gives users clear visibility into agent reasoning and actions as they unfold in real time.
Learn LangGraph basics¶
To get acquainted with LangGraph's key concepts and features, complete the following LangGraph basics tutorials series:

Build a basic chatbot
Add tools
Add memory
Add human-in-the-loop controls
Customize state
Time travel
In completing this series of tutorials, you will build a support chatbot in LangGraph that can:

✅ Answer common questions by searching the web
✅ Maintain conversation state across calls
✅ Route complex queries to a human for review
✅ Use custom state to control its behavior
✅ Rewind and explore alternative conversation paths
"""
docs = [
    {"id": 0, "content": input_text}, 
    {"id": 1, "content": input_text1}
    ]

#result = map_step.invoke({"documents": docs})


#print('Result keys = ', result.keys())

#for i, summary in enumerate(result["summaries"]):
#    print(f"\n=== Doc {i} ===")
#    print("Summary:", summary["summary"])
#    print("Explanation:", summary["explanation"])
    
# This is the summary node
map_reduce_chain = map_step | reduce_summaries


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# We will share an LLM for each step of the generate -> update -> review cycle
# You may want to consider using Opus or another more powerful model for this
taxonomy_generation_llm = model # the same model will be used to define the taxonomy too
## Initial generation

print('taxonomy template inputs' , hub.pull("wfh/tnt-llm-taxonomy-generation").input_variables)
taxonomy_generation_prompt = hub.pull("wfh/tnt-llm-taxonomy-generation").partial(
    use_case="Generate the taxonomy that can be used to label the user intent in the conversation.",
)

print('After using partial to fix the use_case')
print(taxonomy_generation_prompt.input_variables)
print(taxonomy_generation_prompt.format(cluster_description_length=20, cluster_name_length=10, data_xml='data_xml', explanation_length = 20, max_num_clusters = 10))

print('='*100)

# A chain that starts with the prompt taxonomy template and followed by model and then followed by an OutputParse
taxa_gen_llm_chain = (
    taxonomy_generation_prompt | taxonomy_generation_llm | StrOutputParser()
).with_config(run_name="GenerateTaxonomy")

# We add to the chain another parser, that will extract information from the output of the previous chain
# The previous chain returns an XML and the function parse_taxa converts it into a dictionary
generate_taxonomy_chain = taxa_gen_llm_chain | parse_taxa

#result = map_step.invoke({"documents": docs})
#summaries = result["summaries"]

#data_xml = ""
#for i, s in enumerate(summaries):
#    #data_xml += f"<doc><id>{i}</id><conversations>{s['summary']}</conversations><explanation>{s['explanation']}</explanation></doc>\n"
#    data_xml += f"<conversations>{s['summary']}</conversations>\n"

#taxonomy_input = {
#    "data_xml": data_xml,
#    "cluster_description_length": 30,
#    "cluster_name_length": 20,
#    "explanation_length": 40,
#    "max_num_clusters": 10
#}

#taxonomy = generate_taxonomy_chain.invoke(taxonomy_input)
#print('T'*100)
#print(taxonomy)
#print(parse_taxa(taxonomy))


# Update taxonomy in real time as new datasets arrive
taxonomy_update_prompt = hub.pull("wfh/tnt-llm-taxonomy-update")
taxa_update_llm_chain = (
    taxonomy_update_prompt | taxonomy_generation_llm | StrOutputParser()
).with_config(run_name="UpdateTaxonomy")
update_taxonomy_chain = taxa_update_llm_chain | parse_taxa



taxonomy_review_prompt = hub.pull("wfh/tnt-llm-taxonomy-review")
taxa_review_llm_chain = (
    taxonomy_review_prompt | taxonomy_generation_llm | StrOutputParser()
).with_config(run_name="ReviewTaxonomy")
review_taxonomy_chain = taxa_review_llm_chain | parse_taxa


def generate_taxonomy(state, config):
    return invoke_taxonomy_chain(
        generate_taxonomy_chain, state, config, state["minibatches"][0]
    )
    
def update_taxonomy(state, config):
    which_mb = len(state["clusters"]) % len(state["minibatches"])
    return invoke_taxonomy_chain(
        update_taxonomy_chain, state, config, state["minibatches"][which_mb]
    )
    
def review_taxonomy(state, config):
    batch_size = config["configurable"].get("batch_size", 200)
    original = state["documents"]
    indices = list(range(len(original)))
    random.shuffle(indices)
    return invoke_taxonomy_chain(
        review_taxonomy_chain, state, config, indices[:batch_size]
    )


from langgraph.graph import StateGraph

graph = StateGraph(TaxonomyGenerationState)
graph.add_node("summarize", map_reduce_chain)
graph.add_node("get_minibatches", get_minibatches)
graph.add_node("generate_taxonomy", generate_taxonomy)
graph.add_node("update_taxonomy", update_taxonomy)
graph.add_node("review_taxonomy", review_taxonomy)
graph.add_edge("summarize", "get_minibatches")
graph.add_edge("get_minibatches", "generate_taxonomy")
graph.add_edge("generate_taxonomy", "update_taxonomy")


def should_review(state):
    num_minibatches = len(state["minibatches"])
    num_revisions = len(state["clusters"])
    if num_revisions < num_minibatches:
        return "update_taxonomy"
    return "review_taxonomy"

graph.add_conditional_edges(
    "update_taxonomy",
    should_review,
    # Optional (but required for the diagram to be drawn correctly below)
    {"update_taxonomy": "update_taxonomy", "review_taxonomy": "review_taxonomy"},
)
graph.set_finish_point("review_taxonomy")
graph.set_entry_point("summarize")
app = graph.compile()

png_graph = app.get_graph().draw_mermaid_png()
with open("my_graph.png", "wb") as f:
    f.write(png_graph)

use_case = (
    "Generate the taxonomy that can be used to label the news article that would benefit the user."
)
stream = app.stream(
    {"documents": docs},
    {
        "configurable": {
            "use_case": use_case,
            # Optional:
            "batch_size": 10,
            "suggestion_length": 30,
            "cluster_name_length": 10,
            "cluster_description_length": 30,
            "explanation_length": 20,
            "max_num_clusters": 25,
        },
        "max_concurrency": 2,
        "recursion_limit": 50,
    },
)
for step in stream:
    node, state = next(iter(step.items()))
    print(node, str(state)[:20] + " ...")
    
    
from IPython.display import Markdown
def format_taxonomy_md(clusters):
    md = "## Final Taxonomy\\n\\n"
    md += "| ID | Name | Description |\\n"
    md += "|----|------|-------------|\\n"
    # Iterate over each inner list of dictionaries
    for cluster_list in clusters:
        for label in cluster_list:
            id = label["id"]
            name = label["name"].replace("|", "\\\\|")  # Escape any pipe characters within the content
            description = label["description"].replace("|", "\\\\|")  # Escape any pipe characters
            md += f"| {id} | {name} | {description} |\\n"
    return md
markdown_table = format_taxonomy_md(step['review_taxonomy']['clusters'])
print(markdown_table)

with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown_table)