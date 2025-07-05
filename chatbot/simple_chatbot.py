from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import OllamaLLM



chat_history = [SystemMessage(content='You are an AI assistant, answer the user\'s questions to the best of your ability.')]
llm = OllamaLLM(model="mistral")

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state['messages'])
    
    return {'messages' : [AIMessage(content=response)]}


graph_builder.add_node('assistant', chatbot)
graph_builder.add_edge(START, 'assistant')
graph_builder.add_edge('assistant', END)

graph = graph_builder.compile()


def stream_graph_updates(chat_history: list, user_input: str):
    chat_history.append(HumanMessage(content=user_input))
    for event in graph.stream({"messages": chat_history}):
        assistant_message = event['assistant']['messages'][-1]
        print("Assistant:", assistant_message.content)
        chat_history.append(assistant_message.content)
        


print('Type (quit, exit, q or clear) to terminate the chat ... ')
while True:

    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q", "clear"]:
        print("Goodbye!")
        break
    
    stream_graph_updates(chat_history, user_input)
