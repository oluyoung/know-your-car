from llama_index import SimpleDirectoryReader, GPTListIndex, StorageContext, VectorStoreIndex, LLMPredictor, PromptHelper, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

def construct_index():
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 0.5
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader('./docs').load_data()

    index = VectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist('./storage')

    return index

index = construct_index()

def chatbot(input_text):
    storage_context = StorageContext.from_defaults('./storage')
    index = load_index_from_storage(storage_context)
    # index = VectorStoreIndex.load_from_disk()
    # response = index.query(input_text, response_mode="compact")
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

iface.launch(share=True)
