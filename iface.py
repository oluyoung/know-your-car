import os
import openai
import nest_asyncio
import gradio as gr
import sys
from pathlib import Path
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent

openai.api_key = os.environ["OPENAI_API_KEY"]

nest_asyncio.apply()

MAKES = {
    "TOYOTA": [
        ("AVALON", range(2019, 2023 )),
        ("AYGO", range(2013, 2022))
    ]
}

service_context = ServiceContext.from_defaults(chunk_size=512)

# Ingest Data with Unstructured
def construct_index():
    loader = UnstructuredReader()
    doc_set = {}
    index_set = {}

    for make in MAKES:
        for brand_ in MAKES[make]:
            brand = brand_[0]
            years = brand_[1]

            if not brand in doc_set:
                doc_set[brand] = {}
            if not brand in index_set:
                index_set[brand] = {}

            for year in years:
                year_docs = loader.load_data(
                    file=Path(f"./docs/{make}/{brand}/{year}-{make}-{brand}.pdf"), split_documents=False
                )

                for d in year_docs:
                    d.metadata = { "year": year }

                doc_set[brand][year] = year_docs

                storage_context = StorageContext.from_defaults()
                cur_index = VectorStoreIndex.from_documents(
                    doc_set[brand][year],
                    service_context=service_context,
                    storage_context=storage_context,
                )

                index_set[brand][year] = cur_index

                storage_context.persist(persist_dir=f"./storage/{make}/{brand}/{year}")

    return index_set


# Define chatbot with input text, see if we can check for the MAKE or BRAND or YEAR from the input_text.
# Check if one of the props exist in the input and redirect directly
def chatbot(input_text):
    index_set = {}
    for make in MAKES:
        for brand_ in MAKES[make]:
            brand = brand_[0]
            years = brand_[1]

            if not brand in index_set:
                index_set[brand] = {}

            for year in years:
                storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{make}/{brand}/{year}")
                cur_index = load_index_from_storage(storage_context, service_context=service_context)
                index_set[brand][year] = cur_index

            individual_query_engine_tools = [
                QueryEngineTool(
                    query_engine=index_set[brand][year].as_query_engine(),
                    metadata=ToolMetadata(
                        name=f"vector_index_{make}_{brand}_{year}",
                        description=f"useful for when you want to answer queries about a {make} {brand} {year} model. ",
                    ),
                )
                for year in years
            ]

            query_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=individual_query_engine_tools,
                service_context=service_context,
            )

            query_engine_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="sub_question_query_engine",
                    description="useful for when you want to answer queries about cars through their manuals",
                ),
            )

            tools = individual_query_engine_tools + [query_engine_tool]

            agent = OpenAIAgent.from_tools(tools, verbose=True)

            response = agent.chat(input_text)

            # print(str(response))
            return str(response)


index = construct_index()

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-trained AI Chatbot"
)

iface.launch(share=True)
