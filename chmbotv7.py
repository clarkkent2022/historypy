# -*- coding: utf-8 -*-
"""

**Computer History Bot - Richard Sawey**

A chatbot for research on computing history, a RAG implementation on selections from the archives at the Computer History Museum, 
Mountain View, https://www.computerhistory.org

This script expects a parameter to indicate the LLM to use, a parameter to indidate if the vector stores should be recreated, else
the stores in the vectorpersist and summarypersist directories will be used. 

"""

# load up with stuff we'll need
#pip install openai
#pip install sentence-transformers
#pip install langchain pypdf langchain-openai #tiktoken chromadb
#pip install llama-index-vector-stores-chroma
#pip install llama-index --upgrade

#
# Let's get to work now all the riff raff has been loaded
#
import os, sys, argparse, logging, readline

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description="""
        Thank you for trying HistoryBot, a prototype chat bot designed to provide an interactive
        experience for those researching or interested in computer history. In History Bot's current
        experimental configuration the only source document used is a Computer History Museum docent's
        script. The author wrote this script as part of his responsibilities as a museum docent and the
        script includes details on a range of important computers from history such as ENIAC and the 
        IBM 360.

        This script also assumes the existence of an environment variable OPENAI_API_KEY
        that obviously contains your API key for the LLM.

        This is a simple example of Agentic RAG, and uses a simple Llamaindex router to either query
        the source documents directly or to summarize the documents. First created as part of my
        Stanford TECH16 LLMs for Business class, this code is based on DeepLeaning.AI's course,
        'Building Agentic RAG with Llamaindex':
        https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/

        Richard Sawey  March 2024
        """)

# Define optional switches with help messages
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
# parser.add_argument("--config", "-c", type=str, help="Path to configuration file (optional)")
parser.add_argument("--mode", "-m", choices=["train", "query"], type=str, default="query", 
    help="Mode of 'query' (the default) will use existing sumary/vector stores, mode of 'train' will (re)build) these stores")
parser.add_argument("--sources", "-s", type=str, default="./History",
    help="Path to source documents (required if mode of 'train' invoked, default is History subdirectory). Ensure directory only contains documents you want fed into the beast.")
parser.add_argument("--viewnodes", "-vn", action="store_true", help="View document nodes")

# Parse arguments
args = parser.parse_args()

# Configure logging
if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    logging.debug("Verbose mode enabled")
else:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#
# get API keys
#
from openai import OpenAI
if 'OPENAI_API_KEY' in os.environ:
    llm_api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=llm_api_key)
    logging.debug("OPENAI_API_KEY found")
else:
    logging.critical("Expected to find LLM_API_KEY environment variable, not found, so stopping")


from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, SummaryIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

#
# Change to chatgpt 4 from LlamaIndex default of 3.5
#
# Settings.llm = OpenAI(temperature=0, model_name='gpt-4-turbo-preview', api_key=llm_api_key)
llm = OpenAI(temperature=0, model_name='gpt-4o', api_key=llm_api_key)

#
#  index the supplied documents using llama_index and create a persistent index store
#
#
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

if args.mode == "query":
    #
    # Use the already created vector db
    #
    storage_context = StorageContext.from_defaults(persist_dir="./vectorpersist")
    vector_index = load_index_from_storage(storage_context)
    storage_summary_context = StorageContext.from_defaults(persist_dir="./summarypersist")
    summary_index = load_index_from_storage(storage_summary_context)
else:
    #
    # Input mode must be 'train' so generate new vector dbs of source material
    #
    #documents = SimpleDirectoryReader("./History").load_data()
    #
    # get document(s)
    #
    documents = SimpleDirectoryReader(input_files=["./Tourv5.pdf"]).load_data()
    embed_model = OpenAIEmbedding("text-embedding-3-small")
    #
    # split docs up into nodes
    #
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
    print(nodes[5].get_content())

    #
    # Feed nodes into summary indexer
    #
    summary_index = SummaryIndex(nodes)
    summary_index.storage_context.persist(persist_dir="./summarypersist")
    
    #
    # Feed nodes into vector db
    #
    vector_index = VectorStoreIndex(nodes)
    vector_index.storage_context.persist(persist_dir="./vectorpersist")


summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True, llm=llm
)
vector_query_engine = vector_index.as_query_engine(llm=llm)

logging.info(type(summary_index.ref_doc_info))
if args.viewnodes:
    sum_nodes = summary_index.ref_doc_info
    logging.info(sum_nodes)

#    for key, value in sum_nodes:
#        logging.debug(f'Key: {key}, Value: {value}')
#        logging.debug()
#        logging.debug('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
#        logging.debug()


from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool


summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to the source documents",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context and answers related to the source documents",
)

query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

#response = query_engine.query("Summarize the tour in bullet points, one per computer")
#print(response)

while True:  # Initiates an infinite loop, emulating an ever-ready assistant.
    user_input = input("Please enter your question for HistoryBot or press enter to exit: ")
    if user_input == "":  
        sys.exit("Goodbye! Thanks for using History Bot.")
    else:
        # Process the user's question.
        response = query_engine.query(user_input)
        print(response)
