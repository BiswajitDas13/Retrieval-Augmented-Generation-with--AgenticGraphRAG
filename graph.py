import os
import openai
import pprint
import networkx as nx
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Load the document
try:
    docs = TextLoader("story.txt", encoding='utf-8').load()
except Exception as e:
    print(f"Error loading document: {e}")
    exit(1)

# Initialize OpenAI LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # Set the embedding model
    chunk_size=1000
)

# Set up Neo4j environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"  # Replace with your Neo4j URI
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"  # Replace with your Neo4j password

# Initialize Neo4j Graph
try:
    graph = Neo4jGraph()
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    exit(1)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=".")
docs = text_splitter.split_documents(docs)

# Create Neo4j Vector Store
try:
    db = Neo4jVector.from_documents(docs, embeddings)
    retriever = db.as_retriever()
except Exception as e:
    print(f"Error creating Neo4j vector store: {e}")
    exit(1)

# Create the retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_macbeth_queries",
    "This Retriever agent will help in finding related information from the Macbeth Text",
)

tools = [retriever_tool]

# Define grading function to evaluate document relevance
def grade_documents(state) -> str:
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK RELEVANCE---")

    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm_with_tool = llm.with_structured_output(Grade)

    # Define prompt for grading relevance
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    # Invoke the chain to grade the result
    try:
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
    except Exception as e:
        print(f"Error in grading relevance: {e}")
        return "rewrite"

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"

# Define the agent model to process the state
def agent(state):
    """Invokes the agent model to generate a response based on the current state."""
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the rewrite function to improve the question
def rewrite(state):
    """Transforms the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [HumanMessage(content=f"Improve the following question:\n{question}")]

    try:
        response = llm.invoke(msg)
    except Exception as e:
        print(f"Error in rewriting query: {e}")
        return {"messages": messages}

    return {"messages": [response]}

# Define the generate function to produce an answer
def generate(state):
    """Generate an answer based on the retrieved documents."""
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    # Define prompt for generation
    prompt = PromptTemplate(
        template="Using the context below, answer the question:\nContext: {context}\nQuestion: {question}",
        input_variables=["context", "question"],
    )

    # LLM Chain for generating response
    gpt_llm = llm
    rag_chain = prompt | gpt_llm | StrOutputParser()

    try:
        response = rag_chain.invoke({"context": docs, "question": question})
    except Exception as e:
        print(f"Error in generating response: {e}")
        return {"messages": messages}

    return {"messages": [response]}

# Define the workflow state graph
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

workflow = StateGraph(AgentState)

# Define nodes and edges for the workflow
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})

# Edges taken after the retrieve node is called.
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the graph
graph = workflow.compile()

# Define inputs and run the graph
inputs = {
    "messages": [("user", "How does guilt manifest in both Macbeth and Lady Macbeth after King Duncan's murder?")],
}

# Stream the output from the graph
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")