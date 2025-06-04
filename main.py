from string import Template
from typing import Any, Dict, TypedDict, List
from fastapi import FastAPI
from langchain_core.pydantic_v1 import validator
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from os import getenv
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

llm =  ChatOpenAI(
  temperature=0,
  openai_api_key=getenv("API_KEY"),
  openai_api_base=getenv("BASE_URL"),
   model_name="openai/gpt-3.5-turbo",
 # model_name="google/gemini-2.5-pro-exp-03-25",
  model_kwargs={},
)

#RAG Integration
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


class NewsroomState(TypedDict):
    initial_query: str
    topic: str # e.g., "AI breakthroughs in healthcare"
    research_query: str # e.g., "AI advancements in medical diagnosis"
    research_results: List[Document] # The documents retrieved from ChromaDB
    summary: str
    headline: str
    feedback: str
    decision: str

class EditorDecision(BaseModel):
    """
    Represents the editor's decision regarding a news article's summary and headline.
    """
    decision: str = Field(
        ...,
        description="The editor's decision, one of: 'ACCEPT', 'REVISE_SUMMARY', 'REVISE_HEADLINE', 'REVISE_BOTH'."
    )
    feedback: str = Field(
        ...,
        description="A brief reason or specific feedback for the decision, especially if revision is needed."
    )
    @validator('decision')
    def validate_type(cls, v):
        allowed = {"ACCEPT", "REVISE_SUMMARY", "REVISE_HEADLINE", "REVISE_BOTH"}
        v_upper = v.strip().upper()  # Normalize input
        if v_upper not in allowed:
            raise ValueError(f"Invalid type: {v}. Must be one of {allowed}")
        return v_upper  # Return normalized value
        

def ideator_node(state: NewsroomState):
    query = state["initial_query"]
    logger.info(f"Ideator: Starting with initial query: '{query}'")
    result = llm.invoke([
        {
            "role": "system",
            "content": """You are an expert news editor. Your job is to take a broad news query and suggest a single specific, actionable news topic for a journalist to research. NOTHING ELSE JUST A TOPIC"""
        },
        {"role": "user", "content": query}
    ])
    logger.info(f"Ideator: Generated specific topic: '{result.content}'")
    return {"topic": result.content}


def researcher_node(state: NewsroomState):
    topic = state["topic"]
    retrieved_docs = retriever.invoke(topic)
    logger.info(f"Researcher: Searching for topic: '{topic}'")
    logger.info(f"Researcher: Retrieved {len(retrieved_docs)} documents, realted to topic")

    return {"research_results": retrieved_docs}


def summarizer_node(state: NewsroomState):
    topic = state["topic"]
    researched_docs = state["research_results"]
    parsed_docs = []
    template = Template("Source: $doc_source, Story: $doc_story, Sentiment: $doc_sentiment")

    for doc in researched_docs:
        context_doc = template.substitute(
            doc_source=doc.metadata.get("source", "Unknown"),
            doc_story=doc.page_content,
            doc_sentiment=doc.metadata.get("sentiment_label", "N/A")
        )
        parsed_docs.append(context_doc)
    
    documents_context_str = "\n---\n".join(parsed_docs)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a summarizer tasked with analyzing and summarizing a collection of research documents.

        Only use the content provided in the "Documents" section below. Do not rely on outside knowledge or assumptions.
        Maintain a neutral, factual tone throughout.

        Each document may include sentiment information. If the sentiment is relevant to the content, briefly incorporate it into the summary.
        RETURN ONLY THE SUMMARY."""),
        ("user", "Topic: {topic}\n\nDocuments:\n{documents}") # Use the placeholders here
    ])

    messages = prompt_template.format_messages(
        topic=topic,
        documents=documents_context_str
    )
    logger.info(f"Summarizer: Summarizing {len(researched_docs)} documents for topic: '{topic}'")
    result = llm.invoke(messages)
    logger.info(f"Summarizer: Summary generated.")
    return {"summary": result.content}

def headline_generator_node(state: NewsroomState) -> Dict[str, Any]:
    summary = state["summary"]
    topic = state["topic"] 

    logger.info(f"Headline Generator: Generating headline for summary related to '{topic}'...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a creative news editor. Your task is to write a concise, compelling, and accurate headline based on the provided news summary.
        The headline should be engaging and capture the essence of the summary in 5-10 words.
        Focus on the main subject and action. Do NOT use emojis."""),
        ("human", "News Summary: {summary}\n\nSuggested Headline:")
    ])

    messages = prompt.format_messages(
        summary=summary
    )
    result = llm.invoke(messages)
    logger.info(f"Headline Generator: Generated headline: '{result.content}'")
    return {"headline": result.content}

def editor_node(state: NewsroomState) -> Dict[str, Any]:
    summary = state["summary"]
    headline = state["headline"]
    topic = state["topic"] 

    decision = "REVISE_HEADLINE" # Default or fallback
    feedback = "No specific feedback provided by editor."

    logger.info(f"Editor: Reviewing Headline and Summary for publication")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a meticulous news editor. Your task is to review a generated news summary and headline.
        Assess their quality, factual accuracy based on common knowledge (without external search), conciseness, and engagement.

        Based on your review, you must output a JSON object conforming to the provided schema.
        Provide a brief reason or specific feedback for your decision, especially if revision is needed.
        """),
        ("human", "Topic: {topic}\n\nSummary:\n{summary}\n\nHeadline: {headline}\n\nProvide your decision and feedback in JSON format.")
    ])
    messages = prompt.format_messages(
        topic=topic,
        summary=summary,
        headline=headline
    )

    chain = (
        prompt 
        | llm.bind(response_format={"type": "json_object"})
        | JsonOutputParser(pydantic_object=EditorDecision)
    )

    try:
        result = chain.invoke({
        "topic": topic,
        "summary": summary,
        "headline": headline
    })
        decision = result.get('decision', decision).upper()
        feedback = result.get('feedback', feedback)
        logger.info(f"Editor: Review Completed, Status : '{decision}'", )
        return {"decision": decision, "feedback": feedback}
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"decision": decision, "feedback": feedback}
    except Exception as e:
        logger.error(f"Editor error: {e}")
        return {"decision": decision, "feedback": feedback}
   


def route_editor_decision(state: NewsroomState) -> str:
    decision = state.get("decision")
    if decision == "ACCEPT":
        return END
    elif decision == "REVISE_SUMMARY":
        return "summarizer" # Loop back to summarizer
    elif decision == "REVISE_HEADLINE":
        return "headline_generator" # Loop back to headline generator
    elif decision == "REVISE_BOTH":
        return "summarizer" # Or you could choose to loop back to researcher/ideator if severe
    return END # Fallback

workflow = StateGraph(NewsroomState)
workflow.add_node("ideator", ideator_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("headline_generator", headline_generator_node)
workflow.add_node("editor", editor_node)
workflow.add_edge(START, "ideator")
workflow.add_edge("ideator", "researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", "headline_generator")
workflow.add_edge("headline_generator", "editor")
workflow.add_conditional_edges(
    "editor", 
    lambda state: state.get("decision"),
    {
        "ACCEPT": END,
        "REVISE_SUMMARY": "summarizer",
        "REVISE_HEADLINE": "headline_generator",
        "REVISE_BOTH": "summarizer"
    }
)

orchestrator = workflow.compile()

class chatPayload(BaseModel):
    query: str

app = FastAPI()


@app.post("/research/")
async def research(payload: chatPayload):
    state = orchestrator.invoke({"initial_query":  payload.query})
    return {"Headline" : state.get("headline"), "Summary" : state.get("summary")}
