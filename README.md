# Newsroom AI: Multi-Agent Content Generation with LangGraph & RAG

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-FFD43B?style=flat&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-046DD5?style=flat&logo=replit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-005697?style=flat&logo=chroma&logoColor=white)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-FFD33B?style=flat&logo=huggingface&logoColor=black)

## Table of Contents

-   [Introduction](#introduction)
-   [Features](#features)
-   [Architecture Flow](#architecture-flow)
-   [Technologies Used](#technologies-used)
-   [Setup & Installation](#setup--installation)
    -   [Prerequisites](#prerequisites)
    -   [Clone the Repository](#clone-the-repository)
    -   [Create & Activate Virtual Environment](#create--activate-virtual-environment)
    -   [Install Dependencies](#install-dependencies)
    -   [Prepare Your Data](#prepare-your-data)
    -   [Run the ETL Process (First-Time Setup)](#run-the-etl-process-first-time-setup)
    -   [Start the FastAPI Server](#start-the-fastapi-server)
    -   [Interact with the API](#interact-with-the-api)

## Introduction

In the era of information overload, streamlining content creation is paramount. This project, **Newsroom AI**, presents an innovative solution: an autonomous AI-driven newsroom pipeline. It demonstrates how to orchestrate a team of specialized AI agents using **LangGraph** to transform a high-level query into a concise, well-researched, and editorially reviewed news article.

This system combines Retrieval-Augmented Generation (RAG) with a sophisticated multi-agent architecture, showcasing capabilities in intelligent ideation, contextual research, dynamic content generation, and autonomous quality assurance.

## Features

* **Intelligent Ideation:** An AI agent refines broad user queries into specific, actionable news topics.
* **Contextual Research (RAG):** Utilizes a local **ChromaDB** vector store to retrieve the most relevant information from a news dataset, ensuring factual grounding.
* **Sentiment Analysis Integration:** Incorporates **Hugging Face Transformers** to analyze the sentiment of source articles, enriching metadata for enhanced context during content generation and review.
* **Dynamic Content Creation:** Generates factual summaries and compelling headlines based on retrieved research.
* **Autonomous Editorial Review:** An AI Editor agent conditionally reviews the generated summary and headline, routing them back for revision or marking them as accepted, demonstrating sophisticated decision-making and looping capabilities.
* **Structured Output:** Leverages **Pydantic** for robust and reliable structured output from LLMs, ensuring consistent data flow for conditional logic.
* **Scalable & Stateful Architecture:** Built with **LangGraph** for managing complex agent interactions and maintaining shared state throughout the workflow.
* **API Exposure:** The entire multi-agent pipeline is exposed via a **FastAPI** endpoint, making it easily accessible and integrable.
* **Comprehensive Logging:** Detailed logging throughout the agent workflow for better observability and debugging.

## Architecture Flow

The Newsroom AI operates through a series of interconnected agents, with dynamic routing based on the Editor's decisions.

```mermaid
graph TD
    A[User Input: Broad Query] --> B(Ideator Agent: Generates Specific Topic)
    B --> C(Researcher Agent: Retrieves Relevant Documents)
    C --> D(Summarizer Agent: Creates Summary from Docs)
    D --> E(Headline Generator Agent: Crafts Headline)
    E --> F{Editor Agent: Reviews Content}

    F -- "Decision: ACCEPT" --> G(End: Final News Article)
    F -- "Decision: REVISE_SUMMARY" --> D
    F -- "Decision: REVISE_HEADLINE" --> E
    F -- "Decision: REVISE_BOTH" --> D
````

## Technologies Used

  * **Python 3.9+**
  * **LangChain**: Framework for developing LLM applications.
  * **LangGraph**: For building robust, stateful, and cyclic multi-agent applications.
  * **FastAPI**: For creating a high-performance web API.
  * **ChromaDB**: A lightweight, local vector database for RAG.
  * **Hugging Face `transformers`**: For integrating advanced NLP models (e.g., sentiment analysis).
  * **Hugging Face `datasets`**: For convenient dataset loading.
  * **Pandas**: For data manipulation and ETL processes.
  * **Pydantic**: For defining and enforcing structured LLM outputs.
  * **`sentence-transformers`**: For generating embeddings (`all-MiniLM-L6-v2`).

## Setup & Installation

Follow these steps to get the Newsroom AI running on your local machine.

### Prerequisites

  * Python 3.9 or higher
  * `pip` (Python package installer)
  * `git` (for cloning the repository)

### Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/WajeehAhmed/Newsroom-AI-Enabled.git
cd Newsroom-AI-Enabled
```

### Create & Activate Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**

```bash
python -m venv newsroom_venv
.\newsroom_venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv newsroom_venv
source newsroom_venv/bin/activate
```

### Install Dependencies

Install all the necessary Python packages listed in `requirements.txt`. Pay special attention to the PyTorch installation, as it requires a specific `--index-url` for CPU-only builds.

```bash
pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

### Prepare Your Data

The project uses the `R3troR0b/news-dataset` from Hugging Face. The ETL process (described below) will download, clean, and sample this data into a `news.csv` file, which is then used to populate the vector store. Ensure your ETL script points to the correct paths if you modify this.

### Run the ETL Process (First-Time Setup)

Before running the FastAPI server, you must execute the ETL process to populate your ChromaDB vector store. This involves loading data, performing sentiment analysis, chunking text, and generating embeddings.

Locate the ETL related code (often in a dedicated script or the initial part of `main.py` if designed to run once) and execute it. This will create a `chroma_db` directory in your project root.

### Start the FastAPI Server

Once the `chroma_db` is populated, you can start the FastAPI application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The server will typically run on `http://127.0.0.1:8080`. You should see logs indicating the server has started.

### Interact with the API

You can test the API using tools like `curl`, Postman, or Insomnia. Send a POST request to the `/chat/` endpoint with your query:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"content": "Hi, what’s dangerous about AI?"}' \
  [http://127.0.0.1:8080/chat/](http://127.0.0.1:8080/chat/)
```

Observe the console logs from your running FastAPI server to see the agents in action, and you'll receive a JSON response from the API containing the generated news summary and headline.
