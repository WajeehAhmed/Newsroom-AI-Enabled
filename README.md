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
-   [Project Structure](#project-structure)
-   [Future Enhancements](#future-enhancements)
-   [License](#license)
-   [Contact](#contact)

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
