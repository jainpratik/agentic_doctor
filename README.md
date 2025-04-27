 🩺 Interactive Agentic Doctor

A high-risk, high-reward healthcare AI project exploring the use of knowledge graphs and LLMs to create an explainable, conversational medical assistant.

> **Note:** This was built as an exploratory project with a strong focus on experimentation, explainability, and agentic interaction — even failure was part of the learning process.

---

## 🔍 Overview

This project implements a prototype conversational agent that:
- Combines symbolic reasoning from a Neo4j-based UMLS knowledge graph
- Uses semantic retrieval (FAISS + BioBERT) to ground user queries
- Employs GPT-4 through LangChain for response generation
- Incorporates memory and multi-agent interaction for diagnosis and treatment

---

## 📦 Features

- ✅ Knowledge graph reasoning via Neo4j
- ✅ Retrieval-Augmented Generation (RAG) pipeline with LangChain
- ✅ GPT-4 integration with chain-of-thought prompts
- ✅ Conversation memory (LangChain)
- ✅ Streamlit-based web interface
- ✅ Neo4j - Graph Database
- ✅ Docker + GitHub Actions CI/CD
- ✅ GitHub Pages documentation support


**Create a .env file with the following params and values
**
  NEO4J_URI=
  NEO4J_USER=
  NEO4J_PASSWORD=
  OPENAI_API_KEY=

Run Neo4j Locally using docker and map to a local folder

Download UMLS data
Run Convert_umls_to CSV. Py to generate the necessary csv files
Run load_uml_to_neo4J.y to load this data into yout neo4j


**To run the streamlit ui
**go to folder newdoctorapp

**run the commad
streamlit run app.py

TO test the Reasoning agent


this code still need to be integrated into the streamlit app.


