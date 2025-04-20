import os

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_processors import LLMChain, SequentialChain
from tools.explanation_chain import explanation_chain
from tools.guideline_check import guideline_checker
from tools.medication_recommender import MedicationRecommender
from tools.procedure_explainer import procedure_chain
from tools.prompts import (
    cot_prompt,
    procedure_explainer_prompt,
    rag_prompt,
    summary_chain,
    summary_prompt,
    symptom_prompt,
    test_prompt,
)
from tools.rag_tool import rag_tool
from tools.symptom_checker import SymptomChecker
from tools.symptom_tool import SymptomToolNeo4j
from tools.test_recommender import TestRecommender

# Load environment variables from .env file
load_dotenv()

# Set up the environment variables for Neo4j connection
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set up the environment variables for OpenAI API key

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Set up the environment variables for Pinecone API key and environment
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENV"] = os.getenv("PINECONE_ENV")
# Set up the environment variables for LangChain API key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Set up the environment variables for Pinecone index name
os.environ["PINECONE_INDEX_NAME"] = os.getenv("PINECONE_INDEX_NAME")
# Set up the environment variables for Pinecone namespace
os.environ["PINECONE_NAMESPACE"] = os.getenv("PINECONE_NAMESPACE")


# Neo4j config
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

llm = ChatOpenAI(model="gpt-4", temperature=0)


symptom_chain = LLMChain(
    llm=llm, prompt=symptom_prompt, output_key="diagnosis_explanation"
)

procedure_chain = LLMChain(
    llm=llm, prompt=procedure_explainer_prompt, output_key="procedure_explanation"
)

test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test_explanation")
rag_chain = LLMChain(llm=llm, prompt=rag_prompt, output_key="literature_summary")
cot_chain = LLMChain(llm=llm, prompt=cot_prompt, output_key="reasoning_trace")
# summary = summary_chain.run(lab_text)


# Compose sequential reasoning chain
explanation_chain = SequentialChain(
    chains=[symptom_chain, test_chain, rag_chain],
    input_variables=["symptoms", "diseases", "disease", "tests", "abstracts"],
    output_variables=[
        "diagnosis_explanation",
        "test_explanation",
        "literature_summary",
    ],
    verbose=True,
)

# Tools
symptom_tool = SymptomChecker(URI, USER, PASSWORD).as_tool()
med_tool = MedicationRecommender(URI, USER, PASSWORD).as_tool()
test_tool = TestRecommender(URI, USER, PASSWORD).as_tool()

procedure_tool = Tool(
    name="ProcedureExplainer",
    func=lambda x: procedure_chain.run(procedure=x),
    description="Explains a medical test or procedure",
)


explanation_tool = Tool(
    name="ExplanationAgent",
    func=lambda x: explanation_chain.run(x),
    description="Provides step-by-step clinical reasoning explanation.",
)
cot_chain = LLMChain(llm=llm, prompt=cot_prompt, output_key="reasoning_trace")

cot_tool = Tool(
    name="ChainOfThoughtReasoner",
    func=lambda q: cot_chain.run(question=q),
    description="Performs step-by-step clinical reasoning to reach diagnostic or treatment conclusions.",
)

# Clarifier Tool
clarifier_tool = Tool(
    name="Clarifier",
    func=lambda _: "Your input seems ambiguous. Could you clarify symptoms, duration, or any previous conditions?",
    description="Use when user query lacks specificity or is too broad to respond clearly.",
)

guideline_tool = Tool(
    name="ClinicalGuidelineChecker",
    func=guideline_checker,
    description="Check symptoms and lab values against clinical criteria for sepsis or pneumonia.",
)

tools = [
    symptom_tool,
    med_tool,
    test_tool,
    procedure_tool,
    rag_tool,
    explanation_tool,
    cot_tool,
    clarifier_tool,
    guideline_tool,
]


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)
