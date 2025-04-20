from langchain.prompts import PromptTemplate

# Step 1: Symptom → Disease explanation
symptom_prompt = PromptTemplate(
    input_variables=["symptoms", "diseases"],
    template="""
You are a clinical reasoning AI. A patient presents with the following symptoms:
{symptoms}

The system matched these potential diseases:
{diseases}

Explain which diseases are most likely and why based on the symptoms.
""",
)

# Step 2: Disease → Test explanation
test_prompt = PromptTemplate(
    input_variables=["disease", "tests"],
    template="""
The likely diagnosis is: {disease}

Based on clinical guidelines, these tests were recommended:
{tests}

Explain the diagnostic utility of these tests for confirming or ruling out the disease.
""",
)


procedure_explainer_prompt = PromptTemplate(
    input_variables=["procedure"],
    template="""
You are a medical assistant. Explain the clinical purpose of the following procedure:
Procedure: {procedure}

Include:
- What it is
- When it's ordered
- What it diagnoses
- Any risks or limitations
""",
)


# Step 3: Literature alignment (RAG)
rag_prompt = PromptTemplate(
    input_variables=["disease", "abstracts"],
    template="""
The literature discusses {disease} with the following summary abstracts:
{abstracts}

Summarize how this literature supports or enhances the clinical reasoning above.
""",
)

cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a clinical assistant. Think step by step before providing a final answer.

Question: {question}

Step-by-step reasoning:
1. Identify symptoms
2. Match to diseases
3. Consider test recommendations
4. Cross-reference with literature
5. Summarize conclusion

Answer:
""",
)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the clinical information in this lab report:\n{text}",
)
