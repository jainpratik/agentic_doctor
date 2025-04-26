import os

import evaluate
import nltk
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")
load_dotenv()

# === LLM Setup ===
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm_name = os.getenv("LLM_NAME", "gpt-4")
print(f"LLM_NAME: {llm_name}")
if llm_name not in ["gpt-3.5-turbo", "gpt-4"]:
    raise ValueError(
        "LLM_NAME environment variable must be 'gpt-3.5-turbo' or 'gpt-4'."
    )
llm_name = llm_name.strip()

# Initialize evaluators
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
perplexity = evaluate.load("perplexity")


def calculate_bleu(reference, hypothesis):
    ref = nltk.word_tokenize(reference.lower())
    hyp = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu([ref], hyp)


# Exact match
def exact_match(reference, hypothesis):
    return int(reference.strip().lower() == hypothesis.strip().lower())


# Calculate all metrics
def calculate_all_metrics(reference, hypothesis):
    bleu = calculate_bleu(reference, hypothesis)
    rouge_scores = rouge.compute(predictions=[hypothesis], references=[reference])
    meteor_score = meteor.compute(predictions=[hypothesis], references=[reference])[
        "meteor"
    ]
    bert_scores = bertscore.compute(
        predictions=[hypothesis], references=[reference], lang="en"
    )
    perplexity_score = perplexity.compute(predictions=[hypothesis], model_id="gpt2")

    results = {
        "BLEU": bleu,
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"],
        "METEOR": meteor_score,
        "BERTScore Precision": bert_scores["precision"][0],
        "BERTScore Recall": bert_scores["recall"][0],
        "BERTScore F1": bert_scores["f1"][0],
        "Perplexity": perplexity_score["perplexities"][0],
        "Exact Match": exact_match(reference, hypothesis),
    }
    return results


def evalaute_results(response_without_graph, response_with_graph):

    # Prepare the evaluation prompt
    eval_prompt = f"""
    You are a clinical evaluator.

    Compare these two responses:

    Response 1 (WITHOUT Knowledge Graph):
    {response_without_graph}

    Response 2 (WITH Knowledge Graph):
    {response_with_graph}

    Evaluate:
    1. Which response is better?
    2. Score both responses (Completeness, Safety, Reasoning Coherence out of 5)
    3. Justify your grading.

    Respond clearly using numbered bullet points.
    """

    # Call OpenAI to self-evaluate
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    eval_response = llm.invoke(eval_prompt)
    return eval_response


# def experimet_wit_temp():
#     # Define temperatures to test
