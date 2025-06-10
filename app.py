import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os
import torch
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from transformers import pipeline
from datasets import load_dataset

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Fact Checker", layout="centered")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Initialize NLP model
try:
    dataset = load_dataset("kdave/Indian_Financial_News")
    claim_extractor = pipeline("zero-shot-classification", 
                               model="facebook/bart-large-mnli", 
                               device=0 if device == 'cuda' else -1)
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    raise

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
llm_client = OpenAI(api_key=OPENAI_API_KEY)

# Setup Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "fact-checking-index"

# Delete and recreate index if exists
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

index = pinecone.Index(INDEX_NAME)

# Trusted facts
trusted_facts = [
    {"text": "Government announces PM-KUSUM scheme to provide solar power to farmers", "source": "PIB Press Release, 2023"},
    {"text": "Farmers to receive 50% subsidy on solar pumps under PM-KUSUM", "source": "Agriculture Ministry, 2024"},
    {"text": "Free electricity for farmers implemented in Punjab from 2021", "source": "Punjab Government, 2021"},
    {"text": "No nationwide free electricity scheme announced for farmers in 2025", "source": "Power Ministry statement, 2024"},
    {"text": "Subsidized electricity rates for farmers continue under existing policies", "source": "PIB Fact Check, 2024"}
]

# Embed and upsert facts
st.write("Embedding and indexing trusted facts...")
fact_texts = [fact["text"] for fact in trusted_facts]
embeddings = embedding_model.encode(fact_texts, show_progress_bar=True)

upserts = [
    {'id': str(i), 'values': emb.tolist(), 'metadata': trusted_facts[i]}
    for i, emb in enumerate(embeddings)
]
index.upsert(upserts)
st.success("Indexing complete")

# --- Functions ---

def extract_claim(text):
    candidate_labels = ["financial claim", "policy announcement", "economic statement", "other"]
    result = claim_extractor(text, candidate_labels)

    if result['labels'][0] in ["financial claim", "policy announcement", "economic statement"]:
        return text
    else:
        sentences = text.split('.')
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) > 5:
                result = claim_extractor(sent, candidate_labels)
                if result['labels'][0] in ["financial claim", "policy announcement", "economic statement"]:
                    return sent
        return text

def retrieve_similar_facts(claim, top_k=3):
    embedding = embedding_model.encode(claim).tolist()
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match['metadata'] for match in results['matches']]

def compare_claim_with_facts(claim, facts):
    if not facts:
        return {
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": "No relevant facts found in the database to verify this claim"
        }

    fact_list = "\n".join([f"- {f['text']} (Source: {f['source']})" for f in facts])

    prompt = f"""
Compare the following claim with verified facts and determine if it's true, false, or unverifiable.
Provide clear reasoning based on the evidence.

CLAIM: {claim}

VERIFIED FACTS:
{fact_list}

Possible verdicts:
- "True": The claim is fully supported by verified facts
- "False": The claim contradicts verified facts
- "Unverifiable": The claim cannot be verified with current facts

Return your response in JSON format:
{{
  "verdict": "True|False|Unverifiable",
  "evidence": ["list of relevant facts"],
  "reasoning": "Detailed explanation"
}}
    """
    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"LLM error: {e}")
        return {
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": "Error occurred during LLM evaluation"
        }

def fact_check(text):
    claim = extract_claim(text)
    facts = retrieve_similar_facts(claim)
    result = compare_claim_with_facts(claim, facts)
    return {
        "original_text": text,
        "extracted_claim": claim,
        **result
    }

# --- Streamlit UI ---
st.title("🧐 Financial Claim Fact Checker")
user_input = st.text_area("Enter a financial/policy-related news statement:")

if st.button("Fact Check"):
    if user_input:
        with st.spinner("Verifying claim..."):
            output = fact_check(user_input)
        st.subheader("Result")
        st.write(f"**Original Text:** {output['original_text']}")
        st.write(f"**Extracted Claim:** {output['extracted_claim']}")
        st.write(f"**Verdict:** :blue[{output['verdict']}]")
        st.write("**Evidence Used:**")
        for fact in output['evidence']:
            st.markdown(f"- {fact}")
        st.write("**Reasoning:**")
        st.markdown(output['reasoning'])
    else:
        st.warning("Please enter a statement to fact-check.")
