import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm
import torch
import json
from transformers import pipeline

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize NLP model 
try:
    dataset = load_dataset("kdave/Indian_Financial_News")
    claim_extractor = pipeline("zero-shot-classification", 
                               model="facebook/bart-large-mnli", 
                               device=0 if device == 'cuda' else -1)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Initialize embedding and LLM
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
llm_client = OpenAI(api_key=OPENAI_API_KEY)

# Setup Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "fact-checking-index"

# Delete and recreate index
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
trusted_facts = trusted_facts = [
    {
        "text": "On February 27, 2019, Nifty50 remained bearish due to heightened geopolitical tensions between India and Pakistan, along with weak global cues.",
        "source": "Economic Times, February 2019"
    },
    {
        "text": "Fitch Solutions revised India's economic growth forecast for 2020-21 to 1.8% due to COVID-19's impact on private consumption.",
        "source": "Fitch Solutions Report, March 2020"
    },
    {
        "text": "Fitch noted a potential contraction in private consumption and fixed investments in India in FY21 due to large-scale income losses and weak capital expenditure.",
        "source": "Fitch Solutions, 2020"
    },
    {
        "text": "Fitch revised China’s 2020 GDP growth to 1.1% citing a sharp Q1 contraction of 6.8% y-o-y due to COVID-19.",
        "source": "Fitch Solutions, April 2020"
    },
    {
        "text": "India was the second largest PE/VC deal market in Asia-Pacific in 2019 with over 1,000 deals worth USD 45 billion.",
        "source": "Bain & Company PE Report, 2020"
    },
    {
        "text": "India's PE/VC exit value dropped from USD 17 billion in 2018 to USD 13 billion in 2019, excluding Flipkart.",
        "source": "IVCA-Bain PE Report, 2020"
    },
    {
        "text": "India's share of APAC PE/VC deal market rose to nearly 25% in 2019, with investment values 70% higher than in 2018.",
        "source": "Bain & Company, 2020"
    }
]


# Embed and upsert facts
print("Embedding trusted facts...")
fact_texts = [fact["text"] for fact in trusted_facts]
embeddings = embedding_model.encode(fact_texts, show_progress_bar=True)

print("Indexing facts in vector database...")
upserts = [
    {'id': str(i), 'values': emb.tolist(), 'metadata': trusted_facts[i]}
    for i, emb in enumerate(embeddings)
]
index.upsert(upserts)
print("Indexing complete. Stats:")
print(index.describe_index_stats())



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
        print(f"LLM error: {e}")
        return {
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": "Error occurred during LLM evaluation"
        }

def fact_check(text):
    print(f"\nFact-checking: {text}")
    claim = extract_claim(text)
    print(f"Extracted claim: {claim}")
    
    facts = retrieve_similar_facts(claim)
    print(f"Retrieved {len(facts)} relevant facts")
    
    result = compare_claim_with_facts(claim, facts)
    return {
        "original_text": text,
        "extracted_claim": claim,
        **result
    }

#quering

if __name__ == "__main__":
    sample_inputs = [
    "The Nifty50 continued to remain in bear trap on February 27, as heightened geopolitical tensions between India & Pakistan and weak global cues weighed on traders' sentiment.",
    
    "Fitch Solutions has reduced India's economic growth forecast for 2020-21 to just 1.8 percent due to the economic impact of COVID-19.",
    
    "India's private consumption and fixed investment are expected to contract sharply in 2020-21 according to Fitch Ratings.",
    
    "In Q1 2020, China's real GDP shrank by 6.8% year-on-year, and Fitch has downgraded China's growth forecast for 2020 to 1.1 percent.",
    
    "India secured over 1,000 private equity and venture capital deals worth $45 billion in 2019, making it the second-largest PE/VC market in the Asia-Pacific region.",
    
    "India’s PE/VC exit value fell to USD 13 billion in 2019, down from USD 17 billion in 2018, excluding the Flipkart exit.",
    
    "India’s share in Asia-Pacific's private equity market reached nearly 25% in 2019, with a 70% rise in investment value compared to the previous year."
]

    
    for input_text in sample_inputs:
        result = fact_check(input_text)
        print("\nFact-check Result:")
        print(json.dumps(result, indent=2))
        print("\n" + "="*80 + "\n")
