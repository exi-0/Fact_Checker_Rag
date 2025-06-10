Financial News Fact-Checker
This project is a financial fact-checking system that leverages natural language processing, semantic search, and large language models to determine whether financial news claims are true, false, or unverifiable. It uses a curated set of trusted facts and compares new claims against them using a combination of sentence embeddings, vector search (Pinecone), and OpenAI's language models.

Features
Claim extraction using zero-shot classification (facebook/bart-large-mnli)

Semantic similarity search using SentenceTransformer and Pinecone

GPT-based reasoning to determine factual accuracy

Evaluates claims as True, False, or Unverifiable

Uses a curated dataset of trusted financial statements

Requirements
Python 3.8+

Access to the following APIs:

Pinecone

OpenAI

NVIDIA GPU (optional but recommended for faster embedding and classification)

Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/financial-fact-checker.git
cd financial-fact-checker
Create a virtual environment and activate it

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Create a .env file with the following keys:

ini
Copy
Edit
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
Usage
The script initializes a trusted fact database, embeds and indexes the facts in Pinecone, and uses OpenAI GPT to analyze new claims.

To run the system:

bash
Copy
Edit
python main.py
This will:

Load the Indian Financial News dataset

Initialize sentence transformer and zero-shot classifier

Create or reset the Pinecone index

Embed and index the trusted facts

Run fact-checking on a set of sample claims
