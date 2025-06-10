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
**Installation**
1)Clone the repository
2)Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies
3)pip install -r requirements.txt
Set up environment variables
4)Create a .env file with the following keys:
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
**Usage**
The script initializes a trusted fact database, embeds and indexes the facts in Pinecone, and uses OpenAI GPT to analyze new claims.

**To run the system:**
python main.py

**Project Structure**
main.py – Core script for data ingestion, processing, and fact-checking

requirements.txt – List of dependencies

.env – Environment variables for API keys (not committed)
