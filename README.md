News Fact-Checker
This project is a financial fact-checking system that leverages natural language processing, semantic search, and large language models to determine whether financial news claims are true, false, or unverifiable. It uses a curated set of trusted facts and compares new claims against them using a combination of sentence embeddings, vector search (Pinecone), and OpenAI's language models.

Features
Claim extraction using zero-shot classification (facebook/bart-large-mnli)

Semantic similarity search using SentenceTransformer and Pinecone

GPT-based reasoning to determine factual accuracy

Evaluates claims as True, False, or Unverifiable

Uses a curated dataset of trusted financial statements

pine cone :

![Screenshot 2025-06-10 224324](https://github.com/user-attachments/assets/5608dc3a-fe13-4e1c-b789-d02ceb3a85b8)
Streamlit :
![Screenshot 2025-06-10 230536](https://github.com/user-attachments/assets/24a736ac-a51f-45d9-831d-575f56c5275f)
NLP:
![Screenshot 2025-06-10 224330](https://github.com/user-attachments/assets/f57d0485-bb8f-4bb0-9133-be51190a4165)
Input : ![image](https://github.com/user-attachments/assets/b20094a9-1048-4407-9f28-a091b19f2e97)

Output:
![image](https://github.com/user-attachments/assets/a4c27f9a-d1c1-4597-b48c-011a3b3b36f5)





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
git clone https://github.com/exi-0/Fact_Checker_Rag.git
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
