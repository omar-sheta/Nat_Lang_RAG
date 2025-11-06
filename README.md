📘 RAG Backend Benchmarking (SQuAD Dataset)

This README provides setup and execution steps for benchmarking different vector databases (FAISS, Pinecone, and Azure AI Search) using the SQuAD v1.1 dataset.

All experiments can be run inside Jupyter or VSCode notebooks, so team members can easily reproduce results without command-line work.

⸻

🧠 Project Overview

Goal: Compare retrieval quality and performance across vector databases for a standard Question Answering (QA) task.

Databases:
	•	FAISS – Open-source, local, and CPU/GPU optimized.
	•	Pinecone – Cloud-native vector DB with scalable APIs.
	•	Azure AI Search – Hybrid semantic search engine combining keyword and vector search.

Dataset: SQuAD v1.1 (Stanford Question Answering Dataset)￼

Embedding Model: intfloat/e5-base-v2

⸻

⚙️ Environment Setup

Each teammate should:
	1.	Clone the project or sync with the shared repo.
	2.	Create and activate a virtual environment (e.g., myenv).
	3.	Install dependencies:

pip install sentence-transformers==3.* transformers==4.* faiss-cpu pandas numpy tqdm pinecone-client python-dotenv

	4.	Create a .env file in the project root:

PINECONE_API_KEY=your_pinecone_key_here


⸻

📂 Project Structure

project/
│
├── code/
│   ├── prepare_squad_faiss.py          # Builds FAISS index
│   ├── faiis_eval.py                   # Evaluates FAISS retrieval
│   ├── prepare_squad_pinecone.ipynb    # Uploads data & evaluates Pinecone
│   ├── data_exploration.ipynb          # (Optional) dataset inspection
│   └── prepare_100_contracts.ipynb     # legacy CUAD setup
│
├── Data/
│   ├── SQuAD/dev-v1.1.json             # Dataset file
│   └── squad_prepared/                 # Auto-generated embeddings & indices
│
└── README.md (this file)


⸻

🚀 Running FAISS Experiments (Notebook or VSCode)

Step 1. Prepare the Dataset

Run this cell in a notebook:

!python prepare_squad_faiss.py \
  --input ../Data/SQuAD/dev-v1.1.json \
  --out_dir ../Data/squad_prepared \
  --num_docs 0 \
  --max_tokens 384 \
  --stride 128 \
  --batch_size 96

This will create chunks, embeddings, and FAISS indices.

Step 2. Inspect Dataset

!python faiis_eval.py --inspect --squad ../Data/SQuAD/dev-v1.1.json

Step 3. Evaluate Retrieval

A) Semantic-match (lenient)

!python faiis_eval.py --eval --squad ../Data/SQuAD/dev-v1.1.json --k_list 1,3,5,10

B) String-only (stricter, fairer for comparison)

!python faiis_eval.py --eval --squad ../Data/SQuAD/dev-v1.1.json --k_list 1,3,5,10 --string_only

Optional: Global Search (no per-doc restriction)

!python faiis_eval.py --eval --squad ../Data/SQuAD/dev-v1.1.json --global_eval --k_list 1,3,5,10 --string_only


⸻

☁️ Running Pinecone Experiments

Open and run prepare_squad_pinecone.ipynb.

The notebook steps include:
	1.	Loading the .env file (for PINECONE_API_KEY).
	2.	Initializing the Pinecone index.
	3.	Uploading SQuAD chunks and embeddings.
	4.	Running retrieval and computing Recall@K, EM, and F1.

📦 You can track progress with tqdm progress bars added to cells 10 and 11.

⸻

📊 Comparing Results

After both runs, record these metrics:

Metric	FAISS (Local)	Pinecone (Cloud)
Recall@1	0.999	0.812
Recall@3	1.000	0.873
Recall@5	1.000	0.884
Recall@10	1.000	0.891
MRR	0.999	–
EM	–	0.717
F1	–	0.769
N	10,570	1,000


⸻

💬 Interpreting the Results
	•	FAISS performs near-perfectly on local retrieval tasks (ideal baseline).
	•	Pinecone reflects production-grade results with cloud API latency and real-world scaling.
	•	Azure AI Search (optional) can later be tested with hybrid text + vector retrieval.

⸻

📈 Optional Extensions
	•	Add latency and cost metrics (e.g., ms/query, $/month).
	•	Include Azure AI Search comparison.
	•	Extend to larger datasets (SQuAD train split).

⸻

✅ Team Reproducibility

Every teammate can open the notebooks directly in VSCode → Jupyter Mode and execute sequentially.

Ensure the folder paths (../Data/...) remain consistent. All results (FAISS index, embeddings, and evaluation JSONs) will save automatically under ../Data/squad_prepared/.

⸻

Authors: Team RAG Benchmark – University of Louisville 2025