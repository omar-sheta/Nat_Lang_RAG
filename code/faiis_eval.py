#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS search + SQuAD retrieval evaluation (per‑doc search, semantic match)
"""

import argparse, json, math, os, re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


# --- FIX: Added path to the .npy file ---
INDEX_PATH   = "../Data/squad_prepared/faiss_ip.index"
IDS_CSV      = "../Data/squad_prepared/faiss_ids.csv"
CHUNKS_CSV   = "../Data/squad_prepared/processed_chunks.csv"
EMBEDDINGS_PATH = "../Data/squad_prepared/embeddings.npy"
SQUAD_JSON   = "../Data/SQuAD/dev-v1.1.json"
SAMPLE_INFO  = "../Data/squad_prepared/sample_indices.json"
MODEL_NAME = "intfloat/e5-base-v2"
SIM_THRESHOLD = 0.66 # Cosine similarity threshold for semantic match

def load_index(path):
    try:
        with open(path, "rb") as f:
            return faiss.deserialize_index(f.read())
    except Exception:
        return faiss.read_index(path)

def _norm_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().replace("\u00a0"," ")
    return re.sub(r"\s+"," ", s).strip()

def _chunk_contains_answer_string_match(chunk_text: str, answers: List[str]) -> bool:
    ct = _norm_text(chunk_text)
    for a in answers:
        if a and _norm_text(a) in ct:
            return True
    return False

def embed_query(q: str, model: SentenceTransformer) -> np.ndarray:
    text = f"query: {q}"
    e = model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    return e

# --- Evaluation function (per‑doc FAISS over SQuAD) ---
def evaluate_retrieval(
    joined_df: pd.DataFrame,
    all_embeddings: np.ndarray,
    model: SentenceTransformer,
    squad_path: str,
    sample_info_path: str,
    ks: List[int],
    *,
    faiss_index: faiss.Index = None,
    global_eval: bool = False,
    string_only: bool = False,
) -> Dict[str, Any]:

    with open(squad_path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    # --- Load sampled titles (same as your script) ---
    sampled_titles = None
    if os.path.exists(sample_info_path):
        try:
            info = json.load(open(sample_info_path, "r", encoding="utf-8"))
            idxs = set(info.get("sampled_indices", []))
            titles = []
            for i, art in enumerate(cuad.get("data", [])):
                if i in idxs:
                    titles.append(art.get("title",""))
            sampled_titles = set(titles)
        except Exception:
            sampled_titles = None

    # --- Create mapping: title -> list of FAISS/DataFrame indices (for per-doc mode) ---
    by_title = {}
    if not global_eval:
        for i, row in joined_df.iterrows():
            # i is the integer index (0, 1, 2...) which matches the FAISS index
            by_title.setdefault(row["contract_title"], []).append(i)

    # --- Load all queries (same as your script) ---
    queries = []  # (question, answers[], title)
    for art in cuad.get("data", []):
        title = art.get("title","")
        if sampled_titles is not None and title not in sampled_titles:
            continue
        for para in art.get("paragraphs", []):
            for qa in para.get("qas", []):
                qtext = qa.get("question","")
                ans_texts = [a.get("text","") for a in qa.get("answers", []) if a.get("text","")]
                if qtext and ans_texts:
                    queries.append((qtext, ans_texts, title))

    if not queries:
        return {"error": "No queries found. Check CUAD path or sampling."}

    ks_sorted = sorted(set(ks))
    max_k = max(ks_sorted)
    hits_at_k = {k: 0 for k in ks_sorted}
    ndcg_at_k = {k: 0.0 for k in ks_sorted}
    mrr_total = 0.0
    n = 0 # Total valid queries

    mode = "GLOBAL" if global_eval else "PER-DOC"
    print(f"Running evaluation for {len(queries)} queries in {mode} mode...")
    
    # --- Process one query at a time ---
    for q, ans_list, title in tqdm(queries):
        # 1) Embed the query
        qv = embed_query(q, model)

        # 2) Get candidate indices
        if global_eval:
            # search the full FAISS index
            if faiss_index is None:
                raise ValueError("global_eval=True requires faiss_index.")
            k_to_search = max_k
            D, I = faiss_index.search(qv, k_to_search)
            candidate_indices = list(I[0])
        else:
            # per-doc: restrict to this article’s chunks
            valid_indices = by_title.get(title)
            if not valid_indices:
                continue  # no chunks found for this title
            contract_embeddings = all_embeddings[valid_indices]
            d = contract_embeddings.shape[1]
            temp_index = faiss.IndexFlatIP(d)
            temp_index.add(contract_embeddings)
            k_to_search = min(max_k, len(valid_indices))
            D, I_local = temp_index.search(qv, k_to_search)
            candidate_indices = [valid_indices[i] for i in I_local[0]]

        # 3) Prepare semantic match if enabled
        if not string_only:
            gold_answer_embeddings = model.encode(
                ans_list, normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")

        # 7. Check for relevance (string + semantic)
        first_rel_rank = None
        for rank_pos, global_idx in enumerate(candidate_indices, start=1):
            row = joined_df.iloc[global_idx]
            chunk_text = row["chunk_text"]
            chunk_embedding = all_embeddings[global_idx]  # normalized

            # Method 1: String match (fast, precise)
            is_relevant = _chunk_contains_answer_string_match(chunk_text, ans_list)

            # Method 2: Semantic match (optional)
            if not is_relevant and not string_only:
                similarities = np.dot(gold_answer_embeddings, chunk_embedding)
                if np.max(similarities) > SIM_THRESHOLD:
                    is_relevant = True
            
            if is_relevant:
                if first_rel_rank is None:
                    first_rel_rank = rank_pos
                # Don't break; we need this for nDCG (though this simple
                # version only uses first_rel_rank. Breaking is fine.)
                break # We only care about the *first* hit for R@k and MRR

        # --- End of relevance check ---
        
        n += 1 # This query is valid and was processed
        if first_rel_rank is not None:
            mrr_total += 1.0 / first_rel_rank
            for k in ks_sorted:
                if first_rel_rank <= k:
                    hits_at_k[k] += 1
                    # nDCG@k: for binary relevance, it's 1/log2(rank+1) if hit, 0 otherwise
                    ndcg_at_k[k] += 1.0 / math.log2(first_rel_rank + 1)

    # --- End of loop ---

    results = {
        "total_queries": n,
        "R@k": {k: hits_at_k[k] / max(1,n) for k in ks_sorted},
        "MRR": mrr_total / max(1,n),
        "nDCG@k": {k: ndcg_at_k[k] / max(1,n) for k in ks_sorted},
        "counts": {k: {"hits": hits_at_k[k], "total": n} for k in ks_sorted},
    }
    return results

# --- (inspect_squad) ---
def inspect_squad(squad_path: str, sample_info_path: str):
    with open(squad_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sampled_titles = None
    if os.path.exists(sample_info_path):
        try:
            info = json.load(open(sample_info_path,"r",encoding="utf-8"))
            idxs = set(info.get("sampled_indices", []))
            titles = []
            for i, art in enumerate(raw.get("data", [])):
                if i in idxs:
                    titles.append(art.get("title",""))
            sampled_titles = set(titles)
        except Exception:
            pass

    docs = 0; paras = 0; qas = 0
    q_texts = []; ans_len = []
    for art in raw.get("data", []):
        title = art.get("title","")
        if sampled_titles and title not in sampled_titles:
            continue
        docs += 1
        for para in art.get("paragraphs", []):
            paras += 1
            for qa in para.get("qas", []):
                qas += 1
                q_texts.append(qa.get("question",""))
                for a in qa.get("answers", []):
                    ans_len.append(len(a.get("text","") or ""))

    print("=== SQuAD inspection ===")
    print("Docs (sampled):", docs)
    print("Paragraphs:", paras)
    print("QAs:", qas)
    if q_texts:
        vc = pd.Series(q_texts).value_counts().head(15)
        print("\nTop question texts (≈ categories):")
        for k, v in vc.items():
            print(f"  {v:5d}  {k}")
    if ans_len:
        print("\nAnswer length (chars): mean={:.1f}  p50={:.1f}  p90={:.1f}".format(
            float(np.mean(ans_len)), float(np.median(ans_len)), float(np.percentile(ans_len,90))
        ))
    print("Tip: categories ≈ unique question texts (e.g., 'Limitation of Liability').")


def main():
    parser = argparse.ArgumentParser(description="FAISS search + SQuAD evaluation + inspection (E5)")
    parser.add_argument("--query", type=str, default=None, help="Single query to search")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--eval", action="store_true", help="Run retrieval evaluation")
    parser.add_argument("--inspect", action="store_true", help="Print SQuAD dataset stats for the indexed subset")
    
    # --- THIS IS THE FIX ---
    parser.add_argument("--squad", type=str, default=SQUAD_JSON)
    # --- END OF FIX ---
    
    parser.add_argument("--sample_info", type=str, default=SAMPLE_INFO)
    parser.add_argument("--k_list", type=str, default="1,3,5,10")
    parser.add_argument("--global_eval", action="store_true",
                        help="Search the full FAISS index (no per-doc restriction).")
    parser.add_argument("--string_only", action="store_true",
                        help="Count relevance only via exact string match; disable semantic fallback.")
    args = parser.parse_args()

    if args.inspect:
        inspect_squad(args.squad, args.sample_info)

    print("Loading FAISS index, CSVs, and embeddings...")
    index = load_index(INDEX_PATH) # Keep this for interactive search
    
    all_embeddings = np.load(EMBEDDINGS_PATH)
    
    ids   = pd.read_csv(IDS_CSV)
    chunks= pd.read_csv(CHUNKS_CSV)
    joined = ids.merge(
        chunks,
        on=["chunk_id","contract_title","contract_title_safe","paragraph_index","chunk_index"],
        how="left"
    )

    print("Loading E5 model...")
    model = SentenceTransformer(MODEL_NAME)

    if args.eval:
        ks = [int(x) for x in args.k_list.split(",") if x.strip().isdigit()] or [1,3,5,10]
        print(f"Running evaluation on SQuAD: {args.squad} | Ks={ks} | "
              f"global_eval={args.global_eval} | string_only={args.string_only}")
        res = evaluate_retrieval(
            joined, all_embeddings, model, args.squad, args.sample_info, ks,
            faiss_index=index, global_eval=args.global_eval, string_only=args.string_only
        )
        
        if "error" in res:
            print("ERROR:", res["error"])
        else:
            print("\n=== Retrieval Evaluation (Per‑doc search, semantic‑match) ===")
            print(f"Total queries: {res['total_queries']}")
            print("Recall@k:")
            for k, v in res["R@k"].items():
                print(f"  R@{k}: {v:.3f}")
            print(f"MRR: {res['MRR']:.3f}")
            print("nDCG@k:")
            for k, v in res["nDCG@k"].items():
                print(f"  nDCG@{k}: {v:.3f}")
            print("Counts:")
            for k, c in res["counts"].items():
                print(f"  hits@{k}: {c['hits']} / {c['total']}")
        return

    # --- Interactive search (uses the global index, which is fine) ---
    if args.query:
        qv = embed_query(args.query, model)
        D, I = index.search(qv, args.topk)
        print("\nTop results:")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
            row = joined.iloc[idx]
            print(f"\n[{rank}] cosine={score:.4f}  id={row['chunk_id']}  title={row['contract_title']}")
            txt = str(row['chunk_text'])
            print(txt[:700].replace("\n"," ") + ("…" if len(txt)>700 else ""))
        return

    while True:
        try:
            q = input("\nQuery (blank to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        qv = embed_query(q, model)
        D, I = index.search(qv, 5)
        print("\nTop results:")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
            row = joined.iloc[idx]
            print(f"\n[{rank}] cosine={score:.4f}  id={row['chunk_id']}  title={row['contract_title']}")
            txt = str(row['chunk_text'])
            print(txt[:700].replace("\n"," ") + ("…" if len(txt)>700 else ""))

if __name__ == "__main__":
    main()




    # python faiis_eval.py --eval --squad ../Data/SQuAD/dev-v1.1.json --k_list 1,3,5,10 --global_eval --string_only