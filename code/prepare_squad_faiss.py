#!/usr/bin/env python3
"""
SQuAD → token-chunked embeddings (E5) → FAISS (cosine) index
High-retrieval prep with offsets + dataset inspection.

Requirements:
  pip install sentence-transformers==3.* transformers==4.* faiss-cpu pandas numpy tqdm

Example:
  python prepare_squad_faiss.py \
    --input ../Data/dev-v1.1.json \
    --out_dir ../Data/squad_prepared \
    --num_docs 100 \
    --max_tokens 384 \
    --stride 128 \
    --batch_size 96
"""

import argparse, json, os, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss

faiss.omp_set_num_threads(1)  # macOS stability

# -------------------------
# Utilities
# -------------------------
def clean_text_keep_structure(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def make_safe_id(s: str) -> str:
    """Return a filesystem-safe ID (used for chunk names)."""
    import re
    if s is None:
        return "UNTITLED"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]

def load_squad_docs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        squad_raw = json.load(f)
    docs = []
    for art in squad_raw.get("data", []):
        title = art.get("title", "UNKNOWN_TITLE")
        paras = art.get("paragraphs", [])
        contexts = [p.get("context", "") for p in paras]
        full_text = "\n\n".join(contexts)
        docs.append({"title": title, "paragraphs": contexts, "text": full_text})
    return docs, squad_raw

def build_token_chunker(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def chunk_text(text: str, max_tokens: int = 384, stride: int = 128) -> List[Tuple[str,int,int]]:
        """Return list of (chunk_text, char_start, char_end) in ORIGINAL 'text'."""
        if not text:
            return []
        enc = tok.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        out = []
        step = max(1, max_tokens - stride)
        for start in range(0, len(ids), step):
            end = min(start + max_tokens, len(ids))
            if start >= end:
                break
            win_ids = ids[start:end]
            win_off = offsets[start:end]
            c_start = next((a for a,b in win_off if (b-a)>0), None)
            c_end   = next((b for a,b in reversed(win_off) if (b-a)>0), None)
            if c_start is None or c_end is None:
                continue
            txt = tok.decode(win_ids, skip_special_tokens=True)
            out.append((txt, int(c_start), int(c_end)))
            if end >= len(ids):
                break
        return out
    return chunk_text

def embed_texts_e5_passages(texts: List[str], batch_size: int = 96) -> np.ndarray:
    model_name = "intfloat/e5-base-v2"
    model = SentenceTransformer(model_name)
    # E5 expects "passage: ..." for corpus entries
    passages = ["passage: " + (t if isinstance(t, str) else "") for t in texts]
    emb = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine
    ).astype("float32")
    return emb

def build_faiss_cosine(emb: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(emb)  # harmless if already normalized
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

# -------------------------
# Inspection (lightweight)
# -------------------------
def quick_dataset_report(squad_raw: Dict[str, Any], sampled_titles: set) -> Dict[str, Any]:
    n_docs = 0
    n_paras = 0
    n_qas = 0
    q_texts = []
    ans_len = []
    for art in squad_raw.get("data", []):
        title = art.get("title","")
        if sampled_titles and title not in sampled_titles:
            continue
        n_docs += 1
        for para in art.get("paragraphs", []):
            n_paras += 1
            for qa in para.get("qas", []):
                n_qas += 1
                q_texts.append(qa.get("question",""))
                for a in qa.get("answers", []):
                    t = a.get("text","") or ""
                    ans_len.append(len(t))
    top_qs = pd.Series(q_texts).value_counts().head(10).to_dict() if q_texts else {}
    return {
        "sampled_docs": n_docs,
        "sampled_paragraphs": n_paras,
        "sampled_qas": n_qas,
        "top_question_texts": top_qs,
        "answer_length_mean": float(np.mean(ans_len)) if ans_len else 0.0,
        "answer_length_p50": float(np.median(ans_len)) if ans_len else 0.0,
        "answer_length_p90": float(np.percentile(ans_len, 90)) if ans_len else 0.0,
    }

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to SQuAD JSON (v1.1 or v2.0 dev/train)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--num_docs", type=int, default=100, help="How many documents to sample")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--min_chunk_chars", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SQuAD:", args.input)
    docs, squad_raw = load_squad_docs(args.input)
    print(f"Found {len(docs)} documents.")

    # Sample
    if args.num_docs > 0 and args.num_docs < len(docs):
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(docs))[:args.num_docs]
        sampled = [docs[i] for i in idx]
        sampled_idx = idx.tolist()
    else:
        sampled = docs
        sampled_idx = list(range(len(docs)))

    with open(out_dir / "sample_indices.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "num_docs": args.num_docs,
            "sampled_indices": sampled_idx,
            "model": "intfloat/e5-base-v2",
            "max_tokens": args.max_tokens,
            "stride": args.stride,
        }, f, indent=2)

    # Chunker
    chunker = build_token_chunker("intfloat/e5-base-v2")

    rows = []
    print("Cleaning and chunking…")
    for d_i, doc in enumerate(tqdm(sampled, total=len(sampled))):
        title_raw = doc["title"] or f"DOCUMENT_{d_i}"
        title_safe = make_safe_id(title_raw)
        cleaned = clean_text_keep_structure(doc["text"])
        paragraphs = [p for p in cleaned.split("\n\n") if p.strip()]

        # track paragraph absolute offsets within cleaned doc
        para_starts = []
        pos = 0
        for p in paragraphs:
            j = cleaned.find(p, pos)
            if j == -1: j = pos
            para_starts.append(j)
            pos = j + len(p)

        for p_idx, para in enumerate(paragraphs):
            chunks = chunker(para, max_tokens=args.max_tokens, stride=args.stride)  # (txt, c_start, c_end)
            for c_idx, (chunk_txt, c_start, c_end) in enumerate(chunks):
                if len(chunk_txt) < args.min_chunk_chars:
                    continue
                base_id = f"{title_safe}_{p_idx}_{c_idx}"
                row_id = hashlib.md5(base_id.encode("utf-8")).hexdigest()
                rows.append({
                    "row_id": row_id,
                    "chunk_id": base_id,
                    "contract_title": title_raw,
                    "contract_title_safe": title_safe,
                    "paragraph_index": p_idx,
                    "chunk_index": c_idx,
                    "chunk_text": chunk_txt,
                    "para_char_start": int(c_start),
                    "para_char_end": int(c_end),
                    "doc_char_start": int(para_starts[p_idx] + c_start),
                    "doc_char_end": int(para_starts[p_idx] + c_end),
                })

    if not rows:
        raise RuntimeError("No chunks produced; adjust token limits.")

    df = pd.DataFrame(rows)
    chunks_csv = out_dir / "processed_chunks.csv"
    df.to_csv(chunks_csv, index=False)
    print(f"Saved chunks → {chunks_csv} (rows={len(df)})")

    # Embeddings (E5 passages)
    texts = df["chunk_text"].astype(str).tolist()
    print(f"Embedding {len(texts)} chunks…")
    emb = embed_texts_e5_passages(texts, batch_size=args.batch_size)
    emb = np.ascontiguousarray(emb.astype("float32"))
    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, emb)
    print(f"Saved embeddings → {emb_path} (shape={emb.shape})")

    # FAISS (cosine via IP on normalized vectors)
    print("Building FAISS (cosine)…")
    index = build_faiss_cosine(emb)
    faiss_path = out_dir / "faiss_ip.index"
    with open(faiss_path, "wb") as f:
        f.write(faiss.serialize_index(index))
    print(f"Saved FAISS index → {faiss_path} | dim={emb.shape[1]} | ntotal={index.ntotal}")

    # ID map
    ids_csv = out_dir / "faiss_ids.csv"
    df[["row_id","chunk_id","contract_title","contract_title_safe","paragraph_index","chunk_index"]].to_csv(ids_csv, index=False)
    print(f"Saved ID map → {ids_csv}")

    # Dataset inspection report (on sampled subset)
    sampled_titles = {d["title"] for d in sampled}
    report = quick_dataset_report(squad_raw, sampled_titles)
    with open(out_dir / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved dataset_report.json")
    print("SQuAD FAISS prep complete ✅")

if __name__ == "__main__":
    main()