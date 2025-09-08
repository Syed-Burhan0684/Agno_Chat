# app/db.py
import os
import json
import numpy as np
import faiss
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')

# Where to persist per-user indices & metadata
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(BASE_DIR, '..', 'data', 'indexes')
os.makedirs(INDEX_DIR, exist_ok=True)


class UserStore:
    """
    Per-user FAISS HNSW store. Stores index, docs, and metadata in memory and persists to disk.
    """
    def __init__(self, user_id: str, dim: int, m: int = 32, ef_construction: int = 200):
        self.user_id = user_id
        self.dim = dim
        # HNSW index for inner-product (we store normalized vectors)
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efConstruction = ef_construction
        self.docs: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.count = 0

    def add(self, vectors: np.ndarray, docs: List[str], metas: List[Dict[str, Any]]):
        """
        vectors: (N, dim) float32 and should be L2-normalized before calling
        """
        if vectors is None or len(vectors) == 0:
            return
        self.index.add(vectors)
        self.docs.extend(docs)
        self.meta.extend(metas)
        self.count += vectors.shape[0]

    def search(self, qvec: np.ndarray, top_k: int = 3, ef_search: int = 64):
        if self.count == 0:
            return []
        # set efSearch for performance/quality tradeoff
        try:
            self.index.hnsw.efSearch = ef_search
        except Exception:
            pass
        D, I = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({
                'text': self.docs[idx],
                'meta': self.meta[idx],
                'score': float(score)
            })
        return results

    def persist(self):
        """Write FAISS index + metadata to disk"""
        idx_path = os.path.join(INDEX_DIR, f"{self.user_id}.index")
        meta_path = os.path.join(INDEX_DIR, f"{self.user_id}_meta.json")
        # faiss.write_index requires a real Index (HNSW is fine)
        faiss.write_index(self.index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({'docs': self.docs, 'meta': self.meta}, f, ensure_ascii=False, indent=2)


class VectorDB:
    """
    Loads transactions from transactions.json, builds/loads per-user HNSW indices,
    and exposes retrieve/add operations.
    """
    def __init__(self, path_json: str, dim_expected: int | None = None):
        self.users: Dict[str, UserStore] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.dim = None

        # Load JSON source
        if not os.path.exists(path_json):
            # No seed file -> empty DB
            return

        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Prepare per-user lists for embedding (batch embeddings)
        per_user_texts: Dict[str, List[str]] = {}
        per_user_meta: Dict[str, List[Dict[str, Any]]] = {}
        for user in data.get('users', []):
            uid = user['user_id']
            self.profiles[uid] = user.get('profile', {})
            per_user_texts.setdefault(uid, [])
            per_user_meta.setdefault(uid, [])
            for t in user.get('transactions', []):
                per_user_texts[uid].append(t['text'])
                per_user_meta[uid].append({'user_id': uid, 'doc_id': t.get('id'), 'amount': t.get('amount')})

        # For each user, try to load persisted index; otherwise compute embeddings and build index
        for uid, texts in per_user_texts.items():
            idx_path = os.path.join(INDEX_DIR, f"{uid}.index")
            meta_path = os.path.join(INDEX_DIR, f"{uid}_meta.json")
            if os.path.exists(idx_path) and os.path.exists(meta_path):
                # load index + meta
                idx = faiss.read_index(idx_path)
                # ensure dim consistent
                dim = idx.d
                store = UserStore(uid, dim)
                store.index = idx
                with open(meta_path, 'r', encoding='utf-8') as f:
                    jm = json.load(f)
                store.docs = jm.get('docs', [])
                store.meta = jm.get('meta', [])
                store.count = len(store.docs)
                self.users[uid] = store
                self.dim = dim if self.dim is None else self.dim
                continue

            # else: create embeddings in a single batch for this user
            if not texts:
                continue
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            vecs = np.array([r.embedding for r in resp.data], dtype='float32')
            # normalize for inner product (cosine)
            faiss.normalize_L2(vecs)
            dim = vecs.shape[1]
            # create HNSW UserStore
            store = UserStore(uid, dim)
            metas = per_user_meta[uid]
            store.add(vecs, texts, metas)
            # persist
            store.persist()
            self.users[uid] = store
            self.dim = dim if self.dim is None else self.dim

    def get_profile(self, user_id: str):
        return self.profiles.get(user_id)

    def retrieve(self, user_id: str, query: str, top_k: int = 3, ef_search: int = 64):
        """
        Return list of {text, meta, score} for top_k candidates for this user.
        """
        store = self.users.get(user_id)
        if not store:
            return []
        resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
        q = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(q)
        return store.search(q, top_k=top_k, ef_search=ef_search)

    def add_transaction(self, user_id: str, text: str):
        """
        Add a new transaction: compute embedding, add to user's index, persist.
        """
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        v = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(v)
        if user_id not in self.users:
            dim = v.shape[1]
            self.users[user_id] = UserStore(user_id, dim)
            self.profiles[user_id] = {'current_balance': 0.0, 'monthly_income': 0.0, 'monthly_outgoings': 0.0}
        self.users[user_id].add(v, [text], [{'user_id': user_id, 'doc_id': 'added', 'amount': None}])
        # persist the updated index and metadata
        self.users[user_id].persist()
