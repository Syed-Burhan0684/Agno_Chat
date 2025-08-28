# app/db.py
import os, json
from typing import Dict, List, Any
import numpy as np
import faiss
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
INDEX_DIR = os.path.join(BASE_DIR, "data", "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)


class UserStore:
    def __init__(self, user_id: str, dim: int, m:int=32, ef_construction:int=200):
        self.user_id = user_id
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efConstruction = ef_construction
        self.docs: List[str] = []
        self.meta: List[Dict[str,Any]] = []
        self.count = 0

    def add(self, vectors: np.ndarray, docs: List[str], metas: List[Dict[str,Any]]):
        if vectors is None or vectors.shape[0] == 0: return
        self.index.add(vectors)
        self.docs.extend(docs)
        self.meta.extend(metas)
        self.count += vectors.shape[0]

    def search(self, qvec: np.ndarray, top_k: int=3, ef_search: int=64):
        if self.count == 0: return []
        try:
            self.index.hnsw.efSearch = ef_search
        except Exception:
            pass
        D, I = self.index.search(qvec, top_k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            out.append({'text': self.docs[idx], 'meta': self.meta[idx], 'score': float(score)})
        return out

    def persist(self):
        faiss.write_index(self.index, os.path.join(INDEX_DIR, f"{self.user_id}.index"))
        with open(os.path.join(INDEX_DIR, f"{self.user_id}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({'docs': self.docs, 'meta': self.meta}, f, ensure_ascii=False, indent=2)


class VectorDB:
    """
    Loads seed JSON -> builds per-user HNSW indices (persisted).
    At query-time only the query embedding is computed.
    """
    def __init__(self, path_json: str):
        self.users: Dict[str, UserStore] = {}
        self.profiles: Dict[str, Dict[str,Any]] = {}
        if not os.path.exists(path_json):
            return
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # collect per-user texts & metas
        per_user_texts: Dict[str, List[str]] = {}
        per_user_meta: Dict[str, List[Dict[str,Any]]] = {}
        for user in data.get('users', []):
            uid = user['user_id']
            self.profiles[uid] = user.get('profile', {})
            per_user_texts.setdefault(uid, [])
            per_user_meta.setdefault(uid, [])
            for t in user.get('transactions', []):
                per_user_texts[uid].append(t['text'])
                per_user_meta[uid].append({'user_id': uid, 'doc_id': t.get('id'), 'amount': t.get('amount')})

        # build or load per-user index
        for uid, texts in per_user_texts.items():
            idx_path = os.path.join(INDEX_DIR, f"{uid}.index")
            meta_path = os.path.join(INDEX_DIR, f"{uid}_meta.json")
            if os.path.exists(idx_path) and os.path.exists(meta_path):
                idx = faiss.read_index(idx_path)
                st = UserStore(uid, idx.d)
                st.index = idx
                with open(meta_path, 'r', encoding='utf-8') as f:
                    jm = json.load(f)
                st.docs = jm.get('docs', [])
                st.meta = jm.get('meta', [])
                st.count = len(st.docs)
                self.users[uid] = st
                continue
            if not texts: continue
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            vecs = np.array([r.embedding for r in resp.data], dtype='float32')
            faiss.normalize_L2(vecs)
            st = UserStore(uid, vecs.shape[1])
            st.add(vecs, texts, per_user_meta[uid])
            st.persist()
            self.users[uid] = st

    def get_profile(self, user_id: str):
        return self.profiles.get(user_id)

    def retrieve(self, user_id: str, query: str, top_k:int=3, ef_search:int=64):
        store = self.users.get(user_id)
        if not store: return []
        resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
        q = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(q)
        return store.search(q, top_k=top_k, ef_search=ef_search)

    def add_transaction(self, user_id: str, text: str):
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        v = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(v)
        if user_id not in self.users:
            self.users[user_id] = UserStore(user_id, v.shape[1])
            self.profiles[user_id] = {'current_balance': 0.0, 'monthly_income': 0.0, 'monthly_outgoings': 0.0}
        self.users[user_id].add(v, [text], [{'user_id': user_id, 'doc_id':'added', 'amount': None}])
        self.users[user_id].persist()
