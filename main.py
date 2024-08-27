from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cos_similarity_search(v, V):
    v = np.array(v)
    cos = ((V*v).sum(axis=1))**2 / ((V**2).sum(axis=1) * (v**2).sum())
    near = (-cos).argsort()
    ret = []
    for x in near:
        ret.append([x, cos[x]])
    return ret


with open("./data/laws_with_vectors.pickle", 'rb') as f:
    laws = pickle.load(f)

laws_name = [{'id': law['id'], 'name':law['name']} for law in laws]
vectors = np.array([law['vector'] for law in laws])


@app.get("/laws")
async def root():
    return laws_name


@app.get("/laws/{law_id}")
async def read_law(law_id: int):
    _law = next(item for item in laws if item["id"] == law_id)
    law = dict(_law)
    law.pop('vector')
    return law


@app.get("/near/{law_id}")
async def near(law_id: int):
    law = next(item for item in laws if item["id"] == law_id)
    idxs = cos_similarity_search(law['vector'], vectors)
    near_laws = [{**laws_name[idx[0]], 'score':idx[1], 'ranking':i}
                 for i, idx in enumerate(idxs)]
    return near_laws[:100]
