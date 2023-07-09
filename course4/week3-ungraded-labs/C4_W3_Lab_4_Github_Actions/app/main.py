import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

# Test comment
from sklearn.exceptions import InconsistentVersionWarning
#warnings.simplefilter("error", InconsistentVersionWarning)




try:
    est = pickle.loads("models/wine-95.pkl")
except InconsistentVersionWarning as w:
    print("INCONSISTENT VERSIONS")
    print(w.original_sklearn_version)



app = FastAPI(title="Predicting Wine Class with batching")

# Open classifier in global scope
with open("models/wine-95.pkl", "rb") as file:
    clf = pickle.load(file)


class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}
