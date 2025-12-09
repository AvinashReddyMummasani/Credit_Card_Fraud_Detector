from xgboost import XGBClassifier
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler,QuantileTransformer

model_path=os.path.join(os.getcwd(),"models",'model.pkl')
std_scalar=os.path.join(os.getcwd(),"source",'stscalar.pkl')
qe_transformer=os.path.join(os.getcwd(),"source",'qe.pkl')
with open(model_path,'rb') as f:
    model=pickle.load(f)

with open(std_scalar,'rb') as f:
    sc=pickle.load(f)

with open(qe_transformer,'rb') as f:
    qe=pickle.load(f)

def predict(X):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X = sc.transform(X)
    X = qe.transform(X)
    preds = model.predict(X)

    if hasattr(preds, "tolist"):
        out = preds.tolist()
    else:
        out = list(preds)

    if len(out) == 1:
        v = out[0]
        if hasattr(v, "item"):
            return v.item()
        return v
    return out