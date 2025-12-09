import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="Fraud UI", layout="wide")

st.title("Fraud Detection UI")

api_url = st.text_input("API URL", "http://127.0.0.1:8000")

threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

st.header("Single Prediction")

with st.form("single"):
    text = st.text_area("Enter V1,V2,V3 ... V28,Amount")
    submit = st.form_submit_button("Predict")

if submit:
    try:
        arr = [float(x.strip()) for x in text.split(",")]
        if len(arr) != 29:
            st.error("Exactly 30 values required")
        else:
            payload = {"features": arr, "threshold": threshold}
            r = requests.post(f"{api_url}/predict", json=payload)
            st.json(r.json())
            final_result=r.json()["result"]
            if final_result:
                st.text("Fraud")
            else:
                st.text("Not Fraud")
    except:
        st.error("Invalid input")

st.header("Batch Prediction")

file = st.file_uploader("Upload CSV (no header, 30 columns)")

import math

if file:
    df = pd.read_csv(file, header=None)
    df = df.iloc[1:, :]
    st.write(df)

    if st.button("Predict Batch"):
        rows = df.values.tolist()
        clean_rows = []
        bad_rows = []

        for i, row in enumerate(rows):
            cleaned = []
            ok = True
            for v in row:
                try:
                    fv = float(v)
                except:
                    ok = False
                    break
                if not math.isfinite(fv):
                    ok = False
                    break
                cleaned.append(fv)
            if ok:
                clean_rows.append(cleaned)
            else:
                bad_rows.append(i)

        if bad_rows:
            st.error(f"Invalid numeric values in rows: {bad_rows}")
        else:
            payload = {"transactions": [{"features": r} for r in clean_rows]}
            r = requests.post(f"{api_url}/predict/batch", json=payload, timeout=120)
            resp = r.json()
            res = resp.get("result")

            if isinstance(res, dict) and "predictions" in res:
                preds = res["predictions"]
            elif isinstance(res, list):
                preds = res
            else:
                preds = []

            if len(preds) < len(clean_rows):
                preds = preds + [None] * (len(clean_rows) - len(preds))
            else:
                preds = preds[: len(clean_rows)]

            out_df = pd.DataFrame(clean_rows)
            out_df["result"] = preds

            st.dataframe(out_df)
            csv = out_df.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "predictions.csv")
            st.json(resp)
