PROJECT : Machine Learning Model - Load & Use (Pickle Based)

-----------------------------------------
HOW TO RUN THE PROJECT
-----------------------------------------

1) Clone or download the project folder.

2) Create a virtual environment (optional but recommended).
   Windows  :   venv\Scripts\activate
   Linux/Mac:   source venv/bin/activate

3) Install requirements using:
       pip install -r requirements.txt
4) Create models folder and data folder in root dir.

5) Downlad kaggle credit fraud detection dataset "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" and place creditcard.csv in data folder.

6) Open Jupyter Notebook to explore:
       jupyter notebook
   Run:
       notebooks/exploration.ipynb

7) Go to root directory and run:
     uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

8) In root directory in other terminal run:
     streamlit run ui/app.py


-----------------------------------------
PROJECT STRUCTURE
-----------------------------------------

project/
    ├── api/
    ├── data/                   - Dataset
    ├── notebooks/              - EDA + training notebooks
    ├── source/                    - Python modules for processing
    ├── model/              - Final trained model file
    ├── requirements.txt        - All dependencies
    └── README.txt              


-----------------------------------------
PROJECT IDEA / MOTIVATION
-----------------------------------------

Credit card fraud detection is challenging because millions of
transactions occur daily. Manual identification is impossible.
This project builds an ML-based system that automatically detects
fraudulent transactions based on historical patterns.

-----------------------------------------
WORKFLOW SUMMARY
-----------------------------------------

1) DATA EXPLORATION
   - Loaded dataset, checked imbalance.
   - Visualized distributions, correlation.

2) PREPROCESSING
   - Cleaned data, handled imbalance (SMOTE).
   - Scaled numerical features.
   - Train-test split created.

3) MODEL TRAINING
   - Tried multiple ML algorithms.
   - Tuned hyperparameters.
   - Selected best model based on ROC-AUC score.

4) EVALUATION METRICS
   Accuracy
   Precision / Recall
   F1 Score
   ROC-AUC Curve

5) SAVING MODEL
   Model saved using pickle for offline use:
       pickle.dump(model, open("model.pkl","wb"))


-----------------------------------------
RESULTS (Replace with your actual values)
-----------------------------------------

Accuracy   : 99+
ROC-AUC    : 0.99+
Recall     : High (detects fraud strongly)

-----------------------------------------
END OF FILE
-----------------------------------------
