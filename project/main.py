from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import joblib

import pandas as pd

import uvicorn
 



# 1. Initialize the App

app = FastAPI(title="SwiftLoan AI API", description="XGBoost Powered Loan Prediction")



# 2. Enable CORS (Crucial for the HTML frontend to work)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],  # EXACT frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# 3. Load the winning model

try:

    model = joblib.load('random_forest_model.pkl')

except FileNotFoundError:

    print("❌ Error: loan_model.pkl not found in current directory!")



# 4. Define the Input Data Model

class LoanRequest(BaseModel):

    no_of_dependents: int

    education: str

    self_employed: str

    income_annum: int

    loan_amount: int

    loan_term: int

    cibil_score: int

    residential_assets_value: int

    commercial_assets_value: int

    luxury_assets_value: int

    bank_asset_value: int



@app.get("/")

def home():

    return {"status": "Online", "model": "RF 0.97 F1"}



@app.post("/predict")

def predict_loan(data: LoanRequest):

    try:

        # Convert incoming JSON to a DataFrame

        raw_data = data.dict()

        df = pd.DataFrame([raw_data])

       

        # --- Preprocessing ---

        # 1. Clean and Map strings to numbers

        df['education'] = df['education'].str.strip().map({'Not Graduate': 0, 'Graduate': 1})

        df['self_employed'] = df['self_employed'].str.strip().map({'No': 0, 'Yes': 1})

       

        # 2. Aggregate Assets (Matching your Notebook logic)

        df['Movable_assets'] = df['bank_asset_value'] + df['luxury_assets_value']

        df['Immovable_assets'] = df['residential_assets_value'] + df['commercial_assets_value']

       

        # 3. Select and Order Features (MUST match the order used during xgb.fit)

        # Assuming the order was: no_of_dependents, education, self_employed, income_annum,

        # loan_amount, loan_term, cibil_score, Movable_assets, Immovable_assets

        feature_order = [

            'no_of_dependents', 'education', 'self_employed', 'income_annum',

            'loan_amount', 'loan_term', 'cibil_score',

            'Movable_assets', 'Immovable_assets'

        ]

       

        features = df[feature_order]

       

        # 4. Make Prediction

        prediction = model.predict(features)

        status = "Approved" if int(prediction[0]) == 1 else "Rejected"

       

        return {

            "prediction": status,

            "status_code": int(prediction[0])

        }

       

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)