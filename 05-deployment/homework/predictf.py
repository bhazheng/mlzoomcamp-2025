import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="predict")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class PredictResponse(BaseModel):
    probability: float

def predict_single(customer: Customer):

    customer_dict = customer.dict()
    result = pipeline.predict_proba([customer_dict])[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer)

    # return PredictResponse(probability=prob)
    

    return {
        "probability": prob  # Kunci harus string
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)