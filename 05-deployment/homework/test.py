import requests
url = "http://localhost:9696/predict"
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
response = requests.post(url, json=client).json()

print(response)