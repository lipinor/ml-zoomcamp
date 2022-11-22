import requests

data = {
    "url": "http://bit.ly/mlbookcamp-pants"
}

# url = "http://localhost:8080/2015-03-31/functions/function/invocations" # Local 
url = "https://ogjia4qtta.execute-api.us-east-1.amazonaws.com/test/predict" # AWS Lambda

results = requests.post(url, json=data).json()

print(results)