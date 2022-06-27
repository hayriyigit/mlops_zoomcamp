import requests

ride = {
    "PULocationID": 40,
    "DOLocationID": 50,
    "trip_distance": 40,
}

url = 'http://localhost:9696/predict'
respose = requests.post(url, json = ride)
print(respose.json())