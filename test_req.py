import requests
import json

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('test_picture.jpeg','rb')})

print(resp.status_code)
print(resp.json())
