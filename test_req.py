import requests
import json

resp = requests.post("http://34.64.237.147:5000/predict",
                     files={"file": open('test_picture.jpeg','rb')})

print(resp.status_code)
print(resp.json())
