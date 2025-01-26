import base64
import requests

url = "http://127.0.0.1:8000/predict"
image_path = "hard-test.jpg"

with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post(
    url,
    json={
        "text": "<VQA>Given the following receipt, extract the total amount spent.",
        "image_data": encoded_string,
    },
)
print(response.json())
