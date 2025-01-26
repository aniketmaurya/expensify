# Open expense tracker

The project is a simple expense tracker that allows you to track your expenses by taking a picture of your receipt.


## Installation

1. Clone the repository
```bash
git clone https://github.com/aniketmaurya/expensify.git
cd expensify
```

2. Install maestro
```bash
git clone https://github.com/roboflow/maestro.git
cd maestro
pip install roboflow .
```


## Retrain the model (optional)

1. Update the dataset in the `data` folder
2. Run the `train.ipynb` file


## Deploy the model

### Run the server

The following command deploys the model as a server on port 8000.

```bash
python server.py
```

### Query the server

Send a receipt image to the server and get the total amount spent.

```python
import base64
import requests

url = "http://127.0.0.1:8000/predict"
image_path = "test.jpg"

with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post(url, json={"text": "<VQA>Given the following receipt, extract the total amount spent.", "image_data": encoded_string})
print(response.json())
```

**Output:**
```json
{
  "total": 26.15
}
```
