import base64
import io

import torch
from litserve import LitAPI, LitServer
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_id_or_path = "aniketmaurya/receipt-model-2025"
model_id_or_path = "training/florence-2/4/checkpoints/best"
revision = "main"


class ReceiptAPI(LitAPI):
    def setup(self, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            revision=revision,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base-ft", trust_remote_code=True
        )

    def decode_request(self, request):
        text = request["text"]
        image = request["image_data"]
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image))
        inputs = self.processor(text, image, return_tensors="pt").to(self.device)
        return inputs

    def predict(self, inputs):
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
            )
        return generated_ids

    def encode_response(self, generated_ids):
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return generated_text


if __name__ == "__main__":
    api = ReceiptAPI()
    server = LitServer(api, accelerator="cpu")
    server.run(port=8000)
