from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile
from torch.nn import Softmax

from model import transform, load_model

model = load_model()
app = FastAPI()


@app.post("/analyze")
async def analyze_mri(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        tensor = transform(image)

        with torch.no_grad():
            logits = model(tensor)
            softmax = Softmax()
            probs = softmax(logits)

        return {
            "logits": logits.tolist(),
            "predictions": probs.tolist(),
            "predicted_label": int(torch.argmax(probs)),
        }
    except Exception as e:
        raise HTTPException(500, f'Error: {e}')
